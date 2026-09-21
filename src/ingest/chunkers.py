"""Chunking strategies — built here, CHOSEN in Phase 4 by benchmark.

Nothing in this file decides anything. All three strategies get implemented,
and `chunking.strategy` in configs/experiment.yaml picks between them. The
winner is whichever one produces the best retrieval metrics on the golden set.

────────────────────────────────────────────────────────────────────────
THE DESIGN DECISION THAT SHAPES EVERY SIGNATURE BELOW
────────────────────────────────────────────────────────────────────────
Splitters are PURE TEXT FUNCTIONS. They take a string, they return character
spans. They never see a page number, a breadcrumb, or a book title.

Metadata is bound afterwards, by `bind_metadata()`, which maps character
offsets back onto pages. Two payoffs:

  * Your splitting logic is testable with a one-line string and no PDF.
  * A bug in chunking cannot corrupt citation data, because chunking has no
    access to it.

────────────────────────────────────────────────────────────────────────
THREE THINGS THE DEFAULTS ARE QUIETLY ASSUMING
────────────────────────────────────────────────────────────────────────
1. TOKENS, NOT CHARACTERS — and WHOSE tokens?
   `chunk_size: 512` is in tokens, which requires choosing a tokenizer. We use
   tiktoken's cl100k_base. But our generators are Llama, Qwen and Gemma, whose
   tokenizers differ — a "512-token" chunk here may be ~550 under Llama.
   So cl100k is a consistent YARDSTICK, not ground truth. It matters only that
   every strategy is measured with the same ruler.

2. WHY OVERLAP EXISTS.
   Not redundancy for its own sake. An idea that straddles a boundary is
   otherwise retrievable from neither side: the first chunk has the setup
   without the conclusion, the second has the conclusion without the setup.
   Overlap buys insurance against a boundary landing mid-argument, and costs
   storage plus near-duplicate hits in the result list. `chunk_overlap: 64`
   (~12%) is a convention, NOT a measurement.

3. 512 IS UNTESTED.
   It is inherited from common practice, not derived from this corpus.
   Textbook paragraphs are long and arguments span several of them, so a
   larger chunk may well win. That is a Phase 4 sweep, not a decision.

────────────────────────────────────────────────────────────────────────
WHAT LIVES HERE
────────────────────────────────────────────────────────────────────────
Plumbing:    Chunk/Span types, tokenizer, sentence splitter, metadata binding,
             strategy registry.
Algorithms:  the three split_* functions — where the boundaries actually go.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Protocol

# ───────────────────────────────────────────────────────────────────────
# Types
# ───────────────────────────────────────────────────────────────────────

@dataclass
class Span:
    """A slice of the source text, with offsets preserved.

    Offsets are NOT decoration — they are how `bind_metadata()` recovers which
    page(s) a chunk came from. A splitter that returns text without correct
    offsets produces chunks that cannot be cited.

    Invariant every splitter must uphold: ``source[span.start:span.end]``
    reconstructs ``span.text``, modulo whitespace stripping at the edges.
    """
    text: str
    start: int
    end: int


@dataclass
class ParentChild:
    """A big context block plus the small blocks that index into it.

    The whole idea of parent-child: EMBED the children (small, sharp vectors
    that match one idea precisely), but SEND the parent to the LLM (large
    enough to actually contain the answer).

    A 2,000-token chunk covering four ideas embeds to roughly the average of
    four ideas — a vector near everything and close to nothing. A 256-token
    chunk embeds to one idea sharply. Parent-child decouples *what you match
    on* from *what you read*, so you stop trading precision against
    sufficiency.
    """
    parent: Span
    children: list[Span]


@dataclass
class Chunk:
    """A chunk with everything needed to retrieve, generate from, and cite it."""
    chunk_id: str
    doc_id: str
    book: str                      # short name, e.g. "Géron"
    text: str
    n_tokens: int
    page_start: int                # physical
    page_end: int
    cite_page: int                 # printed when trustworthy, else physical
    cite_kind: str                 # "printed" | "physical"
    breadcrumb: list[str] = field(default_factory=list)
    chunker: str = ""
    parent_id: str | None = None   # set only by parent_child

    def cite(self) -> str:
        """Human-verifiable citation string.

        Says "PDF p." when the book had no trustworthy page offset (Huyen,
        Jurafsky). Being explicit about which number it is beats printing a
        confident wrong one.
        """
        where = " > ".join(self.breadcrumb) if self.breadcrumb else "unsectioned"
        label = "p." if self.cite_kind == "printed" else "PDF p."
        return f"{self.book}, {where}, {label} {self.cite_page}"


class Splitter(Protocol):
    """What the registry accepts. Keeps strategies swappable by config."""
    def __call__(self, text: str, **cfg) -> list[Span]: ...


# Counts units in a string. Injected so splitters can be tested with a trivial
# word counter instead of loading a tokenizer.
Counter_ = Callable[[str], int]

# Maps sentences -> vectors. Injected for the same reason: split_semantic must
# be testable without a 1.2 GB embedding model.
Embedder = Callable[[list[str]], list[list[float]]]


# ───────────────────────────────────────────────────────────────────────
# Plumbing
# ───────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _encoder():
    import tiktoken
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Token count under cl100k_base. The project's standard ruler.

    ``disallowed_special=()`` is REQUIRED here, and the reason is a nice
    illustration of what a corpus about LLMs does to LLM tooling.

    By default tiktoken raises if the input contains a special-token string
    such as ``<|endoftext|>``, on the assumption that such a string appearing
    in user text is an attempt to smuggle a control token into a model. That
    guard is right for prompts and wrong for us: Raschka's "Build a Large
    Language Model (From Scratch)" *discusses* ``<|endoftext|>`` as subject
    matter, so the string appears as ordinary prose. The first full ingestion
    run died on it.

    We are COUNTING tokens for chunk sizing, not feeding a model, so treating
    the string as literal text is correct.

    (Phase 5 note: the same string reaching a generator's prompt is a different
    question, and one the guardrails layer should look at.)

    >>> count_tokens("") == 0
    True
    >>> count_tokens("<|endoftext|>") > 0
    True
    """
    return len(_encoder().encode(text, disallowed_special=()))


_SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def split_sentences(text: str) -> list[Span]:
    """Split into sentences, preserving offsets.

    Deliberately simple regex, not a full NLP sentence tokenizer. It will trip
    on "Fig. 4.1" and "et al." — acceptable, because its only consumer is
    split_semantic, which needs approximate boundaries to measure topic drift,
    not linguistically perfect ones. Adding an NLP dependency for this would
    not change any downstream metric.

    >>> [s.text for s in split_sentences("One thing. Two things! Three?")]
    ['One thing.', 'Two things!', 'Three?']
    """
    spans: list[Span] = []
    pos = 0
    for part in _SENT_END.split(text):
        if not part.strip():
            continue
        start = text.index(part, pos)
        spans.append(Span(part.strip(), start, start + len(part)))
        pos = start + len(part)
    return spans


# ───────────────────────────────────────────────────────────────────────
# Internal helpers for the splitters
# ───────────────────────────────────────────────────────────────────────

def _hard_cut(root: str, start: int, end: int, chunk_size: int,
              count: Counter_) -> list[Span]:
    """Last-resort fixed-width character cut.

    Reached only when every separator has been exhausted and a piece is STILL
    oversized — e.g. a long equation block with no spaces. Without this the
    recursion would terminate holding a piece it cannot split, and either
    return an oversized chunk or loop.

    Width is estimated from the observed chars-per-unit ratio of THIS segment
    rather than a global constant, because that ratio varies wildly: dense
    mathematical notation tokenises very differently from prose.
    """
    seg = root[start:end]
    n = count(seg)
    if n <= chunk_size:
        return [Span(seg.strip(), start, end)] if seg.strip() else []

    width = max(1, int(chunk_size * (len(seg) / max(1, n))))
    out: list[Span] = []
    p = start
    while p < end:
        q = min(end, p + width)
        if root[p:q].strip():
            out.append(Span(root[p:q].strip(), p, q))
        p = q
    return out


def _atomize(root: str, start: int, end: int, chunk_size: int,
             count: Counter_, separators: tuple[str, ...]) -> list[Span]:
    """Break [start,end) into pieces that each fit chunk_size.

    Recursion is over SEPARATORS, not over length. A piece that already fits is
    returned whole — that is what keeps paragraph boundaries intact instead of
    shredding everything down to words.

    Offsets stay absolute (into ``root``) at every level, so no rebasing is
    needed later and citations cannot drift.
    """
    seg = root[start:end]
    if not seg.strip():
        return []

    if count(seg) <= chunk_size:
        return [Span(seg.strip(), start, end)]

    if not separators:
        return _hard_cut(root, start, end, chunk_size, count)

    sep = separators[0]
    if sep == "":
        return _hard_cut(root, start, end, chunk_size, count)

    out: list[Span] = []
    pos = start
    for part in seg.split(sep):
        out.extend(
            _atomize(root, pos, pos + len(part), chunk_size, count, separators[1:])
        )
        pos += len(part) + len(sep)
    return out


def _cos(a: list[float], b: list[float]) -> float:
    """Cosine similarity, 0.0 for a zero vector rather than a ZeroDivisionError."""
    num = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return 0.0 if na == 0 or nb == 0 else num / (na * nb)


def _percentile(vals: list[float], p: float) -> float:
    """Linear-interpolated percentile. Avoids a numpy import for tiny lists."""
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (p / 100.0)
    lo, hi = math.floor(k), math.ceil(k)
    return s[int(k)] if lo == hi else s[lo] + (s[hi] - s[lo]) * (k - lo)


# ═══════════════════════════════════════════════════════════════════════
#   The three splitters
# ═══════════════════════════════════════════════════════════════════════

def split_recursive(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
    count: Counter_ = count_tokens,
    separators: tuple[str, ...] = ("\n\n", "\n", ". ", " ", ""),
) -> list[Span]:
    """Split text by trying progressively finer separators.

    THE CORE IDEA: prefer to break where a human already broke the text.
    Paragraph breaks are better boundaries than line breaks, which are better
    than sentence breaks, which are better than arbitrary word breaks. So walk
    ``separators`` in order and only descend to a finer one when a piece still
    exceeds ``chunk_size``.

    The last separator is ``""`` — a hard character cut. That is the
    give-up case for pathological input (a 900-token equation block with no
    spaces). It must exist, or the function can loop forever on text it cannot
    split.

    Algorithm:
      1. Split on the current separator.
      2. Any piece still over ``chunk_size`` -> recurse with the next separator.
      3. Greedily MERGE adjacent pieces while the total stays within
         ``chunk_size`` (otherwise a page of short lines yields 40 tiny chunks).
      4. Apply ``overlap``: each chunk after the first re-includes roughly
         ``overlap`` units from the end of its predecessor.

    Args:
        text: the section text to split.
        chunk_size: maximum units per chunk, measured by ``count``.
        overlap: units carried over from the previous chunk. Must be < chunk_size.
        count: unit counter. Defaults to tokens; pass ``lambda s: len(s.split())``
            to work in words when testing.
        separators: tried in order, coarsest first.

    Returns:
        Spans in document order. ``start``/``end`` MUST index into ``text`` —
        citations depend on it.

    Raises:
        ValueError: if ``overlap >= chunk_size`` (that would never advance,
            producing an infinite loop rather than an obvious failure).

    >>> words = lambda s: len(s.split())
    >>> [s.text for s in split_recursive("aa bb cc dd ee ff", 2, 0, words)]
    ['aa bb', 'cc dd', 'ee ff']
    >>> [s.text for s in split_recursive("aa bb cc dd", 2, 1, words)]
    ['aa bb', 'bb cc', 'cc dd']
    >>> split_recursive("", 10, 0, words)
    []

    Implementation note: the merge and the overlap are ONE sliding window, not
    two passes. Merging first and then bolting overlap on afterwards double-
    counts units and produces chunks that exceed chunk_size. The window emits
    ``atoms[i:j]``, then rewinds ``i`` by however many trailing atoms are worth
    ``overlap`` units — overlap is specified in UNITS, but must be applied in
    whole ATOMS, and those are different quantities.

    The ``max(i + 1, ...)`` guard guarantees forward progress even when a single
    atom is itself larger than ``overlap``.
    """
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be < chunk_size ({chunk_size}); "
            "otherwise the window never advances and this loops forever"
        )
    if not text.strip():
        return []

    atoms = _atomize(text, 0, len(text), chunk_size, count, tuple(separators))
    if not atoms:
        return []

    out: list[Span] = []
    i, n = 0, len(atoms)
    while i < n:
        # Count the ACTUAL candidate text, not a running sum of atom counts.
        #
        # Summing per-atom counts is wrong because BPE is not additive:
        #   count(A) + count(B) != count(A + B)
        # Two separate effects push it out:
        #   - the separators BETWEEN atoms reappear in text[s:e] but were never
        #     counted, since split() discards them;
        #   - tokens re-merge across a boundary, so the joined string
        #     tokenises differently from its parts.
        # The first full ingestion produced 646-token chunks against a 512
        # limit this way. That would matter silently: mxbai-embed-large has a
        # 512-token context and would truncate the overflow with no error.
        #
        # Re-counting costs O(atoms_per_chunk) encodes of strings bounded by
        # chunk_size - cheap, and it makes "never exceeds chunk_size" a
        # guarantee rather than an estimate.
        j = i
        while j < n:
            if j > i and count(text[atoms[i].start: atoms[j].end]) > chunk_size:
                break
            j += 1

        s, e = atoms[i].start, atoms[j - 1].end
        out.append(Span(text[s:e].strip(), s, e))

        if j >= n:
            break

        # Rewind by whole atoms until we have carried >= `overlap` units.
        back, acc, k = 0, 0, j - 1
        while k > i and acc < overlap:
            acc += count(atoms[k].text)
            back += 1
            k -= 1
        i = max(i + 1, j - back)

    return out


def split_parent_child(
    text: str,
    parent_size: int = 2048,
    child_size: int = 256,
    count: Counter_ = count_tokens,
) -> list[ParentChild]:
    """Split into large parents, each subdivided into small children.

    Retrieval embeds and searches the CHILDREN; generation receives the PARENT.
    See ``ParentChild`` for why that separation matters.

    Algorithm:
      1. Split ``text`` into parents of at most ``parent_size``.
      2. Split each parent into children of at most ``child_size``.
      3. Children carry offsets into the ORIGINAL ``text``, not into the
         parent — otherwise page binding later attributes chunks to the wrong
         pages. This is the easiest thing to get subtly wrong here.

    Both levels can reuse ``split_recursive``; no overlap is needed at the
    child level, because the parent already supplies surrounding context.

    Args:
        text: section text.
        parent_size: max units per parent (what the LLM reads).
        child_size: max units per child (what gets embedded).
        count: unit counter.

    Returns:
        Parents in document order, each with its children.

    Raises:
        ValueError: if ``child_size > parent_size``.

    >>> words = lambda s: len(s.split())
    >>> pcs = split_parent_child("a b c d e f g h", 4, 2, words)
    >>> [p.parent.text for p in pcs]
    ['a b c d', 'e f g h']
    >>> [[c.text for c in p.children] for p in pcs]
    [['a b', 'c d'], ['e f', 'g h']]
    >>> all(c.start >= p.parent.start for p in pcs for c in p.children)
    True
    """
    if child_size > parent_size:
        raise ValueError(
            f"child_size ({child_size}) must be <= parent_size ({parent_size})"
        )

    out: list[ParentChild] = []
    for parent in split_recursive(text, parent_size, 0, count):
        # Children are split from the parent's OWN text, so split_recursive
        # returns offsets relative to that substring. Rebasing by parent.start
        # is what makes them index into the original `text` — skip this and
        # every child is attributed to the wrong page, silently, with citations
        # that look perfectly plausible.
        children = [
            Span(c.text, parent.start + c.start, parent.start + c.end)
            for c in split_recursive(parent.text, child_size, 0, count)
        ]
        out.append(ParentChild(parent=parent, children=children))
    return out


def split_semantic(
    text: str,
    embed: Embedder,
    percentile: int = 95,
    buffer_size: int = 1,
) -> list[Span]:
    """Split where the topic actually changes, measured by embedding distance.

    THE IDEA: instead of counting tokens, embed each sentence and cut where
    consecutive sentences stop resembling each other. A large jump in distance
    is evidence the author moved on.

    THE COST, stated plainly: this embeds EVERY sentence in the corpus before
    a single chunk exists. On ~3,500 pages that is the most expensive strategy
    by a wide margin, and on a CPU-only machine it is the one most likely to be
    impractical. It also cannot respect ``chunk_size`` — chunks come out
    whatever length the topic structure dictates.

    PRIOR (to be falsified in Phase 4): this LOSES to structure-aware recursive
    chunking on our corpus, because our text is already structured — we have a
    real TOC giving genuine section boundaries. Semantic chunking is a way of
    *guessing* boundaries that a publisher already marked for us. Measure it
    anyway; a confirmed negative is a result worth reporting.

    Algorithm:
      1. ``split_sentences(text)``.
      2. Group each sentence with ``buffer_size`` neighbours on each side
         before embedding. A lone sentence embeds noisily; context stabilises
         it. ``buffer_size=1`` means embed sentence i-1 + i + i+1.
      3. ``embed`` the grouped strings.
      4. distance[i] = 1 - cosine_similarity(vec[i], vec[i+1]).
      5. threshold = the ``percentile``-th percentile of all distances.
      6. Cut wherever distance > threshold.

    Note step 5 makes this RELATIVE, not absolute: it always cuts at roughly
    the same *fraction* of gaps regardless of the text. Higher percentile =
    fewer, larger chunks.

    Args:
        text: section text.
        embed: sentences -> vectors. Injected so this is testable without a
            model; Phase 4 passes the real embedder.
        percentile: 0-100. Cut only above this percentile of distances.
        buffer_size: neighbouring sentences included for context.

    Returns:
        Spans in document order, offsets into ``text``.

    >>> def fake(groups):
    ...     # 'cat' sentences -> one direction, everything else -> orthogonal
    ...     return [[1.0, 0.0] if 'at' in g else [0.0, 1.0] for g in groups]
    >>> spans = split_semantic("Cats purr. Cats nap. Dogs bark. Dogs run.",
    ...                        embed=fake, percentile=50, buffer_size=0)
    >>> [s.text for s in spans]
    ['Cats purr. Cats nap.', 'Dogs bark. Dogs run.']

    Implementation note: with 4 sentences there are 3 gaps. Under ``fake`` the
    distances are [0.0, 1.0, 0.0]; the 50th percentile is 0.0, so only the
    middle gap exceeds it and we get 2 chunks.

    Note the threshold is STRICT (``d > thr``). With ``>=``, a text whose gaps
    are all identical — every distance equal to the percentile — would split at
    every single sentence.
    """
    sents = split_sentences(text)
    if not sents:
        return []
    if len(sents) == 1:
        return [Span(text.strip(), 0, len(text))]

    # Embed each sentence together with its neighbours. A lone sentence embeds
    # noisily ("It follows directly." carries almost no topical signal); the
    # surrounding context stabilises the vector so the distances reflect real
    # topic drift rather than sentence-length artefacts.
    groups = [
        " ".join(
            s.text
            for s in sents[max(0, i - buffer_size): min(len(sents), i + buffer_size + 1)]
        )
        for i in range(len(sents))
    ]

    vecs = embed(groups)
    dists = [1.0 - _cos(vecs[i], vecs[i + 1]) for i in range(len(vecs) - 1)]
    if not dists:
        return [Span(text.strip(), 0, len(text))]

    thr = _percentile(dists, percentile)

    def _span(a: int, b: int) -> Span:
        s, e = sents[a].start, sents[b].end
        return Span(text[s:e].strip(), s, e)

    out: list[Span] = []
    first = 0
    for i, d in enumerate(dists):
        if d > thr:
            out.append(_span(first, i))
            first = i + 1
    out.append(_span(first, len(sents) - 1))
    return out


# ───────────────────────────────────────────────────────────────────────
# Registry — resolves configs/experiment.yaml chunking.strategy
# ───────────────────────────────────────────────────────────────────────

CHUNKERS: dict[str, str] = {
    "recursive": "split_recursive",
    "semantic": "split_semantic",
    "parent_child": "split_parent_child",
}

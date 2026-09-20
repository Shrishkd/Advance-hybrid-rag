"""Context assembly — retrieved chunks into a promptable, citable block.

WHY THIS IS NOT STRING CONCATENATION
------------------------------------
Three decisions live here, and each one changes what the generator can do:

  1. HOW MANY chunks to include (the context budget)
  2. HOW to label them so citations can be VERIFIED, not just read
  3. IN WHAT ORDER

None of them is obvious, and getting (2) wrong makes the whole Phase 8 eval
layer depend on an LLM judge for something arithmetic could have checked.

CITATIONS MUST BE MACHINE-CHECKABLE
-----------------------------------
The natural instinct is to ask for citations like "(Bishop, p. 341)". That
reads beautifully and is nearly unverifiable: the model can invent a plausible
book-and-page pair, and checking it needs fuzzy matching against metadata the
model was free to paraphrase.

Instead each chunk is given an explicit id - [S1], [S2] - and the model is
required to cite THOSE. Verification then becomes exact:

    cites [S7] when 5 sources were supplied  -> fabricated, provably
    cites [S3]                               -> resolves to a real chunk we
                                                can name, page and quote

The pretty form is recovered afterwards by expanding [S3] into "Bishop, Ch. 7
- Sparse Kernel Machines, p. 341". Verifiability and readability are separated
rather than traded off.

THE BUDGET IS A HARDWARE CONSTRAINT, NOT A PREFERENCE
------------------------------------------------------
`llama3.2` and `phi4-mini` advertise 128k context windows. That number is
irrelevant here: on 7.4 GB of RAM a 4B model at Q4 occupies ~2.5 GB, and the
KV cache for a long context consumes the rest fast. A large context also costs
latency linearly on CPU.

So the budget is set in TOKENS and enforced, defaulting to ~3,000 - roughly six
512-token chunks plus the question and instructions. `top_n` and the budget are
both experiment variables: D6 showed a reranker's value depends on how many
chunks the generator actually reads, which means the reranker decision and this
one are coupled.

WHY ORDER MATTERS
-----------------
Chunks go in RETRIEVAL ORDER, best first. Models attend unevenly across a long
context - the "lost in the middle" effect - so the most relevant chunk should
not be buried. This is a default worth challenging in Phase 6, not a proven
choice; it is recorded here as a prior.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Source:
    """One retrieved chunk, as the generator will see it."""
    sid: str                      # "S1" - what the model must cite
    book: str
    text: str
    page_start: int
    page_end: int
    cite_page: int                # printed page where trustworthy, else physical
    cite_kind: str                # "printed" | "physical"
    breadcrumb: list[str] = field(default_factory=list)
    chunk_id: str = ""
    n_tokens: int = 0

    def human(self) -> str:
        """Readable citation: 'Géron, Ch. 4 - Training Models, p. 112'.

        Built from breadcrumb + page, never from anything the model wrote, so
        a displayed citation cannot inherit a hallucination.
        """
        parts = [self.book]
        if self.breadcrumb:
            # Deepest two levels: the full chain is often four deep and reads
            # as noise in a citation.
            parts.append(" > ".join(self.breadcrumb[-2:]))
        suffix = "p." if self.cite_kind == "printed" else "PDF p."
        parts.append(f"{suffix} {self.cite_page}")
        return ", ".join(parts)

    def block(self) -> str:
        """The chunk as it appears in the prompt."""
        return f"[{self.sid}] {self.human()}\n{self.text}"


@dataclass
class AssembledContext:
    sources: list[Source]
    text: str
    n_tokens: int
    dropped: int                  # chunks that did not fit the budget

    def by_sid(self) -> dict[str, Source]:
        return {s.sid: s for s in self.sources}


def assemble(
    chunks: list[dict],
    top_n: int = 6,
    token_budget: int = 3000,
    count_tokens=None,
) -> AssembledContext:
    """Turn ranked chunks into a numbered, citable context block.

    Args:
        chunks: retrieved chunks, BEST FIRST. Each needs book/text/page_start/
            page_end/cite_page/cite_kind; breadcrumb and chunk_id optional.
        top_n: hard cap on sources, applied before the budget.
        token_budget: total tokens of chunk text allowed.
        count_tokens: token counter; defaults to the corpus tokenizer so the
            budget is measured in the same unit chunking used.

    Returns:
        AssembledContext. `dropped` counts chunks cut by the budget - a
        non-zero value means the generator did NOT see something retrieval
        found, which is exactly the kind of loss that otherwise gets blamed on
        the model.

    A chunk is included whole or not at all. Truncating mid-chunk would hand
    the model a sentence that stops dead, and the citation would then point at
    a passage containing text the model never saw.

    >>> cs = [dict(book="Bishop", text="alpha "*10, page_start=340,
    ...            page_end=341, cite_page=320, cite_kind="printed",
    ...            breadcrumb=["7. Sparse Kernel Machines", "7.1 Margins"]),
    ...       dict(book="Géron", text="beta "*10, page_start=110,
    ...            page_end=110, cite_page=112, cite_kind="printed",
    ...            breadcrumb=["Chapter 4. Training Models"])]
    >>> ctx = assemble(cs, top_n=2, token_budget=1000, count_tokens=len)
    >>> [s.sid for s in ctx.sources]
    ['S1', 'S2']
    >>> ctx.sources[0].human()
    'Bishop, 7. Sparse Kernel Machines > 7.1 Margins, p. 320'
    >>> ctx.text.splitlines()[0]
    '[S1] Bishop, 7. Sparse Kernel Machines > 7.1 Margins, p. 320'
    >>> # budget of 1 token admits nothing, and says so rather than truncating
    >>> tiny = assemble(cs, top_n=2, token_budget=1, count_tokens=len)
    >>> len(tiny.sources), tiny.dropped
    (0, 2)
    """
    if count_tokens is None:
        from src.ingest.chunkers import count_tokens as _ct
        count_tokens = _ct

    sources: list[Source] = []
    used = 0
    dropped = 0

    for i, c in enumerate(chunks[:top_n], start=1):
        n = c.get("n_tokens") or count_tokens(c["text"])
        if used + n > token_budget:
            # Do NOT break: a later chunk may be small enough to fit. Breaking
            # would let one oversized chunk silently discard everything after
            # it, including chunks the generator could have used.
            dropped += 1
            continue
        sources.append(Source(
            sid=f"S{len(sources) + 1}",
            book=c["book"],
            text=c["text"],
            page_start=c["page_start"],
            page_end=c["page_end"],
            cite_page=c.get("cite_page", c["page_start"] + 1),
            cite_kind=c.get("cite_kind", "physical"),
            breadcrumb=list(c.get("breadcrumb") or []),
            chunk_id=c.get("chunk_id", ""),
            n_tokens=n,
        ))
        used += n

    dropped += max(0, len(chunks) - top_n)
    return AssembledContext(
        sources=sources,
        text="\n\n".join(s.block() for s in sources),
        n_tokens=used,
        dropped=dropped,
    )

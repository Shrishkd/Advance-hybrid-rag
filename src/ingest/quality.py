"""Proxy metrics for PDF extraction quality.

THE CORE PROBLEM
----------------
We want to rank four parsers, but we have no ground-truth transcription of
3,500 pages. So "accuracy" is literally not computable. Anyone who claims an
automated extraction-accuracy score without reference text is measuring
something else and calling it accuracy.

What we CAN do is measure signatures of *damage*. Each metric below detects
one specific way PDF extraction goes wrong. None is meaningful alone; together
they rank the parsers, and a human eyeball on ~20 sampled pages confirms the
ranking is not lying. Both halves are required.

READ THIS BEFORE IMPLEMENTING
-----------------------------
These metrics are DIRECTIONAL — for each one, know whether high is good or
bad before you write it, and put the answer in HIGHER_IS_BETTER below. A
benchmark table that silently mixes directions is worse than no table,
because it looks authoritative while ranking things backwards.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass


# Direction of every metric. parse_bench.py reads this to know which way to
# sort and how to colour the report. Keep it in sync as you add metrics.
HIGHER_IS_BETTER: dict[str, bool] = {
    "chars_per_page": True,          # more text recovered = less silently dropped
    "alpha_ratio": True,             # more letters = less math/garbage soup
    "single_char_token_ratio": False,  # fragments = shattered equations
    "broken_hyphen_rate": False,     # line-break hyphens = destroyed words
    "repeated_line_ratio": False,    # repeated lines = headers/footers as noise
}


@dataclass
class PageQuality:
    """All metrics for one page, from one parser."""
    parser: str
    page_num: int
    chars_per_page: int
    alpha_ratio: float
    single_char_token_ratio: float
    broken_hyphen_rate: float


# ═══════════════════════════════════════════════════════════════════════
# WORKED EXAMPLES — implemented, to show the pattern
# ═══════════════════════════════════════════════════════════════════════

def chars_per_page(text: str) -> int:
    """Count non-whitespace characters recovered from a page.

    Why it matters: the quietest parser failure is not garbled text, it is
    text that never appears at all. A parser silently returning 300 chars
    where another returns 2,400 has dropped most of the page — and nothing
    downstream will ever tell you.

    >>> chars_per_page("hello world")
    10
    >>> chars_per_page("   ")
    0
    """
    return len(re.sub(r"\s", "", text))


def alpha_ratio(text: str) -> float:
    """Fraction of non-whitespace characters that are alphabetic.

    Why it matters: this is the blunt "is this prose or is this soup?"
    detector. Healthy textbook prose runs ~0.75-0.85. A page of shattered
    equations, coordinates and stray digits drops far below that.

    Note ``str.isalpha()`` is Unicode-aware, so Greek letters common in ML
    notation (θ, μ, σ) count as alphabetic. That is deliberate — they are
    real content, not garbage.

    Returns 0.0 for empty input rather than raising: a parser that returned
    nothing should score badly, not crash the benchmark.

    >>> alpha_ratio("hello")
    1.0
    >>> round(alpha_ratio("p(x|θ) = 5"), 3)
    0.375
    >>> alpha_ratio("")
    0.0
    """
    stripped = re.sub(r"\s", "", text)
    if not stripped:
        return 0.0
    return sum(c.isalpha() for c in stripped) / len(stripped)


# ═══════════════════════════════════════════════════════════════════════
# YOUR TURN — three metrics to implement
#
# For each: read the docstring, check the worked example, then write the
# body. The examples are executable tests — verify with:
#
#     .venv\Scripts\python.exe -m pytest src/ingest/quality.py
#
# You are done when that reports 5 passed, 0 failed.
# ═══════════════════════════════════════════════════════════════════════

def single_char_token_ratio(text: str) -> float:
    """Fraction of whitespace-separated tokens that are exactly one character.

    THIS IS THE EQUATION-SHATTER DETECTOR, and probably the single most
    diagnostic metric for our corpus. When a parser fails on mathematical
    notation it does not usually error — it emits the symbols as isolated
    fragments. ``p(x|θ) = Πᵢ N(xᵢ; μ, σ²)`` degrades into ``p x 1 N i``.

    Healthy English prose sits near 0.02-0.05 (mostly "a" and "I").
    A shattered equation block runs 0.4+. Bishop and Goodfellow will expose
    the difference between parsers here more than any other page type.

    Args:
        text: raw extracted page text.

    Returns:
        Ratio in [0.0, 1.0]. Return 0.0 for text with no tokens — an empty
        page is already penalised by chars_per_page; do not double-count it
        here, and definitely do not divide by zero.

    >>> round(single_char_token_ratio("The gradient descent algorithm converges"), 3)
    0.0
    >>> round(single_char_token_ratio("x t 1 N gradient"), 3)
    0.8
    >>> single_char_token_ratio("")
    0.0

    Implementation note: ``text.split()`` splits on any whitespace run and
    drops empties, which is exactly the tokenisation wanted here.
    """
    tokens = text.split()
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if len(t) == 1) / len(tokens)


def broken_hyphen_rate(text: str) -> float:
    """Fraction of tokens that are the first half of a line-break hyphenation.

    Why it matters: PDFs hyphenate words at line ends for typesetting. A
    parser that does not rejoin them emits ``gra-`` and ``dient`` as separate
    tokens. Neither is a real word, so neither is in any embedding model's
    vocabulary, and the concept becomes unretrievable. Across 3,500 pages this
    is thousands of silently destroyed terms.

    Detection signature: a token ending in "-" whose FOLLOWING token starts
    with a lowercase letter. The lowercase condition matters — it separates
    a broken word from a legitimate compound or dash usage:

        "gra- dient"        -> broken          (next token lowercase)
        "state- of-the-art" -> broken
        "self- Attention"   -> NOT broken      (next token capitalised)
        "end -"             -> NOT broken      (no following token)

    Args:
        text: raw extracted page text.

    Returns:
        (count of broken-hyphen tokens) / (total tokens), in [0.0, 1.0].
        0.0 when there are no tokens.

    >>> round(broken_hyphen_rate("gra- dient des- cent"), 3)
    0.5
    >>> broken_hyphen_rate("gradient descent")
    0.0
    >>> broken_hyphen_rate("self- Attention")
    0.0

    Implementation note: detection needs token i and token i+1 together, so
    we scan adjacent pairs via ``zip(tokens, tokens[1:])``. The final token
    has no successor and therefore can never be counted — correct, since a
    trailing hyphen at end-of-page has nothing to rejoin to here.
    """
    tokens = text.split()
    if not tokens:
        return 0.0
    broken = sum(
        1
        for a, b in zip(tokens, tokens[1:])
        if len(a) > 1 and a.endswith("-") and b[:1].islower()
    )
    return broken / len(tokens)


def repeated_line_ratio(pages: list[str], threshold: float = 0.5) -> float:
    """Fraction of all line-instances that are repeating headers/footers.

    NOTE THE DIFFERENT SIGNATURE — this one takes MANY pages, not one.
    That is inherent to what it measures: you cannot tell a running header
    from a normal line by looking at a single page. A line is only a header
    because it recurs. Some properties only exist across a corpus, and the
    function signature should say so honestly.

    Why it matters: "CHAPTER 4. TRAINING MODELS" and a page number appearing
    on every page injects noise into every chunk we ever build. Worse, it
    makes unrelated chunks look similar to each other, because they share
    boilerplate — which actively degrades retrieval.

    Algorithm:
      1. Split every page into lines; strip whitespace; ignore empty lines.
      2. Count how many DISTINCT pages each unique line appears on.
      3. A line is "repeating" if it appears on >= max(2, threshold * len(pages))
         distinct pages.
      4. Return (total instances of repeating lines) / (total line instances).

      The ``max(2, ...)`` floor is load-bearing, and the original spec for this
      function omitted it — a bug caught while implementing. Without it, a
      2-page sample gives ``0.5 * 2 == 1.0``, and since EVERY line appears on
      at least one page, every line would be classified as a running header
      and the function would return 1.0 for any input. A line seen on exactly
      one page cannot be boilerplate by definition, so 2 is the true floor.

      Why ``>=`` rather than ``>``: strict inequality would break the
      recto/verso case this threshold exists to catch. A header printed only
      on right-hand pages appears on exactly half of them — 4 of 8 — which
      ``> 0.5 * 8`` rejects and ``>= 0.5 * 8`` accepts.

    Careful: step 2 counts *pages*, not *occurrences*. A line appearing three
    times on one page is still just one page.

    Args:
        pages: extracted text for several pages of the SAME document.
        threshold: fraction of pages a line must appear on to count as
            boilerplate. 0.5 is deliberately loose — headers sometimes differ
            between recto and verso pages, so a strict 1.0 would miss them.

    Returns:
        Ratio in [0.0, 1.0]. 0.0 if ``pages`` is empty or has no lines.

    >>> pages = [
    ...     "CHAPTER 4\\nGradient descent is\\nan optimisation method",
    ...     "CHAPTER 4\\nThe learning rate\\ncontrols step size",
    ... ]
    >>> round(repeated_line_ratio(pages), 3)
    0.333

    (In that example "CHAPTER 4" appears on 2 of 2 pages, so it is repeating.
    There are 6 line-instances total and 2 belong to the repeating line:
    2/6 = 0.333.)

    Implementation note: the Counter is built from a SET of lines per page,
    so a line repeated three times within one page still counts as one page.
    """
    if not pages:
        return 0.0

    per_page: list[list[str]] = [
        [ln.strip() for ln in page.split("\n") if ln.strip()] for page in pages
    ]
    total = sum(len(lines) for lines in per_page)
    if total == 0:
        return 0.0

    page_counts: Counter[str] = Counter()
    for lines in per_page:
        page_counts.update(set(lines))        # set(): distinct PAGES, not occurrences

    min_pages = max(2, threshold * len(pages))
    repeating = {ln for ln, n in page_counts.items() if n >= min_pages}

    return sum(1 for lines in per_page for ln in lines if ln in repeating) / total


# ═══════════════════════════════════════════════════════════════════════
# Assembly — wired up once the metrics above exist
# ═══════════════════════════════════════════════════════════════════════

def score_page(parser: str, page_num: int, text: str) -> PageQuality:
    """Run every per-page metric and package the result.

    Note repeated_line_ratio is absent here by necessity — it is a
    per-document metric, so parse_bench.py computes it separately.
    """
    return PageQuality(
        parser=parser,
        page_num=page_num,
        chars_per_page=chars_per_page(text),
        alpha_ratio=alpha_ratio(text),
        single_char_token_ratio=single_char_token_ratio(text),
        broken_hyphen_rate=broken_hyphen_rate(text),
    )

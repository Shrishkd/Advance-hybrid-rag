"""Text cleaning — run AFTER parsing, BEFORE chunking.

ORDER MATTERS, AND IT IS NOT ARBITRARY
--------------------------------------
    1. detect_page_offset   needs headers INTACT (page numbers live there)
    2. strip_headers_footers
    3. dehyphenate

Stripping headers before detecting the offset destroys the only evidence of
what page the book thinks it is on. This ordering is enforced by clean_document().

WHAT THIS FILE DOES NOT DO
--------------------------
It does not touch equations. `corpus.preserve_equations: true` was a deliberate
decision: on this corpus the mathematics IS the content, and a question like
"what is the closed-form solution for ridge regression?" is unanswerable if the
formula was stripped as noise. We accept fragmented equation text in exchange
for not losing it.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

# Lines that are nothing but a number - bare page numbers.
_BARE_NUM = re.compile(r"^\s*(\d{1,4})\s*$")
# A number anywhere in a short line, e.g. "41  1.5. Decision Theory".
_EDGE_NUM = re.compile(r"\b(\d{1,4})\b")
# Word broken across a line: "gra-\ndient"
_HYPHEN_BREAK = re.compile(r"(\w+)-\n(\w+)")

# Headers/footers live at the page EDGES. Only these lines are eligible for
# stripping - otherwise a repeated body line (a recurring figure caption, a
# boilerplate warning) would be deleted as furniture.
EDGE_LINES = 3


@dataclass
class PageOffset:
    """Result of page-offset detection.

    TWO numbers, not one, because they fail differently and the distinction is
    actionable:

      coverage  - fraction of pages that yielded ANY page-number evidence.
                  Low means "this book does not print numbers where we look".
      agreement - fraction of the votes cast that agree with the winner.
                  Low means "we found numbers but they contradict each other",
                  i.e. we are reading figure or equation numbers as page numbers.

    Collapsing these into one "confidence" hides which failure occurred, and
    they demand opposite fixes: low coverage means look harder, low agreement
    means we are looking at the wrong thing.
    """
    value: int | None
    agreement: float
    coverage: float
    votes: int

    @property
    def trustworthy(self) -> bool:
        """Whether this offset should be used for citations at all.

        Both thresholds must hold. An offset derived from 3 pages that happen
        to agree is not evidence, and 200 votes split 55/45 is not either.
        When False, cite PHYSICAL pages and say so - a citation known to be
        approximate beats one confidently wrong by 20 pages.
        """
        return self.value is not None and self.agreement >= 0.80 and self.coverage >= 0.50


@dataclass
class CleanStats:
    """What cleaning actually did. Surfaced so it can be sanity-checked."""
    pages: int
    offset: PageOffset
    header_lines_removed: int
    hyphens_joined: int


# ───────────────────────────────────────────────────────────────────────
# 1. Page offset
# ───────────────────────────────────────────────────────────────────────

def _page_number_in(line: str) -> int | None:
    """Extract a plausible printed page number from one edge line.

    Three patterns, because books disagree about where the number goes:

        "103"                        bare, on its own line
        "41  1.5. Decision Theory"   leading, beside a running head  (verso)
        "Chapter 3   |   103"        trailing, beside a running head (recto)

    The original implementation matched only the bare form. Measured
    consequence: Huyen produced 6 votes across 140 pages and Jurafsky 18,
    because both print numbers INLINE with the running head. With almost no
    real votes, stray figure numbers won the mode and yielded nonsense offsets
    (-103 and -91). Coverage collapse is how a mode-based estimator lies.

    The 80-character cap keeps this on page furniture: a body-text line
    beginning with a numeral ("5 shows that...") is far longer than a running
    head, so the cap excludes it.
    """
    s = line.strip()
    if not s:
        return None
    if m := _BARE_NUM.match(s):
        return int(m.group(1))
    if len(s) <= 80:
        if m := re.match(r"^(\d{1,4})\b", s):
            return int(m.group(1))
        if m := re.search(r"\b(\d{1,4})$", s):
            return int(m.group(1))
    return None


def detect_page_offset(pages: list[str], start_physical: int = 0) -> PageOffset:
    """Infer ``printed_page - physical_page`` from numbers in page edges.

    WHY THIS MATTERS: ``toc.py`` records PHYSICAL page indices - the Nth sheet
    in the file. Textbooks print their own numbers, starting after front matter
    (cover, copyright, preface, roman-numeraled contents). That is typically
    10-30 sheets. Cite the physical index and EVERY citation is wrong by a
    constant, so a reader who opens the book does not find the quote. Silent,
    systematic, and fatal to trust.

    Method: for each page, look for a bare integer in the top/bottom few lines.
    Each hit proposes ``offset = printed - physical``. The correct offset is
    constant across the body, so the true value is the MODE; noise (equation
    numbers, figure labels) scatters and does not accumulate.

    Args:
        pages: text for CONSECUTIVE pages.
        start_physical: physical index of ``pages[0]``. Lets a caller pass a
            slice from the middle of a book directly, instead of padding the
            front with blanks - padding silently inflates the denominator and
            makes coverage look terrible. (That mistake cost one debugging
            round; the parameter exists so it cannot recur.)

    Returns:
        A :class:`PageOffset`. **Check ``.trustworthy`` before citing with it.**
        A confidently wrong offset is worse than no offset: every citation in
        the book is then off by a constant, and looks authoritative.

    >>> pgs = ["cover", "", "1\\nchapter one text", "2\\nmore text", "3\\nyet more"]
    >>> detect_page_offset(pgs)
    PageOffset(value=-1, agreement=1.0, coverage=0.75, votes=3)
    """
    votes: Counter[int] = Counter()
    non_empty = 0
    voting_pages = 0

    for i, text in enumerate(pages):
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        if not lines:
            continue                       # a blank page cannot carry a number
        non_empty += 1

        physical = start_physical + i
        edges = lines[:EDGE_LINES] + lines[-EDGE_LINES:]

        # EVERY candidate on the page, deduped. An earlier version stopped at
        # the first number found, which meant a chapter heading in the top
        # edge ("4 Implementing a GPT model") cast a bogus vote and masked the
        # real page number below it. Measured agreement was then 0.58-0.78.
        #
        # The signal being exploited: the TRUE page number yields the SAME
        # offset on every single page, while chapter, figure and equation
        # numbers yield offsets that scatter. Collecting all candidates lets
        # the constant accumulate while the noise spreads thin.
        #
        # Deduped per page via a set so one page contributes at most one vote
        # per distinct offset - otherwise a page showing "119" twice (header
        # and footer) would double-weight itself.
        offsets = {
            n - physical
            for ln in edges
            if (n := _page_number_in(ln)) is not None
        }
        if offsets:
            voting_pages += 1
            votes.update(offsets)

    if not votes:
        return PageOffset(None, 0.0, 0.0, 0)

    offset, hits = votes.most_common(1)[0]
    return PageOffset(
        value=offset,
        # agreement: of pages that offered ANY evidence, how many back the winner
        agreement=round(hits / max(1, voting_pages), 3),
        # coverage: how many non-blank pages offered evidence at all
        coverage=round(voting_pages / max(1, non_empty), 3),
        votes=hits,
    )


# ───────────────────────────────────────────────────────────────────────
# 2. Headers and footers
# ───────────────────────────────────────────────────────────────────────

def find_furniture(pages: list[str], threshold: float = 0.4) -> set[str]:
    """Identify recurring page furniture (running heads, footers).

    Only lines at the page EDGES are considered, for the reason given at
    EDGE_LINES above.

    Threshold is 0.4 rather than 0.5 because textbooks commonly alternate
    running heads between recto and verso (book title on one side, chapter
    title on the other), so each appears on only ~half of pages, and sampling
    noise can push that under 0.5.

    Pure numbers are handled separately by ``strip_page_numbers`` - they differ
    on every page and so would never be caught by a recurrence test.
    """
    counts: Counter[str] = Counter()
    for text in pages:
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        if not lines:
            continue
        counts.update(set(lines[:EDGE_LINES] + lines[-EDGE_LINES:]))

    floor = max(2, threshold * len(pages))
    return {ln for ln, n in counts.items() if n >= floor and not _BARE_NUM.match(ln)}


def strip_furniture(text: str, furniture: set[str]) -> tuple[str, int]:
    """Remove known furniture lines and bare page numbers from page edges.

    Returns (cleaned_text, lines_removed).

    Restricted to the edges even though the furniture set was built from edges:
    a running head like "Decision Theory" could legitimately appear as a body
    line elsewhere, and deleting that would remove real content.
    """
    lines = text.split("\n")
    if not lines:
        return text, 0

    keep: list[str] = []
    removed = 0
    last = len(lines) - 1

    for i, ln in enumerate(lines):
        at_edge = i < EDGE_LINES or i > last - EDGE_LINES
        s = ln.strip()
        if at_edge and s and (s in furniture or _BARE_NUM.match(s)):
            removed += 1
            continue
        keep.append(ln)

    return "\n".join(keep), removed


# ───────────────────────────────────────────────────────────────────────
# 3. De-hyphenation
# ───────────────────────────────────────────────────────────────────────

def dehyphenate(text: str) -> tuple[str, int]:
    """Rejoin words split across a line break by typesetting.

    Why it matters: an unrepaired ``gra-\\ndient`` becomes the tokens ``gra-``
    and ``dient``. Neither is a real word, neither is in any embedding model's
    vocabulary, and the concept is unretrievable. Across 3,500 pages this is
    thousands of silently destroyed terms.

    THE SUBTLETY - when NOT to drop the hyphen. Most breaks are ordinary words
    and the hyphen is pure typesetting:

        "gra-\\ndient"        -> "gradient"          hyphen removed

    But some words are genuinely hyphenated and merely happen to break at the
    hyphen. Removing it corrupts them:

        "state-\\nof-the-art" -> "stateof-the-art"   WRONG
        "state-\\nof-the-art" -> "state-of-the-art"  right

    Heuristic used: if the fragment AFTER the break itself contains a hyphen,
    the word is a compound, so keep the hyphen. Otherwise drop it.

    This is a heuristic and it has known failure modes - "non-\\nlinear" keeps
    no hyphen and yields "nonlinear" (usually fine), while a two-part compound
    like "self-\\nattention" yields "selfattention" (wrong). A dictionary check
    would do better; we deliberately avoid pulling in a wordlist dependency for
    a marginal gain, and note the limitation instead of hiding it.

    Only fires when the following fragment starts lowercase, so a line break
    before a proper noun or a new sentence is left alone.

    Returns (text, joins_made).

    >>> dehyphenate("gra-\\ndient descent")
    ('gradient descent', 1)
    >>> dehyphenate("state-\\nof-the-art model")
    ('state-of-the-art model', 1)
    >>> dehyphenate("end-\\nTo-End")
    ('end-\\nTo-End', 0)
    >>> dehyphenate("no hyphens here")
    ('no hyphens here', 0)
    """
    count = 0

    def _join(m: re.Match[str]) -> str:
        nonlocal count
        head, tail = m.group(1), m.group(2)
        if not tail[:1].islower():
            return m.group(0)                 # proper noun / new sentence: leave
        count += 1
        # Peek past the matched tail for a hyphen, i.e. is this a compound?
        rest = m.string[m.end(2): m.end(2) + 12]
        return f"{head}-{tail}" if rest.startswith("-") else f"{head}{tail}"

    return _HYPHEN_BREAK.sub(_join, text), count


# ───────────────────────────────────────────────────────────────────────
# Pipeline
# ───────────────────────────────────────────────────────────────────────

def clean_document(
    pages: list[str],
    start_physical: int = 0,
    strip_headers: bool = True,
    do_dehyphenate: bool = True,
    detect_offset: bool = True,
) -> tuple[list[str], CleanStats]:
    """Clean a whole document. Enforces the ordering described at module top."""
    offset = (
        detect_page_offset(pages, start_physical)
        if detect_offset
        else PageOffset(None, 0.0, 0.0, 0)
    )

    furniture = find_furniture(pages) if strip_headers else set()
    out: list[str] = []
    removed = joined = 0

    for text in pages:
        if strip_headers:
            text, r = strip_furniture(text, furniture)
            removed += r
        if do_dehyphenate:
            text, j = dehyphenate(text)
            joined += j
        out.append(text)

    return out, CleanStats(
        pages=len(pages),
        offset=offset,
        header_lines_removed=removed,
        hyphens_joined=joined,
    )

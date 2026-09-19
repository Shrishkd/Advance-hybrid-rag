"""Turn a PDF's embedded outline into citable, filterable metadata.

WHY THIS IS THE HIGHEST-LEVERAGE FILE IN PHASE 1
------------------------------------------------
Most RAG pipelines treat a PDF as a flat bag of pages and throw away the
structure the publisher already encoded. Textbook PDFs almost always ship a
real chapter/section tree. Recovering it gives us four things for free:

  1. Chunk boundaries that respect sections, so we never split mid-argument.
  2. Metadata filters ("only search Jurafsky chapters 8-10").
  3. Citations a human can verify: "Géron, Ch. 4 - Training Models, p. 112".
  4. A cheap approximation of semantic chunking, without paying for the
     embeddings that real semantic chunking requires.

MEASURED ON OUR CORPUS (2026-09-19)
-----------------------------------
    Build a LLM (Raschka)        109 entries, depth 3
    Designing ML Systems (Huyen) 105 entries, depth 3
    Hands-On ML (Géron)          166 entries, depth 3
    Pattern Recognition (Bishop) 285 entries, depth 3
    Speech & Language (Jurafsky) 548 entries, depth 4   <- Part/Ch/Sec/Sub
    Deep Learning (Goodfellow)     0 entries            <- NO OUTLINE

So the bet pays off for five of six books. Goodfellow needs a heading-
detection fallback, which is a Phase 2 task (it requires page text, which
this module deliberately does not touch).

KNOWN GOTCHA: THE PAGE-NUMBER OFFSET
------------------------------------
PDF outlines store *physical* page indices - the Nth sheet in the file.
Textbooks number their pages *printed*, and front matter (title, copyright,
preface, roman-numeraled TOC) is usually 10-30 sheets that do not count.

So physical page 112 might be printed page 88. If we cite the physical index,
every citation is quietly wrong by a constant, and a reader who opens the
book will not find what we quoted.

We do NOT solve that here - detecting the offset needs the page text, which
is Phase 2's job. This module records physical pages and exposes
``page_offset`` so Phase 2 can correct it in one place.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass, field

from .parsers import TocEntry


# ───────────────────────────────────────────────────────────────────────
# Cleaning
# ───────────────────────────────────────────────────────────────────────

_DOT_LEADER = re.compile(r"[.…]{2,}\s*\d*\s*$")   # "Introduction ...... 42"
_WS = re.compile(r"\s+")


def normalize_title(title: str) -> str:
    """Strip dot leaders, trailing page numbers and whitespace noise.

    Outline titles are frequently scraped from the printed contents page and
    drag along typographic debris.

    >>> normalize_title("4.2  Gradient Descent .......... 118")
    '4.2 Gradient Descent'
    >>> normalize_title("  Deep   Feedforward\\nNetworks ")
    'Deep Feedforward Networks'
    """
    t = _DOT_LEADER.sub("", title)
    return _WS.sub(" ", t).strip()


# ───────────────────────────────────────────────────────────────────────
# Tree
# ───────────────────────────────────────────────────────────────────────

def build_tree(flat: list[TocEntry]) -> list[TocEntry]:
    """Nest a flat, level-tagged outline into a tree.

    PDF outlines arrive flat: [(1, "Chapter 4", 100), (2, "4.1 Linear", 101), ...].
    A stack keyed on ``level`` reconstructs the hierarchy in one pass.

    Malformed outlines (a level-3 entry with no level-2 parent) are common in
    the wild, so we attach orphans to the nearest available ancestor rather
    than raising. A slightly wrong tree is far more useful than a crash.
    """
    roots: list[TocEntry] = []
    stack: list[TocEntry] = []

    for e in flat:
        node = TocEntry(level=e.level, title=normalize_title(e.title), page=e.page)
        while stack and stack[-1].level >= node.level:
            stack.pop()
        if stack:
            stack[-1].children.append(node)
        else:
            roots.append(node)
        stack.append(node)

    return roots


# ───────────────────────────────────────────────────────────────────────
# Page -> section resolution
# ───────────────────────────────────────────────────────────────────────

@dataclass
class Breadcrumb:
    """Where a page sits in the book's structure, outermost level first.

    ``path`` is a list rather than fixed chapter/section/subsection fields
    because books disagree about depth. Jurafsky nests four levels
    (Part > Chapter > Section > Subsection) while Géron nests three. Forcing
    every book into three slots would mislabel Jurafsky's parts as chapters.
    """
    path: list[str] = field(default_factory=list)

    @property
    def chapter(self) -> str | None:
        return self.path[0] if self.path else None

    @property
    def section(self) -> str | None:
        return self.path[1] if len(self.path) > 1 else None

    @property
    def leaf(self) -> str | None:
        """Most specific known location - what a citation should lead with."""
        return self.path[-1] if self.path else None

    def cite(self, book: str, page: int) -> str:
        """Render a human-verifiable citation.

        >>> bc = Breadcrumb(["Chapter 4 - Training Models", "4.2 Gradient Descent"])
        >>> bc.cite("Géron", 112)
        'Géron, Chapter 4 - Training Models > 4.2 Gradient Descent, p. 112'
        >>> Breadcrumb([]).cite("Goodfellow", 240)
        'Goodfellow, unsectioned, p. 240'
        """
        where = " > ".join(self.path) if self.path else "unsectioned"
        return f"{book}, {where}, p. {page}"


class SectionIndex:
    """Resolve any page number to its position in the book's hierarchy.

    Built once per document, then queried per chunk during ingestion.

    THE ALGORITHM, AND THE BUG IT EXISTS TO AVOID
    ---------------------------------------------
    The obvious implementation keeps one sorted list per level and asks each
    independently: "last level-1 entry at or before page p? last level-2?"

    That is WRONG, and wrong in a way that looks plausible in testing. If
    chapter 8 begins on page 359 but its first numbered section begins on
    page 385, then page 379 resolves to chapter 8 - correct - and to the last
    level-2 entry before it, which belongs to *chapter 7*. The citation reads
    "8. Graphical Models > 7.2. Relevance Vector Machines". Both halves are
    individually defensible; together they are nonsense. This was observed on
    Bishop and on Raschka before the fix.

    The correct approach walks BACKWARDS from the nearest preceding entry,
    accepting an ancestor only when it is strictly shallower than the last one
    accepted. That guarantees a genuine ancestor chain: a section is only
    reported if it actually sits inside the reported chapter. Pages before a
    chapter's first section correctly yield the chapter alone.

    Binary search rather than a linear scan: this is called once per CHUNK,
    and there will be ~50,000 chunks re-generated on every chunking
    experiment.
    """

    _SENTINEL_LEVEL = 1 << 30

    def __init__(self, flat: list[TocEntry], page_offset: int = 0) -> None:
        """
        Args:
            flat: outline entries as returned by a parser's ``extract_toc``.
            page_offset: printed_page - physical_page. Phase 2 detects this;
                0 means "cite physical pages and accept the discrepancy".
        """
        self.page_offset = page_offset

        # Stable sort by page. Stability matters: a chapter and its first
        # section frequently start on the SAME page, and the outline lists
        # the chapter first. A stable sort preserves that, so the backward
        # walk sees the section before the chapter and nests them correctly.
        self._entries: list[TocEntry] = sorted(
            (
                TocEntry(level=e.level, title=normalize_title(e.title), page=e.page)
                for e in flat
            ),
            key=lambda e: e.page,
        )
        self._pages: list[int] = [e.page for e in self._entries]

    def breadcrumb(self, physical_page: int) -> Breadcrumb:
        """Resolve a physical page index to its ancestor chain.

        Returns an empty Breadcrumb when the document has no outline, or when
        the page precedes the first entry (front matter). Empty is the honest
        answer - better an unsectioned citation than a fabricated one.
        """
        i = bisect_right(self._pages, physical_page) - 1
        chain: list[str] = []
        deepest = self._SENTINEL_LEVEL

        while i >= 0 and deepest > 1:
            e = self._entries[i]
            if e.level < deepest:       # strictly shallower => a real ancestor
                chain.append(e.title)
                deepest = e.level
            i -= 1

        chain.reverse()                 # collected leaf-first; cite root-first
        return Breadcrumb(path=chain)

    def printed_page(self, physical_page: int) -> int:
        """Convert a physical page index to the number printed on the page."""
        return physical_page + self.page_offset

    def __len__(self) -> int:
        return len(self._entries)

    def __bool__(self) -> bool:
        """False when the document shipped no outline (e.g. Goodfellow)."""
        return bool(self._entries)


# ───────────────────────────────────────────────────────────────────────
# Reporting
# ───────────────────────────────────────────────────────────────────────

def summarize(flat: list[TocEntry]) -> dict[str, int]:
    """TOC statistics for the Phase 1 report.

    A parser recovering 0 entries from a book that visibly has chapters has
    failed at something the text metrics cannot see - which is why TOC
    recovery is scored as its own column.
    """
    return {
        "total_entries": len(flat),
        "chapters": sum(1 for e in flat if e.level == 1),
        "sections": sum(1 for e in flat if e.level == 2),
        "subsections": sum(1 for e in flat if e.level >= 3),
        "max_depth": max((e.level for e in flat), default=0),
    }

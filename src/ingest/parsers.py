"""PDF parser adapters — three libraries behind one interface.

WHY THIS FILE EXISTS
--------------------
Phase 1 compares PDF extraction libraries. Comparing them fairly requires
that they be *interchangeable* — same input, same call, same return type —
so that any difference in the benchmark comes from the parser and not from
how we happened to call it.

Hence one Protocol and four adapters. Adding a fifth candidate (Docling,
Marker) later means writing one class, not touching the benchmark.

WHY FOUR ADAPTERS FOR THREE LIBRARIES
-------------------------------------
PyMuPDF appears twice: once with default extraction and once with
``sort=True``. That flag reorders text blocks by position on the page
rather than by their order in the PDF's internal content stream.

That distinction matters enormously here. Several of our books use
multi-column layouts and floating figure captions. Without sorting, you
can get text in content-stream order, which may interleave columns into
nonsense. With sorting, blocks come out in reading order. Treating these
as two separate candidates lets the benchmark tell us whether it matters
for OUR books, instead of us guessing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable


# ───────────────────────────────────────────────────────────────────────
# Return types
# ───────────────────────────────────────────────────────────────────────

@dataclass
class PageExtraction:
    """One page, extracted by one parser.

    ``elapsed_ms`` is recorded per page because parser speed is a real
    selection criterion: ~3,500 pages means a parser that is 10x slower
    costs hours on every full re-ingest, and we will re-ingest often.
    """
    parser: str
    page_num: int          # 0-indexed, matching the libraries' own convention
    text: str
    elapsed_ms: float
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class TocEntry:
    """One node of a PDF's embedded table of contents.

    This is the highest-leverage metadata in the project. A textbook PDF
    usually ships a real chapter/section tree, which gives us — for free —
    section-aware chunk boundaries, filterable metadata, and citations that
    read "Géron, Ch. 4 — Training Models, p. 112" instead of "chunk_8412".
    """
    level: int             # 1 = chapter, 2 = section, 3 = subsection...
    title: str
    page: int              # 1-indexed, as PDF outlines store it
    children: list["TocEntry"] = field(default_factory=list)


# ───────────────────────────────────────────────────────────────────────
# The interface every candidate must satisfy
# ───────────────────────────────────────────────────────────────────────

@runtime_checkable
class PdfParser(Protocol):
    """Minimum surface a parser candidate must expose to be benchmarked."""

    name: str

    def page_count(self, path: Path) -> int: ...

    def extract_page(self, path: Path, page_num: int) -> PageExtraction: ...

    def extract_toc(self, path: Path) -> list[TocEntry]:
        """Return a FLAT list of TOC entries, or [] if the PDF has none.

        Flat, not nested: nesting is reconstructible from ``level``, and a
        flat list is far easier to compare across parsers. ``toc.py`` builds
        the tree when one is actually needed.
        """
        ...


# ───────────────────────────────────────────────────────────────────────
# Adapters
# ───────────────────────────────────────────────────────────────────────

class PyMuPDFParser:
    """PyMuPDF (fitz) — C-backed, fast, and the only candidate with
    first-class TOC support.

    Prior expectation: this wins. It is fast enough to re-ingest 3,500
    pages casually, and ``get_toc()`` is the feature the rest of the
    pipeline is built around. But 'prior' is not 'proven' — that is what
    the benchmark is for.
    """

    def __init__(self, sort: bool = False) -> None:
        self.sort = sort
        self.name = "pymupdf_sorted" if sort else "pymupdf"

    def page_count(self, path: Path) -> int:
        import fitz
        with fitz.open(path) as doc:
            return doc.page_count

    def extract_page(self, path: Path, page_num: int) -> PageExtraction:
        import fitz
        t0 = time.perf_counter()
        try:
            with fitz.open(path) as doc:
                text = doc[page_num].get_text("text", sort=self.sort)
            err = None
        except Exception as e:                      # noqa: BLE001
            text, err = "", f"{type(e).__name__}: {e}"
        return PageExtraction(
            parser=self.name,
            page_num=page_num,
            text=text,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            error=err,
        )

    def extract_toc(self, path: Path) -> list[TocEntry]:
        import fitz
        try:
            with fitz.open(path) as doc:
                # get_toc() -> [[level, title, page], ...]
                return [
                    TocEntry(level=lvl, title=title.strip(), page=page)
                    for lvl, title, page in doc.get_toc(simple=True)
                ]
        except Exception:                           # noqa: BLE001
            return []


class PdfPlumberParser:
    """pdfplumber — slowest of the three, but the strongest layout and
    table geometry.

    Worth benchmarking despite the speed cost: if our books lose tables
    under PyMuPDF, a hybrid strategy (fast parser for prose, pdfplumber
    for table-heavy pages) becomes a legitimate option. We can only know
    that by measuring.
    """

    name = "pdfplumber"

    def page_count(self, path: Path) -> int:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            return len(pdf.pages)

    def extract_page(self, path: Path, page_num: int) -> PageExtraction:
        import pdfplumber
        t0 = time.perf_counter()
        try:
            with pdfplumber.open(path) as pdf:
                text = pdf.pages[page_num].extract_text() or ""
            err = None
        except Exception as e:                      # noqa: BLE001
            text, err = "", f"{type(e).__name__}: {e}"
        return PageExtraction(
            parser=self.name,
            page_num=page_num,
            text=text,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            error=err,
        )

    def extract_toc(self, path: Path) -> list[TocEntry]:
        # pdfplumber exposes no outline API. Returning [] is the honest
        # answer and is itself a benchmark result: choosing pdfplumber
        # would mean sourcing the TOC from somewhere else.
        return []


class PyPdfParser:
    """pypdf — pure Python, no native dependency.

    Included as the NAIVE BASELINE. This (via ``PyPDFLoader``) is what most
    RAG tutorials use without a second thought. Having it in the table lets
    the report quantify what that default actually costs — which is a more
    useful finding than simply asserting that it is bad.
    """

    name = "pypdf"

    def page_count(self, path: Path) -> int:
        from pypdf import PdfReader
        return len(PdfReader(path).pages)

    def extract_page(self, path: Path, page_num: int) -> PageExtraction:
        from pypdf import PdfReader
        t0 = time.perf_counter()
        try:
            text = PdfReader(path).pages[page_num].extract_text() or ""
            err = None
        except Exception as e:                      # noqa: BLE001
            text, err = "", f"{type(e).__name__}: {e}"
        return PageExtraction(
            parser=self.name,
            page_num=page_num,
            text=text,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            error=err,
        )

    def extract_toc(self, path: Path) -> list[TocEntry]:
        from pypdf import PdfReader
        entries: list[TocEntry] = []
        try:
            reader = PdfReader(path)

            def walk(items, level: int = 1) -> None:
                for item in items:
                    if isinstance(item, list):
                        walk(item, level + 1)       # nested sub-outline
                    else:
                        try:
                            page = reader.get_destination_page_number(item) + 1
                        except Exception:           # noqa: BLE001
                            continue
                        entries.append(
                            TocEntry(level=level, title=str(item.title).strip(), page=page)
                        )

            walk(reader.outline)
        except Exception:                           # noqa: BLE001
            return []
        return entries


# ───────────────────────────────────────────────────────────────────────
# Registry — parse_bench.py iterates this
# ───────────────────────────────────────────────────────────────────────

ALL_PARSERS: dict[str, PdfParser] = {
    "pymupdf": PyMuPDFParser(sort=False),
    "pymupdf_sorted": PyMuPDFParser(sort=True),
    "pdfplumber": PdfPlumberParser(),
    "pypdf": PyPdfParser(),
}


def get_parser(name: str) -> PdfParser:
    """Resolve a parser by the name used in configs/experiment.yaml."""
    if name not in ALL_PARSERS:
        raise KeyError(f"Unknown parser {name!r}. Available: {sorted(ALL_PARSERS)}")
    return ALL_PARSERS[name]

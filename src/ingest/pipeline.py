"""Phase 2 orchestrator: PDF -> cleaned text -> sections -> chunks -> JSONL.

Run:
    .venv\\Scripts\\python.exe -m src.ingest.pipeline
    .venv\\Scripts\\python.exe -m src.ingest.pipeline --strategy parent_child
    .venv\\Scripts\\python.exe -m src.ingest.pipeline --force

Reads configs/experiment.yaml + configs/corpus.yaml. Writes
data/chunks/<strategy>/<short>.jsonl and updates data/processed/manifest.json.

THE BINDING PROBLEM THIS FILE SOLVES
------------------------------------
Splitters are pure text functions returning character offsets. Chunks need
page numbers and breadcrumbs. `PageMap` is the bridge: it records where each
page begins in the assembled text, so a character offset resolves back to the
physical page it came from, and from there to a citation.
"""

from __future__ import annotations

import argparse
import json
import sys
from bisect import bisect_right
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import yaml
from rich.console import Console

from .chunkers import (Chunk, Span, count_tokens, split_parent_child,
                       split_recursive, split_semantic)
from .clean import clean_document
from .manifest import PIPELINE_VERSION, DocRecord, Manifest, content_hash
from .parsers import TocEntry, get_parser
from .toc import SectionIndex

console = Console()

EXPERIMENT = Path("configs/experiment.yaml")
CORPUS = Path("configs/corpus.yaml")
CHUNK_DIR = Path("data/chunks")


# ───────────────────────────────────────────────────────────────────────
# Offset -> page
# ───────────────────────────────────────────────────────────────────────

class PageMap:
    """Maps a character offset in assembled text back to its physical page.

    Pages are joined with a separator, and we record where each begins. A
    binary search then answers "which page is character N on?".

    Chunks routinely SPAN pages, so `range_for` returns both endpoints. A chunk
    covering the bottom of p119 and top of p120 should cite p119 - where the
    passage starts - which is why `page_start` drives the citation.
    """

    SEP = "\n\n"

    def __init__(self, pages: list[str], first_physical: int) -> None:
        self._starts: list[int] = []
        self._pages: list[int] = []
        parts: list[str] = []
        pos = 0
        for i, txt in enumerate(pages):
            self._starts.append(pos)
            self._pages.append(first_physical + i)
            parts.append(txt)
            pos += len(txt) + len(self.SEP)
        self.text = self.SEP.join(parts)

    def page_at(self, offset: int) -> int:
        i = max(0, bisect_right(self._starts, offset) - 1)
        return self._pages[i] if self._pages else 0

    def range_for(self, span: Span) -> tuple[int, int]:
        return self.page_at(span.start), self.page_at(max(span.start, span.end - 1))


# ───────────────────────────────────────────────────────────────────────
# Sections
# ───────────────────────────────────────────────────────────────────────

def build_sections(flat: list[TocEntry], n_pages: int) -> list[tuple[int, int]]:
    """Page ranges [start, end) between consecutive TOC boundaries.

    `respect_section_bounds: true` means a chunk must never straddle a section
    boundary, so we chunk each section independently. Using every TOC entry
    (not just chapters) gives fine-grained sections, which is what we want:
    a chunk that mixes the end of "4.1 Linear Regression" with the start of
    "4.2 Gradient Descent" answers neither question well.

    With no outline, the whole document is one section.
    """
    bounds = sorted({e.page for e in flat if 0 <= e.page < n_pages})
    if not bounds:
        return [(0, n_pages)]
    if bounds[0] > 0:
        bounds.insert(0, 0)          # front matter before the first entry
    return [(a, b) for a, b in zip(bounds, bounds[1:] + [n_pages]) if b > a]


# ───────────────────────────────────────────────────────────────────────
# Per-document ingestion
# ───────────────────────────────────────────────────────────────────────

def ingest_document(pdf: Path, meta: dict, cfg: dict) -> tuple[list[Chunk], list[Chunk], dict]:
    """Parse, clean, section and chunk one book.

    Returns (retrievable_chunks, parent_chunks, stats). `parent_chunks` is
    non-empty only for parent_child, where children are what get embedded and
    parents are what the LLM eventually reads.
    """
    ccfg, chcfg = cfg["corpus"], cfg["chunking"]
    strategy = chcfg["strategy"]
    parser = get_parser(ccfg["parser"])

    n_pages = parser.page_count(pdf)
    raw = [parser.extract_page(pdf, i).text for i in range(n_pages)]

    cleaned, cstats = clean_document(
        raw,
        start_physical=0,
        strip_headers=ccfg.get("strip_headers_footers", True),
        do_dehyphenate=ccfg.get("dehyphenate", True),
        detect_offset=False,          # offsets are pre-measured in corpus.yaml
    )

    flat = parser.extract_toc(pdf) if ccfg.get("use_embedded_toc", True) else []
    offset = meta.get("page_offset")
    idx = SectionIndex(flat, page_offset=offset or 0)
    cite_kind = "printed" if offset is not None else "physical"

    chunks: list[Chunk] = []
    parents: list[Chunk] = []
    doc_id = content_hash(pdf)
    short = meta["short"]
    seq = 0

    for p0, p1 in build_sections(flat, n_pages):
        pm = PageMap(cleaned[p0:p1], p0)
        if not pm.text.strip():
            continue

        def make(span: Span, kind: str, parent_id: str | None = None) -> Chunk:
            nonlocal seq
            ps, pe = pm.range_for(span)
            seq += 1

            # Front matter has no printed Arabic page number, so applying the
            # offset there produces nonsense: Bishop's offset is -19, making
            # physical page 13 render as "p. -6". Arithmetically right,
            # semantically absurd, and it would appear in a real citation.
            # Below 1, fall back to citing the physical PDF page.
            printed = ps + (offset or 0)
            if offset is None or printed < 1:
                # Physical citations render 1-BASED: `ps` is a 0-based index,
                # but a PDF reader labels the first sheet "1". Citing "PDF p. 0"
                # is correct internally and confusing to the human who has to
                # verify it. Only the display value shifts - page_start/page_end
                # stay 0-based, so golden-set labels are unaffected.
                cite_page, kind_ = ps + 1, "physical"
            else:
                cite_page, kind_ = printed, "printed"

            return Chunk(
                chunk_id=f"{short}:{kind}:{seq:06d}",
                doc_id=doc_id,
                book=short,
                text=span.text,
                n_tokens=count_tokens(span.text),
                page_start=ps,
                page_end=pe,
                cite_page=cite_page,
                cite_kind=kind_,
                breadcrumb=idx.breadcrumb(ps).path,
                chunker=strategy,
                parent_id=parent_id,
            )

        if strategy == "parent_child":
            pc_cfg = chcfg["parent_child"]
            for pc in split_parent_child(
                pm.text, pc_cfg["parent_size"], pc_cfg["child_size"], count_tokens
            ):
                parent = make(pc.parent, "par")
                parents.append(parent)
                chunks.extend(make(c, "chd", parent.chunk_id) for c in pc.children)

        elif strategy == "recursive":
            chunks.extend(
                make(s, "rec")
                for s in split_recursive(
                    pm.text, chcfg["chunk_size"], chcfg["chunk_overlap"], count_tokens
                )
            )

        elif strategy == "semantic":
            # Unblocked in Phase 4: the embedder registry now exists.
            #
            # COST, stated before it is incurred. Recursive chunking is pure
            # string arithmetic and runs in seconds. Semantic chunking embeds
            # EVERY SENTENCE in the corpus first - roughly 90k sentences here
            # against 5.8k chunks, so about 15x the embedding work of a full
            # corpus pass, or several hours on this CPU. It is the single most
            # expensive preprocessing step in the project.
            #
            # Which embedder is deliberately NOT the D3 winner: the chunker
            # must be decidable before the embedder, or D2 and D3 become
            # circular. We use the config's current embedding.model and record
            # it, so the dependency is visible rather than hidden.
            sem_cfg = chcfg.get("semantic", {})
            from src.embed.ollama_embedder import get_embedder
            embedder = get_embedder(cfg["embedding"]["model"])
            chunks.extend(
                make(s, "sem")
                for s in split_semantic(
                    pm.text, embedder,
                    percentile=sem_cfg.get("breakpoint_percentile", 95),
                    buffer_size=sem_cfg.get("buffer_size", 1),
                )
            )
        else:
            raise KeyError(f"unknown chunking.strategy: {strategy!r}")

    return chunks, parents, {
        "n_pages": n_pages,
        "toc_entries": len(flat),
        "header_lines_removed": cstats.header_lines_removed,
        "hyphens_joined": cstats.hyphens_joined,
    }


# ───────────────────────────────────────────────────────────────────────
# Runner
# ───────────────────────────────────────────────────────────────────────

def _write(path: Path, chunks: list[Chunk]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 2 ingestion.")
    ap.add_argument("--strategy", help="override chunking.strategy")
    ap.add_argument("--force", action="store_true", help="reprocess everything")
    args = ap.parse_args()

    cfg = yaml.safe_load(EXPERIMENT.read_text(encoding="utf-8"))
    corpus = yaml.safe_load(CORPUS.read_text(encoding="utf-8"))["books"]
    if args.strategy:
        cfg["chunking"]["strategy"] = args.strategy
    strategy = cfg["chunking"]["strategy"]

    raw_dir = Path(cfg["corpus"]["raw_dir"])
    man = Manifest.load()
    plan = man.plan(raw_dir)

    out_dir = CHUNK_DIR / strategy

    if args.force:
        todo = sorted(raw_dir.glob("*.pdf"))
    else:
        # Content changes always force reprocessing, whatever the strategy.
        todo = list(plan.new + plan.changed)
        known = {p.resolve() for p in todo}

        # Everything else is judged PER STRATEGY, not per document. A book can
        # be current under `recursive` while its `parent_child` chunks are
        # missing or stale. Judging by document alone meant that after a
        # version bump, `pipeline` then `pipeline --strategy parent_child`
        # reprocessed only the first: the first run stamped the document
        # current, so the second saw "unchanged" and skipped, leaving stale
        # chunks on disk under a manifest claiming they were fresh.
        for pdf in sorted(raw_dir.glob("*.pdf")):
            meta = corpus.get(pdf.name)
            if not meta or pdf.resolve() in known:
                continue
            rec = man.records.get(content_hash(pdf))
            missing = not (out_dir / f"{meta['short']}.jsonl").exists()
            if missing or rec is None or rec.strategy_stale(strategy):
                todo.append(pdf)

    # Report the REAL queue, not the document-level plan. Printing
    # plan.summary() before the per-strategy additions produced
    # "processing 0" immediately followed by 8,944 chunks being written -
    # a log line that makes every later run harder to trust.
    console.print(
        f"[cyan]{plan.summary()}[/cyan]  strategy={strategy}  "
        f"[bold]-> processing {len(todo)}[/bold]"
    )

    if not todo:
        console.print(
            f"[green]nothing to do — manifest current and {strategy} chunks exist[/green]"
        )
        return 0
    total = 0

    for pdf in todo:
        meta = corpus.get(pdf.name)
        if meta is None:
            console.print(f"[yellow]skip {pdf.name}: not in corpus.yaml[/yellow]")
            continue

        chunks, parents, st = ingest_document(pdf, meta, cfg)
        _write(out_dir / f"{meta['short']}.jsonl", chunks)
        if parents:
            _write(out_dir / f"{meta['short']}.parents.jsonl", parents)

        toks = [c.n_tokens for c in chunks] or [0]

        # Regression guard. An oversized chunk is a SILENT failure downstream:
        # mxbai-embed-large has a 512-token context and truncates without
        # error, so retrieval just quietly degrades. Surface it loudly here.
        limit = (
            cfg["chunking"]["parent_child"]["child_size"]
            if strategy == "parent_child"
            else cfg["chunking"]["chunk_size"]
        )
        over = [c for c in chunks if c.n_tokens > limit]

        console.print(
            f"  [green]{meta['short']:12s}[/green] pages={st['n_pages']:4d} "
            f"toc={st['toc_entries']:4d} chunks={len(chunks):5d} "
            f"parents={len(parents):4d} "
            f"tok(avg/max)={sum(toks)//len(toks):4d}/{max(toks):5d} "
            f"hyphens={st['hyphens_joined']:4d} furniture={st['header_lines_removed']:5d}"
        )
        if over:
            console.print(
                f"       [red]WARNING {len(over)} chunks exceed limit {limit} "
                f"(max {max(c.n_tokens for c in over)})[/red]"
            )
        if chunks:
            console.print(f"       e.g. {chunks[len(chunks)//2].cite()}")

        # Merge, don't replace: a document may have chunks under several
        # strategies and we want the counts for all of them side by side.
        doc_id = content_hash(pdf)
        prior = man.records.get(doc_id)
        counts = dict(prior.n_chunks) if prior else {}
        counts[strategy] = len(chunks)
        versions = dict(prior.strategy_versions) if prior else {}
        versions[strategy] = PIPELINE_VERSION

        man.upsert(DocRecord(
            doc_id=doc_id, filename=pdf.name, short=meta["short"],
            title=meta["title"], file_size=pdf.stat().st_size,
            n_pages=st["n_pages"], toc_entries=st["toc_entries"],
            page_offset=meta.get("page_offset"), parser=cfg["corpus"]["parser"],
            pipeline_version=PIPELINE_VERSION,
            ingested_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            n_chunks=counts,
            strategy_versions=versions,
        ))
        total += len(chunks)

    man.save()
    console.print(f"[bold green]{total} chunks -> {out_dir}[/bold green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())

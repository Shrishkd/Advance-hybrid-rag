"""Incremental ingestion manifest — process only what actually changed.

WHY CONTENT HASHES, NOT FILENAMES
---------------------------------
This project re-ingests constantly. Phase 4 alone sweeps three chunking
strategies across four embedding models, and each sweep rebuilds chunks for
~3,500 pages. Reprocessing unchanged books every time is wasted minutes that
compound into hours.

The naive approach keys on filename + mtime. It fails in both directions, and
this project has already hit one of them:

  * SAME NAME, NEW CONTENT. On 2026-09-19 the Goodfellow PDF was swapped for a
    different edition carrying an embedded TOC. Had the filename been reused,
    a filename-keyed manifest would have reported "unchanged" and silently kept
    serving chunks from a book that no longer existed on disk.

  * NEW NAME, SAME CONTENT. Renaming or re-downloading a file changes mtime and
    path while the bytes are identical. A filename-keyed manifest reprocesses
    800 pages for nothing.

Hashing the bytes answers the only question that matters — *is this the same
document?* — and is cheap: ~100 ms for a 26 MB PDF.

WHY A PIPELINE VERSION TOO
--------------------------
Content hashing alone is insufficient, and this is the subtle half. If we fix a
bug in the de-hyphenator, every PDF is byte-identical but every derived chunk
is now WRONG. The document did not change; our interpretation of it did.

So a document needs reprocessing when EITHER:
    the content hash changed        (the input changed), OR
    the pipeline version changed    (our reading of the input changed)

Bump PIPELINE_VERSION whenever cleaning, chunking or metadata logic changes in
a way that alters output. Forgetting to bump it is how you end up debugging a
retrieval bug that is really a stale-cache bug.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# ── BUMP THIS when cleaning/chunking/metadata logic changes output ──────
#   v1  initial: pymupdf (D1), TOC breadcrumbs, dehyphenation, header strip
#   v2  chunk sizing now counts the real merged text instead of summing atom
#       counts (BPE is not additive) - v1 emitted 646-token chunks against a
#       512 limit. Every v1 chunk is wrong, so every document must reprocess.
#   v3  citations fall back to physical pages when printed would be < 1. Front
#       matter has no printed Arabic number, so Bishop (offset -19) rendered
#       physical p13 as "p. -6". Affects cite_page/cite_kind only; chunk text
#       and page ranges are unchanged, so existing golden labels stay valid.
#   v4  physical citations render 1-based ("PDF p. 1" for the first sheet), to
#       match what a PDF reader shows. Page RANGES in metadata stay 0-based;
#       only the human-facing cite_page shifts. Golden labels unaffected.
PIPELINE_VERSION = "v4"

MANIFEST_PATH = Path("data/processed/manifest.json")
_HASH_CHUNK = 1 << 20      # 1 MiB


# ───────────────────────────────────────────────────────────────────────
# Hashing
# ───────────────────────────────────────────────────────────────────────

def content_hash(path: Path) -> str:
    """SHA-256 of a file's bytes, truncated to 16 hex chars.

    Streamed rather than read whole: these PDFs reach 26 MB and there is only
    ~5.5 GB of usable RAM on this machine, most of which is spoken for by
    models. Never load a whole file when a stream will do.

    16 hex chars = 64 bits. Collision probability across a corpus of tens of
    documents is negligible, and short IDs keep the manifest readable by a
    human, which matters because this file gets committed and diffed.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(_HASH_CHUNK):
            h.update(block)
    return h.hexdigest()[:16]


# ───────────────────────────────────────────────────────────────────────
# Records
# ───────────────────────────────────────────────────────────────────────

@dataclass
class DocRecord:
    """One ingested document, as recorded in the manifest."""
    doc_id: str                 # == content_hash; stable across renames
    filename: str
    short: str                  # citation name, e.g. "Géron"
    title: str
    file_size: int
    n_pages: int
    toc_entries: int
    page_offset: int | None     # printed - physical; None = not yet detected
    parser: str
    pipeline_version: str
    ingested_at: str
    n_chunks: dict[str, int] = field(default_factory=dict)      # strategy -> count
    strategy_versions: dict[str, str] = field(default_factory=dict)  # strategy -> version

    @property
    def is_stale(self) -> bool:
        """True when our processing logic moved on since this was ingested."""
        return self.pipeline_version != PIPELINE_VERSION

    def strategy_stale(self, strategy: str) -> bool:
        """True when THIS strategy's chunks predate the current pipeline.

        Document-level staleness is not enough, and the gap bit us: running
        `recursive` then `parent_child` after a version bump reprocessed only
        the first. The recursive run stamped the DOCUMENT as current, so the
        parent_child run saw "unchanged" and skipped - leaving stale chunks on
        disk while the manifest claimed they were fresh.

        Chunks are per-strategy, so freshness must be tracked per-strategy.
        """
        return self.strategy_versions.get(strategy) != PIPELINE_VERSION


@dataclass
class IngestPlan:
    """What a run must actually do. The whole point of the manifest."""
    new: list[Path] = field(default_factory=list)
    changed: list[Path] = field(default_factory=list)      # same name, new bytes
    stale: list[Path] = field(default_factory=list)        # our code changed
    unchanged: list[Path] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)       # doc_ids no longer on disk

    @property
    def to_process(self) -> list[Path]:
        return self.new + self.changed + self.stale

    def summary(self) -> str:
        """Document-level classification only.

        Deliberately does NOT report a processing count. Chunks are per
        strategy, so the real queue is decided by the caller after checking
        per-strategy freshness - this plan cannot know it. Printing a count
        here produced "processing 0" immediately before 8,944 chunks were
        written.
        """
        return (
            f"new={len(self.new)} changed={len(self.changed)} "
            f"stale={len(self.stale)} unchanged={len(self.unchanged)} "
            f"removed={len(self.removed)}"
        )


# ───────────────────────────────────────────────────────────────────────
# Manifest
# ───────────────────────────────────────────────────────────────────────

class Manifest:
    """Persistent record of what has been ingested, keyed by content hash."""

    def __init__(self, records: dict[str, DocRecord] | None = None) -> None:
        self.records: dict[str, DocRecord] = records or {}

    # ---- persistence ----------------------------------------------------

    @classmethod
    def load(cls, path: Path = MANIFEST_PATH) -> "Manifest":
        """Load, or return an empty manifest if none exists.

        A missing manifest is the normal first-run state, not an error.
        """
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls({k: DocRecord(**v) for k, v in raw.get("documents", {}).items()})

    def save(self, path: Path = MANIFEST_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pipeline_version": PIPELINE_VERSION,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "documents": {k: asdict(v) for k, v in self.records.items()},
        }
        # indent=2 + sorted keys: this file is committed, so its diffs should
        # be readable rather than one 40 KB line.
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

    # ---- planning -------------------------------------------------------

    def plan(self, raw_dir: Path) -> IngestPlan:
        """Compare disk against the manifest and decide what to process.

        Classification, in priority order:
          new       - hash absent from the manifest
          changed   - filename known, but its bytes now hash differently
          stale     - same bytes, but PIPELINE_VERSION moved on
          unchanged - skip entirely
          removed   - in the manifest, absent from disk
        """
        plan = IngestPlan()
        seen: set[str] = set()
        by_name = {r.filename: r for r in self.records.values()}

        for pdf in sorted(raw_dir.glob("*.pdf")):
            h = content_hash(pdf)
            seen.add(h)

            if h in self.records:
                (plan.stale if self.records[h].is_stale else plan.unchanged).append(pdf)
            elif pdf.name in by_name:
                # Same filename, different bytes: the file was replaced in place.
                plan.changed.append(pdf)
            else:
                plan.new.append(pdf)

        plan.removed = [h for h in self.records if h not in seen]
        return plan

    # ---- mutation -------------------------------------------------------

    def upsert(self, rec: DocRecord) -> None:
        self.records[rec.doc_id] = rec

    def drop(self, doc_ids: list[str]) -> None:
        """Forget documents no longer on disk.

        Called explicitly rather than automatically inside plan(): removing a
        book must also invalidate its chunks and vectors downstream, so the
        caller decides when that cascade is safe to run.
        """
        for d in doc_ids:
            self.records.pop(d, None)

    def __len__(self) -> int:
        return len(self.records)

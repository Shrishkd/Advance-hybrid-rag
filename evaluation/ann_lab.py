"""D4 experiment: Flat vs HNSW vs IVF vs IVFPQ on our real vectors.

    python -m evaluation.ann_lab
    python -m evaluation.ann_lab --embedder nomic --strategy recursive --queries 200

Writes reports/04a_ann_lab.md.

WHY THIS RUNS BEFORE THE GOLDEN SET EXISTS
------------------------------------------
Every other Phase 4 experiment needs labelled questions. This one does not,
because ANN recall is measured against EXACT search, not against human labels:

    recall@k = |approx_top_k  INTERSECT  flat_top_k| / k

That answers "how much does approximation cost?" - a different question from
"is retrieval finding the right passages?", which needs the golden set. They
are frequently conflated. An index with recall@10 = 0.98 against flat is
reproducing flat's answers well; if flat's answers are bad, 0.98 is 98% of
bad.

VECTOR CACHING
--------------
Embedding ~6k chunks on a CPU-only box takes minutes. Vectors are cached to
data/processed/vectors/<embedder>_<strategy>.npy, keyed so a different
embedder or chunking strategy never silently reuses the wrong array.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.report_utils import decision_section          # noqa: E402
from src.embed.corpus_cache import corpus_vectors, is_cached  # noqa: E402
from src.index.faiss_lab import evaluate_index          # noqa: E402

console = Console()

EXPERIMENT = Path("configs/experiment.yaml")
REPORT = Path("reports/04a_ann_lab.md")
KINDS = ("flat", "hnsw", "ivf", "ivfpq")


def load_chunks(strategy: str) -> list[dict]:
    d = Path("data/chunks") / strategy
    rows: list[dict] = []
    for f in sorted(d.glob("*.jsonl")):
        if f.name.endswith(".parents.jsonl"):
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def get_vectors(embedder_name: str, strategy: str, texts: list[str]) -> np.ndarray:
    """Embed the corpus, or reuse a cached array.

    Delegates to src/embed/corpus_cache so this lab and the retrieval lab can
    never disagree about the cache key. They did not share it originally, and
    two copies of a cache key is a silent-wrong bug waiting to happen.
    """
    def log(done: int, total: int, secs: float) -> None:
        if done == -1:
            console.print("[yellow]stale vector cache - re-embedding[/yellow]")
        else:
            console.print(f"  [dim]{done}/{total}  {secs:.0f}s[/dim]")

    if is_cached(embedder_name, strategy, len(texts)):
        console.print(f"[dim]reusing cached vectors for "
                      f"{embedder_name}/{strategy}[/dim]")
    else:
        console.print(f"[cyan]embedding {len(texts)} chunks with "
                      f"{embedder_name}[/cyan]")
    return corpus_vectors(embedder_name, strategy, texts, progress=log)


def main() -> int:
    ap = argparse.ArgumentParser(description="ANN index comparison (D4).")
    ap.add_argument("--embedder", default=None)
    ap.add_argument("--strategy", default=None)
    ap.add_argument("--queries", type=int, default=200,
                    help="held-out corpus vectors used as probe queries")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    cfg = yaml.safe_load(EXPERIMENT.read_text(encoding="utf-8"))
    embedder = args.embedder or cfg["embedding"]["model"]
    strategy = args.strategy or cfg["chunking"]["strategy"]

    chunks = load_chunks(strategy)
    if not chunks:
        console.print(f"[red]no chunks for strategy {strategy!r}[/red]")
        return 1

    vectors = get_vectors(embedder, strategy, [c["text"] for c in chunks])

    # Probe queries are corpus vectors themselves. That is standard practice
    # for ANN benchmarking: the question is how faithfully each index
    # reproduces exact search over THIS distribution, and corpus vectors are
    # drawn from exactly that distribution. Real user queries come from a
    # different one, which is why this measures approximation error only and
    # is NOT a substitute for the golden-set experiments.
    rng = np.random.default_rng(7)
    qi = rng.choice(len(vectors), size=min(args.queries, len(vectors)), replace=False)
    queries = vectors[qi]

    console.print(f"[cyan]corpus={len(vectors)} dim={vectors.shape[1]} "
                  f"queries={len(queries)} k={args.k}[/cyan]\n")

    rows, truth = [], None
    for kind in KINDS:
        stats, ids = evaluate_index(
            kind, vectors, queries, k=args.k, cfg=cfg["index"], exact_truth=truth
        )
        if kind == "flat":
            truth = ids                  # every approximate index scored against this
        rows.append(stats.row())
        console.print(
            f"  [green]{kind:7s}[/green] recall={stats.recall_at_k:.4f} "
            f"p50={stats.search_ms_p50:7.3f}ms p95={stats.search_ms_p95:7.3f}ms "
            f"build={stats.build_ms:7.1f}ms mem={stats.memory_mb:6.1f}MB"
        )

    import pandas as pd
    df = pd.DataFrame(rows)

    flat = next(r for r in rows if r["index"] == "flat")
    speedups = [
        f"- **{r['index']}**: {flat['p50_ms'] / max(r['p50_ms'], 1e-9):.2f}x "
        f"{'faster' if r['p50_ms'] < flat['p50_ms'] else 'SLOWER'} than flat, "
        f"recall@{args.k} {r['recall@k']:.4f}, "
        f"{r['mem_mb'] / max(flat['mem_mb'], 1e-9):.2f}x memory"
        for r in rows if r["index"] != "flat"
    ]

    decision = decision_section(REPORT, [
        "## Decision",
        "",
        "- **Chosen index:** _fill in_",
        "- Record as **D4** in `plan.md`; set `index.type` in",
        "  `configs/experiment.yaml` and upgrade its tag to `[PROVEN]`.",
        "",
    ])

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        "\n".join([
            "# Phase 4a — ANN Index Comparison (D4)",
            "",
            f"Corpus: **{len(vectors)}** chunks (`{strategy}`), "
            f"embedder `{embedder}`, dim {vectors.shape[1]}.",
            f"Probes: {len(queries)} held-out corpus vectors, k={args.k}.",
            "",
            "**Recall is measured against exact flat search**, not against human",
            "labels. It answers *how much does approximation cost*, not *is",
            "retrieval good*. The latter needs the golden set and lives in",
            "`04_retrieval_lab.md`.",
            "",
            df.to_markdown(index=False),
            "",
            "## Relative to exact search",
            "",
            *speedups,
            "",
            *decision,
        ]),
        encoding="utf-8",
    )
    console.print(f"\n[bold green]wrote {REPORT}[/bold green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())

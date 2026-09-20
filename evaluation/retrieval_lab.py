"""Phase 4 retrieval lab — scoring retrieval configs against the golden set.

    python -m evaluation.retrieval_lab --embed-only          # precompute vectors
    python -m evaluation.retrieval_lab --experiment embedder # D3 bake-off
    python -m evaluation.retrieval_lab --experiment chunker  # D2 bake-off

WHAT THIS MEASURES, AND WHY IT IS CHEAP
---------------------------------------
Every number here comes from comparing a ranked list of chunks against
labelled (book, page_range) ground truth. No LLM judge, no cost, no rate
limit, no self-preference bias, perfectly reproducible. That is what lets
Phase 4 sweep dozens of configurations while Phase 8's judged metrics run only
on the final few.

Contrast with `04a_ann_lab.md`, which measured recall against EXACT SEARCH -
"how much does approximation cost". This file measures recall against HUMAN
LABELS - "is retrieval finding the right passages". Different questions.

ONE VARIABLE AT A TIME
----------------------
Each experiment sweeps exactly one axis and pins everything else to the
current value in configs/experiment.yaml:

    embedder  sweep nomic|bge-m3|embeddinggemma|mxbai, pin chunker+index
    chunker   sweep recursive|semantic|parent_child, pin the winning embedder

The index is pinned to `flat` throughout, by D4. That is not incidental: an
approximate index would inject its own error into every row and make a
two-point difference between embedders unattributable.
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

from evaluation.golden_schema import GoldenItem, load                # noqa: E402
from evaluation.retrieval_metrics import (                           # noqa: E402
    RetrievedChunk, aggregate, count_relevant_in_corpus,
    precision_ceiling_at_k, score_one,
)
from src.retrieve.fusion import (                                    # noqa: E402
    reciprocal_rank_fusion, weighted_fusion,
)
from evaluation.report_utils import decision_section                # noqa: E402
from evaluation.results_db import record                             # noqa: E402
from src.embed.corpus_cache import corpus_vectors, is_cached         # noqa: E402

console = Console()

EXPERIMENT = Path("configs/experiment.yaml")
GOLDEN = Path("data/golden/golden_50.jsonl")
SYNTHETIC = Path("data/golden/synthetic_retrieval.jsonl")

DATASETS = {"golden": GOLDEN, "synthetic": SYNTHETIC}
REPORT_DIR = Path("reports")
PERQ_DIR = REPORT_DIR / "perq"
KS = (1, 3, 5, 10, 20)


# ───────────────────────────────────────────────────────────────────────
# Corpus loading
# ───────────────────────────────────────────────────────────────────────

def load_chunks(strategy: str) -> list[dict]:
    """Load every chunk for a strategy, in a stable order.

    Sorted by filename so the order is reproducible across runs and machines.
    This order is what the vector cache is keyed against - see corpus_cache.

    PARENT-CHILD IS SCORED ON THE CHILD, AND THAT IS A REAL CHOICE
    --------------------------------------------------------------
    `*.parents.jsonl` is skipped, so parent_child is scored on the CHILD
    chunks - the things actually embedded and matched. The alternative,
    scoring on the parent each child belongs to, would credit a hit with the
    parent's much wider page span. A 2048-token parent covers roughly four
    times the pages of a 512-token recursive chunk, so it would overlap
    ground-truth ranges far more often and post a higher recall for reasons
    that have nothing to do with retrieval quality.

    That would break comparability with `recursive` - and comparing chunkers
    is the entire point of D2. The honest measure of retrieval is what the
    retriever matched. What the GENERATOR then reads (the parent) is a
    separate question, measured in Phase 5 as context sufficiency.
    """
    d = Path("data/chunks") / strategy
    rows: list[dict] = []
    for f in sorted(d.glob("*.jsonl")):
        if f.name.endswith(".parents.jsonl"):
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def flat_search(vectors: np.ndarray, query: np.ndarray, k: int):
    """Exact inner-product search, in numpy. Byte-identical to IndexFlatIP.

    WHY NOT FAISS HERE
    ------------------
    faiss-cpu and torch each bundle their own OpenMP runtime, and importing
    both into one process on Windows raises:

        OMP: Error #15: Initializing libiomp5md.dll, but found
        libomp140.x86_64.dll already initialized

    The documented workaround, KMP_DUPLICATE_LIB_OK=TRUE, is described by
    Intel as unsupported and liable to "silently produce incorrect results".
    Setting that inside a measurement pipeline is exactly the wrong trade -
    this project's recurring failure mode is numbers that are wrong but
    plausible, and that flag manufactures them.

    The real observation is that D4 already settled this: flat exact search
    over L2-normalised vectors IS a matrix multiply. faiss adds nothing at
    5,796 rows. Verified equal on this corpus, with faiss at 1.50 ms/query
    and numpy at 1.73 ms/query - a 0.2 ms difference against 1,000+ ms of
    generation.

    faiss remains in use in evaluation/ann_lab.py, which compares HNSW/IVF/PQ
    and never imports torch. The two live in separate processes, which is the
    clean fix rather than a flag that disables a safety check.
    """
    sims = (query @ vectors.T).ravel()
    if k >= len(sims):
        order = np.argsort(-sims)
    else:
        part = np.argpartition(-sims, k)[:k]
        order = part[np.argsort(-sims[part])]
    return order, sims[order]


def as_retrieved(chunk: dict) -> RetrievedChunk:
    """Project a stored chunk down to what the metrics need."""
    return RetrievedChunk(
        book=chunk["book"],
        page_start=chunk["page_start"],
        page_end=chunk["page_end"],
        chunk_id=chunk.get("chunk_id", ""),
    )


# ───────────────────────────────────────────────────────────────────────
# Which items can be scored  ← SHRISH WRITES THIS
# ───────────────────────────────────────────────────────────────────────

def scorable_items(items: list[GoldenItem]) -> list[GoldenItem]:
    """Select the golden items that retrieval metrics may be averaged over.

    THE PROBLEM THIS SOLVES
    -----------------------
    `recall_at_k` returns 0.0 when an item has no ground-truth contexts
    (retrieval_metrics.py, "if not contexts: return 0.0"). Our 2 unanswerable
    items have no contexts BY DESIGN - that is what makes them unanswerable.

    Average them in and every configuration we ever test carries a constant
    2/50 = 4-point penalty that no retrieval improvement can remove. Worse,
    it is a penalty for doing the RIGHT thing: there is nothing correct to
    retrieve, so "finding nothing" is success, and scoring it as recall 0.0
    inverts the meaning.

    Unanswerable items are not scored here. They are scored in Phase 7/8 on
    whether the system REFUSED - a generation question, not a retrieval one.

    The ambiguous items are different and must stay IN: their discussion is
    genuinely in the corpus (that is why they carry contexts and
    expected_behaviour == "clarify"), so retrieval can and should be measured
    on them. Only the response behaviour differs.

    Args:
        items: the full golden set.

    Returns:
        The subset whose retrieval performance is meaningful - i.e. those
        that have at least one labelled ground-truth context.

    Worked example:
        >>> from evaluation.golden_schema import ContextRef as C
        >>> mk = lambda q, ctx, ans: GoldenItem(
        ...     qid=q, question="", question_type="factual", difficulty="easy",
        ...     answerable=ans, ground_truth_answer="a",
        ...     ground_truth_contexts=ctx)
        >>> pool = [
        ...     mk("a", [C("Bishop", 10, 11)], True),    # normal   -> keep
        ...     mk("b", [C("Huyen", 40, 40)], True),     # ambiguous-> keep
        ...     mk("c", [], False),                      # unanswer -> DROP
        ... ]
        >>> [i.qid for i in scorable_items(pool)]
        ['a', 'b']
    """
    return [i for i in items if i.ground_truth_contexts]


# ───────────────────────────────────────────────────────────────────────
# A single retrieval run
# ───────────────────────────────────────────────────────────────────────

def embed_corpus(embedder: str, strategy: str, chunks: list[dict],
                 backend: str = "ollama") -> np.ndarray:
    """Embed (or load) the corpus, logging progress for the slow path."""
    def log(done: int, total: int, secs: float) -> None:
        if done == -1:
            console.print("[yellow]stale vector cache — re-embedding[/yellow]")
            return
        console.print(f"  [dim]{done}/{total}  {secs:.0f}s[/dim]")

    if not is_cached(embedder, strategy, len(chunks), backend):
        console.print(f"[cyan]embedding {len(chunks)} chunks with "
                      f"{embedder} (minutes, not seconds)[/cyan]")
    return corpus_vectors(embedder, strategy, [c["text"] for c in chunks],
                          progress=log, backend=backend)


def run_config(
    embedder: str,
    strategy: str,
    items: list[GoldenItem],
    top_k: int = 20,
    mode: str = "dense",
    w_dense: float = 0.5,
    rrf_k: int = 60,
    rerank: str | None = None,
    backend: str = "ollama",
) -> tuple[dict, dict]:
    """Score one retrieval configuration against the golden set.

    Args:
        mode: dense | bm25 | hybrid_rrf | hybrid_weighted.
            The dense side always uses flat exact search (D4), so any
            difference between modes is attributable to the SIGNALS being
            fused, never to index approximation error.
        w_dense: weight on the dense list for hybrid_weighted. The sparse
            side gets 1 - w_dense.
        rrf_k: RRF smoothing constant; 60 is the published default.
        rerank: cross-encoder name, or None. The reranker REORDERS the whole
            stage-1 top_k rather than truncating it, so recall@top_k is
            identical with and without reranking - by construction. That is
            deliberate: it puts stage 1's recall ceiling in the same table as
            the reranked numbers, making it obvious that a reranker can only
            move documents UP, never retrieve one stage 1 missed.

    Returns:
        (aggregate_metrics, extras) where extras carries latency, the
        label-implied precision ceiling, and per-question rows.
    """
    chunks = load_chunks(strategy)
    if not chunks:
        raise FileNotFoundError(f"no chunks for strategy {strategy!r}")

    needs_dense = mode != "bm25"
    needs_sparse = mode != "dense"

    vectors = emb = None
    dim = 0
    if needs_dense:
        vectors = embed_corpus(embedder, strategy, chunks, backend)
        dim = int(vectors.shape[1])                      # D4: exact, zero error
        from src.embed.ollama_embedder import get_embedder
        emb = get_embedder(embedder)

    bm25 = None
    if needs_sparse:
        from src.index.bm25_index import BM25Index
        t0 = time.perf_counter()
        bm25 = BM25Index([c["text"] for c in chunks])
        console.print(f"  [dim]bm25 index built in "
                      f"{time.perf_counter()-t0:.1f}s[/dim]")

    # Retrieve deeper than top_k on each arm before fusing. Fusing two lists
    # truncated at top_k would discard a document that ranked 25th on one arm
    # and 2nd on the other - exactly the agreement RRF exists to reward.
    arm_k = top_k * 3

    reranker = None
    if rerank:
        from src.retrieve.rerank import CrossEncoderReranker
        reranker = CrossEncoderReranker(rerank)
        if not reranker.local_ok:
            console.print(f"[yellow]{rerank} needs ~{reranker.ram_mb} MB — "
                          "intended for Colab, not this box[/yellow]")
        console.print(f"  [dim]reranker {rerank} (~{reranker.ram_mb} MB)[/dim]")

    per_q: list[dict] = []
    rows: list[dict] = []
    lat: list[float] = []
    rerank_ms: list[float] = []
    ceil5: list[float] = []

    for it in items:
        t0 = time.perf_counter()
        # embed_query applies the model's query prefix; embed_documents applies
        # the document prefix. Using the wrong one is the classic silent bug -
        # no error, just worse numbers.
        dense_ids: list[int] = []
        dense_scored: list[tuple[int, float]] = []
        if needs_dense:
            qv = emb.embed_query(it.question).reshape(1, -1).astype(np.float32)
            nn, sims = flat_search(vectors, qv, arm_k)
            dense_ids = [int(i) for i in nn]
            dense_scored = [(int(i), float(sc)) for i, sc in zip(nn, sims)]

        sparse_ids: list[int] = []
        sparse_scored: list[tuple[int, float]] = []
        if needs_sparse:
            order, scores = bm25.search(it.question, arm_k)
            sparse_ids = [int(i) for i in order]
            sparse_scored = [(int(i), float(sc)) for i, sc in zip(order, scores)]

        if mode == "dense":
            final = dense_ids[:top_k]
        elif mode == "bm25":
            final = sparse_ids[:top_k]
        elif mode == "hybrid_rrf":
            final = [d for d, _ in reciprocal_rank_fusion(
                [dense_ids, sparse_ids], k=rrf_k, top_n=top_k)]
        elif mode == "hybrid_weighted":
            final = [d for d, _ in weighted_fusion(
                dense_scored, sparse_scored,
                w_dense=w_dense, w_sparse=1.0 - w_dense, top_n=top_k)]
        else:
            raise KeyError(f"unknown retrieval mode {mode!r}")

        if reranker is not None:
            final, rstats = reranker.rerank(
                it.question, final, [chunks[i]["text"] for i in final],
                top_n=len(final),          # reorder, do not truncate
            )
            rerank_ms.append(rstats.total_ms)

        lat.append((time.perf_counter() - t0) * 1000)

        retrieved = [as_retrieved(chunks[i]) for i in final]
        ceil5.append(precision_ceiling_at_k(
            count_relevant_in_corpus(chunks, it.ground_truth_contexts), 5))
        m = score_one(retrieved, it.ground_truth_contexts, ks=KS)
        per_q.append(m)
        rows.append({
            "qid": it.qid,
            **{k: round(v, 6) for k, v in m.items()},
            "type": it.question_type,
            "difficulty": it.difficulty,
            "top_book": retrieved[0].book if retrieved else "-",
            "gt_books": ",".join(sorted({c.book for c in it.ground_truth_contexts})),
        })

    agg = aggregate(per_q)
    ceiling5 = sum(ceil5) / len(ceil5) if ceil5 else 0.0
    # Precision normalised by what the LABELS allow. Reporting raw precision
    # alone invites reading 0.25 as "75% wrong" when the maximum is 0.71.
    agg["prec@5_ceiling"] = round(ceiling5, 4)
    agg["prec@5_vs_ceiling"] = round(
        agg["precision@5"] / ceiling5 if ceiling5 else 0.0, 4)
    if rerank_ms:
        agg["rerank_ms_p50"] = round(float(np.percentile(rerank_ms, 50)), 1)
    extras = {
        "n_scored": len(items),
        "n_chunks": len(chunks),
        "dim": dim,
        "query_ms_p50": float(np.percentile(lat, 50)),
        "query_ms_p95": float(np.percentile(lat, 95)),
        "rows": rows,
    }
    return agg, extras


# ───────────────────────────────────────────────────────────────────────
# Experiments
# ───────────────────────────────────────────────────────────────────────

def experiment_embedder(cfg: dict, items: list[GoldenItem], candidates: list[str],
                        backend: str = "ollama"):
    """D3: sweep the embedder, pin chunker and index.

    All candidates MUST come from one backend. Ollama's quantised weights and
    a GPU's fp16 produce different vectors from the same checkpoint, so a
    mixed run would compare quantisation as much as it compares models.
    corpus_cache refuses to compute non-ollama vectors locally, which makes
    that mistake loud instead of silent.
    """
    strategy = cfg["chunking"]["strategy"]
    results, perq = [], {}
    for name in candidates:
        console.print(f"\n[bold cyan]── {name} ──[/bold cyan]")
        try:
            agg, extras = run_config(name, strategy, items,
                                     cfg["retrieval"]["top_k"],
                                     mode=cfg["retrieval"]["mode"],
                                     rrf_k=cfg["retrieval"].get("rrf_k", 60),
                                     backend=backend)
        except Exception as e:                            # noqa: BLE001
            console.print(f"[red]{name} failed: {e}[/red]")
            continue
        results.append({"config": name, **agg, **{
            k: extras[k] for k in ("dim", "query_ms_p50", "query_ms_p95")}})
        perq[name] = extras["rows"]
        console.print(
            f"  recall@10={agg['recall@10']:.4f}  mrr={agg['mrr']:.4f}  "
            f"ndcg@10={agg['ndcg@10']:.4f}  p50={extras['query_ms_p50']:.0f}ms"
        )
    return results, strategy, perq


def experiment_retrieval(cfg: dict, items: list[GoldenItem], embedder: str):
    """D5: sweep the retrieval mode, pin embedder / chunker / index.

    NOTE ON ORDERING. Strictly, D5 should run on the embedder D3 chooses.
    Running it earlier against the current config value is still useful - it
    validates the fusion code and gives an early read on whether lexical
    signal helps at all - but the result is PROVISIONAL until D3 lands and
    this is re-run. The report says so explicitly.
    """
    strategy = cfg["chunking"]["strategy"]
    top_k = cfg["retrieval"]["top_k"]
    rrf_k = cfg["retrieval"].get("rrf_k", 60)
    results, perq = [], {}
    for mode in ("dense", "bm25", "hybrid_rrf", "hybrid_weighted"):
        console.print()
        console.print("[bold cyan]-- " + mode + " --[/bold cyan]")
        try:
            agg, extras = run_config(embedder, strategy, items, top_k,
                                     mode=mode, rrf_k=rrf_k)
        except Exception as e:                            # noqa: BLE001
            console.print(f"[red]{mode} failed: {e}[/red]")
            continue
        results.append({"config": mode, **agg, **{
            k: extras[k] for k in ("dim", "query_ms_p50", "query_ms_p95")}})
        perq[mode] = extras["rows"]
        console.print(
            f"  recall@10={agg['recall@10']:.4f}  mrr={agg['mrr']:.4f}  "
            f"ndcg@10={agg['ndcg@10']:.4f}  p50={extras['query_ms_p50']:.0f}ms"
        )
    return results, strategy, perq


def experiment_chunker(cfg: dict, items: list[GoldenItem], embedder: str,
                       strategies: list[str]):
    """D2: sweep the chunker, pin embedder / retrieval mode / index.

    Each strategy produces a DIFFERENT number of chunks, so each needs its own
    vector cache entry. corpus_cache keys on (embedder, strategy) precisely so
    this sweep cannot accidentally score one chunker with another's vectors.
    """
    top_k = cfg["retrieval"]["top_k"]
    mode = cfg["retrieval"]["mode"]
    results, perq = [], {}
    for strat in strategies:
        if not (Path("data/chunks") / strat).exists():
            console.print("[yellow]skip " + strat + " - no chunks on disk. "
                          "Run the ingest pipeline for it first.[/yellow]")
            continue
        console.print()
        console.print("[bold cyan]-- " + strat + " --[/bold cyan]")
        try:
            agg, extras = run_config(embedder, strat, items, top_k, mode=mode,
                                     rrf_k=cfg["retrieval"].get("rrf_k", 60))
        except Exception as e:                            # noqa: BLE001
            console.print("[red]" + strat + " failed: " + str(e) + "[/red]")
            continue
        results.append({"config": strat, "n_chunks": extras["n_chunks"], **agg,
                        **{k: extras[k] for k in ("dim", "query_ms_p50",
                                                  "query_ms_p95")}})
        perq[strat] = extras["rows"]
        console.print(
            f"  chunks={extras['n_chunks']}  recall@10={agg['recall@10']:.4f}  "
            f"mrr={agg['mrr']:.4f}  ndcg@10={agg['ndcg@10']:.4f}"
        )
    return results, perq


def experiment_rerank(cfg: dict, items: list[GoldenItem], embedder: str,
                      candidates: list[str]):
    """D6: sweep the reranker, pin stage 1 to the D5 winner."""
    strategy = cfg["chunking"]["strategy"]
    top_k = cfg["retrieval"]["top_k"]
    mode = cfg["retrieval"]["mode"]
    results, perq = [], {}
    for name in candidates:
        console.print()
        console.print("[bold cyan]-- rerank=" + str(name) + " --[/bold cyan]")
        try:
            agg, extras = run_config(
                embedder, strategy, items, top_k, mode=mode,
                rrf_k=cfg["retrieval"].get("rrf_k", 60),
                rerank=None if name == "none" else name,
            )
        except Exception as e:                            # noqa: BLE001
            console.print("[red]" + str(name) + " failed: " + str(e) + "[/red]")
            continue
        results.append({"config": name, **agg, **{
            k: extras[k] for k in ("dim", "query_ms_p50", "query_ms_p95")}})
        perq[name] = extras["rows"]
        console.print(
            f"  recall@1={agg['recall@1']:.4f}  recall@5={agg['recall@5']:.4f}  "
            f"mrr={agg['mrr']:.4f}  ndcg@10={agg['ndcg@10']:.4f}  "
            f"p50={extras['query_ms_p50']:.0f}ms"
        )
    return results, strategy, perq


def write_report(path: Path, title: str, axis: str, pinned: str,
                 results: list[dict], items: list[GoldenItem], n_total: int,
                 experiment: str = "", cfg: dict | None = None,
                 dataset_path: Path | None = None,
                 perq: dict | None = None, dataset_name: str = "golden") -> None:
    import pandas as pd
    perq = perq or {}

    # Persist BEFORE formatting. The markdown report is rewritten every run and
    # keeps no history; the database is the only place a regression can be seen.
    if experiment:
        for r in results:
            # Per-question scores enable PAIRED significance tests later.
            # Aggregates are lossy: once you have only the mean, you can never
            # ask "on how many questions did A actually beat B?" - and that is
            # the question with the statistical power in it.
            rows = perq.get(str(r["config"]))
            if rows:
                pq = PERQ_DIR / (f"{experiment}_{dataset_name}_"
                                 f"{str(r['config']).replace('/', '_')}.jsonl")
                pq.parent.mkdir(parents=True, exist_ok=True)
                with pq.open("w", encoding="utf-8") as fh:
                    for row in rows:
                        print(json.dumps(row), file=fh)

            metrics = {k: v for k, v in r.items() if k != "config"}
            record(experiment, str(r["config"]), metrics,
                   cfg=cfg or {}, pinned={"note": pinned},
                   n_scored=len(items), dataset=dataset_path or GOLDEN)

    keep = ["config", "recall@1", "recall@5", "recall@10", "recall@20",
            "precision@5", "prec@5_vs_ceiling", "mrr", "ndcg@10",
            "rerank_ms_p50", "n_chunks", "dim", "query_ms_p50"]
    # Column set must come from the UNION of all rows, not results[0].
    # In the rerank sweep the first row is `none`, which has no
    # rerank_ms_p50 - keying off row 0 silently dropped the reranker's
    # latency from the table entirely, while the report still rendered
    # perfectly. Missing values become NaN, which is the honest display.
    present = {k for r in results for k in r}
    df = pd.DataFrame(results)[[c for c in keep if c in present]]
    df = df.round(4).sort_values("recall@10", ascending=False)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([
        f"# {title}",
        "",
        f"Swept: **{axis}**.  Pinned: {pinned}.",
        f"Dataset: `{(dataset_path or GOLDEN).name}`.",
        f"Scored on **{len(items)} of {n_total}** items "
        f"({n_total - len(items)} unanswerable excluded — they have no "
        "ground-truth contexts, so recall on them is 0.0 by construction and "
        "would apply a constant penalty to every row).",
        "",
        "Metrics are judge-free: computed from labelled `(book, page_range)`",
        "overlap. Index pinned to `flat` per **D4**, so no approximation error",
        "is mixed into these differences.",
        "",
        df.to_markdown(index=False),
        "",
        "## Caveats that belong next to these numbers",
        "",
        "- **n is small.** With ~48 scored items a 5-point gap is inside the",
        "  noise band. Treat anything under ~5 points as a tie and break it on",
        "  cost, context window, or latency instead.",
        "- **17 of 50 questions name their source book** (e.g. \"According to",
        "  Huyen…\"). Author names do not appear in chunk text, so that hint",
        "  pays off only once metadata filtering exists — it is dead weight in",
        "  the query here, and slightly understates every embedder.",
        "- **`mxbai` has a 512-token context window** against 512-token chunks",
        "  plus a query prefix, so it truncates. That is a real property of the",
        "  model at our chunk size, not a bug — but it means mxbai is being",
        "  judged on a chunk size chosen before it was in the running.",
        "",
        *decision_section(path, [
            "## Decision",
            "",
            "- **Chosen:** _fill in_",
            "- Record in `plan.md`; update `configs/experiment.yaml` and",
            "  upgrade the tag to `[PROVEN]`.",
            "",
        ]),
    ]), encoding="utf-8")
    console.print(f"\n[bold green]wrote {path}[/bold green]")


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 4 retrieval lab.")
    ap.add_argument("--experiment",
                    choices=["embedder", "chunker", "retrieval", "rerank"],
                    default="embedder")
    ap.add_argument("--rerankers", default="none,minilm-l6,minilm-l12")
    ap.add_argument("--strategies", default="recursive,parent_child,semantic")
    ap.add_argument("--backend", default="ollama",
                    help="ollama (local) or st (Colab sentence-transformers). "
                         "Never mix backends inside one experiment.")
    ap.add_argument("--dataset", choices=["golden", "synthetic"], default="golden",
                    help="golden = 48 hand-labelled, absolute numbers you can "
                         "quote. synthetic = ~250 auto-labelled, enough power "
                         "to RANK configs. Never quote synthetic as quality.")
    ap.add_argument("--embed-only", action="store_true",
                    help="precompute vectors for all candidates, score nothing")
    ap.add_argument("--embedders", default="nomic,bge-m3,embeddinggemma,mxbai")
    ap.add_argument("--strategy", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(EXPERIMENT.read_text(encoding="utf-8"))
    if args.strategy:
        cfg["chunking"]["strategy"] = args.strategy
    strategy = cfg["chunking"]["strategy"]
    candidates = [c.strip() for c in args.embedders.split(",") if c.strip()]

    # --embed-only deliberately needs neither the golden set nor
    # scorable_items(). It is the multi-hour half of the work, so it must be
    # startable before the scoring half is finished.
    if args.embed_only:
        chunks = load_chunks(strategy)
        console.print(f"[cyan]{len(chunks)} chunks ({strategy})[/cyan]")
        for name in candidates:
            if is_cached(name, strategy, len(chunks)):
                console.print(f"[green]{name}: cached[/green]")
                continue
            t0 = time.perf_counter()
            try:
                embed_corpus(name, strategy, chunks)
                console.print(f"[green]{name}: done in "
                              f"{(time.perf_counter()-t0)/60:.1f} min[/green]")
            except Exception as e:                        # noqa: BLE001
                console.print(f"[red]{name} failed: {e}[/red]")
        return 0

    dataset = DATASETS[args.dataset]

    def report_path(name: str) -> Path:
        """Reports are per-DATASET, not just per-experiment.

        Writing synthetic results over the golden report would leave a table
        of 250-item numbers sitting under a Decision section that cites
        48-item numbers - internally contradictory, and the kind of document
        that gets quoted later by someone who did not run it.
        """
        if args.dataset == "golden":
            return REPORT_DIR / name
        stem, ext = name.rsplit(".", 1)
        return REPORT_DIR / f"{stem}_{args.dataset}.{ext}"
    if not dataset.exists():
        console.print(f"[red]no dataset at {dataset}[/red]")
        return 1
    all_items = load(dataset)
    items = scorable_items(all_items)
    console.print(f"[cyan]{args.dataset}: {len(all_items)} items, "
                  f"{len(items)} scorable for retrieval[/cyan]")
    if args.dataset == "synthetic":
        # Said loudly on every run, because the one way to misuse this set is
        # to quote its absolute numbers as retrieval quality. Questions were
        # written while looking at the passage, so their wording leaks into
        # the query and flatters the lexical arm in particular.
        console.print("[yellow]SYNTHETIC: use for RANKING configs, not for "
                      "absolute quality. Single-hop only; no multi-hop or "
                      "cross-document coverage.[/yellow]")

    if args.experiment == "chunker":
        embedder = candidates[0]
        strats = [c.strip() for c in args.strategies.split(",") if c.strip()]
        results, perq = experiment_chunker(cfg, items, embedder, strats)
        if not results:
            console.print("[red]no successful runs[/red]")
            return 1
        write_report(
            report_path("04e_chunker_bakeoff.md"),
            "Phase 4e — Chunking Strategy Bake-off (D2)",
            "chunking strategy",
            f"embedder `{embedder}`, retrieval `{cfg['retrieval']['mode']}` "
            f"(D5), index `flat` (D4). Parent-child scored on CHILD chunks.",
            results, items, len(all_items),
            experiment="chunker", cfg=cfg, dataset_path=dataset,
            perq=perq, dataset_name=args.dataset,
        )
        return 0

    if args.experiment == "rerank":
        embedder = candidates[0]
        names = [c.strip() for c in args.rerankers.split(",") if c.strip()]
        results, strategy, perq = experiment_rerank(cfg, items, embedder, names)
        if not results:
            console.print("[red]no successful runs[/red]")
            return 1
        write_report(
            report_path("04d_reranking.md"),
            "Phase 4d — Cross-encoder Reranking (D6)",
            "reranker",
            f"stage 1 `{cfg['retrieval']['mode']}` (D5) top_k="
            f"{cfg['retrieval']['top_k']}, embedder `{embedder}`, "
            f"chunker `{strategy}`, index `flat` (D4)",
            results, items, len(all_items),
            experiment="rerank", cfg=cfg, dataset_path=dataset,
            perq=perq, dataset_name=args.dataset,
        )
        return 0

    if args.experiment == "retrieval":
        embedder = candidates[0]
        results, strategy, perq = experiment_retrieval(cfg, items, embedder)
        if not results:
            console.print("[red]no successful runs[/red]")
            return 1
        write_report(
            report_path("04c_hybrid_retrieval.md"),
            "Phase 4c — Retrieval Mode: dense vs BM25 vs hybrid (D5)",
            "retrieval mode",
            f"embedder `{embedder}` (PROVISIONAL — pending D3), "
            f"chunker `{strategy}`, index `flat` (D4)",
            results, items, len(all_items),
            experiment="retrieval", cfg=cfg, dataset_path=dataset,
            perq=perq, dataset_name=args.dataset,
        )
        return 0

    if args.experiment == "embedder":
        results, strategy, perq = experiment_embedder(
            cfg, items, candidates, backend=args.backend)
        if not results:
            console.print("[red]no successful runs[/red]")
            return 1
        write_report(
            report_path("04b_embedder_bakeoff.md"),
            "Phase 4b — Embedding Bake-off (D3)",
            "embedding model",
            f"chunker `{strategy}`, index `flat` (D4), retrieval "
            f"`{cfg['retrieval']['mode']}` (D5), backend `{args.backend}`",
            results, items, len(all_items),
            experiment="embedder", cfg=cfg, dataset_path=dataset,
            perq=perq, dataset_name=args.dataset,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

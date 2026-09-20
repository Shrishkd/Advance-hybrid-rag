"""FAISS index lab — the D4 experiment: Flat vs HNSW vs IVF vs IVFPQ.

WHY FAISS AND NOT CHROMA FOR THIS
---------------------------------
Chroma hides its index behind an HNSW-only default. That is fine for serving
and useless for the question we are actually asking, which is *what does each
ANN strategy cost and buy on OUR vectors*. FAISS is the only option that lets
all four run over byte-identical inputs, which is the only way the comparison
means anything.

THE STATED PRIOR, TO BE FALSIFIED
---------------------------------
At ~6k-9k chunks, exact flat search should win outright: brute force over
that many vectors is milliseconds, and every approximate index trades recall
for a speedup we do not need. The valuable output of this experiment is
therefore NOT "which index is best" but:

    * a measured recall/latency/memory curve, and
    * the CROSSOVER POINT where approximation starts to pay.

"We measured it and exact search is sufficient at our scale" is a stronger
finding than adopting HNSW because it sounds advanced.

MEASURING RECALL WITHOUT GROUND TRUTH
-------------------------------------
ANN recall is self-referential and needs no golden set: the exact index IS
the ground truth. recall@k = overlap between what the approximate index
returns and what flat returns for the same query. That is why this experiment
can run NOW, before the golden set is curated - it measures approximation
error, not retrieval quality. Those are different questions and conflating
them is a common mistake.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass
class IndexStats:
    """One index configuration's measured cost and quality."""
    kind: str
    build_ms: float
    search_ms_p50: float
    search_ms_p95: float
    memory_mb: float
    recall_at_k: float          # vs exact flat search; 1.0 for flat itself
    params: dict

    def row(self) -> dict:
        return {
            "index": self.kind,
            "recall@k": round(self.recall_at_k, 4),
            "p50_ms": round(self.search_ms_p50, 3),
            "p95_ms": round(self.search_ms_p95, 3),
            "build_ms": round(self.build_ms, 1),
            "mem_mb": round(self.memory_mb, 1),
            "params": ",".join(f"{k}={v}" for k, v in self.params.items()) or "-",
        }


def build_index(kind: str, vectors: np.ndarray, cfg: dict) -> tuple:
    """Construct one FAISS index over `vectors`.

    All use inner product, which equals cosine similarity because the
    embedder L2-normalises. Using L2 distance here instead would silently
    change what "nearest" means relative to how the models were trained.

    Returns:
        (index, effective_params) - the params ACTUALLY used, not the ones
        requested. Several are clamped to the corpus size below, and a report
        that prints the requested value while the index used another is a lie
        that no test would catch.
    """
    import faiss

    n, d = vectors.shape
    eff: dict = {}

    if kind == "flat":
        idx = faiss.IndexFlatIP(d)

    elif kind == "hnsw":
        p = cfg.get("hnsw", {})
        idx = faiss.IndexHNSWFlat(d, p.get("M", 32), faiss.METRIC_INNER_PRODUCT)
        idx.hnsw.efConstruction = p.get("ef_construction", 200)
        idx.hnsw.efSearch = p.get("ef_search", 64)
        eff = {"M": p.get("M", 32),
               "ef_construction": idx.hnsw.efConstruction,
               "ef_search": idx.hnsw.efSearch}

    elif kind == "ivf":
        p = cfg.get("ivf", {})
        # nlist cannot exceed the number of training points, and FAISS warns
        # below ~39 points per centroid. Clamp rather than emit a wall of
        # warnings or, worse, train a degenerate quantiser.
        nlist = min(p.get("nlist", 256), max(1, n // 39))
        quant = faiss.IndexFlatIP(d)
        idx = faiss.IndexIVFFlat(quant, d, nlist, faiss.METRIC_INNER_PRODUCT)
        idx.train(vectors)
        idx.nprobe = min(p.get("nprobe", 16), nlist)
        eff = {"nlist": nlist, "nprobe": idx.nprobe}

    elif kind == "ivfpq":
        p = cfg.get("ivfpq", {})
        nlist = min(p.get("nlist", 256), max(1, n // 39))
        m = p.get("m", 64)
        # m must divide the embedding dimension exactly - otherwise FAISS
        # aborts. Fall back to the largest divisor that fits.
        if d % m:
            m = next((c for c in (64, 32, 16, 8, 4) if d % c == 0), 1)
        quant = faiss.IndexFlatIP(d)
        nbits = p.get("nbits", 8)
        idx = faiss.IndexIVFPQ(quant, d, nlist, m, nbits)
        idx.train(vectors)
        idx.nprobe = min(p.get("nprobe", 16), nlist)
        eff = {"nlist": nlist, "m": m, "nbits": nbits, "nprobe": idx.nprobe,
               # Each of the m sub-quantizers learns 2**nbits centroids from n
               # points. FAISS wants >=39 points per centroid; below that the
               # codebooks are undertrained and recall suffers. This number
               # explains ivfpq's recall directly, so it belongs in the report.
               "pts_per_centroid": round(n / (2 ** nbits), 1)}

    else:
        raise KeyError(f"unknown index kind {kind!r}")

    idx.add(vectors)
    return idx, eff


def _memory_mb(idx, vectors: np.ndarray, kind: str) -> float:
    """Approximate resident size of the index.

    FAISS does not expose a portable byte count, so we serialise it. That is
    honest and comparable across kinds, which is what the experiment needs -
    the interesting number is IVFPQ's compression ratio versus flat, not an
    exact RSS figure.
    """
    import faiss

    try:
        return len(faiss.serialize_index(idx)) / 1e6
    except Exception:                                   # noqa: BLE001
        return vectors.nbytes / 1e6


def evaluate_index(
    kind: str,
    vectors: np.ndarray,
    queries: np.ndarray,
    k: int = 10,
    cfg: dict | None = None,
    exact_truth: np.ndarray | None = None,
) -> tuple[IndexStats, np.ndarray]:
    """Build one index, time it, and measure its recall against exact search.

    Args:
        vectors: corpus, shape (n, d), L2-normalised float32.
        queries: shape (q, d), same normalisation.
        k: neighbours retrieved per query.
        exact_truth: flat-search results to compare against. Pass the result
            from the "flat" run so every approximate index is scored against
            the SAME reference.

    Returns:
        (stats, neighbour_ids) - ids so the caller can reuse flat's output as
        truth for the remaining kinds.
    """
    cfg = cfg or {}

    t0 = time.perf_counter()
    idx, eff_params = build_index(kind, vectors, cfg)
    build_ms = (time.perf_counter() - t0) * 1000

    # Time queries individually: p95 is the number that matters for a chat UI,
    # and batching would hide per-query variance behind throughput.
    lat: list[float] = []
    ids = np.zeros((len(queries), k), dtype=np.int64)
    for i, q in enumerate(queries):
        t1 = time.perf_counter()
        _, nn = idx.search(q.reshape(1, -1), k)
        lat.append((time.perf_counter() - t1) * 1000)
        ids[i] = nn[0]

    if exact_truth is None:
        recall = 1.0                      # this IS the reference
    else:
        hits = sum(
            len(set(ids[i]) & set(exact_truth[i])) for i in range(len(queries))
        )
        recall = hits / (len(queries) * k)

    stats = IndexStats(
        kind=kind,
        build_ms=build_ms,
        search_ms_p50=float(np.percentile(lat, 50)),
        search_ms_p95=float(np.percentile(lat, 95)),
        memory_mb=_memory_mb(idx, vectors, kind),
        recall_at_k=recall,
        params=eff_params,
    )
    return stats, ids

"""The production retriever — Phase 4's winning config, as one reusable object.

WHY THIS FILE EXISTS
--------------------
Until Phase 6, retrieval lived inside `evaluation/generation_lab.retrieve_all` as a
batch loop. Three consumers need it now:

    the generation lab   batch, over the golden set
    the LangGraph nodes  one query at a time, possibly several times per question
                         (CRAG re-retrieves; multi-hop retrieves per sub-question)
    the served app       one query at a time, forever (Phase 9)

If each carried its own copy, they would drift - a different `arm_k`, a forgotten
query prefix - and the graph's numbers would stop being comparable to Phase 5's
for reasons that have nothing to do with the graph. One implementation, three
callers.

WHAT IS FROZEN HERE, AND WHY EACH IS NOT A FREE PARAMETER
---------------------------------------------------------
    embedder       embeddinggemma           D3
    retrieval      hybrid_rrf, k=60         D5
    index          flat (numpy)             D4
    chunker        recursive                D2
    arm depth      top_k * 3 per arm        so RRF sees agreement past top_k

All read from configs/experiment.yaml, so changing a decision is one config edit,
not a code change.

LOAD ONCE
---------
Construction loads 5,796 chunk texts, their vectors and a BM25 index (~2 s). A
graph that built a retriever per node call would pay that on every hop. Build one
and pass it in.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.retrieve.fusion import reciprocal_rank_fusion


def _load_chunks(strategy: str) -> list[dict]:
    """Same order as evaluation.retrieval_lab.load_chunks - the vector cache
    is keyed against this order, so it must not diverge."""
    d = Path("data/chunks") / strategy
    rows: list[dict] = []
    for f in sorted(d.glob("*.jsonl")):
        if f.name.endswith(".parents.jsonl"):
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


class HybridRetriever:
    """Dense (flat) + BM25, fused by RRF. One query in, ranked chunks out."""

    def __init__(self, cfg: dict, strategy: str | None = None) -> None:
        from src.embed.corpus_cache import corpus_vectors
        from src.embed.ollama_embedder import get_embedder
        from src.index.bm25_index import BM25Index

        self.strategy = strategy or cfg["chunking"]["strategy"]
        self.embedder_name = cfg["embedding"]["model"]
        self.top_k = cfg["retrieval"]["top_k"]
        self.rrf_k = cfg["retrieval"].get("rrf_k", 60)

        self.chunks = _load_chunks(self.strategy)
        if not self.chunks:
            raise FileNotFoundError(f"no chunks for strategy {self.strategy!r}")
        self.vectors = corpus_vectors(
            self.embedder_name, self.strategy, [c["text"] for c in self.chunks]
        )
        self.emb = get_embedder(self.embedder_name)
        self.bm25 = BM25Index([c["text"] for c in self.chunks])

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """Ranked chunks for one query, best first.

        Each arm retrieves `top_k * 3` before fusion. Truncating both arms at
        top_k first would discard a document ranked 25th by one arm and 2nd by
        the other - exactly the agreement RRF exists to reward.
        """
        k = top_k or self.top_k
        arm_k = k * 3
        qv = self.emb.embed_query(query).reshape(1, -1).astype(np.float32)
        sims = (qv @ self.vectors.T).ravel()
        if arm_k >= len(sims):
            dense = np.argsort(-sims)
        else:
            part = np.argpartition(-sims, arm_k)[:arm_k]
            dense = part[np.argsort(-sims[part])]
        sparse, _ = self.bm25.search(query, arm_k)
        fused = reciprocal_rank_fusion(
            [[int(i) for i in dense], [int(i) for i in sparse]],
            k=self.rrf_k, top_n=k,
        )
        return [self.chunks[d] for d, _ in fused]

"""Cross-encoder reranking — Phase 4.5 (D6).

BI-ENCODER vs CROSS-ENCODER, AND WHY THE SECOND STAGE EXISTS
------------------------------------------------------------
Everything up to this point has used a BI-ENCODER: the query and each chunk
are embedded SEPARATELY, into the same space, and compared by cosine
similarity. That separation is what makes it fast - 5,796 chunk vectors are
computed once, offline, and reused for every query forever.

It is also what makes it weak. The chunk was embedded before your question
existed. Its vector is a single fixed summary of the passage, and it cannot
emphasise the part you asked about, because it has never seen the question.

A CROSS-ENCODER concatenates them:

    [CLS] what is the Adam optimizer [SEP] ...passage text... [SEP]  ->  score

Every query token attends to every passage token. The model can notice that
"Adam" in the question refers to the same "Adam" in paragraph three and not
to the "Adam" in a citation. That is a strictly richer computation, and it
scores meaningfully better on ranking benchmarks.

The cost is brutal and unavoidable: the score depends on the PAIR, so nothing
can be precomputed. Scoring the whole corpus per query means 5,796 forward
passes. At roughly 4 ms each on this CPU that is ~23 seconds per question.

Hence the two-stage design, which is the only reason this is practical:

    stage 1  hybrid_rrf over 5,796 chunks   ->  top 20     (cheap, high recall)
    stage 2  cross-encoder over those 20    ->  top 5      (costly, precise)

Stage 1 optimises RECALL - get the right passage into the pool at all. Stage 2
optimises PRECISION - put it at rank 1. A reranker cannot recover a document
stage 1 never retrieved, which is exactly why D5 chose the mode with the best
recall@20 rather than the best recall@1.

THE CEILING THIS IMPLIES
------------------------
Reranking the top 20 can never exceed hybrid_rrf's recall@20 = 0.8438. That
number is the hard upper bound on anything this stage can deliver, and it is
the honest way to read the D6 table: the question is not "how good is the
reranker" but "how much of the 0.8438 already in the pool does it lift to the
top".

MODEL CHOICE AND RAM, STATED BEFORE LOADING ANYTHING
-----------------------------------------------------
    ms-marco-MiniLM-L-6-v2    22M params   ~90 MB   ~4 ms/pair   runs here
    bge-reranker-v2-m3       568M params  ~2.2 GB  ~90 ms/pair   Colab only

At 7.4 GB total RAM the large model would load but leave nothing for Ollama,
and at 90 ms/pair a 20-document rerank costs 1.8 s per question. It is
benchmarked on Colab in Phase 10, not here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

# Registry of reranker models. Name -> (HF id, approx RAM MB, runs_locally).
RERANKERS: dict[str, tuple[str, int, bool]] = {
    "minilm-l6": ("cross-encoder/ms-marco-MiniLM-L-6-v2", 90, True),
    "minilm-l12": ("cross-encoder/ms-marco-MiniLM-L-12-v2", 140, True),
    "bge-v2-m3": ("BAAI/bge-reranker-v2-m3", 2200, False),
}


@dataclass
class RerankStats:
    """What one rerank pass cost."""
    model: str
    n_pairs: int
    total_ms: float

    @property
    def ms_per_pair(self) -> float:
        return self.total_ms / self.n_pairs if self.n_pairs else 0.0


class CrossEncoderReranker:
    """Reorder a candidate list by query-document cross-attention.

    Lazily loaded: constructing this class does not pull the model, so the
    registry above can be inspected (and RAM costs reported) without paying
    for a download.
    """

    def __init__(self, name: str = "minilm-l6", max_length: int = 512) -> None:
        if name not in RERANKERS:
            raise KeyError(f"unknown reranker {name!r}; have {sorted(RERANKERS)}")
        self.name = name
        self.hf_id, self.ram_mb, self.local_ok = RERANKERS[name]
        self.max_length = max_length
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.hf_id, max_length=self.max_length)
        return self._model

    def score(self, query: str, docs: list[str]) -> list[float]:
        """Relevance score per document. Higher is better.

        Scores are raw logits, NOT probabilities and NOT comparable across
        queries - the same caveat as BM25. Use them to order within one
        query's candidate list and nothing else.
        """
        if not docs:
            return []
        pairs = [(query, d) for d in docs]
        return [float(s) for s in self.model.predict(pairs, show_progress_bar=False)]

    def rerank(
        self, query: str, candidates: list[int], texts: list[str], top_n: int = 10
    ) -> tuple[list[int], RerankStats]:
        """Reorder `candidates` (corpus indices) by cross-encoder score.

        Args:
            query: the user question, unmodified. No prefix - cross-encoders
                are trained on raw query/passage pairs, unlike the asymmetric
                bi-encoders which need "search_query: " and friends.
            candidates: corpus indices from stage 1, in stage-1 order.
            texts: the chunk text for each candidate, same order.
            top_n: how many to keep.

        Returns:
            (reordered indices, stats)
        """
        t0 = time.perf_counter()
        scores = self.score(query, texts)
        elapsed = (time.perf_counter() - t0) * 1000

        order = sorted(range(len(candidates)), key=lambda i: -scores[i])
        stats = RerankStats(self.name, len(texts), elapsed)
        return [candidates[i] for i in order[:top_n]], stats

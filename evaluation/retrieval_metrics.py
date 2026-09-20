"""Retrieval metrics — recall@k, precision@k, MRR, nDCG@k.

NO LLM JUDGE REQUIRED. THAT IS THE WHOLE POINT.
-----------------------------------------------
Everything here is computed from labelled page ranges and a ranked list of
chunks. No model call, no cost, no rate limit, no judge bias, and perfectly
reproducible. That is what makes Phase 4 able to sweep dozens of retrieval
configurations cheaply, while Phase 8's judged metrics run only on the final
few.

If a metric needs an LLM, it does not belong in this file.

HOW A "HIT" IS DECIDED
----------------------
A retrieved chunk counts as relevant when its [page_start, page_end] OVERLAPS
a labelled ground-truth range for the SAME book.

Three deliberate choices in that sentence:

  OVERLAP, not containment - an answer spanning a page boundary must not be
      scored as a miss just because the chunk starts one page early.
  PAGE RANGES, not chunk ids - chunk ids change with every chunking strategy;
      page ranges do not. This is what keeps ONE golden set valid across every
      Phase 4 experiment. Label by chunk id and the labels die the first time
      you change chunk_size.
  SAME BOOK - a page range without its book is meaningless; page 141 exists in
      all six.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    """The minimum a retriever must return for scoring."""
    book: str
    page_start: int
    page_end: int
    chunk_id: str = ""


def is_relevant(chunk: RetrievedChunk, contexts: list) -> bool:
    """Does this chunk overlap any labelled ground-truth range?

    `contexts` items need `.book`, `.page_start`, `.page_end` (ContextRef) or
    the equivalent dict keys.

    >>> from types import SimpleNamespace as S
    >>> gt = [S(book="Bishop", page_start=100, page_end=105)]
    >>> is_relevant(RetrievedChunk("Bishop", 104, 108), gt)
    True
    >>> is_relevant(RetrievedChunk("Bishop", 106, 110), gt)
    False
    >>> is_relevant(RetrievedChunk("Géron", 100, 105), gt)
    False
    """
    for c in contexts:
        book = c.book if hasattr(c, "book") else c["book"]
        ps = c.page_start if hasattr(c, "page_start") else c["page_start"]
        pe = c.page_end if hasattr(c, "page_end") else c["page_end"]
        if chunk.book == book and ps <= chunk.page_end and chunk.page_start <= pe:
            return True
    return False


def _rel_flags(retrieved: list[RetrievedChunk], contexts: list) -> list[int]:
    return [1 if is_relevant(c, contexts) else 0 for c in retrieved]


# ───────────────────────────────────────────────────────────────────────
# Metrics
# ───────────────────────────────────────────────────────────────────────

def recall_at_k(retrieved: list[RetrievedChunk], contexts: list, k: int) -> float:
    """Fraction of ground-truth RANGES hit by the top-k.

    Note the denominator: distinct labelled ranges, not retrieved chunks. For
    a cross-document question with two ranges, finding both scores 1.0 and
    finding only one scores 0.5 - which is exactly the behaviour that makes
    cross-document questions worth having. A dense retriever that returns four
    excellent chunks from a single book scores 0.5 here, and should.

    >>> from types import SimpleNamespace as S
    >>> gt = [S(book="A", page_start=1, page_end=2), S(book="B", page_start=9, page_end=9)]
    >>> r = [RetrievedChunk("A", 1, 1), RetrievedChunk("A", 2, 2)]
    >>> recall_at_k(r, gt, 10)
    0.5
    """
    if not contexts:
        return 0.0
    top = retrieved[:k]
    hit = 0
    for c in contexts:
        book = c.book if hasattr(c, "book") else c["book"]
        ps = c.page_start if hasattr(c, "page_start") else c["page_start"]
        pe = c.page_end if hasattr(c, "page_end") else c["page_end"]
        if any(r.book == book and ps <= r.page_end and r.page_start <= pe for r in top):
            hit += 1
    return hit / len(contexts)


def precision_at_k(retrieved: list[RetrievedChunk], contexts: list, k: int) -> float:
    """Fraction of the top-k that are relevant.

    Precision matters here for a reason specific to RAG: every irrelevant
    chunk consumes context-window budget and gives the generator another
    chance to be distracted. High recall with low precision means the model is
    reading four wrong passages to find one right one.

    >>> from types import SimpleNamespace as S
    >>> gt = [S(book="A", page_start=1, page_end=2)]
    >>> r = [RetrievedChunk("A", 1, 1), RetrievedChunk("B", 5, 5)]
    >>> precision_at_k(r, gt, 2)
    0.5
    """
    top = retrieved[:k]
    if not top:
        return 0.0
    return sum(_rel_flags(top, contexts)) / len(top)


def mrr(retrieved: list[RetrievedChunk], contexts: list) -> float:
    """Reciprocal rank of the FIRST relevant chunk.

    Rewards putting something right at the very top. Useful precisely because
    it ignores everything after the first hit - which mirrors how a reranker
    plus a small top_n actually consumes the list.

    >>> from types import SimpleNamespace as S
    >>> gt = [S(book="A", page_start=1, page_end=2)]
    >>> mrr([RetrievedChunk("B",9,9), RetrievedChunk("A",1,1)], gt)
    0.5
    >>> mrr([RetrievedChunk("B",9,9)], gt)
    0.0
    """
    for i, flag in enumerate(_rel_flags(retrieved, contexts), start=1):
        if flag:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[RetrievedChunk], contexts: list, k: int) -> float:
    """Normalised Discounted Cumulative Gain over binary relevance.

    Unlike recall, nDCG cares WHERE the relevant chunks landed: a hit at rank
    1 is worth more than the same hit at rank 9, discounted by 1/log2(rank+1).

    The ideal DCG assumes as many relevant chunks as could exist in the top-k.
    With binary relevance and an unknown total pool, we cap that at
    min(k, number of retrieved relevant) - so nDCG=1.0 means "every relevant
    chunk you found was ranked as high as it possibly could have been", not
    "you found everything". Recall answers the second question; keep them
    separate.

    >>> from types import SimpleNamespace as S
    >>> gt = [S(book="A", page_start=1, page_end=1)]
    >>> ndcg_at_k([RetrievedChunk("A",1,1), RetrievedChunk("B",9,9)], gt, 2)
    1.0
    >>> round(ndcg_at_k([RetrievedChunk("B",9,9), RetrievedChunk("A",1,1)], gt, 2), 4)
    0.6309
    """
    flags = _rel_flags(retrieved[:k], contexts)
    if not any(flags):
        return 0.0
    dcg = sum(f / math.log2(i + 1) for i, f in enumerate(flags, start=1))
    n_ideal = min(k, sum(flags))
    idcg = sum(1 / math.log2(i + 1) for i in range(1, n_ideal + 1))
    return dcg / idcg if idcg else 0.0


def precision_ceiling_at_k(n_relevant_in_corpus: int, k: int) -> float:
    """The best precision@k any retriever could achieve for this item.

    WHY THIS EXISTS
    ---------------
    precision@k is bounded by how many relevant chunks EXIST, not by skill.
    An item whose answer lives on a single page may have only 2 overlapping
    chunks in the whole corpus; at k=5 its precision can never exceed 0.4 even
    if both are ranked 1st and 2nd.

    Reporting raw precision@5 therefore mixes two things: how well the
    retriever ranked, and how narrowly the item was labelled. Measured on our
    golden set the ceiling is 0.71, so a raw 0.25 is 34% of achievable - a
    very different story from "25% correct".

    Use this to normalise. Do NOT use it to excuse a low score: the gap
    between measured and ceiling is real, recoverable headroom.

    >>> precision_ceiling_at_k(2, 5)
    0.4
    >>> precision_ceiling_at_k(18, 5)
    1.0
    >>> precision_ceiling_at_k(0, 5)
    0.0
    """
    return min(n_relevant_in_corpus, k) / k


def count_relevant_in_corpus(corpus: list, contexts: list) -> int:
    """How many chunks in the whole corpus overlap this item's labels.

    `corpus` items need .book/.page_start/.page_end (RetrievedChunk) or the
    equivalent dict keys - the same duck-typing as is_relevant.

    >>> from types import SimpleNamespace as S
    >>> gt = [S(book="A", page_start=1, page_end=2)]
    >>> corpus = [RetrievedChunk("A",1,1), RetrievedChunk("A",2,3),
    ...           RetrievedChunk("B",1,1)]
    >>> count_relevant_in_corpus(corpus, gt)
    2
    """
    out = 0
    for c in corpus:
        if isinstance(c, dict):
            c = RetrievedChunk(c["book"], c["page_start"], c["page_end"])
        if is_relevant(c, contexts):
            out += 1
    return out


# ───────────────────────────────────────────────────────────────────────
# Aggregation
# ───────────────────────────────────────────────────────────────────────

def score_one(retrieved: list[RetrievedChunk], contexts: list,
              ks: tuple[int, ...] = (1, 3, 5, 10, 20)) -> dict[str, float]:
    """All metrics for a single question."""
    out: dict[str, float] = {}
    for k in ks:
        out[f"recall@{k}"] = recall_at_k(retrieved, contexts, k)
        out[f"precision@{k}"] = precision_at_k(retrieved, contexts, k)
    out["mrr"] = mrr(retrieved, contexts)
    out["ndcg@10"] = ndcg_at_k(retrieved, contexts, 10)
    return out


def aggregate(per_question: list[dict[str, float]]) -> dict[str, float]:
    """Mean each metric across questions.

    A plain (macro) mean: every question counts equally regardless of how many
    ground-truth ranges it has. The alternative - pooling hits corpus-wide -
    would let multi-context questions dominate the average simply for carrying
    more labels.
    """
    if not per_question:
        return {}
    keys = per_question[0].keys()
    return {k: sum(q[k] for q in per_question) / len(per_question) for k in keys}

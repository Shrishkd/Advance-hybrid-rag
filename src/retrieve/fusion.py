"""Hybrid fusion — combining dense and lexical result lists.

THE PROBLEM FUSION SOLVES
-------------------------
Dense search returns cosine similarities in [-1, 1]. BM25 returns unbounded
scores whose scale depends on corpus statistics and query length. A document
scoring 0.83 dense and 14.7 lexical cannot be combined by addition - the
numbers are not in the same units, and BM25 would dominate every time simply
because its numbers are bigger.

Two ways out, and we benchmark both:

  RRF (rank-based)     ignores scores entirely, uses only position.
                       Scale-free by construction. No tuning.
  Weighted (score-based) normalises each list, then blends with a weight.
                       Can outperform RRF when tuned, at the cost of having
                       a knob that must be tuned per corpus.

RRF is the default because "no free parameter" is a real virtue in a project
that has to defend every decision. A weight that happens to work on 50
questions is not evidence of much.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def reciprocal_rank_fusion(
    rankings: list[list[int]], k: int = 60, top_n: int = 10
) -> list[tuple[int, float]]:
    """Fuse ranked ID lists by Reciprocal Rank Fusion.

        score(d) = sum over lists of  1 / (k + rank(d))

    with rank 1-based. A document ranked 1st contributes 1/61; ranked 10th,
    1/70. The differences are deliberately small, which is the point: RRF
    rewards documents that appear CONSISTENTLY across retrievers more than it
    rewards one retriever's confident top hit.

    WHY k=60. The constant flattens the curve. Without it (k=0), rank 1 scores
    1.0 and rank 2 scores 0.5 - a single retriever's top result would dominate
    any amount of agreement elsewhere. k=60 is the value from the original
    Cormack et al. paper and is near-universally used unchanged; it is a
    default we adopt knowingly, not a tuned parameter.

    Args:
        rankings: one ranked list of document ids per retriever, best first.
        k: smoothing constant.
        top_n: how many fused results to return.

    Returns:
        [(doc_id, score)] sorted best-first.

    Ties break on ascending doc id. That matters more than it looks: a doc at
    ranks (1,2) and another at (2,1) score IDENTICALLY, because float addition
    is commutative. RRF genuinely cannot distinguish them, and pretending
    otherwise would be false precision.

    >>> dense  = [5, 1, 3]        # 5 top in both lists
    >>> sparse = [5, 9, 1]
    >>> [d for d, _ in reciprocal_rank_fusion([dense, sparse], top_n=3)]
    [5, 1, 9]
    >>> # 5 wins (1st in both). 1 beats 9 despite a worse best-rank,
    >>> # because it appears in BOTH lists - that is the whole point of RRF.
    >>> [d for d, _ in reciprocal_rank_fusion([[7], [8]], top_n=2)]
    [7, 8]
    """
    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked[:top_n]


def _minmax(scores: np.ndarray) -> np.ndarray:
    """Scale to [0, 1]. A constant list maps to all-ones, not NaN.

    >>> _minmax(np.array([1.0, 1.0])).tolist()
    [1.0, 1.0]
    """
    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo < 1e-12:
        return np.ones_like(scores)
    return (scores - lo) / (hi - lo)


def weighted_fusion(
    dense: list[tuple[int, float]],
    sparse: list[tuple[int, float]],
    w_dense: float = 0.5,
    w_sparse: float = 0.5,
    top_n: int = 10,
) -> list[tuple[int, float]]:
    """Fuse by normalised score rather than rank.

    Each list is min-max normalised WITHIN ITSELF first, which is what makes
    the blend meaningful. Skip that and BM25's larger raw numbers swamp cosine
    similarity regardless of the weights.

    The normalisation is also this method's weakness: it is computed over the
    retrieved slice, not the whole corpus, so the same document can normalise
    differently depending on what else came back. RRF has no such dependence,
    which is why it is the default.

    A document missing from one list contributes 0 from that side rather than
    being excluded - absence is weak evidence, not disqualifying.

    >>> d = [(1, 0.9), (2, 0.6), (3, 0.5)]      # cosine-ish scale
    >>> s = [(2, 12.0), (3, 8.0), (4, 4.0)]     # BM25 scale, 13x larger
    >>> [doc for doc, _ in weighted_fusion(d, s, top_n=3)]
    [2, 1, 3]
    >>> # doc 2 wins on being good in BOTH, not on raw magnitude - which is
    >>> # exactly what the per-list normalisation buys.
    """
    out: dict[int, float] = defaultdict(float)

    for lst, w in ((dense, w_dense), (sparse, w_sparse)):
        if not lst:
            continue
        ids = [i for i, _ in lst]
        norm = _minmax(np.asarray([s for _, s in lst], dtype=np.float64))
        for doc_id, s in zip(ids, norm):
            out[doc_id] += w * float(s)

    ranked = sorted(out.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked[:top_n]

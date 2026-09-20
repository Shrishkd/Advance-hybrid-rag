"""BM25 lexical retrieval — the other half of hybrid search.

WHY LEXICAL SEARCH STILL MATTERS IN 2026
----------------------------------------
Dense embeddings capture meaning and blur specifics. That is usually a virtue
and occasionally fatal. Our corpus is dense with exact tokens that MUST match
literally:

    ReLU, Adam, BLEU, L-BFGS, k-means, CKY, PCFG, softmax, AdaGrad

Ask "what is the Adam optimizer?" and a dense retriever happily returns
passages about optimisers in general, because "Adam" and "RMSProp" and "SGD"
all live in the same neighbourhood. BM25 does not care about neighbourhoods:
it matches the string "Adam" and ranks by how unusual that string is.

Rare terms are exactly where BM25 wins, and a technical textbook is made of
rare terms. That is the basis for the prior that hybrid beats dense-only here.

TOKENISATION IS A REAL DECISION, NOT BOILERPLATE
------------------------------------------------
BM25 matches tokens, so how we split text determines what can be found at all.
Naive `text.lower().split()` leaves punctuation attached ("adam," != "adam")
and silently destroys hyphenated technical terms.

We lowercase, then split on non-alphanumeric while KEEPING hyphens inside
words, so "k-means" and "self-attention" survive as single tokens. Getting
this wrong does not raise - it just quietly makes some terms unsearchable.
"""

from __future__ import annotations

import re

import numpy as np

# Keep intra-word hyphens ("k-means"), drop everything else.
_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase and split into BM25 terms.

    >>> tokenize("The Adam optimizer, k-means, and ReLU!")
    ['the', 'adam', 'optimizer', 'k-means', 'and', 'relu']
    >>> tokenize("self-attention vs. multi-head attention")
    ['self-attention', 'vs', 'multi-head', 'attention']
    """
    return _TOKEN.findall(text.lower())


class BM25Index:
    """Okapi BM25 over the chunk corpus.

    Deliberately NOT stemmed and NOT stopword-filtered.

    Stemming would conflate "training" with "train", which is usually helpful
    and here is risky: in this corpus "train" (verb) and "training set" (noun)
    are distinct ideas, and "Bayesian"/"Bayes" should not collapse. Stopword
    removal would delete the "of" in "mixture of experts" and the "a" in
    "a priori".

    Both are options worth benchmarking in Phase 4 rather than assumed - but
    the default is the one that destroys the least information.
    """

    def __init__(self, texts: list[str], ids: list[str] | None = None) -> None:
        from rank_bm25 import BM25Okapi

        self.ids = ids or [str(i) for i in range(len(texts))]
        self.corpus_tokens = [tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Return (indices, scores) for the top-k documents.

        Scores are raw BM25 and are NOT comparable across queries - BM25 has
        no fixed range, unlike cosine similarity's [-1, 1]. This is precisely
        why hybrid fusion should use RANKS (RRF) rather than raw scores: you
        cannot meaningfully add a 14.7 to a 0.83.
        """
        scores = np.asarray(self.bm25.get_scores(tokenize(query)), dtype=np.float32)
        if k >= len(scores):
            order = np.argsort(scores)[::-1]
        else:
            # argpartition is O(n) vs O(n log n); at 6k docs per query, on a
            # CPU-only box, that difference is worth having.
            part = np.argpartition(scores, -k)[-k:]
            order = part[np.argsort(scores[part])[::-1]]
        return order, scores[order]

    def __len__(self) -> int:
        return len(self.corpus_tokens)

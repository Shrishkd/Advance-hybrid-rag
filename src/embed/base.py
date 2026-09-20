"""Embedder interface — one contract, several backends, swappable by config.

THE SILENT 10-20% RECALL BUG THIS FILE EXISTS TO PREVENT
========================================================
Most modern retrieval embedders are ASYMMETRIC: they were trained with
different prefixes for queries and for documents, because a question and the
passage answering it are not the same kind of text.

    nomic-embed-text   "search_query: "  /  "search_document: "
    e5 family          "query: "         /  "passage: "
    embeddinggemma     "task: search result | query: " / "title: none | text: "
    bge-m3             (none - trained symmetrically)

Forget the prefixes and NOTHING fails. No exception, no warning. You get
vectors, you get neighbours, you get plausible-looking results - just worse
ones than the model can deliver. Then you benchmark four embedders, and the
one whose prefixes you happened to get right "wins".

That is a benchmark measuring your configuration care, not model quality.

HOW BIG IS THE EFFECT? NOT YET MEASURED HERE.
A first probe (60 corpus chunks, nomic, SECTION TITLES as queries) found no
meaningful difference: recall@1 0.550 with prefixes vs 0.567 without, MRR
0.673 vs 0.671. That is inside noise at n=60.

Do NOT read that as "prefixes do not matter". The probe used section titles,
which are short and keyword-like - not the natural-question-vs-passage
asymmetry prefixes are trained for. It is a weak proxy, and it settles
nothing. The real measurement is Phase 4 against the golden set's actual
questions, and it belongs in reports/04_retrieval_lab.md.

Applying the prefixes remains correct regardless: it is what the model
authors specify, it costs nothing, and "no measured harm" is not a reason to
deviate from a model's documented interface. But the magnitude claim stays
open until it is measured properly.

So prefixes are NOT the caller's responsibility. They live in
configs/models.yaml beside the model they belong to, and `embed_query` and
`embed_documents` apply them. A caller cannot forget what it never had to
remember.

WHY TWO METHODS INSTEAD OF ONE
------------------------------
`embed(texts)` would be a smaller API and would make the asymmetry invisible
again. Two methods force every call site to state which side it is on, which
is exactly the distinction that matters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import yaml

MODELS_YAML = Path("configs/models.yaml")


@runtime_checkable
class Embedder(Protocol):
    """Contract every embedding backend must satisfy."""

    name: str          # key in configs/models.yaml -> embedders
    dim: int           # vector dimensionality
    max_ctx: int       # token limit; chunks beyond this are TRUNCATED silently

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed passages for indexing. Shape (len(texts), dim), float32."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query. Shape (dim,), float32."""
        ...


def load_spec(name: str) -> dict:
    """Read one embedder's spec from configs/models.yaml.

    Keeping specs in config rather than code means adding a candidate to the
    Phase 4 bake-off is a YAML edit, and the prefixes travel with the model
    that needs them.
    """
    cfg = yaml.safe_load(MODELS_YAML.read_text(encoding="utf-8"))
    specs = cfg["embedders"]
    if name not in specs:
        raise KeyError(f"unknown embedder {name!r}. Available: {sorted(specs)}")
    return specs[name]


def l2_normalize(v: np.ndarray) -> np.ndarray:
    """Scale rows to unit length so inner product == cosine similarity.

    FAISS's IndexFlatIP maximises inner product. On unit vectors that is
    exactly cosine similarity, which is what these models are trained for.
    Skip this and IP rewards long vectors - documents get ranked partly by
    magnitude, which encodes nothing meaningful about relevance.

    Zero vectors are left alone rather than producing NaN; a NaN silently
    poisons every downstream distance it touches.
    """
    if v.ndim == 1:
        n = np.linalg.norm(v)
        return v if n == 0 else (v / n).astype(np.float32)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (v / n).astype(np.float32)

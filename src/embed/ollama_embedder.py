"""Ollama-backed embedder — the Phase 4 bake-off workhorse.

All four local candidates (nomic, bge-m3, embeddinggemma, mxbai) are served
through Ollama, so one class covers the whole comparison. Differences between
them - dimensionality, context window, query/document prefixes - are DATA in
configs/models.yaml, not code.

TRUNCATION IS THE OTHER SILENT FAILURE
--------------------------------------
Each model has a token limit and exceeding it does not raise; the tail is
simply dropped. mxbai-embed-large caps at 512 tokens, which our 512-token
chunks sit exactly against, and parent-child PARENTS are 2048 tokens - four
times over. Embedding those with mxbai would silently discard three quarters
of every parent.

So we count tokens and warn. A warning we can act on beats a number we cannot
explain three phases later.
"""

from __future__ import annotations

import warnings

import numpy as np

from .base import Embedder, l2_normalize, load_spec


class OllamaEmbedder:
    """Embedder backed by a locally served Ollama model."""

    def __init__(self, name: str, normalize: bool = True, batch_size: int = 16) -> None:
        spec = load_spec(name)
        self.name = name
        self.model = spec["name"]
        self.dim = spec["dim"]
        self.max_ctx = spec.get("max_ctx", 512)
        self.query_prefix = spec.get("query_prefix", "")
        self.doc_prefix = spec.get("doc_prefix", "")
        self.normalize = normalize
        # Small batches deliberately: 7.4 GB total RAM, most of it spoken for.
        self.batch_size = batch_size
        self._warned = False

    # ---- internals ------------------------------------------------------

    def _check_length(self, texts: list[str]) -> None:
        """Warn once if inputs exceed the model's context window."""
        if self._warned:
            return
        from src.ingest.chunkers import count_tokens

        over = [t for t in texts[:64] if count_tokens(t) > self.max_ctx]
        if over:
            self._warned = True
            worst = max(count_tokens(t) for t in over)
            warnings.warn(
                f"{self.name}: {len(over)}+ inputs exceed max_ctx={self.max_ctx} "
                f"(worst {worst} tokens). Ollama TRUNCATES silently - the tail "
                f"of every such chunk is not embedded at all.",
                stacklevel=2,
            )

    def _embed(self, texts: list[str]) -> np.ndarray:
        import ollama

        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i: i + self.batch_size]
            r = ollama.embed(model=self.model, input=batch)
            out.extend(r["embeddings"])
        arr = np.asarray(out, dtype=np.float32)
        return l2_normalize(arr) if self.normalize else arr

    # ---- interface ------------------------------------------------------

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        self._check_length(texts)
        return self._embed([self.doc_prefix + t for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([self.query_prefix + text])[0]


def get_embedder(name: str, **kw) -> Embedder:
    """Resolve configs/experiment.yaml `embedding.model` to an instance."""
    spec = load_spec(name)
    if spec.get("provider") == "openai":
        raise NotImplementedError(
            f"{name} needs an OpenAI API key. The spec is in models.yaml with "
            "enabled:false - set it and add an OpenAI backend to use it."
        )
    return OllamaEmbedder(name, **kw)

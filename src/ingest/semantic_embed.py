"""Two-pass semantic chunking — offload the embedding, keep the algorithm.

THE PROBLEM
-----------
`split_semantic` embeds every sentence in the corpus before a single chunk
exists. Measured on this corpus that is **78,729 sentence groups**:

    this CPU, ~2 groups/s     ~10.9 hours
    Colab T4, ~400 groups/s   ~3.3 minutes

Ten hours to produce one of three candidates for D2 is not a sensible use of
the machine, and the comparison is worthless if it never runs.

THE INSIGHT THAT MAKES OFFLOADING CLEAN
----------------------------------------
`split_semantic(text, embed, ...)` takes `embed` as an injected callable, and
WHICH groups get embedded is decided before any vector is used:

    1-3  split into sentences, group with neighbours   <- produces the groups
    4-6  distances, percentile threshold, cut          <- consumes the vectors

So the set of strings to embed does not depend on the embeddings. That allows
two passes over identical inputs:

    PASS 1 (local, fast)   RecordingEmbedder returns dummy vectors and records
                           every group it was asked for. The chunks it produces
                           are meaningless and are discarded - only the recorded
                           groups matter.
    ---- Colab embeds the recorded groups ----
    PASS 2 (local, fast)   CachedGroupEmbedder looks the vectors up by content
                           hash. Real boundaries, no local inference.

KEYED BY CONTENT, NOT POSITION
------------------------------
Groups are keyed by a hash of their text, exactly as queries are. A positional
key would silently return the wrong vector if the corpus, the sentence
splitter or `buffer_size` changed between passes; a content hash raises
instead. Any of those changing means pass 1 must be re-run anyway.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def group_key(text: str) -> str:
    """Content hash for one grouped-sentence string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class RecordingEmbedder:
    """Pass 1: record what would be embedded, embed nothing.

    Returns a constant vector for every group. That makes the resulting chunk
    boundaries arbitrary - every distance is 0, so nothing exceeds the
    percentile threshold and the whole document becomes one span. The chunks
    from this pass MUST be discarded; only `self.groups` is meaningful.
    """

    def __init__(self, dim: int = 8) -> None:
        self.groups: list[str] = []
        self.dim = dim

    def __call__(self, groups: list[str]) -> list[list[float]]:
        self.groups.extend(groups)
        return [[1.0] + [0.0] * (self.dim - 1) for _ in groups]

    # split_semantic accepts a bare callable, but the Embedder protocol is
    # also satisfied so this can stand in wherever one is expected.
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self(texts), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray(self([text])[0], dtype=np.float32)

    def save(self, path: Path) -> int:
        """Write the de-duplicated groups. Returns how many were written.

        De-duplication matters: textbooks repeat boilerplate, and identical
        groups embed identically, so shipping duplicates would pay for the
        same vector more than once.
        """
        seen: dict[str, str] = {}
        for g in self.groups:
            seen.setdefault(group_key(g), g)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for k, g in sorted(seen.items()):
                print(json.dumps({"k": k, "text": g}, ensure_ascii=False), file=f)
        return len(seen)


class CachedGroupEmbedder:
    """Pass 2: serve precomputed vectors, looked up by content hash."""

    def __init__(self, keys_path: Path, vecs_path: Path) -> None:
        keys = json.loads(Path(keys_path).read_text(encoding="utf-8"))
        vecs = np.load(vecs_path)
        if len(keys) != len(vecs):
            raise ValueError(f"{len(keys)} keys but {len(vecs)} vectors")
        self._by_key = {k: vecs[i] for i, k in enumerate(keys)}
        self.dim = int(vecs.shape[1])

    def __call__(self, groups: list[str]) -> list[np.ndarray]:
        out = []
        for g in groups:
            v = self._by_key.get(group_key(g))
            if v is None:
                raise KeyError(
                    "no cached vector for a sentence group. The corpus, the "
                    "sentence splitter or buffer_size changed since the "
                    "recording pass - re-run pass 1 and re-embed."
                )
            out.append(v)
        return out

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self(texts), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray(self([text])[0], dtype=np.float32)

"""Corpus embedding with an on-disk cache.

WHY THIS IS A SHARED MODULE AND NOT A HELPER IN ONE SCRIPT
----------------------------------------------------------
Embedding 5,796 chunks on a CPU-only box takes roughly 40 minutes per model.
Phase 4 needs the same vectors from several places - the ANN lab, the embedder
bake-off, the chunker bake-off, and later the served application. If each one
carried its own copy of the caching logic they would eventually disagree about
the cache key, and the failure mode is silent: you would compare two embedders
using one embedder's vectors and get a beautiful, meaningless table.

THE CACHE KEY IS THE WHOLE POINT
--------------------------------
Vectors are keyed by `<embedder>_<strategy>.npy`. BOTH parts matter:

    embedder  - a different model means a different vector space entirely
    strategy  - a different chunker means different rows, and a different count

A row-count check guards the strategy half. Nothing can guard the embedder
half except the filename, which is why the embedder name is in it.

ROW ORDER IS LOAD-BEARING
-------------------------
Row i of the returned array must correspond to chunk i of the input list, for
as long as the cache file exists. Search returns row indices; the caller maps
those back to chunk metadata by position. Re-order the corpus without
re-embedding and every citation silently points at the wrong page.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

VEC_DIR = Path("data/processed/vectors")


def cache_path(embedder_name: str, strategy: str,
               backend: str = "ollama") -> Path:
    """Where this (embedder, strategy, backend) triple's vectors live.

    BACKEND IS PART OF THE KEY, AND THIS IS NOT PEDANTRY
    -----------------------------------------------------
    Ollama serves quantised GGUF weights. sentence-transformers on a GPU runs
    fp16/fp32 of the same checkpoint. They are the SAME MODEL and they do NOT
    produce the same vectors.

    That matters because of what D3 is: a comparison between embedders. If
    nomic's vectors came from Ollama and bge-m3's from a Colab GPU, the table
    would be measuring "nomic quantised vs bge-m3 at full precision" - two
    variables at once, and the confound is invisible in the output. Every
    number would look fine.

    So Colab-produced vectors get their own filename suffix and can never be
    silently mixed with locally-produced ones. A D3 run must draw all four
    candidates from the SAME backend.

    The default keeps existing ollama caches valid - the suffix is added only
    for non-ollama backends.
    """
    suffix = "" if backend == "ollama" else f"__{backend}"
    return VEC_DIR / f"{embedder_name}_{strategy}{suffix}.npy"


def is_cached(embedder_name: str, strategy: str, n_expected: int,
              backend: str = "ollama") -> bool:
    """True when usable vectors already exist, without loading the array.

    Reads only the .npy header, so checking twelve cache entries costs
    nothing. Used to report what a run will actually have to compute before
    it starts spending forty minutes per model.
    """
    p = cache_path(embedder_name, strategy, backend)
    if not p.exists():
        return False
    try:
        # mmap_mode reads the header and maps the file without pulling the
        # array into RAM. The obvious alternative, np.lib.format's private
        # _read_array_header, was REMOVED in numpy 2.x - it silently raised
        # AttributeError here, so this function always returned False and the
        # "cached" fast path never ran. Public API only.
        return np.load(p, mmap_mode="r").shape[0] == n_expected
    except Exception:                                    # noqa: BLE001
        return False


def corpus_vectors(
    embedder_name: str,
    strategy: str,
    texts: list[str],
    *,
    progress=None,
    force: bool = False,
    backend: str = "ollama",
) -> np.ndarray:
    """Embed `texts`, or reuse the cached array for this (embedder, strategy).

    Args:
        embedder_name: key in configs/models.yaml `embedders`.
        strategy: chunking strategy the texts came from.
        texts: chunk texts, in corpus order.
        progress: optional callable(done, total, elapsed_s). Called with
            done == -1 to signal a stale cache is being discarded.
        force: re-embed even when a valid cache exists.

    Returns:
        float32 array, shape (len(texts), dim), L2-normalised by the embedder
        so inner product equals cosine similarity.
    """
    VEC_DIR.mkdir(parents=True, exist_ok=True)
    path = cache_path(embedder_name, strategy, backend)

    if path.exists() and not force:
        v = np.load(path)
        if len(v) == len(texts):
            return v
        # Row-count mismatch means the corpus changed under the cache.
        # Re-embed rather than silently score against stale vectors.
        if progress:
            progress(-1, len(texts), 0.0)

    if backend != "ollama":
        # Non-ollama vectors are produced elsewhere (Colab) and imported.
        # Computing them here would silently fall back to the wrong backend
        # and reintroduce the confound this key exists to prevent.
        raise FileNotFoundError(
            f"no {backend} vectors for {embedder_name}/{strategy} at {path}. "
            "Produce them with notebooks/embed_colab.ipynb and import with "
            "scripts/import_colab_vectors.py."
        )

    from src.embed.ollama_embedder import get_embedder

    emb = get_embedder(embedder_name)
    t0 = time.perf_counter()
    step = max(1, len(texts) // 20)
    parts: list[np.ndarray] = []
    for i in range(0, len(texts), step):
        parts.append(emb.embed_documents(texts[i: i + step]))
        if progress:
            progress(min(i + step, len(texts)), len(texts), time.perf_counter() - t0)

    v = np.vstack(parts).astype(np.float32)

    # Write to a temp file and replace, so an interrupted run (or a full disk)
    # cannot leave a truncated .npy that later loads as a valid short array.
    #
    # np.save APPENDS ".npy" when the target path does not already end in it.
    # An earlier version used path.with_suffix(".npy.tmp"), so numpy wrote
    # "...npy.tmp.npy" while the rename looked for "...npy.tmp" - which
    # destroyed a two-hour bge-m3 run at the final line. Writing through an
    # open file handle avoids the whole class of problem: numpy never rewrites
    # a name it was not given.
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as fh:
        np.save(fh, v)
    tmp.replace(path)
    return v

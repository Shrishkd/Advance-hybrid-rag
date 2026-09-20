"""sentence-transformers embedder — the query side of the `st` backend.

WHY THIS FILE HAD TO EXIST
--------------------------
Colab produces DOCUMENT vectors with sentence-transformers at fp16 on a GPU.
Queries are embedded at search time, locally. Until this module existed the lab
embedded those queries with Ollama no matter which backend supplied the
documents - so `--backend st` compared:

    documents:  fp16 HuggingFace weights
    queries:    quantised GGUF weights

Both come from the same checkpoint, so the vectors land in approximately the
same space and cosine similarity returns plausible numbers. Nothing raises and
nothing looks wrong.

HOW BIG IS THE ERROR? MEASURED, AFTER THIS MODULE WAS WRITTEN.
--------------------------------------------------------------
When this file was created the docstring asserted "a systematic quantisation
error that varies by model". That was a guess stated as fact. Measured on our
5,796 chunks, cosine between the Ollama-quantised and fp16 vector of the SAME
chunk:

    nomic           mean 1.00000   min 0.99725
    bge-m3          mean 0.99999   min 0.99964
    embeddinggemma  mean 0.99999   min 0.99654
    mxbai           mean 1.00000   min 0.99758

And end to end on 250 questions, recall@10 was IDENTICAL to four decimals for
all four models across the two backends; only MRR moved, in the fourth decimal.
So the error is real but tiny, and mixing backends would have been very nearly
harmless here. The original claim overstated it.

THE RULE STILL STANDS: one backend per run, on BOTH sides.

Not because the error is large - it is not - but because its size was unknown
until it was measured, and it cost nothing to be correct. A comparison whose
validity depends on an unmeasured assumption is not a comparison. The
measurement above is now a finding (quantisation is free for retrieval at this
scale) rather than a liability.

COST, STATED BEFORE LOADING
---------------------------
These are full-precision HuggingFace weights on a CPU-only box:

    nomic-embed-text-v1.5     137M params   ~550 MB   needs trust_remote_code
    bge-m3                    568M params   ~2.2 GB
    mxbai-embed-large-v1      335M params   ~1.3 GB
    embeddinggemma-300m       300M params   ~1.2 GB   gated on HF

One at a time fits in 7.4 GB. All four at once does not, so the lab loads and
releases per configuration rather than holding a registry of live models.

Queries are short and few (48 golden, 250 synthetic), so CPU inference here is
seconds - the expensive half of the work is the 5,796-document corpus, which
is exactly the half Colab did.
"""

from __future__ import annotations

import numpy as np

from src.embed.base import l2_normalize, load_spec

# Ollama registry name -> (HF checkpoint, needs trust_remote_code)
HF_IDS: dict[str, tuple[str, bool]] = {
    "nomic": ("nomic-ai/nomic-embed-text-v1.5", True),
    "bge-m3": ("BAAI/bge-m3", False),
    "embeddinggemma": ("google/embeddinggemma-300m", False),
    "mxbai": ("mixedbread-ai/mxbai-embed-large-v1", False),
}


class STEmbedder:
    """Local sentence-transformers embedder matching the Colab document pass.

    Prefixes come from configs/models.yaml - the SAME source the notebook
    reads - so query and document sides cannot drift apart. That matters for
    asymmetric models: nomic embeds documents behind "search_document: " and
    queries behind "search_query: ", and swapping them degrades retrieval with
    no error at all.
    """

    def __init__(self, name: str, batch_size: int = 32) -> None:
        if name not in HF_IDS:
            raise KeyError(f"no HF mapping for {name!r}; have {sorted(HF_IDS)}")
        self.name = name
        self.hf_id, self._trc = HF_IDS[name]
        self.batch_size = batch_size

        spec = load_spec(name)
        self.dim = spec["dim"]
        self.max_ctx = spec.get("max_ctx", 512)
        self.query_prefix = spec.get("query_prefix", "") or ""
        self.doc_prefix = spec.get("doc_prefix", "") or ""
        self._model = None

    @property
    def model(self):
        """Lazy: constructing the class must not trigger a 2 GB download."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            kw = {"trust_remote_code": True} if self._trc else {}
            self._model = SentenceTransformer(self.hf_id, device="cpu", **kw)
            if self._model.max_seq_length:
                self._model.max_seq_length = min(
                    self.max_ctx, self._model.max_seq_length
                )
        return self._model

    def _encode(self, texts: list[str]) -> np.ndarray:
        v = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,   # matches the notebook exactly
            show_progress_bar=False,
        ).astype(np.float32)
        # Belt and braces: the lab's flat search is an inner product and is
        # only cosine similarity if rows are unit-norm.
        return l2_normalize(v)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.doc_prefix + t for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([self.query_prefix + text])[0]


def get_st_embedder(name: str, **kw) -> STEmbedder:
    return STEmbedder(name, **kw)


class CachedQueryEmbedder:
    """Query vectors precomputed elsewhere, looked up by content hash.

    WHY LOOKUP RATHER THAN LOCAL INFERENCE
    --------------------------------------
    The `st` backend's documents were embedded on a GPU. Its queries must come
    from the same weights at the same precision, and encoding them locally
    means downloading ~5 GB of HuggingFace checkpoints to process 300 short
    strings - one of which is gated. The notebook embeds the queries in the
    same pass as the documents instead, and this class reads the result.

    KEYED BY TEXT, NOT BY qid
    -------------------------
    A qid is a label whose meaning someone can change; a content hash is not.
    Edit a golden question and its key changes, so the stale vector cannot be
    reused silently - the lookup raises instead. That is the behaviour we want:
    a loud failure beats scoring a rewritten question against the embedding of
    its previous wording.
    """

    def __init__(self, name: str, backend: str = "st") -> None:
        import json

        from src.embed.corpus_cache import VEC_DIR

        spec = load_spec(name)
        self.name, self.backend = name, backend
        self.dim = spec["dim"]
        self.max_ctx = spec.get("max_ctx", 512)

        keys_path = VEC_DIR / f"query_keys__{backend}.json"
        vecs_path = VEC_DIR / f"{name}_queries__{backend}.npy"
        if not (keys_path.exists() and vecs_path.exists()):
            raise FileNotFoundError(
                f"no {backend} query vectors for {name!r}. Run "
                "scripts/export_for_colab.py, embed with "
                "notebooks/embed_colab.ipynb (it embeds queries too), then "
                "scripts/import_colab_vectors.py."
            )
        keys = json.loads(keys_path.read_text(encoding="utf-8"))
        vecs = np.load(vecs_path)
        if len(keys) != len(vecs):
            raise ValueError(
                f"{len(keys)} query keys but {len(vecs)} vectors for {name}"
            )
        self._by_key = {k: vecs[i] for i, k in enumerate(keys)}

    def embed_query(self, text: str) -> np.ndarray:
        from scripts.export_for_colab import query_key

        k = query_key(text.strip())
        v = self._by_key.get(k)
        if v is None:
            raise KeyError(
                f"no cached {self.backend} vector for this question "
                f"(key {k}). It was added or edited after the last export - "
                "re-export and rerun the notebook."
            )
        return v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError(
            "CachedQueryEmbedder serves queries only; documents come from the "
            "corpus vector cache."
        )

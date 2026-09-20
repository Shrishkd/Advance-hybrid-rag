"""Generate notebooks/embed_colab.ipynb.

The notebook is generated rather than hand-written as JSON because .ipynb is a
hostile format to edit by hand: one unescaped newline in a source array and
the file will not open, with no useful error. Regenerate with:

    python scripts/build_colab_notebook.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

OUT = Path("notebooks/embed_colab.ipynb")

# Ollama name -> (HF checkpoint, needs trust_remote_code, gated on HF)
HF_IDS = {
    "nomic": ("nomic-ai/nomic-embed-text-v1.5", True, False),
    "bge-m3": ("BAAI/bge-m3", False, False),
    "embeddinggemma": ("google/embeddinggemma-300m", False, True),
    "mxbai": ("mixedbread-ai/mxbai-embed-large-v1", False, False),
}


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip().splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.strip().splitlines(keepends=True)}


def main() -> int:
    spec = yaml.safe_load(Path("configs/models.yaml").read_text(encoding="utf-8"))
    embedders = spec["embedders"]

    models_py = {}
    for name, (hf, trc, gated) in HF_IDS.items():
        e = embedders[name]
        models_py[name] = {
            "hf": hf,
            "trust_remote_code": trc,
            "gated": gated,
            "query_prefix": e.get("query_prefix", ""),
            "doc_prefix": e.get("doc_prefix", ""),
            "max_ctx": e.get("max_ctx", 512),
            "dim": e.get("dim"),
        }
    models_json = json.dumps(models_py, indent=4, ensure_ascii=False)

    cells = [
        md("""
# Phase 4 — Embedding the corpus on a T4

Produces `nomic`, `bge-m3`, `embeddinggemma` and `mxbai` vectors for the RAG
chatbot's Phase 4.1 embedder bake-off (D3).

**Runtime → Change runtime type → T4 GPU.** Without a GPU this is no faster
than the laptop it is replacing.

### Before you start

1. Run `python scripts/export_for_colab.py --strategy recursive` locally.
2. Upload **both** files from `build/colab/` when prompted below.

### What this notebook does and does not receive

It receives chunk **text** and a line index. It does **not** receive book
titles, authors, chapters, page numbers or chunk ids — those stay on the local
machine, so ground-truth labels never leave it. The uploaded text is still
verbatim copyrighted textbook content; the last cell deletes it.

### Why all four models run here, including nomic

`nomic` vectors already exist locally from Ollama. They are **not reused**.
Ollama serves quantised GGUF; this notebook runs fp16 on GPU. Same checkpoint,
different numbers. Mixing the two would make D3 a comparison of *quantisation*
as much as of *models* — two variables at once, invisible in the output.

All four are therefore produced by the same backend, and land under a distinct
`__st` filename so they can never be mixed with the Ollama set.
"""),
        md("## 1 — Check the GPU"),
        code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "NO GPU — set Runtime > Change runtime type > T4"
"""),
        md("## 2 — Install"),
        code("""
# EmbeddingGemma is a gemma3_text architecture and needs sentence-transformers
# >=5.0 plus a matching transformers. Unpinned "-U" usually gets there, but
# pinning turns a confusing "unrecognised model type" into an install error.
!pip -q install -U "sentence-transformers>=5.0.0" einops
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
"""),
        md("""
## 3 — Upload the corpus

Upload from `build/colab/`: `recursive_texts.jsonl`, `recursive_manifest.json`,
`queries.jsonl`, and — if you want semantic chunking measured (D2) —
`semantic_groups.jsonl`.

`queries.jsonl` holds only the benchmark questions — no book, page or answer.
They are embedded here because query and document vectors must come from the
same backend; encoding 300 short strings locally would otherwise mean
downloading ~5 GB of weights.
"""),
        code("""
import json, pathlib
from google.colab import files

up = files.upload()

texts_file = next(f for f in up if f.endswith("_texts.jsonl"))
man_file   = next(f for f in up if f.endswith("_manifest.json"))
query_file = next((f for f in up if f.endswith("queries.jsonl")), None)
sem_file   = next((f for f in up if f.endswith("semantic_groups.jsonl")), None)

manifest = json.loads(pathlib.Path(man_file).read_text(encoding="utf-8"))
STRATEGY = manifest["strategy"]

rows = [json.loads(l) for l in pathlib.Path(texts_file).read_text(encoding="utf-8").splitlines() if l.strip()]

# Sort by the exported index. Upload order is not guaranteed, and a silently
# reordered corpus makes every vector point at the wrong chunk.
rows.sort(key=lambda r: r["i"])
assert [r["i"] for r in rows] == list(range(len(rows))), "index gap — bad upload"

TEXTS = [r["text"] for r in rows]
assert len(TEXTS) == manifest["n_chunks"], (len(TEXTS), manifest["n_chunks"])
print(f"{len(TEXTS)} chunks, strategy={STRATEGY}")

# Queries are embedded here too. They MUST come from the same backend as the
# documents: fp16 documents scored against quantised queries is a silent,
# model-dependent error in every similarity, and nothing raises.
QKEYS, QTEXTS = [], []
if query_file:
    qrows = [json.loads(l) for l in pathlib.Path(query_file).read_text(encoding="utf-8").splitlines() if l.strip()]
    QKEYS  = [r["k"] for r in qrows]
    QTEXTS = [r["text"] for r in qrows]
    print(f"{len(QTEXTS)} queries")
else:
    print("no queries.jsonl uploaded - documents only")

# Semantic chunking (D2) needs a vector per sentence GROUP before any chunk
# exists: 63k of them, ~11 hours on the laptop CPU, ~3 minutes here. Only the
# chunking embedder needs to do this, not all four.
SEM_KEYS, SEM_TEXTS = [], []
if sem_file:
    srows = [json.loads(l) for l in pathlib.Path(sem_file).read_text(encoding="utf-8").splitlines() if l.strip()]
    SEM_KEYS  = [r["k"] for r in srows]
    SEM_TEXTS = [r["text"] for r in srows]
    print(f"{len(SEM_TEXTS)} sentence groups for semantic chunking")
"""),
        md("""
## 4 — Verify the corpus fingerprint

Recomputed here with the same function as the exporter. If this does not match,
stop: the upload is not the corpus the local machine will score against.
"""),
        code("""
import hashlib

def corpus_fingerprint(texts):
    h = hashlib.sha256()
    for i, t in enumerate(texts):
        h.update(str(i).encode()); h.update(b"\\x00")
        h.update(t.encode("utf-8")); h.update(b"\\x00")
    return h.hexdigest()

FP = corpus_fingerprint(TEXTS)
assert FP == manifest["corpus_fingerprint"], "FINGERPRINT MISMATCH — re-upload"
print("fingerprint OK:", FP[:16], "...")
"""),
        md("""
## 5 — Model registry

Prefixes are copied from `configs/models.yaml`. They are **not** decoration:
asymmetric models are trained with distinct query and document prefixes, and
the local pipeline applies these exact strings. Using different ones here
would embed documents into a different region than local queries land in.

`embeddinggemma` is **gated** on Hugging Face — accept the licence on the model
page and add an `HF_TOKEN` Colab secret, or it will be skipped.
"""),
        code(f"""
MODELS = {models_json}

# Which model performs the semantic-chunking sentence pass. This is the D3
# winner, and the dependency is deliberate and recorded: semantic chunking
# needs an embedder to produce chunks, while D3 compares embedders over
# chunks. Pinning one embedder for chunk PRODUCTION breaks the circularity
# visibly rather than hiding it.
SEMANTIC_MODEL = "embeddinggemma"

# Run a SUBSET. Leave empty to embed all four.
# Vectors already imported locally do not need regenerating: the local importer
# installs whatever is in the zip and leaves the rest alone. If only
# embeddinggemma is missing, set ONLY = ["embeddinggemma"] and this notebook
# takes about three minutes instead of fifteen.
ONLY = []

if ONLY:
    MODELS = {{k: v for k, v in MODELS.items() if k in ONLY}}

for k, v in MODELS.items():
    print(f"{{k:16s}} {{v['hf']:42s}} ctx={{v['max_ctx']:<6d}} gated={{v['gated']}}")
"""),
        md("""
## 5b — Gated-model access check

`embeddinggemma` is gated under the Gemma licence. Two things are needed, and
this cell checks BOTH before the embedding loop rather than after it:

1. Accept the licence at <https://huggingface.co/google/embeddinggemma-300m>
   while signed in.
2. Add your HF token as a Colab secret named `HF_TOKEN` (key icon in the left
   sidebar) and enable notebook access for it.

The token stays in Colab's secret store. Do not paste it into a cell, into the
repo, or into a chat — a cell's output is saved with the notebook.
"""),
        code("""
try:
    from google.colab import userdata
    HF_TOKEN = userdata.get("HF_TOKEN")
except Exception as e:
    HF_TOKEN = None
    print("no Colab secret available:", e)

print("HF_TOKEN present:", bool(HF_TOKEN))

gated = [k for k, v in MODELS.items() if v["gated"]]
if gated:
    if not HF_TOKEN:
        print(f"MISSING TOKEN - {gated} will be SKIPPED. Add the HF_TOKEN secret.")
    else:
        # Cheap probe: fetch one small config file. Fails fast and clearly on a
        # licence that has not been accepted, instead of after a model download.
        from huggingface_hub import hf_hub_download
        for g in gated:
            try:
                hf_hub_download(MODELS[g]["hf"], "config.json", token=HF_TOKEN)
                print(f"access OK: {MODELS[g]['hf']}")
            except Exception as e:
                print(f"NO ACCESS to {MODELS[g]['hf']}: {type(e).__name__}: {e}")
                print("  -> accept the licence on the model page, then rerun.")
"""),
        md("## 6 — Embed"),
        code("""
import gc, time, numpy as np, torch
from sentence_transformers import SentenceTransformer

# HF_TOKEN comes from cell 5b, which already verified access.
BATCH = 64
results, queries_out, semantic_out, skipped = {}, {}, {}, {}

for name, spec in MODELS.items():
    if spec["gated"] and not HF_TOKEN:
        skipped[name] = "gated on HF and no HF_TOKEN secret set"
        print(f"SKIP {name}: {skipped[name]}")
        continue

    print(f"\\n=== {name} ({spec['hf']}) ===")
    t0 = time.time()
    try:
        kw = {"trust_remote_code": True} if spec["trust_remote_code"] else {}
        if HF_TOKEN:
            kw["token"] = HF_TOKEN
        model = SentenceTransformer(spec["hf"], device="cuda", **kw)
        model.max_seq_length = min(spec["max_ctx"], model.max_seq_length or spec["max_ctx"])

        docs = [spec["doc_prefix"] + t for t in TEXTS]
        vecs = model.encode(
            docs,
            batch_size=BATCH,
            convert_to_numpy=True,
            normalize_embeddings=True,   # inner product == cosine, as locally
            show_progress_bar=True,
        ).astype(np.float32)

        assert vecs.shape[0] == len(TEXTS), vecs.shape
        results[name] = vecs

        if SEM_TEXTS and name == SEMANTIC_MODEL:
            # DOCUMENT prefix: these groups are passages, not questions.
            sv = model.encode(
                [spec["doc_prefix"] + t for t in SEM_TEXTS],
                batch_size=BATCH, convert_to_numpy=True,
                normalize_embeddings=True, show_progress_bar=True,
            )
            # float16 halves a ~194 MB download. Only cosine DISTANCES between
            # neighbouring groups are used, then reduced to a percentile
            # threshold - fp16 is far more precision than that needs.
            semantic_out["vecs"] = sv.astype(np.float16)
            print(f"  semantic groups: {sv.shape} (fp16)")

        if QTEXTS:
            # QUERY prefix, not the document one. Asymmetric models are trained
            # with different prefixes for each side and swapping them degrades
            # retrieval with no error at all.
            qv = model.encode(
                [spec["query_prefix"] + q for q in QTEXTS],
                batch_size=BATCH, convert_to_numpy=True,
                normalize_embeddings=True, show_progress_bar=False,
            ).astype(np.float32)
            assert qv.shape[0] == len(QTEXTS), qv.shape
            queries_out[name] = qv
            print(f"  queries: {qv.shape}")
        print(f"{name}: {vecs.shape} in {time.time()-t0:.0f}s "
              f"(max_seq_length={model.max_seq_length})")
    except Exception as e:
        skipped[name] = f"{type(e).__name__}: {e}"
        print(f"FAILED {name}: {skipped[name]}")
    finally:
        # Free VRAM before the next model. A T4 has 15 GB and bge-m3 is large;
        # without this the third or fourth model can OOM. `model` may not
        # exist if SentenceTransformer() itself raised, hence the guard -
        # a NameError here would abort the whole sweep inside a finally block.
        try:
            del model
        except NameError:
            pass
        gc.collect(); torch.cuda.empty_cache()
        print(f"  VRAM free: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB")

print("\\nembedded:", sorted(results), "| skipped:", skipped)
"""),
        md("""
## 7 — Sanity check before download

Two checks that catch the mistakes which otherwise produce a perfectly
plausible, completely meaningless benchmark table.
"""),
        code("""
import numpy as np, itertools

for name, v in results.items():
    n = np.linalg.norm(v, axis=1)
    print(f"{name:16s} shape={str(v.shape):14s} norm[{n.min():.4f},{n.max():.4f}] "
          f"finite={np.isfinite(v).all()}")

# Two models must not produce identical vectors. If they do, the loop
# reassigned the name but kept the previous model loaded.
for a, b in itertools.combinations(sorted(results), 2):
    if results[a].shape == results[b].shape and np.allclose(results[a][:50], results[b][:50], atol=1e-6):
        raise RuntimeError(f"{a} and {b} are IDENTICAL — same model embedded twice")
print("\\nall distinct — OK")
"""),
        md("## 8 — Package and download"),
        code("""
import json, pathlib, shutil, numpy as np

out = pathlib.Path("colab_vectors"); out.mkdir(exist_ok=True)
for name, v in results.items():
    np.save(out / f"{name}_{STRATEGY}__st.npy", v)
for name, v in queries_out.items():
    np.save(out / f"{name}_queries__st.npy", v)
if QKEYS:
    (out / "query_keys__st.json").write_text(json.dumps(QKEYS), encoding="utf-8")
if semantic_out:
    np.save(out / "semantic_groups.npy", semantic_out["vecs"])
    (out / "semantic_group_keys.json").write_text(json.dumps(SEM_KEYS), encoding="utf-8")

# The sidecar the importer requires. Without it the local script refuses.
(out / "corpus_fingerprint.json").write_text(json.dumps({
    "corpus_fingerprint": FP,
    "strategy": STRATEGY,
    "n_chunks": len(TEXTS),
    "backend": "st",
    "models": {k: list(v.shape) for k, v in results.items()},
    "queries": {k: list(v.shape) for k, v in queries_out.items()},
    "skipped": skipped,
}, indent=2), encoding="utf-8")

shutil.make_archive("colab_vectors", "zip", out)
print("size:", pathlib.Path("colab_vectors.zip").stat().st_size / 1e6, "MB")

from google.colab import files
files.download("colab_vectors.zip")
"""),
        md("""
## 9 — Delete the uploaded corpus

Run this before disconnecting. It removes the textbook text from the Colab VM;
the vectors you downloaded are derived numbers, not redistributable prose.
"""),
        code("""
import pathlib, gc
for p in (texts_file, man_file, query_file, sem_file):
    if p is None:
        continue
    pathlib.Path(p).unlink(missing_ok=True)
TEXTS = None; rows = None; docs = None
gc.collect()
print("uploaded corpus deleted from the VM")
print("\\nNext, locally:")
print("  python scripts/import_colab_vectors.py ~/Downloads/colab_vectors.zip")
"""),
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT} ({len(cells)} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

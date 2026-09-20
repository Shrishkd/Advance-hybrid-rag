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
!pip -q install -U sentence-transformers einops
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
"""),
        md("""
## 3 — Upload the corpus

Upload `recursive_texts.jsonl` **and** `recursive_manifest.json`.
"""),
        code("""
import json, pathlib
from google.colab import files

up = files.upload()

texts_file = next(f for f in up if f.endswith("_texts.jsonl"))
man_file   = next(f for f in up if f.endswith("_manifest.json"))

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

for k, v in MODELS.items():
    print(f"{{k:16s}} {{v['hf']:42s}} ctx={{v['max_ctx']:<6d}} gated={{v['gated']}}")
"""),
        md("## 6 — Embed"),
        code("""
import gc, time, numpy as np, torch
from sentence_transformers import SentenceTransformer

try:
    from google.colab import userdata
    HF_TOKEN = userdata.get("HF_TOKEN")
except Exception:
    HF_TOKEN = None

BATCH = 64
results, skipped = {}, {}

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

# The sidecar the importer requires. Without it the local script refuses.
(out / "corpus_fingerprint.json").write_text(json.dumps({
    "corpus_fingerprint": FP,
    "strategy": STRATEGY,
    "n_chunks": len(TEXTS),
    "backend": "st",
    "models": {k: list(v.shape) for k, v in results.items()},
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
for p in (texts_file, man_file):
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

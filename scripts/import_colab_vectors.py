"""Install Colab-produced vectors, refusing anything that does not verify.

    python scripts/import_colab_vectors.py ~/Downloads/colab_vectors
    python scripts/import_colab_vectors.py ~/Downloads/colab_vectors.zip

WHY THIS IS NOT JUST A FILE COPY
---------------------------------
Vectors arriving from another machine are the highest-risk artifact in this
project. Every failure mode is silent:

    wrong order        -> every citation points at the wrong page
    wrong row count    -> caught, but only if something checks
    wrong model        -> a beautiful D3 table comparing a model to itself
    wrong strategy     -> vectors from parent_child scored as recursive
    not normalised     -> inner product stops equalling cosine similarity

None of these raise. All of them produce a plausible report. So nothing is
copied into data/processed/vectors until it verifies:

  1. the sidecar fingerprint matches the CURRENT local corpus,
  2. the row count matches the local chunk count,
  3. rows are unit-norm to within tolerance,
  4. no two candidate models produced identical arrays.

Check 4 catches the specific accident of a notebook loop that reassigns the
model but keeps embedding with the previous one - which produces four valid
files, four plausible tables, and a completely meaningless D3.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.export_for_colab import corpus_fingerprint, load_chunks  # noqa: E402
from src.embed.corpus_cache import VEC_DIR, cache_path                # noqa: E402

CANDIDATES = ("nomic", "bge-m3", "embeddinggemma", "mxbai")


def verify(arr: np.ndarray, n_expected: int, name: str) -> list[str]:
    """Structural checks on one vector array."""
    errs: list[str] = []
    if arr.ndim != 2:
        errs.append(f"{name}: expected 2-D, got shape {arr.shape}")
        return errs
    if arr.shape[0] != n_expected:
        errs.append(f"{name}: {arr.shape[0]} rows, corpus has {n_expected}")
    if not np.isfinite(arr).all():
        errs.append(f"{name}: contains NaN or inf")
    norms = np.linalg.norm(arr, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        errs.append(
            f"{name}: rows are not L2-normalised "
            f"(min {norms.min():.4f}, max {norms.max():.4f}). Inner product "
            "would stop equalling cosine similarity."
        )
    return errs


def main() -> int:
    ap = argparse.ArgumentParser(description="Import Colab vectors.")
    ap.add_argument("source", help="directory or .zip from the notebook")
    ap.add_argument("--strategy", default="recursive")
    ap.add_argument("--backend", default="st")
    args = ap.parse_args()

    src = Path(args.source).expanduser()
    tmp: tempfile.TemporaryDirectory | None = None
    if src.suffix == ".zip":
        tmp = tempfile.TemporaryDirectory()
        with zipfile.ZipFile(src) as z:
            z.extractall(tmp.name)
        src = Path(tmp.name)
    if not src.is_dir():
        print(f"not a directory or zip: {src}")
        return 1

    chunks = load_chunks(args.strategy)
    if not chunks:
        print(f"no local chunks for {args.strategy!r} — nothing to verify against")
        return 1
    local_fp = corpus_fingerprint([c["text"] for c in chunks])
    n = len(chunks)

    # 1. Fingerprint. Refuse before reading a single vector.
    sidecar = src / "corpus_fingerprint.json"
    if not sidecar.exists():
        print("MISSING corpus_fingerprint.json — the notebook must write it.")
        print("Refusing to import unverifiable vectors.")
        return 1
    remote = json.loads(sidecar.read_text(encoding="utf-8"))
    if remote.get("corpus_fingerprint") != local_fp:
        print("FINGERPRINT MISMATCH — refusing to import.")
        print(f"  local : {local_fp[:24]}...")
        print(f"  remote: {str(remote.get('corpus_fingerprint'))[:24]}...")
        print()
        print("The corpus changed since export, or the upload was reordered.")
        print("Re-run scripts/export_for_colab.py and redo the Colab run.")
        return 1
    print(f"fingerprint OK ({n} chunks)")

    # 2-3. Structural checks, all models before installing any.
    arrays: dict[str, np.ndarray] = {}
    errors: list[str] = []
    for name in CANDIDATES:
        f = src / f"{name}_{args.strategy}__{args.backend}.npy"
        if not f.exists():
            print(f"  [skip] {name}: no file")
            continue
        arr = np.load(f)
        errs = verify(arr, n, name)
        errors.extend(errs)
        if not errs:
            arrays[name] = arr
            print(f"  [ok]   {name}: {arr.shape} dim={arr.shape[1]}")
        else:
            for e in errs:
                print(f"  [FAIL] {e}")

    # 4. Two models producing identical vectors means the notebook loop
    #    reassigned the name but kept the previous model loaded.
    names = sorted(arrays)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if arrays[a].shape == arrays[b].shape and np.allclose(
                arrays[a][:50], arrays[b][:50], atol=1e-6
            ):
                errors.append(
                    f"{a} and {b} produced IDENTICAL vectors — the notebook "
                    "almost certainly embedded twice with the same model."
                )

    if errors:
        print(f"\n{len(errors)} problem(s); nothing installed.")
        for e in errors:
            print(f"  - {e}")
        return 1
    if not arrays:
        print("\nno usable vector files found.")
        return 1

    VEC_DIR.mkdir(parents=True, exist_ok=True)
    for name, arr in arrays.items():
        dst = cache_path(name, args.strategy, args.backend)
        np.save(dst, arr.astype(np.float32))
        print(f"installed {dst}  ({arr.nbytes/1e6:.1f} MB)")

    print(f"\n{len(arrays)} model(s) installed. Run D3 with:")
    print(f"  python -m evaluation.retrieval_lab --experiment embedder "
          f"--backend {args.backend}")
    if tmp:
        tmp.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())

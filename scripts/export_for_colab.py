"""Export chunk texts for a Colab embedding run.

    python scripts/export_for_colab.py --strategy recursive

Writes build/colab/<strategy>_texts.jsonl and a manifest.

WHAT LEAVES THE MACHINE, AND WHY THAT IS A DECISION NOT A DETAIL
------------------------------------------------------------------
`CLAUDE.md` states: "The corpus never leaves the machine. data/raw/ holds
copyrighted textbooks." This script breaks that rule deliberately and under
instruction, so it exports the MINIMUM that makes embedding possible:

    exported      chunk text, and a line index
    NOT exported  book, author, chapter, section, page numbers, chunk ids

Metadata stays local. What lands on Colab is unlabelled prose with no
indication of which book it came from or where. That does not make it
non-copyrighted - it is still verbatim textbook text - but it removes the
citation scaffolding that would make the dump useful to anyone else, and it
keeps ground-truth page labels off the external machine entirely.

The notebook deletes its inputs before disconnecting.

ORDER INTEGRITY IS THE WHOLE GAME
----------------------------------
Row i of the returned vector array must correspond to chunk i of the LOCAL
corpus. If the order drifts - a file renamed, a chunker rerun, a shuffled
upload - every vector maps to the wrong chunk, every citation points at the
wrong page, and nothing raises. The retrieval numbers would still look
plausible.

So this writes a `corpus_fingerprint`: a hash over the ordered texts. The
import script recomputes it against the current local corpus and REFUSES to
install vectors whose fingerprint does not match.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT_DIR = Path("build/colab")


def load_chunks(strategy: str) -> list[dict]:
    """Must match evaluation.retrieval_lab.load_chunks EXACTLY."""
    d = Path("data/chunks") / strategy
    rows: list[dict] = []
    for f in sorted(d.glob("*.jsonl")):
        if f.name.endswith(".parents.jsonl"):
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def corpus_fingerprint(texts: list[str]) -> str:
    """Order-sensitive hash of the corpus.

    Hashes each text with its index, so a reordering changes the fingerprint
    even though the multiset of texts is identical. A plain hash of the
    concatenation would also catch reordering, but including the index makes
    the intent explicit and survives refactors of the join.
    """
    h = hashlib.sha256()
    for i, t in enumerate(texts):
        h.update(str(i).encode())
        h.update(b"\x00")
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="Export chunk texts for Colab.")
    ap.add_argument("--strategy", default="recursive")
    args = ap.parse_args()

    chunks = load_chunks(args.strategy)
    if not chunks:
        print(f"no chunks for strategy {args.strategy!r}")
        return 1
    texts = [c["text"] for c in chunks]
    fp = corpus_fingerprint(texts)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    texts_path = OUT_DIR / f"{args.strategy}_texts.jsonl"
    with texts_path.open("w", encoding="utf-8") as f:
        for i, t in enumerate(texts):
            # Text and index only. No book, no page, no chunk id.
            f.write(json.dumps({"i": i, "text": t}, ensure_ascii=False) + "\n")

    manifest = {
        "strategy": args.strategy,
        "n_chunks": len(texts),
        "corpus_fingerprint": fp,
        "expected_outputs": [
            f"{name}_{args.strategy}__st.npy"
            for name in ("nomic", "bge-m3", "embeddinggemma", "mxbai")
        ],
    }
    (OUT_DIR / f"{args.strategy}_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    mb = texts_path.stat().st_size / 1e6
    print(f"{len(texts)} chunks -> {texts_path}  ({mb:.1f} MB)")
    print(f"fingerprint: {fp[:16]}...")
    print(f"manifest   : {OUT_DIR / (args.strategy + '_manifest.json')}")
    print()
    print("Upload BOTH files to the Colab notebook. Metadata (book, page,")
    print("chapter, chunk_id) stays on this machine.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

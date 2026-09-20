"""Synthetic retrieval set — statistical power for Phase 4, for free.

    python -m evaluation.generate_synthetic --n 250
    python -m evaluation.generate_synthetic --n 250 --model gpt-oss:20b-cloud

Writes data/golden/synthetic_retrieval.jsonl.

THE PROBLEM THIS SOLVES
-----------------------
The golden set is capped at 50 items, of which 48 are scorable. At that size a
5-point difference between two configs is inside the noise band, so D2/D3/D6
could easily produce a "winner" that is a coin flip. The honest fix is more
questions, and the cap exists because HUMAN CURATION is expensive.

But curation is expensive for one reason: verifying that a written ANSWER is
correct and complete. Retrieval metrics do not use the answer at all. They use
only:

    question  ->  (book, page_range)

WHY THE GROUND TRUTH IS CORRECT BY CONSTRUCTION
------------------------------------------------
If a question is generated FROM a specific chunk, that chunk's page range is
its ground truth automatically. Nobody has to label anything, because the
label is the provenance. That makes a 250-item set essentially free, and it
means these labels are more reliable than hand-written ones, not less.

WHAT THIS SET IS NOT
--------------------
Three limits, stated up front so nobody quotes these numbers as if they were
the golden set:

  1. NO ANSWERS ARE VERIFIED. This set may ONLY be used for judge-free
     retrieval metrics. It cannot measure faithfulness, correctness or any
     Phase 8 quality metric.

  2. VOCABULARY LEAKAGE. A model writing a question while looking at a passage
     reuses that passage's wording, which BM25 then matches trivially. Expect
     absolute scores here to run HIGHER than the golden set, and expect the
     lexical arm to be flattered. Use this set for RANKING configurations
     against each other, and the golden set for absolute numbers.

  3. SINGLE-HOP ONLY. Each question comes from one chunk, so nothing here
     tests multi-hop or cross-document retrieval - the very capabilities the
     golden set showed we are worst at. Those stay hand-written.

The division of labour: the golden 50 says HOW GOOD retrieval is; the
synthetic 250 says WHICH CONFIG IS BETTER with enough samples to mean it.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import yaml
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm.client import OllamaClient                              # noqa: E402

console = Console()

OUT = Path("data/golden/synthetic_retrieval.jsonl")
MIN_CHARS = 400          # skip stubs: headers, captions, half-empty pages
MAX_CHARS = 2400         # keep the prompt small and the question focused

PROMPT = """You are writing retrieval benchmark questions from a machine \
learning textbook passage.

Write ONE question that:
- is answerable ONLY from this passage, not from general ML knowledge
- a student would plausibly ask
- does NOT quote rare phrases from the passage verbatim (paraphrase instead)
- does NOT mention the book, author, page, "the passage" or "the text"

PASSAGE:
{passage}

Reply with JSON only: {{"question": "..."}}"""


def usable(text: str) -> bool:
    """Is this chunk substantial enough to generate a question from?

    Rejects the two things that produce useless questions: stubs with no
    content, and chunks that are mostly mathematical notation, where a
    generated question tends to be about symbols rather than ideas.
    """
    if not (MIN_CHARS <= len(text) <= MAX_CHARS):
        return False
    letters = sum(c.isalpha() or c.isspace() for c in text)
    return letters / len(text) >= 0.75


def parse_question(raw: str) -> str | None:
    """Pull the question out of a model reply, tolerating stray prose."""
    m = re.search(r'\{.*?\}', raw, re.S)
    if m:
        try:
            q = json.loads(m.group(0)).get("question", "").strip()
            if q:
                return q
        except json.JSONDecodeError:
            pass
    # Fallback: a bare line ending in '?' is good enough to salvage.
    for line in raw.splitlines():
        line = line.strip().strip('"')
        if line.endswith("?") and len(line) > 20:
            return line
    return None


def leaks_source(question: str) -> bool:
    """Reject questions that name the book or refer to 'the passage'.

    Both make the item unrepresentative: the first tests metadata filtering
    rather than retrieval, the second is not a question a user would ask.
    """
    bad = ("passage", "text above", "this text", "the book", "the chapter",
           "bishop", "goodfellow", "huyen", "géron", "geron", "raschka",
           "jurafsky", "according to the")
    q = question.lower()
    return any(b in q for b in bad)


def load_chunks(strategy: str) -> list[dict]:
    d = Path("data/chunks") / strategy
    rows: list[dict] = []
    for f in sorted(d.glob("*.jsonl")):
        if f.name.endswith(".parents.jsonl"):
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate the synthetic retrieval set.")
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--model", default=None, help="default: models.yaml bulk tier")
    ap.add_argument("--strategy", default=None)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path("configs/experiment.yaml").read_text(encoding="utf-8"))
    models = yaml.safe_load(Path("configs/models.yaml").read_text(encoding="utf-8"))
    strategy = args.strategy or cfg["chunking"]["strategy"]
    model = args.model or models["cloud"]["bulk"]["name"]

    chunks = [c for c in load_chunks(strategy) if usable(c["text"])]
    console.print(f"[cyan]{len(chunks)} usable chunks ({strategy}); "
                  f"generating {args.n} with {model}[/cyan]")

    # Sample WITHOUT replacement and stratify by book, so one long textbook
    # cannot supply half the benchmark. Book balance matters more here than in
    # the golden set precisely because nobody is eyeballing these.
    rng = random.Random(args.seed)
    by_book: dict[str, list[dict]] = {}
    for c in chunks:
        by_book.setdefault(c["book"], []).append(c)
    for v in by_book.values():
        rng.shuffle(v)

    books = sorted(by_book)
    # Oversample the pool 2.5x. A third of candidates are rejected - the model
    # returns an unparseable reply, or writes a question naming the book - and
    # a pool sized exactly to `n` therefore runs dry short of target. Measured
    # on the first run: 250 requested, 252 sampled, 82 rejected, 170 produced.
    # Rejected items cost nothing to replace because the LLM cache makes a
    # rerun free for everything already generated.
    per_book = int(args.n * 2.5) // len(books) + 1
    pool: list[dict] = []
    for b in books:
        pool.extend(by_book[b][:per_book])
    rng.shuffle(pool)

    client = OllamaClient()
    out: list[dict] = []
    skipped = {"no_question": 0, "leaked": 0, "error": 0}
    t0 = time.perf_counter()

    for c in pool:
        if len(out) >= args.n:
            break
        try:
            # temperature > 0 so 250 passages do not yield 250 identically
            # phrased questions; cache is keyed on params so reruns are stable.
            # max_tokens=800, not 200. gpt-oss emits a reasoning trace and
            # IGNORES think=False, so a tight budget is consumed entirely by
            # reasoning and the answer comes back EMPTY - no error, just
            # nothing. Measured: 200 tokens produced ~950 chars of reasoning
            # and zero output on 6/6 attempts. The budget must cover the
            # reasoning AND the one-line answer.
            raw = client.chat(model, PROMPT.format(passage=c["text"]),
                              max_tokens=800, temperature=0.7).text
        except Exception as e:                            # noqa: BLE001
            skipped["error"] += 1
            if skipped["error"] <= 3:
                console.print(f"[red]{type(e).__name__}: {e}[/red]")
            continue

        q = parse_question(raw)
        if not q:
            skipped["no_question"] += 1
            continue
        if leaks_source(q):
            skipped["leaked"] += 1
            continue

        out.append({
            "qid": f"syn-{len(out)+1:04d}",
            "question": q,
            "question_type": "synthetic_single_hop",
            "difficulty": "unknown",
            "answerable": True,
            "expected_behaviour": "answer",
            "ground_truth_answer": None,      # deliberately absent - see module docstring
            "ground_truth_contexts": [{
                "book": c["book"],
                "page_start": c["page_start"],
                "page_end": c["page_end"],
            }],
            "notes": "synthetic; retrieval metrics ONLY; label is provenance",
            "_source_chunk_id": c.get("chunk_id", ""),
        })
        if len(out) % 25 == 0:
            console.print(f"  [dim]{len(out)}/{args.n}  "
                          f"{time.perf_counter()-t0:.0f}s[/dim]")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    console.print(f"\n[bold green]{len(out)} items -> {OUT}[/bold green]")
    console.print(f"by book: {dict(Counter(r['ground_truth_contexts'][0]['book'] for r in out))}")
    console.print(f"skipped: {skipped}")
    console.print("[yellow]Retrieval metrics only. No answers are verified.[/yellow]")
    return 0


if __name__ == "__main__":
    sys.exit(main())

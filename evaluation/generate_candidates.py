"""Phase 3: draft golden-set candidates from REAL chunks.

Run:
    .venv\\Scripts\\python.exe -m evaluation.generate_candidates
    .venv\\Scripts\\python.exe -m evaluation.generate_candidates --n 80 --seed 7

Writes data/golden/candidates.jsonl for human curation.

WHY GENERATE FROM CHUNKS RATHER THAN FROM THE MODEL'S HEAD
----------------------------------------------------------
Asking an LLM for "50 questions about machine learning" produces questions
about ML in general — many unanswerable from OUR six books, and none with
ground-truth page labels. Generating FROM a sampled chunk means:

  * the question is answerable from this corpus BY CONSTRUCTION, and
  * the ground-truth context is known before the question exists, because we
    know which chunk produced it.

That second point is the whole game. Labelling contexts by hand afterwards is
the most tedious hour in the project; here it comes free.

WHAT THIS DELIBERATELY DOES NOT GENERATE
----------------------------------------
Three types are excluded because a model looking at one passage cannot produce
them honestly, and a bad question of these types is worse than none:

  cross_document - needs two books compared; a chunk-local view cannot see
                   that Bishop and Géron treat regularisation differently.
  unanswerable   - must be plausibly NEAR the corpus but genuinely absent.
                   A model asked to invent one tends to produce something
                   either obviously absurd or accidentally answerable.
  ambiguous      - deliberately underspecified, to test whether the system
                   asks for clarification instead of guessing. Requires
                   intent a generator does not have.

These three types live in data/golden/manual.jsonl and are merged in by curate.py.

CURATION CONTRACT
-----------------
Every record carries `keep: null`. Set it true/false (and fix anything wrong)
then run the validator. Candidates are DRAFTS — the model hallucinates
confident answers, and reviewing them against `source_excerpt` is the point.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingest.quality import alpha_ratio, single_char_token_ratio  # noqa: E402
from src.llm.client import OllamaClient                              # noqa: E402

console = Console()

CHUNK_DIR = Path("data/chunks/recursive")
OUT = Path("data/golden/candidates.jsonl")
MODEL = "gpt-oss:120b-cloud"          # free tier, strongest available

# Types drawn from ONE chunk vs TWO. See module docstring.
SINGLE = {"factual": 14, "definition": 12, "explanation": 14,
          "numerical": 10, "specific_source": 10}
PAIRED = {"multi_hop": 8, "comparison": 6, "synthesis": 6}


# ───────────────────────────────────────────────────────────────────────
# Chunk selection
# ───────────────────────────────────────────────────────────────────────

@dataclass
class Candidate:
    chunk_id: str
    book: str
    text: str
    page_start: int
    page_end: int
    breadcrumb: list[str]


# Sections that have real breadcrumbs but contain no teachable content.
# Found by inspecting the sampling dry-run: a "multi-hop" job drew Jurafsky's
# BIBLIOGRAPHICAL AND HISTORICAL NOTES, which would have produced a question
# about citation history rather than about machine learning. Requiring a
# non-empty breadcrumb is NOT enough - bibliographies, exercises and indexes
# all have perfectly good breadcrumbs.
_NON_CONTENT = re.compile(
    r"bibliograph|historical note|references|further reading|exercise|"
    r"\bindex\b|acknowledg|preface|table of contents|notation|"
    r"about the author|colophon|copyright|errata",
    re.I,
)

# Draft/editorial markers. Jurafsky's edition is a draft and carries these.
_DRAFT_CRUFT = re.compile(r"not yet updated|TODO|\[\s*citation needed\s*\]", re.I)


def usable(rec: dict) -> bool:
    """Is this chunk worth building a benchmark question from?

    Reuses the Phase 1 damage metrics rather than inventing new ones — they
    already detect exactly the chunks that produce bad questions.

    Rejects:
      * short chunks - not enough substance for a real question
      * low alpha_ratio - equation soup or a table; a question from it would
        test parsing luck, not retrieval
      * high single_char_token_ratio - fragmented maths
      * no breadcrumb - front matter, contents pages
      * non-content sections - bibliography, exercises, index (see above)
      * draft cruft - editorial placeholders, not prose
    """
    t = rec["text"]
    crumbs = rec.get("breadcrumb") or []
    if not crumbs:
        return False
    if _NON_CONTENT.search(" > ".join(crumbs)):
        return False
    if _DRAFT_CRUFT.search(t):
        return False
    return (
        rec["n_tokens"] >= 250
        and alpha_ratio(t) >= 0.78
        and single_char_token_ratio(t) <= 0.12
    )


def load_pool() -> dict[str, list[Candidate]]:
    """Load usable chunks, grouped by book."""
    pool: dict[str, list[Candidate]] = {}
    for f in sorted(CHUNK_DIR.glob("*.jsonl")):
        if f.name.endswith(".parents.jsonl"):
            continue
        keep: list[Candidate] = []
        total = 0
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            total += 1
            r = json.loads(line)
            if usable(r):
                keep.append(Candidate(
                    chunk_id=r["chunk_id"], book=r["book"], text=r["text"],
                    page_start=r["page_start"], page_end=r["page_end"],
                    breadcrumb=r["breadcrumb"],
                ))
        if keep:
            pool[keep[0].book] = keep
            console.print(f"  {keep[0].book:12s} {len(keep):5d} usable / {total:5d}")
    return pool


# ───────────────────────────────────────────────────────────────────────
# Prompting
# ───────────────────────────────────────────────────────────────────────

TYPE_BRIEF = {
    "factual": "a specific fact stated in the passage (a name, date, property, or claim)",
    "definition": "what a technical term means ('What is X?')",
    "explanation": "why or how a mechanism works, requiring a few sentences to answer",
    "numerical": "a number, formula, equation, or quantitative relationship",
    "specific_source": (
        "a question that NAMES THE BOOK OR AUTHOR explicitly, e.g. "
        "'According to Bishop, ...' or 'How does Geron describe ...'. "
        "This tests metadata-filtered retrieval"
    ),
    "multi_hop": (
        "a question that CANNOT be answered from either passage alone - it must "
        "require a fact from the first AND a fact from the second, chained together"
    ),
    "comparison": "a question contrasting two concepts, one drawn from each passage",
    "synthesis": (
        "a question whose answer must COMBINE both passages into something neither "
        "states on its own"
    ),
}

SYSTEM = """You write benchmark questions for evaluating a retrieval-augmented \
question-answering system built over machine-learning textbooks.

Rules, all mandatory:
1. The question must be answerable using ONLY the passage(s) given.
2. Write it as a REAL USER would type it to an ML chatbot. Never refer to "the \
passage", "the text", "the excerpt", "above", or "this section" - the user \
cannot see any passage. Bad: "According to the text, what is dropout?" \
Good: "What is dropout and why does it help?"
3. It must be specific enough to have one defensible answer, not open-ended chat.
4. The answer must be stated or directly derivable from the passage(s). Do not \
use outside knowledge, and do not invent detail that is not there.
5. Difficulty: "easy" = one sentence answers it; "medium" = needs a short \
paragraph or combining two statements; "hard" = needs real reasoning or \
several linked facts.
6. Keep the answer UNDER 70 WORDS. It is a grading reference, not an essay.

Return ONLY a JSON object, no prose and no code fences:
{"question": "...", "answer": "...", "difficulty": "easy|medium|hard"}"""


def build_prompt(qtype: str, cands: list[Candidate]) -> str:
    parts = [f"Write ONE question of this kind: {TYPE_BRIEF[qtype]}.", ""]
    for i, c in enumerate(cands, 1):
        label = f"PASSAGE {i}" if len(cands) > 1 else "PASSAGE"
        where = " > ".join(c.breadcrumb) if c.breadcrumb else "unsectioned"
        parts += [f"--- {label} (from '{c.book}', section: {where}) ---", c.text, ""]
    if len(cands) > 1:
        parts.append(
            "The question MUST require BOTH passages. If a reader could answer it "
            "from only one, it is wrong for this task."
        )
    if qtype == "specific_source":
        parts.append(f"Name the source explicitly in the question: '{cands[0].book}'.")
    return "\n".join(parts)


_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)


def parse_json(text: str) -> dict | None:
    """Extract a JSON object from model output.

    Models wrap JSON in code fences or add a sentence before it despite being
    told not to. Rather than fail the whole run on formatting, strip fences and
    fall back to the outermost braces.
    """
    if m := _FENCE.search(text):
        text = m.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    if (a := text.find("{")) != -1 and (b := text.rfind("}")) > a:
        try:
            return json.loads(text[a: b + 1])
        except json.JSONDecodeError:
            pass

    # Salvage a TRUNCATED object. gpt-oss spends ~60 tokens on reasoning even
    # with think=False, so a long paired-passage answer can hit the token cap
    # mid-string, leaving JSON that is valid up to the cut. Rather than discard
    # a good question because its reference answer was clipped, close the open
    # string and brace and keep what arrived. The answer is a DRAFT for human
    # review anyway, so a slightly short one is still useful.
    if a != -1:
        frag = text[a:].rstrip()
        if '"question"' in frag:
            repaired = frag.rstrip(",: \n")
            if repaired.count('"') % 2:      # unterminated string
                repaired += '"'
            repaired += "}"
            try:
                d = json.loads(repaired)
                d["_truncated"] = True
                return d
            except json.JSONDecodeError:
                return None
    return None


# ───────────────────────────────────────────────────────────────────────
# Generation
# ───────────────────────────────────────────────────────────────────────

def plan_samples(pool: dict[str, list[Candidate]], rng: random.Random
                 ) -> list[tuple[str, list[Candidate]]]:
    """Decide (question_type, chunks) pairs, balanced across books.

    Deliberately NOT proportional to corpus size. Jurafsky has ~6x Geron's
    chunks; a proportional sample would leave Geron with 2-3 questions and tell
    us almost nothing about retrieval from it. A benchmark should cover the
    corpus, not mirror its shape.
    """
    books = sorted(pool)
    jobs: list[tuple[str, list[Candidate]]] = []

    for qtype, n in SINGLE.items():
        for i in range(n):
            b = books[i % len(books)]
            jobs.append((qtype, [rng.choice(pool[b])]))

    for qtype, n in PAIRED.items():
        for i in range(n):
            b = books[i % len(books)]
            # Two chunks from DIFFERENT sections, else "multi-hop" is two
            # halves of one continuous argument and hops nowhere.
            a, c = rng.sample(pool[b], 2)
            for _ in range(8):
                if a.breadcrumb != c.breadcrumb and abs(a.page_start - c.page_start) > 5:
                    break
                c = rng.choice(pool[b])
            jobs.append((qtype, [a, c]))

    rng.shuffle(jobs)
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser(description="Draft golden-set candidates.")
    ap.add_argument("--n", type=int, default=0, help="cap candidates (0 = all planned)")
    ap.add_argument("--seed", type=int, default=7, help="sampling seed, for reproducibility")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    console.print("[cyan]loading chunk pool[/cyan]")
    pool = load_pool()
    if not pool:
        console.print("[red]no usable chunks — run src.ingest.pipeline first[/red]")
        return 1

    rng = random.Random(args.seed)
    jobs = plan_samples(pool, rng)
    if args.n:
        jobs = jobs[: args.n]

    console.print(f"[cyan]generating {len(jobs)} candidates with {args.model}[/cyan]")
    client = OllamaClient()
    out: list[dict] = []
    failed = 0

    for i, (qtype, cands) in enumerate(jobs, 1):
        try:
            r = client.chat(
                args.model, build_prompt(qtype, cands),
                # 500 was too tight: gpt-oss spends ~60 tokens reasoning even
                # with think=False, and paired-passage answers ran past the
                # cap mid-JSON. That produced a ~25% unparseable rate.
                system=SYSTEM, temperature=0.7, max_tokens=1200, think=False,
            )
        except Exception as e:                                   # noqa: BLE001
            console.print(f"  [red]{i:3d} {qtype}: {e}[/red]")
            failed += 1
            continue

        data = parse_json(r.text)
        if not data or "question" not in data:
            console.print(f"  [yellow]{i:3d} {qtype}: unparseable[/yellow]")
            failed += 1
            continue

        out.append({
            "qid": f"cand-{i:03d}",
            "question": data["question"].strip(),
            "question_type": qtype,
            "difficulty": data.get("difficulty", "medium"),
            "answerable": True,
            "ground_truth_answer": data.get("answer", "").strip(),
            "ground_truth_contexts": [
                {"book": c.book, "page_start": c.page_start, "page_end": c.page_end}
                for c in cands
            ],
            "notes": "",
            # ---- curation aids, stripped when promoted to the golden set ----
            "keep": None,                      # <- set true/false
            "_source_chunk_ids": [c.chunk_id for c in cands],
            "_source_sections": [" > ".join(c.breadcrumb) for c in cands],
            "_source_excerpt": [c.text[:400] for c in cands],
            "_cached": r.cached,
        })

        tag = "cached" if r.cached else f"{r.latency_ms:.0f}ms"
        console.print(f"  [green]{i:3d}[/green] {qtype:16s} {cands[0].book:11s} "
                      f"[dim]{tag}[/dim] {data['question'][:62]}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    from collections import Counter
    console.print(f"\n[bold green]{len(out)} candidates -> {OUT}[/bold green]"
                  + (f"  [red]({failed} failed)[/red]" if failed else ""))
    console.print("by type: " + ", ".join(
        f"{k}={v}" for k, v in sorted(Counter(r["question_type"] for r in out).items())))
    console.print("by book: " + ", ".join(
        f"{k}={v}" for k, v in sorted(Counter(
            c["book"] for r in out for c in r["ground_truth_contexts"]).items())))
    console.print(
        "\n[yellow]Next:[/yellow] curate to 50 by setting \"keep\" true/false, "
        "then hand-write the cross_document / unanswerable / ambiguous items."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

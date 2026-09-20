"""Keyword search over the chunk corpus — for writing hand-labelled questions.

Run:
    python -m evaluation.find_chunks "regulariz"            # prefix/substring
    python -m evaluation.find_chunks "LoRA" -w              # whole word only
    python -m evaluation.find_chunks "dropout" --book Géron --book Goodfellow
    python -m evaluation.find_chunks "attention|self-attention" --regex -n 5

SUBSTRING vs WHOLE WORD - READ THIS BEFORE TRUSTING A COUNT
-----------------------------------------------------------
The default is a case-insensitive SUBSTRING match, which is what you want for
stems: "regulariz" finds regularization, regularized and regularisation.

It is badly wrong for short acronyms. A real example from this corpus:

    "LoRA"        -> 26 hits across ALL SIX books, including Bishop (2006)
    "\\bLoRA\\b"   ->  2 hits, both in Raschka (2024)

The 26 came from "exp-LORA-tion". Bishop predates LoRA by fifteen years. Use
-w for acronyms, and be suspicious of any hit in a book that predates the
concept - that is the cheapest available sanity check on a search result.

WHAT THIS IS FOR
----------------
Three golden-set question types cannot be generated from a single chunk and
must be written by hand (see generate_candidates.py):

  cross_document - needs the SAME topic located in TWO books, so you can ask
                   how their treatments differ. This tool finds both sides.
  unanswerable   - must be plausibly near the corpus but genuinely absent.
                   Searching for a term and getting NOTHING is the evidence
                   that it is absent - far better than assuming.
  ambiguous      - helps you find terms that legitimately mean different
                   things in different books, which is what makes a question
                   genuinely ambiguous rather than merely vague.

Output gives the book, PHYSICAL page range and section for every hit, which is
exactly the ground-truth label format the golden set expects.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingest.toc import is_content_section  # noqa: E402

console = Console()
CHUNK_DIR = Path("data/chunks/recursive")


def search(pattern: str, books: list[str] | None, use_regex: bool,
           per_book: int, context: int, whole_word: bool = False) -> None:
    if use_regex:
        src = pattern
    else:
        src = re.escape(pattern)
        if whole_word:
            src = rf"\b{src}\b"
    rx = re.compile(src, re.I)
    hits_by_book: dict[str, list[dict]] = {}

    for f in sorted(CHUNK_DIR.glob("*.jsonl")):
        if f.name.endswith(".parents.jsonl"):
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if books and r["book"] not in books:
                continue
            # Contents pages list every topic in the book, so they match EVERY
            # query while explaining nothing. Without this they dominate the
            # results and bury the real hits.
            if not is_content_section(r.get("breadcrumb") or []):
                continue
            if not rx.search(r["text"]):
                continue
            hits_by_book.setdefault(r["book"], []).append(r)

    if not hits_by_book:
        console.print(f"[yellow]no matches for {pattern!r}[/yellow]")
        console.print(
            "[dim]For an 'unanswerable' question, zero hits across all six books "
            "is the evidence you want — but search a few phrasings before "
            "concluding a topic is genuinely absent.[/dim]"
        )
        return

    total = sum(len(v) for v in hits_by_book.values())
    console.print(
        f"[cyan]{total} chunks match {pattern!r}[/cyan]  "
        + ", ".join(f"{k}={len(v)}" for k, v in sorted(hits_by_book.items()))
    )

    for book in sorted(hits_by_book):
        console.print(f"\n[bold]{book}[/bold]")
        for r in hits_by_book[book][:per_book]:
            sec = " > ".join(r["breadcrumb"]) or "unsectioned"
            cite = "p." if r["cite_kind"] == "printed" else "PDF p."
            console.print(
                f"  [green]pages {r['page_start']}-{r['page_end']}[/green] "
                f"[dim]({cite} {r['cite_page']})[/dim]  {sec[:72]}"
            )
            m = rx.search(r["text"])
            lo = max(0, m.start() - context)
            snippet = r["text"][lo: m.end() + context].replace("\n", " ")
            console.print(f"     [dim]...{snippet}...[/dim]")

    # The label line you can paste straight into manual.jsonl.
    console.print("\n[yellow]ground_truth_contexts template:[/yellow]")
    for book in sorted(hits_by_book):
        r = hits_by_book[book][0]
        console.print(
            f'  {{"book": "{book}", "page_start": {r["page_start"]}, '
            f'"page_end": {r["page_end"]}}},'
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="Search chunks for question writing.")
    ap.add_argument("pattern")
    ap.add_argument("--book", action="append", dest="books")
    ap.add_argument("--regex", action="store_true")
    ap.add_argument("-w", "--word", action="store_true",
                    help="whole-word match — USE THIS FOR ACRONYMS (see module docs)")
    ap.add_argument("-n", "--per-book", type=int, default=3)
    ap.add_argument("-c", "--context", type=int, default=110)
    args = ap.parse_args()

    if not CHUNK_DIR.exists():
        console.print(f"[red]{CHUNK_DIR} missing — run src.ingest.pipeline[/red]")
        return 1
    search(args.pattern, args.books, args.regex, args.per_book, args.context, args.word)
    return 0


if __name__ == "__main__":
    sys.exit(main())

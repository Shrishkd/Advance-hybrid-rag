"""Phase 3 curation: candidates + manual items -> golden_50.jsonl.

Workflow:
    1. generate_candidates.py  ->  data/golden/candidates.jsonl   (keep: null)
    2. review candidates: set "keep": true/false, fixing questions/answers/pages as needed
    3. add the 3 manual types to data/golden/manual.jsonl
    4. curate.py --promote     ->  data/golden/golden_50.jsonl

Commands:
    python -m evaluation.curate --review            # read candidates in the terminal
    python -m evaluation.curate --review --type multi_hop
    python -m evaluation.curate --stats             # curation progress
    python -m evaluation.curate --promote           # build + validate the golden set

WHY THE CANDIDATES ARE DRAFTS, NOT ANSWERS
------------------------------------------
gpt-oss wrote these from a passage it was shown. It will occasionally:
  * state an answer more confidently than the passage supports,
  * write a question answerable from general ML knowledge without retrieval
    at all (which measures nothing about our system),
  * produce a "multi-hop" question that one passage alone actually answers.

The third is the most damaging, because a false multi-hop item makes a naive
retriever look as good as a multi-hop-capable one. Check each paired question
against BOTH excerpts and ask: could passage 1 alone answer this? If yes, it
is not multi-hop - reject or rewrite it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.golden_schema import (  # noqa: E402
    QUESTION_TYPES, ContextRef, GoldenItem, report, save,
)

console = Console()

CANDIDATES = Path("data/golden/candidates.jsonl")
MANUAL = Path("data/golden/manual.jsonl")
GOLDEN = Path("data/golden/golden_50.jsonl")

MAX_ITEMS = 50          # hard cap set by Shrish


def load_raw(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()
    ]


def to_item(rec: dict) -> GoldenItem:
    """Strip curation-only fields (leading underscore) and build a GoldenItem."""
    clean = {k: v for k, v in rec.items() if not k.startswith("_") and k != "keep"}
    clean["ground_truth_contexts"] = [
        ContextRef(**c) for c in clean.get("ground_truth_contexts", [])
    ]
    return GoldenItem(**clean)


def cmd_review(recs: list[dict], qtype: str | None, only_undecided: bool) -> None:
    shown = 0
    for r in recs:
        if qtype and r["question_type"] != qtype:
            continue
        if only_undecided and r.get("keep") is not None:
            continue
        shown += 1
        mark = {True: "[green]KEEP[/green]", False: "[red]DROP[/red]"}.get(
            r.get("keep"), "[yellow]????[/yellow]"
        )
        console.print(f"\n[bold]{r['qid']}[/bold] {mark} "
                      f"[cyan]{r['question_type']}[/cyan]/{r['difficulty']}")
        console.print(f"  Q: {r['question']}")
        console.print(f"  A: {r['ground_truth_answer'][:300]}")
        for c, sec, ex in zip(r["ground_truth_contexts"],
                              r.get("_source_sections", []),
                              r.get("_source_excerpt", [])):
            console.print(f"  [dim]<- {c['book']} p{c['page_start']}-{c['page_end']} "
                          f"| {sec[:70]}[/dim]")
            console.print(f"     [dim]{ex[:220].replace(chr(10), ' ')}[/dim]")
    console.print(f"\n[cyan]{shown} shown[/cyan]")


def cmd_stats(recs: list[dict]) -> None:
    kept = [r for r in recs if r.get("keep") is True]
    dropped = [r for r in recs if r.get("keep") is False]
    undecided = [r for r in recs if r.get("keep") is None]
    manual = load_raw(MANUAL)

    console.print(f"candidates : {len(recs)}")
    console.print(f"  [green]keep      : {len(kept)}[/green]")
    console.print(f"  [red]drop      : {len(dropped)}[/red]")
    console.print(f"  [yellow]undecided : {len(undecided)}[/yellow]")
    console.print(f"manual     : {len(manual)}")
    total = len(kept) + len(manual)
    colour = "green" if total <= MAX_ITEMS else "red"
    console.print(f"[{colour}]total -> golden set: {total} / {MAX_ITEMS}[/{colour}]")

    have = Counter(r["question_type"] for r in kept + manual)
    missing = QUESTION_TYPES - set(have)
    if missing:
        console.print(f"[yellow]types not yet covered: {sorted(missing)}[/yellow]")


def cmd_promote(recs: list[dict]) -> int:
    kept = [r for r in recs if r.get("keep") is True]
    manual = load_raw(MANUAL)
    combined = kept + manual

    if not combined:
        console.print("[red]nothing to promote — no candidate has keep=true[/red]")
        return 1

    if len(combined) > MAX_ITEMS:
        console.print(
            f"[red]{len(combined)} items exceeds the {MAX_ITEMS} cap. "
            "Drop some before promoting.[/red]"
        )
        return 1

    items: list[GoldenItem] = []
    errors: list[str] = []
    for i, rec in enumerate(combined, 1):
        try:
            it = to_item(rec)
        except TypeError as e:
            errors.append(f"{rec.get('qid', f'#{i}')}: malformed record — {e}")
            continue
        it.qid = f"g-{i:03d}"          # renumber so ids are contiguous and stable
        errors.extend(it.validate())
        items.append(it)

    if errors:
        console.print("[red]validation failed:[/red]")
        for e in errors:
            console.print(f"  - {e}")
        return 1

    # A curated golden set is hours of human judgement and is NOT reproducible
    # from candidates.jsonl alone - items can be edited or added
    # elsewhere (review.html exports straight to this path). Overwriting it
    # silently would be the most expensive data loss in the project, so keep a
    # timestamped copy first.
    if GOLDEN.exists():
        stamp = time.strftime("%Y%m%d-%H%M%S")
        backup = GOLDEN.with_name(f"{GOLDEN.stem}.{stamp}.bak.jsonl")
        backup.write_bytes(GOLDEN.read_bytes())
        console.print(f"[yellow]existing golden set backed up -> {backup}[/yellow]")

    save(GOLDEN, items)
    console.print(f"[bold green]{len(items)} items -> {GOLDEN}[/bold green]\n")
    console.print(report(items))

    # Statistical-power warning. With a 50-item cap across 10 types, thin cells
    # are inevitable; the honest response is to report per-type results as
    # indicative and lean on the larger synthetic set for retrieval metrics.
    console.print(
        "\n[yellow]Reminder:[/yellow] at n=50 a ~5-point gap between configs is "
        "noise. Judge-free retrieval metrics run on the larger synthetic set."
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Curate the golden set.")
    ap.add_argument("--review", action="store_true")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--promote", action="store_true")
    ap.add_argument("--type", help="filter --review by question_type")
    ap.add_argument("--undecided", action="store_true", help="--review only keep=null")
    args = ap.parse_args()

    recs = load_raw(CANDIDATES)
    if not recs:
        console.print(f"[red]no candidates at {CANDIDATES}[/red] — "
                      "run evaluation.generate_candidates first")
        return 1

    if args.promote:
        return cmd_promote(recs)
    if args.review:
        cmd_review(recs, args.type, args.undecided)
        return 0
    cmd_stats(recs)
    return 0


if __name__ == "__main__":
    sys.exit(main())

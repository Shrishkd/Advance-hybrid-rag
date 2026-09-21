"""Regression check over results.db.

    python -m evaluation.compare_runs --experiment retrieval
    python -m evaluation.compare_runs --experiment retrieval --metric mrr
    python -m evaluation.compare_runs --history retrieval --config hybrid_rrf

WHY A SEPARATE COMMAND AND NOT PART OF THE LAB
-----------------------------------------------
The lab answers "which config is best RIGHT NOW". This answers "did anything
get worse since last time", which is a different question and the one that
catches the expensive class of bug: a refactor that quietly degrades retrieval
while every test still passes and every report still renders.

Exits with code 1 on regression, so it can gate a pre-commit hook or a CI job.
It only reports: it never installs hooks or modifies the repository.

THE THRESHOLD IS A TRIPWIRE, NOT A VERDICT
-------------------------------------------
Default 0.02 absolute. At n=48 that is well inside the noise band, so a trip
means "go look", not "you broke it". Absolute rather than relative because a
2-point fall matters equally at 0.82 and at 0.20, and relative thresholds make
small numbers effectively unfalsifiable.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.results_db import compare, history                  # noqa: E402

console = Console()


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare experiment runs.")
    ap.add_argument("--experiment", help="check every config in this experiment")
    ap.add_argument("--history", help="print past runs for this experiment")
    ap.add_argument("--config", help="restrict --history to one config")
    ap.add_argument("--metric", default="recall@10")
    ap.add_argument("--threshold", type=float, default=0.02)
    args = ap.parse_args()

    if args.history:
        rows = history(args.history, args.config, limit=30)
        if not rows:
            console.print("[yellow]no runs recorded[/yellow]")
            return 0
        for r in rows:
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(r["ts"]))
            val = r.get(args.metric)
            shown = f"{val:.4f}" if isinstance(val, (int, float)) else "-"
            console.print(
                f"{when}  {r['config']:18s} {args.metric}={shown}  "
                f"n={r['n_scored']:<4d} git={r['git_sha']:<16s} "
                f"ds={r['dataset_ver']}"
            )
        return 0

    if not args.experiment:
        ap.error("pass --experiment or --history")

    configs = sorted({r["config"] for r in history(args.experiment, limit=200)})
    if not configs:
        console.print(f"[yellow]no runs for {args.experiment!r}[/yellow]")
        return 0

    regressed = False
    for c in configs:
        bad, msg = compare(args.experiment, c, args.metric, args.threshold)
        colour = "red" if bad else "green"
        console.print(f"[{colour}]{'REGRESSED' if bad else 'ok       '}[/{colour}] {msg}")
        regressed |= bad

    if regressed:
        console.print(f"\n[red]regression beyond {args.threshold} — "
                      "investigate before recording a decision[/red]")
    return 1 if regressed else 0


if __name__ == "__main__":
    sys.exit(main())

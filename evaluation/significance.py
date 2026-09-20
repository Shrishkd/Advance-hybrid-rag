"""Paired significance testing for retrieval comparisons.

    python -m evaluation.significance --a hybrid_rrf --b hybrid_weighted \
        --experiment retrieval --dataset synthetic --metric mrr

WHY "IS 5 POINTS NOISE?" IS THE WRONG QUESTION
------------------------------------------------
Throughout this project we have used a rule of thumb: at n=48, a gap under ~5
points is noise. That rule is built on the standard error of an UNPAIRED
proportion, and it is far too conservative here, because our comparisons are
PAIRED: every configuration answers the SAME questions.

The unpaired view asks "could two samples of 48 questions differ this much by
chance?" The paired view asks the much sharper question: "on how many
individual questions did A beat B, and by how much?" If A beats B on 30
questions, loses on 4 and ties on 14, that is strong evidence even though the
means differ by only 3 points.

Two tests, because they fail differently:

  BOOTSTRAP   resamples questions with replacement and reports a confidence
              interval on the mean difference. Answers "how big is the effect,
              and could it be zero?"
  SIGN TEST   counts wins and losses, ignoring magnitude. Answers "does A beat
              B consistently, or does one weird question carry the average?"

A result that passes the bootstrap but fails the sign test is usually a single
outlier question doing all the work. That is worth knowing before a config
value changes.

WHAT THIS CANNOT FIX
--------------------
Significance is not validity. The synthetic set's questions were written while
looking at the passage, which flatters lexical retrieval. A difference can be
real, reproducible and highly significant ON THAT SET while being an artifact
of how the set was built. The test tells you the gap is not chance; it does not
tell you the gap means what you want it to mean.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

console = Console()

PERQ_DIR = Path("reports/perq")


def perq_path(experiment: str, dataset: str, config: str) -> Path:
    """Where one config's per-question scores live."""
    safe = config.replace("/", "_")
    return PERQ_DIR / f"{experiment}_{dataset}_{safe}.jsonl"


def load_perq(experiment: str, dataset: str, config: str) -> dict[str, dict]:
    """qid -> metric dict."""
    p = perq_path(experiment, dataset, config)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found — re-run the experiment; per-question scores are "
            "written alongside the report."
        )
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["qid"]] = r
    return out


def paired_bootstrap(
    a: np.ndarray, b: np.ndarray, n_boot: int = 10000, seed: int = 7
) -> tuple[float, float, float, float]:
    """Bootstrap CI on mean(a) - mean(b), resampling QUESTIONS not scores.

    Resampling questions (rows) rather than the two score lists independently
    is what makes this paired: each draw keeps A's and B's result for the same
    question together, so the shared per-question difficulty cancels out
    instead of inflating the variance.

    Returns:
        (observed_diff, ci_low, ci_high, p_two_sided)

    >>> a = np.array([1.0]*20 + [0.0]*5)
    >>> b = np.array([0.0]*20 + [1.0]*5)
    >>> d, lo, hi, p = paired_bootstrap(a, b, n_boot=2000)
    >>> round(d, 2), p < 0.01
    (0.6, True)
    """
    rng = np.random.default_rng(seed)
    diff = a - b
    observed = float(diff.mean())
    n = len(diff)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    # Two-sided p: how often does a bootstrap sample land on the other side of
    # zero from the observed effect.
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return observed, float(lo), float(hi), float(min(p, 1.0))


def sign_test(a: np.ndarray, b: np.ndarray) -> tuple[int, int, int, float]:
    """Count wins/losses/ties and give an exact binomial p.

    Ties are DISCARDED, which is the standard treatment: a question where both
    configs score identically carries no information about which is better.
    That matters here because retrieval metrics tie constantly - both configs
    find the passage, both score 1.0.

    >>> a = np.array([1.0]*12 + [0.0]*2 + [0.5]*6)
    >>> b = np.array([0.0]*12 + [1.0]*2 + [0.5]*6)
    >>> w, l, t, p = sign_test(a, b)
    >>> w, l, t, round(p, 4)
    (12, 2, 6, 0.0129)
    >>> # 12W/2L over n=14: p = 2 * (C(14,0)+C(14,1)+C(14,2)) / 2**14.
    >>> # The 6 ties are excluded from n, which is why they do not dilute it.
    """
    from math import comb

    d = a - b
    wins = int((d > 0).sum())
    losses = int((d < 0).sum())
    ties = int((d == 0).sum())
    n = wins + losses
    if n == 0:
        return wins, losses, ties, 1.0
    k = min(wins, losses)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return wins, losses, ties, float(min(2 * tail, 1.0))


def compare(experiment: str, dataset: str, cfg_a: str, cfg_b: str,
            metric: str = "recall@10") -> dict:
    """Paired comparison of two configs on one metric."""
    A, B = load_perq(experiment, dataset, cfg_a), load_perq(experiment, dataset, cfg_b)
    shared = sorted(set(A) & set(B))
    if not shared:
        raise ValueError("no overlapping qids — were these run on the same dataset?")

    a = np.array([A[q][metric] for q in shared], dtype=float)
    b = np.array([B[q][metric] for q in shared], dtype=float)

    obs, lo, hi, p_boot = paired_bootstrap(a, b)
    wins, losses, ties, p_sign = sign_test(a, b)
    return {
        "n": len(shared), "metric": metric,
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "diff": obs, "ci_low": lo, "ci_high": hi, "p_bootstrap": p_boot,
        "wins": wins, "losses": losses, "ties": ties, "p_sign": p_sign,
    }


def matrix(experiment: str, dataset: str, configs: list[str],
           metric: str = "recall@10") -> list[dict]:
    """All pairwise comparisons, with the correction counted automatically.

    WHY THIS EXISTS RATHER THAN RUNNING THE PAIRS BY HAND
    ------------------------------------------------------
    Comparing k configs means k*(k-1)/2 tests. Four embedders is six. Running
    them one command at a time makes it easy to pass --n-comparisons 2 out of
    habit, or to quietly forget the pairs that came back null. Counting the
    pairs in code removes both mistakes.

    NON-TRANSITIVITY IS NORMAL, NOT A BUG
    -------------------------------------
    You will see A > C significantly while A ~ B and B ~ C. That is not a
    contradiction: B sits between them and the sample resolves the large gap
    but not the two small ones. "Not significant" means "this sample cannot
    tell them apart", never "they are equal".

    The practical reading: configs that cannot be separated on quality should
    be separated on COST - latency, dimension, memory, context window.
    """
    out = []
    pairs = [(a, b) for i, a in enumerate(configs) for b in configs[i + 1:]]
    alpha = 0.05 / max(1, len(pairs))
    for a, b in pairs:
        try:
            r = compare(experiment, dataset, a, b, metric)
        except FileNotFoundError:
            continue
        r["a"], r["b"] = a, b
        r["alpha"] = alpha
        r["significant"] = (
            not (r["ci_low"] <= 0 <= r["ci_high"])
            and max(r["p_bootstrap"], r["p_sign"]) < alpha
        )
        out.append(r)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Paired significance test.")
    ap.add_argument("--experiment", default="retrieval")
    ap.add_argument("--dataset", default="golden")
    ap.add_argument("--a")
    ap.add_argument("--b")
    ap.add_argument("--matrix", help="comma-separated configs; all pairs")
    ap.add_argument("--metric", default="recall@10")
    ap.add_argument("--n-comparisons", type=int, default=1,
                    help="how many metrics/pairs this investigation tests. "
                         "Applies a Bonferroni correction.")
    args = ap.parse_args()

    if args.matrix:
        cfgs = [c.strip() for c in args.matrix.split(",") if c.strip()]
        rows = matrix(args.experiment, args.dataset, cfgs, args.metric)
        if not rows:
            console.print("[red]no per-question files found for those configs[/red]")
            return 1
        console.print(f"[bold]{args.metric}[/bold], {args.dataset}, " f"n={rows[0]['n']}, {len(rows)} pairs, " f"corrected alpha={rows[0]['alpha']:.4f} ")
        for r in rows:
            mark = "[green]SIG  [/green]" if r["significant"] else "[dim]ns   [/dim]"
            console.print(
                f"  {mark} {r['a']:16s} vs {r['b']:16s} "
                f"{r['diff']:+.4f}  CI[{r['ci_low']:+.4f},{r['ci_high']:+.4f}]  "
                f"{r['wins']}W/{r['losses']}L/{r['ties']}T  p={r['p_sign']:.4f}"
            )
        console.print()
        console.print("[dim]ns = this sample cannot separate them, NOT "
                      "'they are equal'. Separate ties on cost instead.[/dim]")
        return 0

    if not (args.a and args.b):
        ap.error("pass --a and --b, or --matrix")
    r = compare(args.experiment, args.dataset, args.a, args.b, args.metric)

    console.print(f"\n[bold]{args.a}[/bold] vs [bold]{args.b}[/bold] "
                  f"on [cyan]{args.metric}[/cyan]  (n={r['n']}, {args.dataset})\n")
    console.print(f"  mean {args.a:18s} {r['mean_a']:.4f}")
    console.print(f"  mean {args.b:18s} {r['mean_b']:.4f}")
    console.print(f"  difference              {r['diff']:+.4f}  "
                  f"95% CI [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]")
    console.print(f"  bootstrap p             {r['p_bootstrap']:.4f}")
    console.print(f"  sign test               {r['wins']}W / {r['losses']}L / "
                  f"{r['ties']}T   p={r['p_sign']:.4f}")

    # MULTIPLE COMPARISONS.
    # Testing recall@1, recall@10 and MRR and then reporting whichever
    # cleared 0.05 is p-hacking, and it is easy to do honestly: you run
    # three commands, two say no difference, one says significant, and the
    # third is the one that reaches the report.
    #
    # Bonferroni divides the threshold by the number of tests. It is
    # conservative - it assumes independence, and recall@1/recall@10/MRR
    # are strongly correlated - so failing it is not proof of no effect.
    # But clearing the uncorrected threshold alone is not enough evidence
    # to move a config value.
    alpha = 0.05 / max(1, args.n_comparisons)
    if args.n_comparisons > 1:
        console.print(f"  corrected alpha         {alpha:.4f}  "
                      f"(Bonferroni, {args.n_comparisons} comparisons)")

    crosses_zero = r["ci_low"] <= 0 <= r["ci_high"]
    if crosses_zero:
        console.print("\n[yellow]CI includes zero — the gap is consistent with "
                      "no real difference.[/yellow]")
    elif r["p_sign"] > 0.05:
        console.print("\n[yellow]CI excludes zero but the sign test does not "
                      "agree — likely a few large wins rather than a "
                      "consistent edge.[/yellow]")
    elif max(r["p_bootstrap"], r["p_sign"]) >= alpha:
        console.print(
            chr(10) + f"[yellow]Significant uncorrected, but does NOT "
            f"survive correction for {args.n_comparisons} comparisons. "
            "Not enough to move a config value.[/yellow]")
    else:
        console.print(f"\n[green]Both tests agree: {args.a} differs from "
                      f"{args.b} on {args.metric}.[/green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Phase 8 judge validation — does gpt-oss:120b agree with a human, per family?

    python -m evaluation.judge_agreement

Reads data/golden/judge_labels.jsonl (exported from judge_labels.html) and the judge's
stored verdicts, writes reports/08_judge_validation.md.

WHAT WOULD COUNT AS SELF-PREFERENCE
-----------------------------------
Overall agreement is not the question. The judge (gpt-oss:120b) and the production
generator (gpt-oss:20b) are the same family, so the specific failure to look for is
LENIENCY TOWARD ITS OWN FAMILY: the judge saying "supported" where the human said
partial/unsupported, more often on gpt-oss answers than on llama answers.

    over-credit rate = P(judge "supported" | human NOT "supported")

computed separately per generator. If gpt-oss's over-credit rate is clearly above
llama's, the judge flatters its family and gpt-oss's faithfulness (0.87) is inflated.

HONEST ABOUT n
--------------
28 citations split across two groups cannot prove the absence of a small bias. What it
can catch is a LARGE one - the kind that would reverse D9. The report says which.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

LABELS = Path("data/golden/judge_labels.jsonl")
REPORT = Path("reports/08_judge_validation.md")


def judge_verdicts() -> dict[tuple[str, str, str], str]:
    out = {}
    for model in ("gpt-oss:20b-cloud", "llama3.2:3b"):
        p = Path(f"reports/perq/generation_golden_{model.replace(':', '_')}.jsonl")
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                for sid, v in (r.get("faith_verdicts") or {}).items():
                    out[(model, r["qid"], sid.upper())] = v
    return out


def cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    """Agreement corrected for chance, over the 3 labels.

    >>> cohen_kappa([("a","a"),("b","b"),("a","a"),("b","b")])
    1.0
    """
    if not pairs:
        return None
    cats = sorted({x for p in pairs for x in p})
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    pe = sum((sum(a == c for a, _ in pairs) / n) * (sum(b == c for _, b in pairs) / n)
             for c in cats)
    return 1.0 if pe == 1 else round((po - pe) / (1 - pe), 3)


def main() -> int:
    if not LABELS.exists():
        print(f"no labels yet - open data/golden/judge_labels.html, label every "
              f"citation, Export, and save as {LABELS}")
        return 1
    human = [json.loads(l) for l in LABELS.read_text(encoding="utf-8").splitlines() if l.strip()]
    J = judge_verdicts()

    lines = ["# Phase 8 — Judge validation (gpt-oss:120b vs human)", "",
             f"{len(human)} citations labelled blind. The judge is the same family as the "
             "production generator (gpt-oss:20b), so the test is for LENIENCY toward "
             "its own family.", "",
             "| generator | n | exact agreement | Cohen's κ | over-credit rate* |",
             "|---|---:|---:|---:|---:|"]
    stats = {}
    for model in ("gpt-oss:20b-cloud", "llama3.2:3b"):
        pairs = [(h["label"], J[(model, h["qid"], h["sid"].upper())])
                 for h in human if h["model"] == model
                 and (model, h["qid"], h["sid"].upper()) in J]
        n = len(pairs)
        agree = sum(a == b for a, b in pairs) / n if n else 0
        not_sup = [(a, b) for a, b in pairs if a != "supported"]
        over = (sum(b == "supported" for _, b in not_sup) / len(not_sup)) if not_sup else None
        stats[model] = over
        lines.append(f"| {model} | {n} | {agree:.2f} | {cohen_kappa(pairs)} | "
                     f"{'-' if over is None else f'{over:.2f} ({len(not_sup)} cases)'} |")
    lines += ["", "\\* P(judge says *supported* | human said partial or unsupported).", ""]

    g, l = stats.get("gpt-oss:20b-cloud"), stats.get("llama3.2:3b")
    if g is None or l is None:
        verdict = ("Too few non-supported citations in one group to compare leniency. "
                   "Agreement is reported; self-preference is NOT ruled out.")
    elif g - l >= 0.3:
        verdict = ("**Self-preference detected.** The judge over-credits gpt-oss answers "
                   f"far more than llama answers ({g:.2f} vs {l:.2f}). gpt-oss's "
                   "faithfulness score (0.87) is inflated and D9 should be revisited.")
    else:
        verdict = (f"**No large self-preference.** Over-credit on gpt-oss ({g:.2f}) is not "
                   f"materially above llama ({l:.2f}). At this n a SMALL bias cannot be "
                   "ruled out, but not one large enough to reverse D9 - which was also "
                   "corroborated judge-free (citation placement: 0/36 dumped vs 37/48).")
    lines += ["## Verdict", "", verdict, ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())

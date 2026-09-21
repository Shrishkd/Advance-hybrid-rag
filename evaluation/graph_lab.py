"""Phase 6 ablation lab — one graph config at a time, scored two ways.

    python -m evaluation.graph_lab --tag baseline
    python -m evaluation.graph_lab --tag decompose --nodes decompose
    python -m evaluation.graph_lab --tag decompose --nodes decompose --judge gpt-oss:120b-cloud

Writes reports/06_<tag>.md and reports/perq/graph_<tag>.jsonl.

TWO LAYERS, BECAUSE A NODE CAN HELP IN EITHER
----------------------------------------------
    retrieval   recall@10 of the FINAL documents the generator saw, against the
                labelled (book, page_range) ground truth. Judge-free. Answers:
                did decomposition / expansion FIND the passages naive retrieval
                missed?
    answer      usable / grounded / faithfulness, as in Phase 5. Answers: did
                finding them turn into a better answer?

They can disagree, and the disagreement is informative. A node that raises recall
but not answer quality means the generator is the bottleneck; the reverse means the
node helped by REMOVING noise rather than adding evidence.

THE TARGET IS NOT THE AVERAGE
-----------------------------
Phase 4 left three question types weak - multi_hop 0.375, comparison 0.400,
synthesis 0.467 recall@10 - because each needs evidence from more than one
passage. Every table here is broken out by type. A node that lifts those three and
leaves the rest flat is a success an overall mean would hide; one that lifts the
mean by helping the already-easy types is not the fix Phase 6 exists to find.

At n=4-6 per type, per-type cells are INDICATIVE only. Significance is computed
on the pooled multi-passage group (multi_hop + comparison + synthesis, n=14) and
on the full set.

EVERY RUN IS PAIRED AGAINST THE BASELINE
----------------------------------------
Same 50 questions, same retriever, same generator, same prompt. The only
difference is the toggled node, so a per-question paired test is valid - the
design principle that has carried every decision since D5.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.golden_schema import load                             # noqa: E402
from evaluation.retrieval_metrics import RetrievedChunk, score_one    # noqa: E402
from src.generate.citations import verify                             # noqa: E402
from src.generate.prompt import leaked_reasoning                      # noqa: E402

console = Console()

EXPERIMENT = Path("configs/experiment.yaml")
GOLDEN = Path("data/golden/golden_50.jsonl")
PERQ = Path("reports/perq")
MULTI = {"multi_hop", "comparison", "synthesis"}


def score_row(it, out: dict) -> dict:
    """Judge-free scores for one finished graph state."""
    answer = out.get("answer", "") or ""
    sids = [s["sid"] for s in out.get("sources", [])]
    rep = verify(answer, sids)
    refused = bool(out.get("refused"))
    should_refuse = it.expected_behaviour == "refuse"
    leak = leaked_reasoning(answer)
    words = len(answer.split())
    usable = bool((refused and should_refuse) or (
        not should_refuse and not refused and rep.grounded and not leak and words <= 250))

    docs = out.get("documents", []) or []
    rec = None
    if it.ground_truth_contexts:
        rc = [RetrievedChunk(d["book"], d["page_start"], d["page_end"],
                             d.get("chunk_id", "")) for d in docs]
        rec = score_one(rc, it.ground_truth_contexts, ks=(10,))["recall@10"]

    return {
        "qid": it.qid, "type": it.question_type, "question": it.question,
        "answer": answer, "sources_json": out.get("sources", []),
        "trace": out.get("trace", []), "n_docs": len(docs),
        "recall@10": rec, "usable": usable, "grounded": rep.grounded,
        "refused": refused, "should_refuse": should_refuse,
        "wrong_refusal": refused and not should_refuse,
        "leaked_reasoning": leak, "answer_words": words,
    }


def summarise(rows: list[dict]) -> dict:
    def m(key, subset=None):
        vals = [r[key] for r in (subset or rows) if r.get(key) is not None]
        return round(float(np.mean([float(v) for v in vals])), 4) if vals else None
    multi = [r for r in rows if r["type"] in MULTI]
    return {
        "usable": m("usable"), "recall@10": m("recall@10"),
        "usable_multi": m("usable", multi), "recall@10_multi": m("recall@10", multi),
        "wrong_refusals": int(sum(r["wrong_refusal"] for r in rows)),
        "mean_docs": round(float(np.mean([r["n_docs"] for r in rows])), 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 6 graph ablation.")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--nodes", default="", help="comma-separated node toggles")
    ap.add_argument("--judge", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--transform-model", default=None,
                    help="model for query-side nodes (decompose). Separating it "
                         "matters: the first decompose run tested 'decomposition "
                         "BY a 1B model', not decomposition as a technique.")
    args = ap.parse_args()

    from src.graph.rag_graph import GraphConfig, build_graph
    from src.llm.client import OllamaClient
    from src.retrieve.hybrid import HybridRetriever

    cfg = yaml.safe_load(EXPERIMENT.read_text(encoding="utf-8"))
    nodes = {n.strip(): True for n in args.nodes.split(",") if n.strip()}
    gcfg = GraphConfig.from_experiment(cfg, nodes=nodes)
    if args.transform_model:
        gcfg.transform_model = args.transform_model
    client = OllamaClient()
    graph = build_graph(HybridRetriever(cfg), client, gcfg)

    items = load(GOLDEN)
    if args.limit:
        items = items[:args.limit]
    console.print(f"[cyan]{args.tag}: nodes={sorted(nodes) or 'none (baseline)'} "
                  f"| {len(items)} items | {gcfg.generator}[/cyan]")

    rows, lat, errors = [], [], []
    for it in items:
        t0 = time.perf_counter()
        try:
            out = graph.invoke({"question": it.question})
        except Exception as e:                            # noqa: BLE001
            # One transient failure (a DNS outage killed the first decompose
            # run on question ~49 of 50) must not discard the other 49 answers.
            # Record it and move on. The failed item is EXCLUDED from every
            # metric rather than scored as unusable - a network error is not a
            # property of the graph config, and counting it as a quality
            # failure would bias a paired comparison against whichever config
            # happened to be running when the network dropped.
            errors.append(it.qid)
            console.print(f"[red]{it.qid}: {type(e).__name__}: {str(e)[:100]}[/red]")
            continue
        lat.append(time.perf_counter() - t0)
        rows.append(score_row(it, out))
    if errors:
        console.print(f"[yellow]{len(errors)} item(s) errored and are excluded: "
                      f"{errors}. Re-run to fill them; completed work is cached.[/yellow]")

    if args.judge:
        from src.generate.faithfulness import judge_answer
        from src.generate.prompt import load_template
        jt = load_template("judge_faithfulness_v0",
                           required=("{question}", "{answer}", "{sources}"))
        for r in rows:
            if r["should_refuse"] or r["refused"] or not r["answer"]:
                continue
            fr = judge_answer(client, args.judge, jt, r["question"], r["answer"],
                              r["sources_json"])
            if fr and not fr.parse_failed:
                r["faith"] = fr.supported_rate

    agg = summarise(rows)
    if args.judge:
        fs = [r["faith"] for r in rows if r.get("faith") is not None]
        agg["faith"] = round(float(np.mean(fs)), 4) if fs else None
    agg["wall_s_p50"] = round(float(np.median(lat)), 1) if lat else None
    agg["errored"] = len(errors)

    PERQ.mkdir(parents=True, exist_ok=True)
    with (PERQ / f"graph_{args.tag}.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            print(json.dumps(r, ensure_ascii=False), file=f)

    by_type: dict[str, list] = {}
    for r in rows:
        by_type.setdefault(r["type"], []).append(r)
    lines = [f"# Phase 6 — `{args.tag}`", "",
             f"Nodes: `{sorted(nodes) or 'none (baseline)'}` · generator "
             f"`{gcfg.generator}` · transform `{gcfg.transform_model}` · "
             f"top_n={gcfg.top_n} · n={len(items)}", "",
             "| metric | value |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in agg.items()]
    lines += ["", "## By question type (indicative at n=4-10 per cell)", "",
              "| type | n | recall@10 | usable |", "|---|---:|---:|---:|"]
    for t, rs in sorted(by_type.items()):
        recs = [r["recall@10"] for r in rs if r["recall@10"] is not None]
        lines.append(f"| {t}{' *' if t in MULTI else ''} | {len(rs)} | "
                     f"{np.mean(recs):.3f} | {np.mean([r['usable'] for r in rs]):.2f} |"
                     if recs else f"| {t} | {len(rs)} | - | "
                     f"{np.mean([r['usable'] for r in rs]):.2f} |")
    lines += ["", "\\* multi-passage types - the target of Phase 6."]
    out_path = Path(f"reports/06_{args.tag}.md")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    console.print(f"  {agg}")
    console.print(f"[bold green]wrote {out_path}[/bold green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())

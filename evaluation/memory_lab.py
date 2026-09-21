"""Memory lab — does a pronoun-only follow-up still retrieve the right pages?

    python -m evaluation.memory_lab

Writes reports/06_memory.md.

THE TEST
--------
Turn 1 is a golden question with labelled pages. Turn 2 is a follow-up containing
NO topic words at all - "Why is that important?". Turn 2's retrieval is scored
against turn 1's pages, since that is what "that" refers to.

    memory OFF   retrieval sees the literal words "Why is that important?"
    memory ON    condensation rewrites it into a standalone question first

This is the Phase 9 acceptance criterion made measurable: a chatbot whose turn 2
collapses is not a chatbot. It also exercises the checkpointer end to end - the
ON condition runs two real invocations on one thread, so a state leak between
turns (stale sub-queries, stale grades) would show up here, not in production.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.golden_schema import load                             # noqa: E402
from evaluation.retrieval_metrics import RetrievedChunk, score_one    # noqa: E402

console = Console()

FOLLOW_UPS = [
    "Why is that important?",
    "Can you explain it in simpler terms?",
    "What is the intuition behind it?",
    "How is that used in practice?",
]
SINGLE = {"factual", "definition", "explanation", "numerical", "specific_source"}


def recall(docs, contexts) -> float:
    rc = [RetrievedChunk(d["book"], d["page_start"], d["page_end"]) for d in docs]
    return score_one(rc, contexts, ks=(10,))["recall@10"]


def main() -> int:
    from langgraph.checkpoint.memory import MemorySaver
    from src.graph.rag_graph import GraphConfig, build_graph
    from src.llm.client import OllamaClient
    from src.retrieve.hybrid import HybridRetriever

    cfg = yaml.safe_load(Path("configs/experiment.yaml").read_text(encoding="utf-8"))
    retriever = HybridRetriever(cfg)
    client = OllamaClient()
    gcfg = GraphConfig.from_experiment(cfg, nodes={"memory": True})
    graph = build_graph(retriever, client, gcfg, checkpointer=MemorySaver())

    items = [it for it in load(Path("data/golden/golden_50.jsonl"))
             if it.question_type in SINGLE and it.ground_truth_contexts]
    # Only follow up on questions turn 1 actually answers well - otherwise a
    # turn-2 miss could be inherited from turn 1 rather than caused by memory.
    items = [it for it in items
             if recall(retriever.retrieve(it.question), it.ground_truth_contexts) == 1.0][:12]

    rows = []
    for i, it in enumerate(items):
        fu = FOLLOW_UPS[i % len(FOLLOW_UPS)]
        off = recall(retriever.retrieve(fu), it.ground_truth_contexts)

        thread = {"configurable": {"thread_id": f"mem-{it.qid}"}}
        graph.invoke({"question": it.question}, thread)
        st = graph.invoke({"question": fu}, thread)
        on = recall(st["documents"], it.ground_truth_contexts)
        rows.append({"qid": it.qid, "turn1": it.question, "follow_up": fu,
                     "condensed": st["question"], "recall_off": off, "recall_on": on,
                     "history_len": len(st.get("history") or [])})
        console.print(f"  {it.qid}: off={off:.2f} on={on:.2f}  -> {st['question'][:80]}")

    off_m = float(np.mean([r["recall_off"] for r in rows]))
    on_m = float(np.mean([r["recall_on"] for r in rows]))
    Path("reports/perq").mkdir(parents=True, exist_ok=True)
    with open("reports/perq/memory.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            print(json.dumps(r, ensure_ascii=False), file=f)

    lines = ["# Phase 6 — Conversation memory", "",
             f"{len(rows)} two-turn conversations. Turn 2 is a pronoun-only follow-up; "
             "its retrieval is scored against turn 1's labelled pages.", "",
             "| condition | turn-2 recall@10 |", "|---|---:|",
             f"| memory OFF (retrieve on the literal follow-up) | {off_m:.3f} |",
             f"| memory ON (condense, then retrieve) | {on_m:.3f} |", "",
             "## What condensation produced", "",
             "| turn 1 | follow-up | condensed | off | on |", "|---|---|---|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['turn1'][:60]} | {r['follow_up']} | {r['condensed'][:70]} | "
                     f"{r['recall_off']:.2f} | {r['recall_on']:.2f} |")
    Path("reports/06_memory.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    console.print(f"[bold]turn-2 recall@10: OFF {off_m:.3f}  ->  ON {on_m:.3f}[/bold]")
    console.print(f"history length after 2 turns (expect 4): "
                  f"{sorted({r['history_len'] for r in rows})}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

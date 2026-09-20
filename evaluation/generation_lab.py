"""Phase 5 generation lab — D9, judge-free.

    python -m evaluation.generation_lab --models llama3.2:3b,qwen3:4b
    python -m evaluation.generation_lab --models gpt-oss:20b-cloud --top-n 3

Writes reports/05_generators.md.

WHAT THIS MEASURES WITHOUT AN LLM JUDGE
----------------------------------------
Phase 8 owns faithfulness and answer relevancy, which need a cloud judge and
therefore cost money and time. Leaning on it here would make every generation
experiment slow. So this file measures only what arithmetic can:

    citation integrity   does every [S<n>] resolve to a source we supplied?
    citation coverage    did it cite at all, and how many sources did it use?
    refusal behaviour    does it emit NOT_IN_SOURCES on unanswerable items,
                         and does it WRONGLY refuse answerable ones?
    operations           latency p50/p95, tokens in/out, error rate
    format compliance    did it obey the length instruction?

Every one is deterministic and free. What is deliberately NOT here: whether
the answer is CORRECT, or whether a cited claim is actually supported by the
chunk it points at. Both need a judge. An answer can cite [S2] perfectly and
say something [S2] contradicts.

THE TRAP THIS LAYOUT AVOIDS
---------------------------
Fabrication rate alone is a broken metric: a model that never cites anything
scores a perfect 0.0. So `has_citations` is reported beside it, and the
headline number is `grounded` - cited at least once AND every citation
resolves. Optimising one of these in isolation produces a worse system.

RETRIEVAL IS FROZEN
-------------------
Every model sees the SAME retrieved chunks, from the Phase 4 winning config
(embeddinggemma + hybrid_rrf + flat + recursive). Retrieval runs once per
question and is reused across generators, so any difference in the table is
the generator and nothing else - and the run is much cheaper.
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

from evaluation.golden_schema import GoldenItem, load                 # noqa: E402
from evaluation.report_utils import decision_section                  # noqa: E402
from evaluation.results_db import record                              # noqa: E402
from evaluation.retrieval_lab import (                                # noqa: E402
    embed_corpus, flat_search, load_chunks,
)
from src.generate.citations import verify                             # noqa: E402
from src.generate.context import assemble                             # noqa: E402
from src.generate.prompt import build_prompt, is_refusal, load_template  # noqa: E402
from src.llm.client import OllamaClient                               # noqa: E402
from src.retrieve.fusion import reciprocal_rank_fusion                # noqa: E402

console = Console()

EXPERIMENT = Path("configs/experiment.yaml")
GOLDEN = Path("data/golden/golden_50.jsonl")
REPORT = Path("reports/05_generators.md")


def retrieve_all(items: list[GoldenItem], cfg: dict, top_k: int) -> dict:
    """Retrieve once per question, reuse for every generator.

    Freezing retrieval is what makes this a generator comparison rather than a
    confounded end-to-end one. It is also the difference between one retrieval
    pass and one per model.
    """
    from src.embed.ollama_embedder import get_embedder
    from src.index.bm25_index import BM25Index

    strategy = cfg["chunking"]["strategy"]
    embedder = cfg["embedding"]["model"]
    chunks = load_chunks(strategy)
    vectors = embed_corpus(embedder, strategy, chunks)
    emb = get_embedder(embedder)
    bm25 = BM25Index([c["text"] for c in chunks])
    arm_k = top_k * 3

    out: dict[str, list[dict]] = {}
    for it in items:
        qv = emb.embed_query(it.question).reshape(1, -1).astype(np.float32)
        dense_ids, _ = flat_search(vectors, qv, arm_k)
        sparse_ids, _ = bm25.search(it.question, arm_k)
        fused = reciprocal_rank_fusion(
            [[int(i) for i in dense_ids], [int(i) for i in sparse_ids]],
            k=cfg["retrieval"].get("rrf_k", 60), top_n=top_k,
        )
        out[it.qid] = [chunks[d] for d, _ in fused]
    return out


def run_model(model: str, items: list[GoldenItem], retrieved: dict,
              template: str, top_n: int, budget: int,
              client: OllamaClient, max_tokens: int) -> tuple[dict, list[dict]]:
    """Generate an answer per question and score it without a judge."""
    rows: list[dict] = []
    lat: list[float] = []
    errors = 0

    for it in items:
        ctx = assemble(retrieved[it.qid], top_n=top_n, token_budget=budget)
        prompt = build_prompt(it.question, ctx.text, template)

        t0 = time.perf_counter()
        try:
            resp = client.chat(model, prompt, max_tokens=max_tokens,
                               temperature=0.0)
            answer = resp.text or ""
        except Exception as e:                            # noqa: BLE001
            errors += 1
            if errors <= 2:
                console.print(f"[red]{model}: {type(e).__name__}: {e}[/red]")
            answer = ""
        ms = (time.perf_counter() - t0) * 1000
        lat.append(ms)

        rep = verify(answer, [s.sid for s in ctx.sources])
        refused = is_refusal(answer)
        # `answerable` means the evidence is in the corpus; expected_behaviour
        # says what the system should DO. Only "refuse" items should refuse -
        # the "clarify" (ambiguous) items have evidence and must not be
        # declined. Collapsing those two would score a correct clarification
        # as a failure.
        should_refuse = it.expected_behaviour == "refuse"

        rows.append({
            "qid": it.qid,
            "type": it.question_type,
            "expected_behaviour": it.expected_behaviour,
            "n_sources": len(ctx.sources),
            "ctx_tokens": ctx.n_tokens,
            "dropped": ctx.dropped,
            "answer_words": len(answer.split()),
            "refused": refused,
            "should_refuse": should_refuse,
            "correct_refusal": refused and should_refuse,
            "wrong_refusal": refused and not should_refuse,
            "missed_refusal": (not refused) and should_refuse,
            "latency_ms": round(ms, 1),
            "cached": bool(getattr(resp, "cached", False)) if not errors else False,
            **rep.row(),
        })

    def mean(key: str) -> float:
        vals = [r[key] for r in rows]
        return float(np.mean([float(v) for v in vals])) if vals else 0.0

    answerable = [r for r in rows if not r["should_refuse"]]
    refusable = [r for r in rows if r["should_refuse"]]

    agg = {
        "grounded_rate": round(mean("grounded"), 4),
        "has_citations_rate": round(mean("has_citations"), 4),
        "fabrication_rate": round(mean("fabrication_rate"), 4),
        "n_fabricated_total": int(sum(r["n_fabricated"] for r in rows)),
        "correct_refusals": f"{sum(r['correct_refusal'] for r in rows)}/{len(refusable)}",
        "wrong_refusals": int(sum(r["wrong_refusal"] for r in rows)),
        "mean_sources_cited": round(
            float(np.mean([r["n_cited"] for r in answerable])) if answerable else 0.0, 2),
        "answer_words_median": int(np.median([r["answer_words"] for r in rows])),
        "latency_p50_ms": round(float(np.percentile(lat, 50)), 0),
        "latency_p95_ms": round(float(np.percentile(lat, 95)), 0),
        "errors": errors,
    }
    return agg, rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 5 generation lab (D9).")
    ap.add_argument("--models", default="llama3.2:3b,qwen3:4b,phi4-mini")
    ap.add_argument("--prompt", default="answer_v0")
    ap.add_argument("--top-n", type=int, default=6)
    ap.add_argument("--budget", type=int, default=3000)
    ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument("--limit", type=int, default=0, help="first N items (smoke test)")
    args = ap.parse_args()

    cfg = yaml.safe_load(EXPERIMENT.read_text(encoding="utf-8"))
    items = load(GOLDEN)
    if args.limit:
        items = items[:args.limit]
    template = load_template(args.prompt)
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    console.print(f"[cyan]{len(items)} golden items | prompt={args.prompt} | "
                  f"top_n={args.top_n} budget={args.budget}[/cyan]")
    console.print(f"[dim]frozen retrieval: {cfg['embedding']['model']} + "
                  f"{cfg['retrieval']['mode']} + {cfg['chunking']['strategy']}[/dim]")

    retrieved = retrieve_all(items, cfg, cfg["retrieval"]["top_k"])
    client = OllamaClient()

    results, perq = [], {}
    for m in models:
        console.print()
        console.print("[bold cyan]-- " + m + " --[/bold cyan]")
        agg, rows = run_model(m, items, retrieved, template, args.top_n,
                              args.budget, client, args.max_tokens)
        results.append({"config": m, **agg})
        perq[m] = rows
        console.print(
            f"  grounded={agg['grounded_rate']:.2f} cites={agg['has_citations_rate']:.2f} "
            f"fabrication={agg['fabrication_rate']:.3f} "
            f"refusals={agg['correct_refusals']} wrong={agg['wrong_refusals']} "
            f"p50={agg['latency_p50_ms']:.0f}ms errors={agg['errors']}"
        )

    if not results:
        return 1

    import pandas as pd
    keep = ["config", "grounded_rate", "has_citations_rate", "fabrication_rate",
            "n_fabricated_total", "correct_refusals", "wrong_refusals",
            "mean_sources_cited", "answer_words_median",
            "latency_p50_ms", "latency_p95_ms", "errors"]
    df = pd.DataFrame(results)[keep].sort_values("grounded_rate", ascending=False)

    PERQ = Path("reports/perq")
    PERQ.mkdir(parents=True, exist_ok=True)
    for m, rows in perq.items():
        with (PERQ / f"generation_golden_{m.replace(':', '_')}.jsonl").open(
                "w", encoding="utf-8") as f:
            for r in rows:
                print(json.dumps(r), file=f)
        record("generation", m, {k: v for k, v in
                                 next(x for x in results if x["config"] == m).items()
                                 if k != "config" and not isinstance(v, str)},
               cfg=cfg, pinned={"prompt": args.prompt, "top_n": args.top_n},
               n_scored=len(items), dataset=GOLDEN)

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join([
        "# Phase 5 — Generator Comparison (D9)",
        "",
        f"{len(items)} golden items · prompt `{args.prompt}` · top_n={args.top_n} "
        f"· budget={args.budget} tokens",
        "",
        f"Retrieval FROZEN at the Phase 4 winner "
        f"(`{cfg['embedding']['model']}` + `{cfg['retrieval']['mode']}` + "
        f"`{cfg['chunking']['strategy']}` + `flat`) and computed once, so every "
        "row differs only by generator.",
        "",
        "**No LLM judge.** Every column is deterministic and free. Whether an "
        "answer is CORRECT, or whether a cited claim is actually supported by "
        "the chunk it points at, is Phase 8's question.",
        "",
        df.to_markdown(index=False),
        "",
        "## Reading this table",
        "",
        "- **grounded_rate** — cited at least once AND every citation resolves."
        " The headline number.",
        "- **fabrication_rate alone is a trap**: a model that never cites"
        " scores a perfect 0.0. Always read it beside `has_citations_rate`.",
        "- **correct_refusals** — of the items whose `expected_behaviour` is"
        " `refuse`. The 2 `ambiguous` items expect `clarify`, not refusal, and"
        " are counted as wrong refusals if declined.",
        "- **wrong_refusals** — declined a question the corpus can answer. The"
        " expensive failure: an unhelpful system that looks safe.",
        "",
        *decision_section(REPORT, [
            "## Decision",
            "",
            "- **Chosen generator:** _fill in_",
            "- Record as **D9** in `plan.md`; set `generation.model_role` in",
            "  `configs/experiment.yaml`.",
            "",
        ]),
    ]), encoding="utf-8")
    console.print(f"\n[bold green]wrote {REPORT}[/bold green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""The chatbot API — Phase 9. The earned configuration, served.

    uvicorn src.serve.api:app --port 8000

ONE CODE PATH FOR USERS AND FOR EVALUATION
------------------------------------------
Everything a user gets comes from the same graph the Phase 6 labs measured, built
from configs/experiment.yaml. No setting here re-decides anything: generator (D9),
retrieval (D3/D4/D5), chunker (D2), context budget (Phase 5b) and the graph nodes
all come from that file. Changing a decision is one config edit and a restart, and
the served app then provably matches the reported numbers.

MEMORY
------
Conversation state lives in a LangGraph SQLite checkpointer, keyed by session id.
It survives a server restart, and every turn runs `start_turn` (which clears last
turn's sub-queries and grades) then `condense` (which rewrites a pronoun follow-up
into a standalone question before retrieval).

CITATIONS ARE EXPANDED HERE, FROM METADATA
------------------------------------------
The model writes [S3]. This server replaces it with "Bishop, 7.1 Margins, p. 320"
using the SOURCE's metadata - never text the model produced - so a rendered
citation cannot inherit a hallucination. An [S9] that resolves to nothing is shown
flagged, not silently dropped.

GUARDRAILS (Phase 7) plug in at two points, input before the graph and output after
it, without this file changing.
"""

from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

EXPERIMENT = Path("configs/experiment.yaml")
CHECKPOINTS = Path("data/processed/checkpoints.sqlite")
_CITE = re.compile(r"\[\s*(S\d+(?:\s*,\s*S\d+)*)\s*\]", re.I)

app = FastAPI(title="Advanced Hybrid RAG Chatbot", version="1.0")
_state: dict = {}


class ChatIn(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=100)
    message: str = Field(..., min_length=1, max_length=2000)


class Source(BaseModel):
    sid: str
    cite: str
    text: str


class ChatOut(BaseModel):
    answer: str                  # citations expanded for display
    raw_answer: str              # exactly what the model wrote
    standalone_question: str     # what was actually searched for
    sources: list[Source]
    cited: list[str]
    fabricated: list[str]        # [S<n>] tags that resolve to nothing
    refused: bool
    blocked: bool = False        # stopped by an input guardrail
    trace: list[str]
    latency_ms: float


def expand_citations(answer: str, sources: list[dict]) -> str:
    """[S3] -> (Bishop, 7.1 Margins, p. 320), from source metadata only."""
    by_sid = {s["sid"].upper(): s["cite"] for s in sources}

    def repl(m: re.Match) -> str:
        names = []
        for sid in re.findall(r"S\d+", m.group(1), re.I):
            sid = sid.upper()
            names.append(by_sid.get(sid, f"{sid} — NO SUCH SOURCE"))
        return "(" + "; ".join(names) + ")"

    return _CITE.sub(repl, answer)


def guard_input(message: str, first_turn: bool = True) -> tuple[bool, str]:
    """Phase 7 hook. Returns (allowed, reason). Default: allow."""
    try:
        from src.guardrails.input_guard import check
        return check(message, first_turn=first_turn)
    except ImportError:
        return True, ""


def guard_output(answer: str) -> str:
    """Phase 7 hook. May redact. Default: pass through."""
    try:
        from src.guardrails.output_guard import scrub
        return scrub(answer)
    except ImportError:
        return answer


def production_nodes(cfg: dict) -> dict:
    """Graph nodes for serving: memory is always on - a chatbot without it
    collapses on every follow-up - plus whatever Phase 6 proved."""
    nodes = {"memory": True}
    for n in (cfg.get("graph", {}) or {}).get("nodes", []) or []:
        nodes[n] = True
    return nodes


@app.on_event("startup")
def _startup() -> None:
    from langgraph.checkpoint.sqlite import SqliteSaver
    from src.graph.rag_graph import GraphConfig, build_graph
    from src.llm.client import OllamaClient
    from src.retrieve.hybrid import HybridRetriever

    cfg = yaml.safe_load(EXPERIMENT.read_text(encoding="utf-8"))
    CHECKPOINTS.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(CHECKPOINTS, check_same_thread=False)
    gcfg = GraphConfig.from_experiment(cfg, nodes=production_nodes(cfg))
    retriever = HybridRetriever(cfg)
    try:
        from src.guardrails.input_guard import attach
        attach(retriever)          # scope check reuses the loaded vectors
    except ImportError:
        pass
    _state.update(
        cfg=cfg, gcfg=gcfg, retriever=retriever,
        graph=build_graph(retriever, OllamaClient(), gcfg,
                          checkpointer=SqliteSaver(conn)),
    )


@app.get("/health")
def health() -> dict:
    import urllib.request
    try:
        urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=2)
        ollama = "up"
    except Exception:                                    # noqa: BLE001
        ollama = "DOWN - run scripts/start_ollama.ps1"
    return {"status": "ok" if "graph" in _state else "starting", "ollama": ollama}


@app.get("/config")
def config() -> dict:
    """The earned decisions this server is running, for the UI's info panel."""
    cfg, g = _state["cfg"], _state["gcfg"]
    return {
        "generator": g.generator, "top_n": g.top_n, "budget_tokens": g.budget,
        "embedder": cfg["embedding"]["model"], "retrieval": cfg["retrieval"]["mode"],
        "chunker": cfg["chunking"]["strategy"], "nodes": sorted(g.nodes),
    }


@app.post("/chat", response_model=ChatOut)
def chat(req: ChatIn) -> ChatOut:
    if "graph" not in _state:
        raise HTTPException(503, "server still loading the index")
    t0 = time.perf_counter()

    thread = {"configurable": {"thread_id": req.session_id}}
    # A session with history is mid-conversation: its raw message may be a pronoun
    # follow-up that only makes sense after condensation, so scope is skipped.
    prior = _state["graph"].get_state(thread).values.get("history") or []
    allowed, reason = guard_input(req.message, first_turn=not prior)
    if not allowed:
        return ChatOut(answer=reason, raw_answer="", standalone_question=req.message,
                       sources=[], cited=[], fabricated=[], refused=True,
                       blocked=True, trace=["guard_input:blocked"],
                       latency_ms=(time.perf_counter() - t0) * 1000)

    try:
        st = _state["graph"].invoke({"question": req.message}, thread)
    except Exception as e:                               # noqa: BLE001
        raise HTTPException(502, f"generation failed: {type(e).__name__}: {e}") from e

    from src.generate.citations import verify
    sources = st.get("sources", []) or []
    raw = st.get("answer", "") or ""
    rep = verify(raw, [s["sid"] for s in sources])
    shown = guard_output(expand_citations(raw, sources)) if raw else \
        "I could not produce an answer to that. Please try rephrasing."

    # The trace accumulates across turns in the checkpointer; report this turn's.
    trace = st.get("trace", []) or []
    if "start_turn" in trace:
        trace = trace[len(trace) - 1 - trace[::-1].index("start_turn"):]

    return ChatOut(
        answer=shown, raw_answer=raw,
        standalone_question=st.get("question", req.message),
        sources=[Source(sid=s["sid"], cite=s["cite"], text=s["text"]) for s in sources],
        cited=rep.cited, fabricated=rep.fabricated,
        refused=bool(st.get("refused")), trace=trace,
        latency_ms=round((time.perf_counter() - t0) * 1000, 1),
    )

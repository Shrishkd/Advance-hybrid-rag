"""The RAG graph — Phase 6. One graph, toggleable nodes, ablated one at a time.

WHY ONE GRAPH AND NOT FOUR PIPELINES
------------------------------------
CRAG, Self-RAG, adaptive routing and multi-hop decomposition are usually presented
as rival architectures. They are not. They are CONTROL FLOW over the same retrieval
core: grade the documents and re-retrieve (CRAG), grade the answer and retry
(Self-RAG), decide whether to retrieve at all (adaptive), split a question into hops
(multi-hop). Built as four pipelines, each would carry its own retrieval and
generation, and a difference in the results could come from any of them. Built as
nodes on one graph, each switches on or off with every other part held fixed - which
is the only way an ablation means anything.

THE BASELINE MUST REPRODUCE PHASE 5 EXACTLY
--------------------------------------------
With every node off, this graph is retrieve -> generate, and it must produce the SAME
prompt, byte for byte, as the Phase 5 generation lab at the D9 settings. Because the
LLM cache is keyed on (model, prompt, params), parity is checkable exactly: an
identical prompt is a cache hit returning the identical answer. Any Phase 6 delta
measured against a baseline that drifted from Phase 5 would be confounded by the
port itself.

WHY LANGGRAPH
-------------
The later nodes are CYCLIC - grade, retry, grade again - and LangChain chains are
linear. LangGraph also provides conversation memory as persisted graph state (a
checkpointer), which replaces the deprecated ConversationBufferMemory.

STATE CARRIES A TRACE
---------------------
Every node appends its name to `trace`. An ablation can then say not just that the
score moved, but which path each question actually took - essential once routing
sends different questions down different branches.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, TypedDict

from src.generate.citations import verify
from src.generate.context import assemble
from src.generate.prompt import build_prompt, is_refusal, load_template


_NL = chr(10)
CONDENSE_PROMPT = (
    "Below is a conversation about machine learning, then a follow-up question. "
    "Rewrite the follow-up as ONE standalone question that makes sense without "
    "the conversation: replace pronouns like 'it', 'that' or 'they' with what they "
    "refer to. If the follow-up is already standalone, return it unchanged. "
    "Output only the question."
    + _NL + _NL + "Conversation:" + _NL + "{history}"
    + _NL + _NL + "Follow-up: {question}"
    + _NL + _NL + "Standalone question:"
)


class RAGState(TypedDict, total=False):
    question: str                 # the user's question, as asked
    query: str                    # what retrieval actually searches for
    sub_queries: list[str]        # set by `decompose`; one retrieval each
    grades: list[bool]            # CRAG relevance verdicts, top docs first
    attempt: int                  # CRAG: 0 first retrieval, 1 after a rewrite
    turn: str                     # the user's words this turn, before condensing
    history: Annotated[list[dict], operator.add]   # [{role, content}], all turns
    documents: list[dict]         # ranked retrieved chunks
    answer: str
    sources: list[dict]           # what the generator was shown, with [S<n>] ids
    refused: bool
    trace: Annotated[list[str], operator.add]   # nodes visited, in order


@dataclass
class GraphConfig:
    """Everything that defines a run. Toggles default OFF: the baseline."""
    generator: str = "gpt-oss:20b-cloud"
    prompt: str = "answer_v0"
    top_n: int = 10
    budget: int = 5300
    max_tokens: int = 700
    temperature: float = 0.0
    # Model for query-side nodes (decompose). CLAUDE.md: graph nodes run on
    # tiny LOCAL models; only answer generation goes to cloud. qwen3:0.6b is
    # excluded - it is the reasoning family that leaked its monologue at 4B.
    transform_model: str = "llama3.2:1b"
    # Condensation decides WHAT this turn is about; if it drifts, the turn
    # answers the wrong question. Small models drifted and invented entities in
    # the decomposition runs, so this defaults to the generator (None -> use
    # `generator`) and runs only when history exists.
    condense_model: str | None = None
    # CRAG grader + rewriter. None -> the generator (cloud). A local llama3.2:3b
    # grader froze the machine: it cannot coexist with the embedder and the API
    # in 7.4 GB (see src/graph/crag.py BATCH_GRADE_PROMPT).
    grader_model: str | None = None
    # Phase 6 nodes - each added and ablated one at a time.
    nodes: dict = field(default_factory=dict)

    @classmethod
    def from_experiment(cls, cfg: dict, **overrides) -> "GraphConfig":
        g = cfg.get("generation", {})
        base = cls(
            generator=g.get("model_role", cls.generator),
            top_n=g.get("top_n", cls.top_n),
            budget=g.get("context_budget_tokens", cls.budget),
        )
        for k, v in overrides.items():
            setattr(base, k, v)
        return base


def parse_subqueries(raw: str, question: str, max_n: int = 3) -> list[str]:
    """Turn the decomposer's reply into search queries.

    Tolerates the numbering and bullets small models add despite being told
    not to, and falls back to the original question when nothing usable came
    back - a decomposition failure must degrade to the baseline, never to an
    empty retrieval.

    >>> nl = chr(10)
    >>> parse_subqueries("1. What is L1 regularization?" + nl + "2) What is L2?", "q")
    ['What is L1 regularization?', 'What is L2?']
    >>> parse_subqueries("", "original question")
    ['original question']
    >>> parse_subqueries(nl.join(["- lasso", "- ridge", "- dropout", "- pruning"]), "q")
    ['lasso', 'ridge', 'dropout']
    >>> # capped at 3, and one-character fragments are rejected as junk:
    >>> parse_subqueries(nl.join(["- a", "- b"]), "fallback")
    ['fallback']
    """
    import re
    out = []
    for line in raw.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().strip('"')
        if len(line) >= 3 and not line.lower().startswith(("here are", "queries:", "sure")):
            out.append(line)
    return out[:max_n] or [question]


def make_nodes(retriever, client, gcfg: GraphConfig) -> dict:
    """Node functions closed over their dependencies.

    Closures rather than globals: a node is then a pure function of state, and a test
    can hand it a fake retriever or client without touching anything shared.
    """
    template = load_template(gcfg.prompt)

    def retrieve(state: RAGState) -> dict:
        subs = state.get("sub_queries") or []
        if len(subs) <= 1:
            # Baseline path. MUST stay byte-identical to Phase 5: parity with
            # the measured baseline is what makes every Phase 6 delta valid.
            q = (subs[0] if subs else None) or state.get("query") or state["question"]
            return {"documents": retriever.retrieve(q), "trace": ["retrieve"]}

        # One retrieval per sub-question, fused by RRF over chunk ids. RRF
        # rather than concatenation: a passage relevant to TWO sub-questions is
        # evidence of centrality and should rise, and RRF rewards exactly that
        # agreement - the same reason it won D5.
        from src.retrieve.fusion import reciprocal_rank_fusion
        by_id: dict[str, dict] = {}
        rankings = []
        for sq in subs:
            docs = retriever.retrieve(sq)
            rankings.append([d["chunk_id"] for d in docs])
            for d in docs:
                by_id.setdefault(d["chunk_id"], d)
        fused = reciprocal_rank_fusion(rankings, k=60, top_n=retriever.top_k)
        return {"documents": [by_id[cid] for cid, _ in fused],
                "trace": [f"retrieve x{len(subs)}"]}

    decompose_tpl = None
    if gcfg.nodes.get("decompose"):
        decompose_tpl = load_template("decompose_v0", required=("{question}",))

    def decompose(state: RAGState) -> dict:
        prompt = decompose_tpl.replace("{question}", state["question"])
        resp = client.chat(gcfg.transform_model, prompt, max_tokens=150,
                           temperature=0.0)
        subs = parse_subqueries(resp.text or "", state["question"])
        if gcfg.nodes.get("keep_original"):
            # Always search with the question AS ASKED, alongside the pieces.
            # Measured: small models split 49/50 and 8/8 questions regardless of
            # the prompt's "leave simple questions whole" rule, and fragments of
            # a single-topic question retrieve worse than the whole. With the
            # original's own ranking inside the RRF fusion, a junk sub-query can
            # add noise but cannot REMOVE what the original would have found.
            subs = [state["question"]] + [q for q in subs if q != state["question"]]
        return {"sub_queries": subs, "trace": [f"decompose->{len(subs)}"]}

    def generate(state: RAGState) -> dict:
        ctx = assemble(state["documents"], top_n=gcfg.top_n, token_budget=gcfg.budget)
        # The ORIGINAL question goes into the prompt even when retrieval searched a
        # rewritten `query`: the user asked this, and the answer must address it.
        prompt = build_prompt(state["question"], ctx.text, template)
        resp = client.chat(gcfg.generator, prompt, max_tokens=gcfg.max_tokens,
                           temperature=gcfg.temperature)
        answer = resp.text or ""
        mem = []
        if gcfg.nodes.get("memory"):
            # Store the STANDALONE question, not the raw turn: the next
            # condensation reads cleaner history ("what is dropout?") than a
            # chain of pronouns ("and why?").
            mem = [{"role": "user", "content": state["question"]},
                   {"role": "assistant", "content": answer}]
        return {
            "answer": answer,
            "history": mem,
            "refused": is_refusal(answer),
            "sources": [{"sid": s.sid, "book": s.book, "text": s.text,
                         "cite": s.human()} for s in ctx.sources],
            "trace": ["generate"],
        }

    def start_turn(state: RAGState) -> dict:
        # With a checkpointer, state PERSISTS across turns - so last turn's
        # sub_queries, grades and attempt would leak into this turn's retrieval.
        # Reset every per-turn field before anything else runs.
        return {"turn": state["question"], "sub_queries": [], "grades": [],
                "attempt": 0, "query": "", "trace": ["start_turn"]}

    def condense(state: RAGState) -> dict:
        hist = state.get("history") or []
        if not hist:
            return {"trace": ["condense:skip"]}          # first turn - free
        convo = chr(10).join(f"{m['role']}: {m['content'][:600]}" for m in hist[-6:])
        prompt = (CONDENSE_PROMPT.replace("{history}", convo)
                                 .replace("{question}", state["question"]))
        resp = client.chat(gcfg.condense_model or gcfg.generator, prompt,
                           max_tokens=300, temperature=0.0)
        standalone = (resp.text or "").strip().splitlines()
        standalone = standalone[0].strip() if standalone else state["question"]
        return {"question": standalone or state["question"],
                "trace": ["condense"]}

    from src.graph.crag import (BATCH_GRADE_PROMPT, GRADE_TOP, REWRITE_PROMPT,
                                SNIPPET_CHARS, parse_batch_grades)

    grader = gcfg.grader_model or gcfg.generator

    def grade(state: RAGState) -> dict:
        # ONE batched call for all top passages (see crag.BATCH_GRADE_PROMPT for
        # why this is not five local calls). str.replace, not str.format: chunk
        # text is mathematics full of braces.
        docs = state["documents"][:GRADE_TOP]
        block = (chr(10) + chr(10)).join(
            f"Passage {i}: {d['text'][:SNIPPET_CHARS]}" for i, d in enumerate(docs, 1))
        prompt = (BATCH_GRADE_PROMPT.replace("{question}", state["question"])
                                    .replace("{passages}", block))
        resp = client.chat(grader, prompt, max_tokens=400, temperature=0.0)
        out = parse_batch_grades(resp.text or "", len(docs))
        return {"grades": out, "trace": [f"grade {sum(out)}/{len(out)}"]}

    def rewrite(state: RAGState) -> dict:
        prompt = REWRITE_PROMPT.replace("{question}", state["question"])
        resp = client.chat(grader, prompt, max_tokens=300,
                           temperature=0.0)
        new_q = (resp.text or "").strip().splitlines()[0].strip() if resp.text else ""
        # FUSE the rewrite with the original - never replace it.
        subs = [state["question"]] + ([new_q] if new_q and new_q != state["question"] else [])
        return {"sub_queries": subs, "attempt": 1, "trace": ["rewrite"]}

    return {"retrieve": retrieve, "generate": generate, "decompose": decompose,
            "grade": grade, "rewrite": rewrite,
            "start_turn": start_turn, "condense": condense}


def build_graph(retriever, client, gcfg: GraphConfig, checkpointer=None):
    """Assemble and compile. With no nodes toggled on: retrieve -> generate."""
    from langgraph.graph import END, START, StateGraph

    n = make_nodes(retriever, client, gcfg)
    g = StateGraph(RAGState)
    g.add_node("retrieve", n["retrieve"])
    g.add_node("generate", n["generate"])
    if gcfg.nodes.get("crag"):
        from src.graph.crag import crag_route
        g.add_node("grade", n["grade"])
        g.add_node("rewrite", n["rewrite"])
        g.add_edge("retrieve", "grade")
        g.add_conditional_edges(
            "grade",
            lambda st: crag_route(st.get("grades", []), st.get("attempt", 0)),
            {"generate": "generate", "rewrite": "rewrite"},
        )
        g.add_edge("rewrite", "retrieve")
    else:
        g.add_edge("retrieve", "generate")
    entry = START
    if gcfg.nodes.get("memory"):
        g.add_node("start_turn", n["start_turn"])
        g.add_node("condense", n["condense"])
        g.add_edge(START, "start_turn")
        g.add_edge("start_turn", "condense")
        entry = "condense"
    if gcfg.nodes.get("decompose"):
        g.add_node("decompose", n["decompose"])
        g.add_edge(entry, "decompose")
        g.add_edge("decompose", "retrieve")
    else:
        g.add_edge(entry, "retrieve")
    g.add_edge("generate", END)
    return g.compile(checkpointer=checkpointer)


def citation_report(state: RAGState):
    """Judge-free citation check on a finished state."""
    return verify(state.get("answer", ""), [s["sid"] for s in state.get("sources", [])])

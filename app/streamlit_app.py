"""Streamlit chat UI — Phase 9.

    streamlit run app/streamlit_app.py

A thin client over the FastAPI /chat endpoint. Deliberately holds no RAG logic of its
own: every answer comes from the same served graph the evaluation measures, so what a
user sees is what the reports describe.

Two panels exist so the system is inspectable rather than a black box:
  - "What was searched for" shows the standalone question after condensation - the
    place a multi-turn conversation silently goes wrong if memory fails.
  - "Sources" shows exactly the passages the generator read, with their citations.
"""

from __future__ import annotations

import os
import uuid

import httpx
import streamlit as st

API = os.environ.get("RAG_API", "http://127.0.0.1:8000")

st.set_page_config(page_title="ML Textbook Chatbot", page_icon="📚", layout="wide")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.turns = []


def call(path: str, payload: dict | None = None, timeout: float = 240.0):
    try:
        if payload is None:
            r = httpx.get(f"{API}{path}", timeout=10)
        else:
            r = httpx.post(f"{API}{path}", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json(), None
    except httpx.HTTPError as e:
        return None, str(e)


# ── sidebar: what is this system running? ─────────────────────────────────
with st.sidebar:
    st.header("📚 ML Textbook RAG")
    health, err = call("/health")
    if err:
        st.error(f"API unreachable at {API}.\n\nStart it with:\n\n"
                 "`uvicorn src.serve.api:app --port 8000`")
    else:
        st.caption(f"API: {health['status']} · Ollama: {health['ollama']}")
        cfg, _ = call("/config")
        if cfg:
            st.subheader("Earned configuration")
            st.markdown(
                f"- **Generator**: `{cfg['generator']}`\n"
                f"- **Embedder**: `{cfg['embedder']}`\n"
                f"- **Retrieval**: `{cfg['retrieval']}`\n"
                f"- **Chunker**: `{cfg['chunker']}`\n"
                f"- **Context**: top {cfg['top_n']} · {cfg['budget_tokens']} tokens\n"
                f"- **Graph nodes**: {', '.join(cfg['nodes']) or 'none'}"
            )
            st.caption("Every value above was chosen by a benchmark — see `reports/`.")
    if st.button("🗑️ New conversation", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.turns = []
        st.rerun()
    st.caption(f"session `{st.session_state.session_id[:8]}`")

st.title("Ask the ML textbooks")
st.caption("Géron · Goodfellow · Bishop · Jurafsky · Huyen · Raschka — "
           "every answer is cited to book, section and page.")


def render(turn: dict) -> None:
    with st.chat_message("user"):
        st.write(turn["q"])
    with st.chat_message("assistant"):
        r = turn["r"]
        if r.get("blocked"):
            st.warning(r["answer"])
            return
        st.markdown(r["answer"])
        if r.get("fabricated"):
            st.error(f"⚠️ Cited source(s) that do not exist: {', '.join(r['fabricated'])}")
        meta = f"⏱ {r['latency_ms'] / 1000:.1f}s · path: {' → '.join(r['trace'])}"
        if r["standalone_question"].strip() != turn["q"].strip():
            with st.expander("🔎 What was searched for"):
                st.write(r["standalone_question"])
                st.caption("Your follow-up was rewritten into a standalone question "
                           "using the conversation so far.")
        if r["sources"]:
            with st.expander(f"📖 Sources ({len(r['sources'])} passages read)"):
                for s in r["sources"]:
                    cited = "✅ cited" if s["sid"].upper() in r["cited"] else "not cited"
                    st.markdown(f"**[{s['sid']}] {s['cite']}** — _{cited}_")
                    st.caption(s["text"][:600] + ("…" if len(s["text"]) > 600 else ""))
        st.caption(meta)


for turn in st.session_state.turns:
    render(turn)

if q := st.chat_input("Ask about machine learning, deep learning or NLP…"):
    with st.spinner("Retrieving and reading the textbooks…"):
        res, err = call("/chat", {"session_id": st.session_state.session_id, "message": q})
    if err:
        st.error(f"Request failed: {err}")
    else:
        st.session_state.turns.append({"q": q, "r": res})
        st.rerun()

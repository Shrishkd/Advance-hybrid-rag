"""Ollama client — local and cloud, with a persistent response cache.

TWO NON-OBVIOUS REQUIREMENTS THIS FILE EXISTS TO MEET
=====================================================

1. CACHING IS MANDATORY, NOT AN OPTIMISATION.
   The free Ollama Cloud tier is rate-limited. A single Phase 8 evaluation run
   issues hundreds of judge calls, and we will re-run it after every config
   change. Without a cache, re-running an unchanged experiment burns quota
   re-answering prompts we already have answers to, and eventually just fails.

   The cache also buys REPRODUCIBILITY. Even at temperature 0, a model may be
   re-quantised or updated server-side between runs. A cached response means a
   regression in results.db reflects OUR change, not a silent model swap.

   Key = sha256(model + prompt + params). Any of those changing is a genuinely
   different call.

2. gpt-oss EMITS ITS REASONING, AND IT MUST BE STRIPPED.
   Verified on this account:

       $ ollama run gpt-oss:20b-cloud "Say OK"
       Thinking... The user requests: "Say OK". This is a simple request...
       ...done thinking.  OK

   Feed that to a faithfulness metric and the judge scores the model's
   scratchpad instead of its answer. Every Phase 8 number would be measuring
   the wrong string. Newer Ollama returns reasoning in a separate `thinking`
   field, but not always, so we handle both and strip defensively.

MODEL NAMING: cloud models REQUIRE a ':cloud' suffix. Plain 'glm-5.3' resolves
to a nonexistent LOCAL tag and fails with a confusing "not found". Roles in
configs/models.yaml carry the correct tags; resolve through there, never
hardcode.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import warnings
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CACHE_PATH = Path("data/processed/llm_cache.sqlite")

# gpt-oss wraps reasoning in these markers when it is inlined into content.
_THINK = re.compile(
    r"Thinking\.\.\..*?(?:\.\.\.done thinking\.|<\|end\|>)", re.S | re.I
)
_THINK_TAG = re.compile(r"<think>.*?</think>", re.S | re.I)


def strip_reasoning(text: str) -> str:
    """Remove inline reasoning traces.

    >>> strip_reasoning("Thinking... blah blah ...done thinking.  OK")
    'OK'
    >>> strip_reasoning("<think>hmm</think>Answer")
    'Answer'
    >>> strip_reasoning("plain answer")
    'plain answer'
    """
    text = _THINK.sub("", text)
    text = _THINK_TAG.sub("", text)
    return text.strip()


@dataclass
class LLMResponse:
    text: str
    model: str
    cached: bool
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning: str = ""      # kept for inspection, never scored


class ResponseCache:
    """SQLite-backed prompt -> response cache.

    SQLite rather than files: one artefact, queryable ("how many judge calls
    did that run make?"), and safe under concurrent reads.
    """

    def __init__(self, path: Path = CACHE_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS cache (
                   key TEXT PRIMARY KEY,
                   model TEXT NOT NULL,
                   prompt TEXT NOT NULL,
                   response TEXT NOT NULL,
                   reasoning TEXT DEFAULT '',
                   created_at REAL NOT NULL
               )"""
        )
        self.conn.commit()

    @staticmethod
    def key(model: str, prompt: str, params: dict[str, Any]) -> str:
        blob = json.dumps(
            {"m": model, "p": prompt, "o": params}, sort_keys=True, ensure_ascii=False
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, key: str) -> tuple[str, str] | None:
        row = self.conn.execute(
            "SELECT response, reasoning FROM cache WHERE key = ?", (key,)
        ).fetchone()
        return (row[0], row[1]) if row else None

    def put(self, key: str, model: str, prompt: str, response: str, reasoning: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO cache VALUES (?,?,?,?,?,?)",
            (key, model, prompt, response, reasoning, time.time()),
        )
        self.conn.commit()

    def stats(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT model, COUNT(*) FROM cache GROUP BY model"
        ).fetchall()
        return dict(rows)


class OllamaClient:
    """Chat wrapper over Ollama, local or cloud, with caching and retries."""

    def __init__(
        self,
        cache: ResponseCache | None = None,
        use_cache: bool = True,
        max_retries: int = 4,
    ) -> None:
        self.cache = cache or ResponseCache()
        self.use_cache = use_cache
        self.max_retries = max_retries

    def chat(
        self,
        model: str,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        think: bool = False,
    ) -> LLMResponse:
        """One-shot chat completion.

        Args:
            model: full tag including ':cloud' for cloud models.
            think: whether to let a reasoning model reason. Default False —
                for graders and routers, reasoning is wasted latency on a
                yes/no answer.
        """
        import ollama

        params = {"temperature": temperature, "num_predict": max_tokens,
                  "system": system, "think": think}
        key = ResponseCache.key(model, prompt, params)

        if self.use_cache and (hit := self.cache.get(key)):
            return LLMResponse(
                text=hit[0], model=model, cached=True, latency_ms=0.0, reasoning=hit[1]
            )

        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]

        t0 = time.perf_counter()
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                r = ollama.chat(
                    model=model,
                    messages=messages,
                    think=think,
                    options={"temperature": temperature, "num_predict": max_tokens},
                )
                break

            # ONLY transient failures are retryable. An earlier version caught
            # bare Exception and retried everything, which meant a TypeError
            # from a wrong keyword argument was retried 4 times with backoff -
            # 15 seconds to produce the identical error. Retrying a bug does
            # not fix the bug, it just hides it behind a delay.
            except (TypeError, AttributeError, KeyError):
                raise                                    # programming errors: fail now

            except Exception as e:                       # noqa: BLE001
                last = e
                msg = str(e)
                # 402 means the model is not on our plan. No amount of retrying
                # changes billing - fail immediately with an actionable message.
                if "402" in msg or "Payment Required" in msg:
                    raise RuntimeError(
                        f"{model} is not in the free tier on this account. "
                        "Verified free: gpt-oss:20b-cloud, gpt-oss:120b-cloud."
                    ) from e
                if "not found" in msg.lower():
                    raise RuntimeError(
                        f"model {model!r} not found. Cloud models REQUIRE a "
                        "':cloud' suffix - 'glm-5.3' resolves to a local tag."
                    ) from e
                # Rate limits and network blips ARE worth retrying, with
                # exponential backoff so a burst does not become a storm.
                time.sleep(2 ** attempt)
        else:
            raise RuntimeError(
                f"{model} failed after {self.max_retries} attempts: {last}"
            ) from last

        latency = (time.perf_counter() - t0) * 1000
        msg_obj = r.get("message", {})
        raw = msg_obj.get("content", "") or ""
        reasoning = msg_obj.get("thinking", "") or ""

        # Belt and braces: newer Ollama separates reasoning into `thinking`,
        # older builds inline it. Strip regardless of which we got.
        text = strip_reasoning(raw)
        if not reasoning and raw != text:
            reasoning = raw[: len(raw) - len(text)]

        # A reasoning model can spend the ENTIRE token budget thinking, leaving
        # nothing after the trace is stripped. Observed live: max_tokens=20 on
        # gpt-oss:120b-cloud returned 52 completion tokens, 201 chars of
        # reasoning, and an EMPTY answer. Silent, and downstream it looks like
        # the model had nothing to say rather than like a budget error.
        if not text and reasoning:
            warnings.warn(
                f"{model}: empty answer after stripping {len(reasoning)} chars of "
                f"reasoning (max_tokens={max_tokens}). The budget was consumed by "
                "the reasoning trace - raise max_tokens.",
                stacklevel=2,
            )

        if self.use_cache:
            self.cache.put(key, model, prompt, text, reasoning)

        return LLMResponse(
            text=text,
            model=model,
            cached=False,
            latency_ms=latency,
            prompt_tokens=r.get("prompt_eval_count", 0) or 0,
            completion_tokens=r.get("eval_count", 0) or 0,
            reasoning=reasoning,
        )

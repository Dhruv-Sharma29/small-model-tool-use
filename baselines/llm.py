"""LLM policy: one OpenAI-compatible chat call per step, tools offered natively (llama.cpp --jinja).

Identical for every condition (PLAN.md M3); only the server (model) and the system prompt differ.
temperature 0, seed 0, one tool call per step (parallel_tool_calls off; if a reply still carries
several, the first is used and the rest are counted as dropped), and a fixed output budget.
Transport errors are retried a few times without costing the policy a retry; if they persist the
step is a text reply (counted as invalid by the runner) and the error is recorded.
"""

from __future__ import annotations

import time

import httpx

from tasks.tools import openai_tools

MAX_TOKENS = 512
TRANSPORT_RETRIES = 3


class LLMPolicy:
    def __init__(self, base_url: str, model: str, client: httpx.Client | None = None, max_tokens: int = MAX_TOKENS):
        self.model, self.max_tokens = model, max_tokens
        self.client = client or httpx.Client(base_url=base_url, timeout=600)
        self.tools = openai_tools()

    def act(self, messages: list[dict]) -> dict:
        body = {"model": self.model, "messages": messages, "tools": self.tools, "tool_choice": "auto",
                "parallel_tool_calls": False, "temperature": 0, "seed": 0, "max_tokens": self.max_tokens}
        t0, last_err = time.perf_counter(), None
        for attempt in range(TRANSPORT_RETRIES):
            try:
                r = self.client.post("/v1/chat/completions", json=body)
                r.raise_for_status()
                resp = r.json()
                break
            except (httpx.HTTPError, ValueError) as e:
                last_err = f"{type(e).__name__}: {getattr(getattr(e, 'response', None), 'text', '') or e}"[:300]
                time.sleep(2 * (attempt + 1))
        else:
            return {"text": "", "transport_error": last_err, "latency_s": round(time.perf_counter() - t0, 2)}
        choice = resp["choices"][0]
        msg = choice["message"]
        meta = {"latency_s": round(time.perf_counter() - t0, 2), "usage": resp.get("usage") or {},
                "finish_reason": choice.get("finish_reason")}
        calls = msg.get("tool_calls") or []
        if not calls:
            return {"text": msg.get("content") or "", **meta}
        fn = calls[0]["function"]
        return {"name": fn["name"], "arguments": fn.get("arguments") or "{}", "dropped_calls": len(calls) - 1,
                "content": msg.get("content") or None, **meta}

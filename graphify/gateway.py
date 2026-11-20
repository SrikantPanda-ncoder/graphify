"""LLM calls for graphify through the nCoder gateway (LiteLLM), or simulated replies for local runs.

Selected with GRAPHIFY_BACKEND=gateway. ONE switch decides the mode:
  LLM_SIMULATION=true   no network; canned replies in the shape the description steps expect
  LLM_SIMULATION unset  real call to LITELLM_BASE_URL/chat/completions with LITELLM_API_KEY

Every real call is attributed to GRAPHIFY_CUSTOMER_ID (the gateway refuses unattributed calls).
GRAPHIFY_MODEL names the model as the proxy knows it.
"""

from __future__ import annotations

import json
import os
import re

SIMULATED_SYMBOL = "Simulated description of this symbol."  # kept for callers that want a fixed text
SIMULATED_GROUP = "Simulated description of this group."
SIMULATED_NAME = "Simulated Group"


def simulation_on() -> bool:
    return os.environ.get("LLM_SIMULATION", "").strip().lower() in ("1", "true", "yes", "on")


def simulated_reply(prompt: str) -> str:
    """A JSON object shaped like the reply each graphify prompt asks for: descriptions (per node or group), or community
    names (one name per "<community id>: members" line)."""
    if "naming clusters in a knowledge graph" in prompt:
        ids = re.findall(r"^(-?\d+): ", prompt, re.M)
        return json.dumps({i: f"{SIMULATED_NAME} {i}" for i in ids})
    reply = {}
    # one block per node: "### id: <id>\nname: <label>\nfile: <path>\n"
    for node_id, name, path, why_flag in re.findall(r"^### id: (.+)\nname: (.*)\nfile: (.*?)(\nwhy: yes)?$", prompt, re.M):
        what = f"Simulated description of {name.strip() or node_id.strip()} in {path.strip() or 'no file'}."
        if why_flag:  # a code symbol: what and why in one reply
            reply[node_id.strip()] = {"what": what, "why": f"Simulated reason for {name.strip() or node_id.strip()}."}
        else:
            reply[node_id.strip()] = what
    # one block per group: "### group: <cid>\nname: <name>\nsize: <n>\n..." -> a description that names it
    for group, name, size in re.findall(r"^### group: (.+)\nname: (.*)\nsize: (.*)$", prompt, re.M):
        reply[group.strip()] = f"Simulated description of {name.strip() or group.strip()} ({size.strip() or '?'} symbols)."
    for group in re.findall(r"^### group: (.+)$", prompt, re.M):
        reply.setdefault(group.strip(), SIMULATED_GROUP)
    return json.dumps(reply)


def call_gateway(prompt: str, *, max_tokens: int = 200, usage_out: dict | None = None) -> str:
    """One chat completion through the gateway, or the simulated reply. Returns the reply text."""
    if simulation_on():
        return simulated_reply(prompt)
    import requests

    base = os.environ.get("LITELLM_BASE_URL", "").strip().rstrip("/")
    key = os.environ.get("LITELLM_API_KEY", "").strip()
    customer = os.environ.get("GRAPHIFY_CUSTOMER_ID", "").strip()
    if not (base and key):
        raise RuntimeError("LITELLM_BASE_URL and LITELLM_API_KEY are required unless LLM_SIMULATION=true")
    if not customer:
        raise RuntimeError("GRAPHIFY_CUSTOMER_ID is required: every gateway call is attributed to a customer")
    model = os.environ.get("GRAPHIFY_MODEL", "gemini-2.5-flash").strip()
    resp = requests.post(
        f"{base}/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": model, "messages": [{"role": "user", "content": prompt}],
              "temperature": 0, "max_tokens": max_tokens, "user": customer},
        timeout=120,
    )
    resp.raise_for_status()
    body = resp.json()
    if usage_out is not None:
        usage = body.get("usage") or {}
        usage_out["input"] = usage_out.get("input", 0) + int(usage.get("prompt_tokens", 0) or 0)
        usage_out["output"] = usage_out.get("output", 0) + int(usage.get("completion_tokens", 0) or 0)
    return body["choices"][0]["message"]["content"] or ""

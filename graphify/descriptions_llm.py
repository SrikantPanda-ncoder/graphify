"""LLM-generated node and community descriptions for graph.json (GRAPHIFY_DESCRIPTIONS=llm).

Node descriptions: each node's source range (or a file's first lines) is sent with its name. Community
descriptions: each community's name, size and member names, plus the node descriptions made in this run.
Replies are JSON maps from id to one sentence.

Requests run in parallel (GRAPHIFY_DESCRIPTION_WORKERS, default 8); nodes per request is GRAPHIFY_DESCRIPTION_BATCH (default 15). Progress is printed to stderr after each
batch. The cache, <graphify-out>/descriptions_cache.json, is written after each batch, so an interrupted run
keeps its finished work, and a rebuild only re-asks for nodes whose code changed.

Dry run (GRAPHIFY_DESCRIPTIONS_DRY_RUN=1): counts requests and estimated tokens, sends nothing.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

VERSION = "v3"  # v3: the cache holds {"what", "why"} objects
NODE_BATCH = max(1, int(os.environ.get("GRAPHIFY_DESCRIPTION_BATCH", "15") or 15))  # nodes per request
COMM_BATCH = 8
SNIP_LINES = 60
SNIP_CHARS = 1800
MAX_TOKENS = 16384  # reasoning models spend output tokens before the answer
TEXT_SUFFIXES = {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".json", ".md", ".txt", ".yml", ".yaml",
                 ".css", ".scss", ".html", ".py", ".go", ".rs", ".java", ".kt", ".swift", ".rb", ".php",
                 ".sql", ".graphql", ".gql", ".toml", ".cfg", ".ini", ".sh", ".vue", ".svelte", ""}


class ReplyParseError(ValueError):
    """The model replied, but not with the JSON object requested. Only this is retried or skipped."""


def _line(loc) -> int | None:
    m = re.match(r"^L(\d+)$", str(loc or ""))
    return int(m.group(1)) if m else None


def _snippet(root: Path, node: dict) -> str:
    src = node.get("source_file") or ""
    path = root / src
    if not src or path.suffix.lower() not in TEXT_SUFFIXES or not path.is_file():
        return ""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (UnicodeDecodeError, OSError):
        return ""
    start = node.get("start_line") or _line(node.get("source_location"))
    if not start:
        return ""
    end = node.get("end_line") or (start + SNIP_LINES - 1 if node.get("source_location") == "L1" else start)
    chunk = lines[max(0, start - 1): min(len(lines), end)][:SNIP_LINES]
    return "\n".join(chunk)[:SNIP_CHARS]


def cache_mode(backend: str) -> str:
    """What produced a description. It is part of every cache key, so a cache written in one mode is never read
    in another: 'simulated' (LLM_SIMULATION), 'offline' (templates), or 'gateway:<model>' (real calls)."""
    from graphify.gateway import simulation_on

    name = backend.strip().lower()
    if name in OFFLINE_BACKENDS:
        return "offline"
    if name == "gateway" and simulation_on():
        return "simulated"
    if name == "gateway":
        return "gateway:" + (os.environ.get("GRAPHIFY_MODEL", "").strip() or "gemini-2.5-flash")
    return name


def _key(node: dict, code: str, mode: str) -> str:
    return hashlib.sha256(f"{VERSION}|{mode}|{node['id']}|{code}".encode()).hexdigest()


def _parse(text: str) -> dict:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        raise ReplyParseError("no JSON object in reply")
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError as exc:
        raise ReplyParseError(f"reply is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ReplyParseError("reply is not a JSON object")
    return data


RETRIES = 6  # transient provider errors (rate limits, timeouts, 5xx): wait and retry


def _ask(prompt: str, backend: str, usage: dict) -> dict:
    import time
    from graphify.gateway import GatewayPermanentError
    from graphify.llm import _call_llm
    last = None
    for attempt in range(RETRIES):
        try:
            text = _call_llm(prompt, backend=backend, max_tokens=MAX_TOKENS, usage_out=usage)
            return _parse(text)
        except ReplyParseError:
            raise
        except (ImportError, GatewayPermanentError):
            # A missing module or a setting that is wrong or absent: waiting will not change it. Fail now.
            raise
        except Exception as exc:  # provider errors; a parse failure is handled by the caller
            last = exc
            wait = min(60, 2 ** attempt * 2)
            print(f"[descriptions] provider error ({type(exc).__name__}), retry {attempt + 1}/{RETRIES} in {wait}s",
                  file=sys.stderr, flush=True)
            time.sleep(wait)
    raise last


def _ask_with_split(items: list, build_prompt, backend: str, usage: dict, depth: int = 0) -> dict:
    """Ask for a batch; on a reply-parse failure split it in half and retry (max two levels)."""
    try:
        return _ask(build_prompt(items), backend, usage)
    except ReplyParseError:
        if len(items) <= 1 or depth >= 2:
            return {}
        mid = len(items) // 2
        out = _ask_with_split(items[:mid], build_prompt, backend, usage, depth + 1)
        out.update(_ask_with_split(items[mid:], build_prompt, backend, usage, depth + 1))
        return out


def _wants_why(node: dict) -> bool:
    """A WHY is asked for code symbols (functions, classes, components, hooks), not for files, packages or concepts."""
    return node.get("file_type") == "code" and ("_callable" in node or "_callable_class" in node)


def _node_prompt(batch: list) -> str:
    parts = [
        "You describe pieces of source code or documents in a knowledge graph. For each item, write ONE plain "
        "sentence saying what it does or what it says. For an item marked 'why: yes', reply with an object "
        "{\"what\": <that sentence>, \"why\": <one sentence: why it exists or why it is needed>}; for every other "
        "item reply with the sentence as a string. Use only the text shown; if a snippet is empty, say what the name "
        "suggests and nothing more. Reply ONLY with a JSON object that maps each item id to its value. No markdown.",
        "",
    ]
    for it in batch:
        ask_why = "\nwhy: yes" if it.get("why") else ""
        parts.append(f"### id: {it['id']}\nname: {it['label']}\nfile: {it['file']}{ask_why}\n```\n{it['code']}\n```")
    return "\n".join(parts)


def _comm_prompt(batch: list) -> str:
    parts = [
        "You describe groups of related code in a knowledge graph. For each group, write ONE plain sentence "
        "saying what the group is for, based on its members. Reply ONLY with a JSON object that maps each "
        "group id (as a string) to its sentence. No markdown.",
        "",
    ]
    for g in batch:
        parts.append(f"### group: {g['cid']}\nname: {g['name']}\nsize: {g['size']}\nmembers: {', '.join(g['members'])}")
    return "\n".join(parts)


def _log(msg: str) -> None:
    print(f"[descriptions] {msg}", file=sys.stderr, flush=True)


def _env_key(backend: str) -> str:
    return {"openrouter": "OPENROUTER_API_KEY", "claude": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY",
            "gemini": "GEMINI_API_KEY", "deepseek": "DEEPSEEK_API_KEY", "kimi": "MOONSHOT_API_KEY"}.get(
        backend, f"{backend.upper()}_API_KEY")


OFFLINE_BACKENDS = ("", "none", "off", "disable", "disabled")


def offline_side(nodes: list, root) -> dict:
    """Offline descriptions, no model: a node's description is the first line of its code (its label when it has
    no code); a community's is a template over its name, size and first members. For tests and offline runs only."""
    node_text = {}
    for n in nodes:
        code = _snippet(Path(root), n)
        first = next((ln.strip() for ln in code.splitlines() if ln.strip()), "")
        node_text[n["id"]] = first[:200] or f"{n.get('label', '')}."
    members: dict = defaultdict(list)
    for n in nodes:
        if n.get("community") is not None:
            members[str(n["community"])].append(n)
    comm_text = {}
    for cid, ms in members.items():
        name = next((m.get("community_name") for m in ms if m.get("community_name")), f"Community {cid}")
        labels = ", ".join(m.get("label", "") for m in ms[:5])
        comm_text[cid] = {"description": f"{name}: {len(ms)} symbols, including {labels}.",
                          "members": sorted(m["id"] for m in ms)}
    return {"communities": comm_text, "nodes": node_text}


def offline_allowed() -> bool:
    """Offline descriptions are for tests and local runs; a production run must have a backend."""
    return os.environ.get("APP_ENV", "").strip().lower() not in ("prod", "production")


class DescriptionError(RuntimeError):
    """The LLM could not describe every node and community, or is not configured. Raised, never returned as a gap."""


def generate(graph_json: str, root=None) -> tuple[dict, dict]:
    """Return (dict for descriptions.apply, report). Every node and every community gets a description, or this raises.
    root: the repo folder that the node source_file paths are relative to."""
    graph_path = Path(graph_json).resolve()
    root = Path(root).resolve() if root is not None else graph_path.parent.parent
    cache_path = graph_path.parent / "descriptions_cache.json"
    backend = os.environ.get("GRAPHIFY_BACKEND", "").strip()
    workers = max(1, int(os.environ.get("GRAPHIFY_DESCRIPTION_WORKERS", "8") or 8))
    report: dict = {}

    graph = json.loads(graph_path.read_text())
    nodes = graph.get("nodes", [])
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    pending, node_text, why_text = [], {}, {}
    mode = cache_mode(backend)
    for n in nodes:  # every node is described, trivial and test nodes included
        code = _snippet(root, n)
        k = _key(n, code, mode)
        wants_why = _wants_why(n)
        k = k + "|why" if wants_why else k
        if k in cache:
            node_text[n["id"]] = cache[k]["what"]
            if cache[k].get("why"):
                why_text[n["id"]] = cache[k]["why"]
        else:
            pending.append({"id": n["id"], "label": n.get("label", ""), "file": n.get("source_file") or "",
                            "code": code, "key": k, "why": wants_why})
    report.update(nodes_total=len(nodes), nodes_cached=len(node_text), nodes_to_ask=len(pending))

    if backend.lower() in OFFLINE_BACKENDS:
        if not offline_allowed():
            raise DescriptionError("GRAPHIFY_BACKEND is not set: offline descriptions are not allowed in prod")
        report.update(mode="offline_template", nodes_total=len(nodes), nodes_described=len(nodes))
        _log("GRAPHIFY_BACKEND not set: offline template descriptions (no model call)")
        return offline_side(nodes, root), report
    if backend != "gateway" and not os.environ.get(_env_key(backend), ""):
        raise DescriptionError(f"{_env_key(backend)} is not set: the LLM is always on for graphify descriptions")

    lock = threading.Lock()
    usage_total: dict = {}

    def run(batch, build, label_fn):
        usage: dict = {}
        got = _ask_with_split(batch, build, backend, usage)
        with lock:
            for k, v in usage.items():
                usage_total[k] = usage_total.get(k, 0) + v
        return got

    batches = [pending[i:i + NODE_BATCH] for i in range(0, len(pending), NODE_BATCH)]
    failed_nodes: list = []
    done = 0
    _log(f"{len(pending)} nodes to ask in {len(batches)} requests, {workers} at a time")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run, b, _node_prompt, None): b for b in batches}
        for fut in futures:
            batch = futures[fut]
            got = fut.result()
            with lock:
                for it in batch:
                    reply = got.get(it["id"])
                    if isinstance(reply, dict):
                        what, why = str(reply.get("what") or "").strip(), str(reply.get("why") or "").strip()
                    elif isinstance(reply, str):
                        what, why = reply.strip(), ""
                    else:
                        what, why = "", ""
                    if what:
                        node_text[it["id"]] = what
                        entry = {"what": what, "why": why or None}
                        cache[it["key"]] = entry
                        if why:
                            why_text[it["id"]] = why
                    else:
                        failed_nodes.append(it["id"])
                cache_path.write_text(json.dumps(cache, indent=1))
                done += len(batch)
            _log(f"nodes {done}/{len(pending)} done")

    members: dict = defaultdict(list)
    for n in nodes:
        members[str(n.get("community"))].append(n)
    comm_items = []
    for cid, ms in sorted(members.items(), key=lambda kv: kv[0]):
        if cid == "None":
            continue
        hub = next((m.get("community_name") for m in ms if m.get("community_name")), cid)
        comm_items.append({"cid": cid, "name": hub, "size": len(ms),
                           "members": [f"{m.get('label', '')}: {node_text.get(m['id'], '')}".rstrip(': ')
                                       for m in ms][:30],
                           "ids": sorted(m["id"] for m in ms)})
    comm_text: dict = {}
    failed_comms: list = []
    cbatches = [comm_items[i:i + COMM_BATCH] for i in range(0, len(comm_items), COMM_BATCH)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run, b, _comm_prompt, None): b for b in cbatches}
        for fut in futures:
            batch = futures[fut]
            got = fut.result()
            for it in batch:
                text = got.get(it["cid"])
                if isinstance(text, str) and text.strip():
                    comm_text[it["cid"]] = {"description": text.strip(), "members": it["ids"]}
                else:
                    failed_comms.append(it["cid"])
    _log(f"communities described: {len(comm_text)}/{len(comm_items)}")

    if failed_nodes or failed_comms:
        raise DescriptionError(
            f"the LLM gave no description for {len(failed_nodes)} node(s) {failed_nodes[:5]} and "
            f"{len(failed_comms)} community(ies) {failed_comms[:5]}; the step fails rather than leave gaps")
    report.update(nodes_described=len(node_text), communities_described=len(comm_text), usage=usage_total)
    side = {"communities": comm_text, "nodes": node_text, "why": why_text}
    return side, report

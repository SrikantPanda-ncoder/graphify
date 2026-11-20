"""Builds graphify's graph.json for one checkout, with no LLM. Run as a script, in the graphify interpreter.

    python graphify_cli.py <checkout_dir> <out_dir> [resolution] [ignore_file] [--describe]

With --describe, graphify's description steps run after the graph is built, through the gateway backend
(graphify/gateway.py in this repository). LLM_SIMULATION=true makes them return simulated text, with no network.

The ignore file is ncoder/graphignore/common.graphifyignore, found next to this script. Run from the repository root
with this checkout on PYTHONPATH, or with the graphifyy package installed.

Standalone on purpose: it imports only graphify and the standard library, never the app package, so the graphify
interpreter does not need the service's environment. Mirrors react_parser/ncoder_react_parser/build_graph.py build()
for the no-LLM path. Exit 0 and the graph path on stdout on success.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

COPY_SKIP = ("node_modules", ".git")
HERE = Path(__file__).resolve().parent
DEFAULT_IGNORE = HERE / "graphignore" / "common.graphifyignore"


def _graphify_names(graph, communities, placeholders: dict) -> dict:
    """graphify's own community names, through the gateway backend. Falls back to the placeholders on any failure,
    and says so on stderr."""
    from graphify.llm import generate_community_labels

    backend = os.environ.get("GRAPHIFY_BACKEND", "").strip()
    if backend.lower() in ("", "none", "off", "disable", "disabled"):
        # offline: graphify's deterministic names (the community's hub node), no model call
        from graphify.cluster import label_communities_by_hub
        from graphify.descriptions_llm import offline_allowed

        if not offline_allowed():
            raise RuntimeError("GRAPHIFY_BACKEND is not set: offline community names are not allowed in prod")
        return {**placeholders, **label_communities_by_hub(graph, communities)}
    try:
        labels, source = generate_community_labels(graph, communities, backend=backend, quiet=True)
    except Exception as exc:  # noqa: BLE001 - names are optional; the placeholders still work
        print(f"community names failed ({exc}); keeping placeholders", file=sys.stderr)
        return placeholders
    named = {cid: labels.get(cid) or placeholders[cid] for cid in placeholders}
    print(f"community names: source={source}, named={sum(named[c] != placeholders[c] for c in named)}", file=sys.stderr)
    return named


def build(checkout: Path, out_dir: Path, resolution: float, ignore_file: Path, describe_llm: bool = False) -> Path:
    from graphify.build import build_merge
    from graphify.cluster import cluster
    from graphify.detect import detect
    from graphify.export import to_json
    from graphify.extract import extract

    out_dir.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="react-graphify-"))
    src = work / "src"
    try:
        shutil.copytree(checkout, src, ignore=shutil.ignore_patterns(*COPY_SKIP), symlinks=True)
        shutil.copy(ignore_file, src / ".graphifyignore")

        detection = detect(src)
        code_files = [Path(p) for p in detection["files"].get("code", [])]
        extraction = extract(code_files, root=src)
        graph = build_merge([extraction], graph_path=work / "no-previous-graph.json", dedup=True, root=src)
        communities = cluster(graph, resolution=resolution)
        names = {cid: f"Community {cid}" for cid in communities}
        if describe_llm:
            names = _graphify_names(graph, communities, names)

        graph_path = out_dir / "graph.json"
        to_json(graph, communities, str(graph_path), force=True, community_labels=names)
        return graph_path
    finally:
        shutil.rmtree(work, ignore_errors=True)


def describe(graph_path: Path, checkout: Path) -> dict:
    """Run graphify's description steps through the gateway backend and write them into graph.json."""
    import json

    from graphify import descriptions, descriptions_llm

    side, report = descriptions_llm.generate(str(graph_path), root=checkout)
    descriptions.apply(str(graph_path), side)
    return report


def main(argv: list[str]) -> int:
    import json

    do_describe = "--describe" in argv
    argv = [a for a in argv if a != "--describe"]
    if len(argv) < 3:
        print("usage: graphify_cli.py <checkout_dir> <out_dir> [resolution] [ignore_file] [--describe]", file=sys.stderr)
        return 2
    checkout, out_dir = Path(argv[1]).resolve(), Path(argv[2]).resolve()
    resolution = float(argv[3]) if len(argv) > 3 else 1.0
    ignore_file = Path(argv[4]) if len(argv) > 4 else DEFAULT_IGNORE
    if not checkout.is_dir():
        print(f"checkout not found: {checkout}", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)
    # graphify's community naming and its caches resolve paths against the current directory. Run from the output
    # folder, so a run started inside a repository never leaves graphify-out/ in that repository.
    with contextlib.chdir(out_dir):
        graph_path = build(checkout, out_dir, resolution, ignore_file, describe_llm=do_describe)
        print(graph_path)
        if do_describe:
            report = describe(graph_path, checkout)
            print("DESCRIBE " + json.dumps(report, default=str), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

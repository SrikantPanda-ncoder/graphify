"""The run script's final line (ncoder/graphify_cli.py, summarize) reports the counts in graph.json.

Run from the repository root: ``python -m pytest tests``.
"""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("graphify_cli", ROOT / "ncoder" / "graphify_cli.py")
cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cli)


def test_summary_counts_nodes_edges_communities_and_descriptions(tmp_path):
    graph = {
        "nodes": [{"id": "a", "description": "A does x."}, {"id": "b", "description": "B does y."}, {"id": "c"}],
        "links": [{"source": "a", "target": "b"}, {"source": "b", "target": "c"}],
        "communities": [{"id": 0, "name": "Group 0"}],
    }
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(graph))
    assert cli.summarize(path) == "SUMMARY nodes=3 edges=2 communities=1 described=2"


def test_summary_without_descriptions_or_communities(tmp_path):
    path = tmp_path / "graph.json"
    path.write_text(json.dumps({"nodes": [{"id": "a"}], "links": []}))
    assert cli.summarize(path) == "SUMMARY nodes=1 edges=0 communities=0 described=0"

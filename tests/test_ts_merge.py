"""The TypeScript relations pass must not overwrite an existing edge (graphify/ts_relations.py, merge_new_edges).

The graph holds one edge per pair of nodes, so a new relation on an already-linked pair would replace the old one.
These tests check that the old edge keeps its relation and the new one is recorded on it, and that new pairs are added.

Run from the repository root: ``python -m pytest tests``.
"""

from graphify.ts_relations import merge_new_edges


def _edge(src, tgt, rel):
    return {"source": src, "target": tgt, "relation": rel}


def test_new_pair_is_added():
    existing = [_edge("a", "b", "calls")]
    added, annotated = merge_new_edges(existing, [_edge("a", "c", "renders")])
    assert [e["relation"] for e in added] == ["renders"]
    assert annotated == 0


def test_existing_pair_keeps_its_relation_and_records_the_new_one():
    existing = [_edge("header", "hook", "calls")]
    added, annotated = merge_new_edges(existing, [_edge("header", "hook", "calls_hook")])
    assert added == []
    assert annotated == 1
    assert existing[0]["relation"] == "calls"
    assert existing[0]["ts_relations"] == ["calls_hook"]


def test_reverse_direction_counts_as_the_same_pair():
    existing = [_edge("a", "b", "imports")]
    added, annotated = merge_new_edges(existing, [_edge("b", "a", "renders")])
    assert added == []
    assert annotated == 1
    assert existing[0]["ts_relations"] == ["renders"]


def test_same_relation_recorded_once_on_one_edge():
    existing = [_edge("a", "b", "contains")]
    _, annotated = merge_new_edges(existing, [_edge("a", "b", "uses_store"), _edge("a", "b", "uses_store")])
    assert annotated == 2
    assert existing[0]["ts_relations"] == ["uses_store"]


def test_two_new_relations_on_a_new_pair_add_one_edge_and_record_the_second():
    added, annotated = merge_new_edges([], [_edge("a", "b", "renders"), _edge("a", "b", "calls_hook")])
    assert [e["relation"] for e in added] == ["renders"]
    assert annotated == 1
    assert added[0]["ts_relations"] == ["calls_hook"]


def test_no_existing_edge_is_removed_by_any_merge():
    existing = [_edge("a", "b", "calls"), _edge("c", "d", "contains"), _edge("e", "f", "imports")]
    new = [_edge("a", "b", "renders"), _edge("c", "d", "uses_store"), _edge("g", "h", "tested_by")]
    added, _ = merge_new_edges(existing, new)
    assert [(e["source"], e["target"], e["relation"]) for e in existing] == [
        ("a", "b", "calls"), ("c", "d", "contains"), ("e", "f", "imports"),
    ]
    assert [e["relation"] for e in added] == ["tested_by"]

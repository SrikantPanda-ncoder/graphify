"""The description cache is keyed by mode (graphify/descriptions_llm.py: cache_mode, _key).

A description written in simulated mode must never be read in a real-model run, and the reverse. No network is used.

Run from the repository root: ``python -m pytest tests``.
"""

import pytest

import graphify.descriptions_llm as dl

NODE = {"id": "src_app_main_main"}
CODE = "def main():\n    return 1\n"


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in ("LLM_SIMULATION", "GRAPHIFY_MODEL", "GRAPHIFY_BACKEND"):
        monkeypatch.delenv(name, raising=False)
    yield


def test_simulated_mode_is_named_simulated(monkeypatch):
    monkeypatch.setenv("LLM_SIMULATION", "true")
    assert dl.cache_mode("gateway") == "simulated"


def test_real_gateway_mode_names_the_model(monkeypatch):
    assert dl.cache_mode("gateway") == "gateway:gemini-2.5-flash"
    monkeypatch.setenv("GRAPHIFY_MODEL", "my-model")
    assert dl.cache_mode("gateway") == "gateway:my-model"


def test_offline_templates_have_their_own_mode():
    assert dl.cache_mode("") == "offline"
    assert dl.cache_mode("none") == "offline"


def test_the_same_node_gets_a_different_key_in_each_mode():
    keys = {dl._key(NODE, CODE, m) for m in ("simulated", "offline", "gateway:gemini-2.5-flash")}
    assert len(keys) == 3


def test_the_same_mode_gives_the_same_key():
    assert dl._key(NODE, CODE, "simulated") == dl._key(NODE, CODE, "simulated")


def test_a_simulated_entry_is_not_found_by_a_real_run(monkeypatch, tmp_path):
    """generate() looks entries up by key: a simulated cache entry must miss in a real-model run."""
    monkeypatch.setenv("LLM_SIMULATION", "true")
    simulated_key = dl._key(NODE, CODE, dl.cache_mode("gateway"))
    monkeypatch.delenv("LLM_SIMULATION")
    real_key = dl._key(NODE, CODE, dl.cache_mode("gateway"))
    cache = {simulated_key: {"what": "Simulated text.", "why": None}}
    assert real_key not in cache

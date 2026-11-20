"""The simulated gateway reply (graphify/gateway.py, simulated_reply) picks its answer by the prompt's opening line.

A node batch can carry code snippets that contain the naming prompt's phrase (graphify's own source does), so a
substring check over the whole prompt sent such a batch down the naming branch and returned no descriptions.

Run from the repository root: ``python -m pytest tests``.
"""

import json

from graphify.gateway import simulated_reply


def _node_prompt(item_id, code):
    return (
        "You describe pieces of source code or documents in a knowledge graph. For each item, write ONE plain sentence.\n"
        "\n"
        f"### id: {item_id}\n"
        "name: helper()\n"
        "file: src/helper.py\n"
        "```\n"
        f"{code}\n"
        "```"
    )


def test_node_batch_is_answered_even_when_a_snippet_has_the_naming_phrase():
    snippet = 'if "You are naming clusters in a knowledge graph" in prompt:\n    return reply'
    reply = json.loads(simulated_reply(_node_prompt("node_a", snippet)))
    assert list(reply) == ["node_a"]
    assert reply["node_a"].startswith("Simulated description of helper()")


def test_naming_prompt_still_gets_community_names():
    prompt = (
        "You are naming clusters in a knowledge graph. For each community below, return a short name.\n"
        "0: alpha, beta\n"
        "1: gamma\n"
    )
    reply = json.loads(simulated_reply(prompt))
    assert set(reply) == {"0", "1"}
    assert reply["0"].startswith("Simulated Group")

"""The gateway (graphify/gateway.py) and the description retry loop (graphify/descriptions_llm.py).

Covered: the gateway uses only the standard library; missing settings and 4xx answers are permanent errors, while
429, 5xx and network errors are transient; the retry loop gives up at once on a permanent error or a missing module,
and retries transient errors. No network: urlopen and the model call are replaced, and sleeps are recorded.

Run from the repository root: ``python -m pytest tests``.
"""

import io
import json
import time
import urllib.error

import pytest

import graphify.descriptions_llm as dl
import graphify.gateway as gw


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in ("LLM_SIMULATION", "LITELLM_BASE_URL", "LITELLM_API_KEY", "GRAPHIFY_CUSTOMER_ID", "GRAPHIFY_MODEL"):
        monkeypatch.delenv(name, raising=False)
    yield


def _no_network(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("no network call expected")
    monkeypatch.setattr(gw.urllib.request, "urlopen", boom)


def test_gateway_module_has_no_third_party_import():
    import inspect
    src = inspect.getsource(gw)
    assert "import requests" not in src


def test_missing_settings_are_config_errors_and_make_no_call(monkeypatch):
    _no_network(monkeypatch)
    with pytest.raises(gw.GatewayConfigError, match="LITELLM_BASE_URL"):
        gw.call_gateway("hello")
    monkeypatch.setenv("LITELLM_BASE_URL", "https://gateway.example")
    monkeypatch.setenv("LITELLM_API_KEY", "k")
    with pytest.raises(gw.GatewayConfigError, match="GRAPHIFY_CUSTOMER_ID"):
        gw.call_gateway("hello")


def test_simulation_needs_no_settings(monkeypatch):
    monkeypatch.setenv("LLM_SIMULATION", "true")
    _no_network(monkeypatch)
    assert gw.call_gateway("hello").startswith("{")


def _answer(body: dict):
    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

    return _Resp(json.dumps(body).encode("utf-8"))


def test_successful_call_posts_json_and_reads_the_reply(monkeypatch):
    monkeypatch.setenv("LITELLM_BASE_URL", "https://gateway.example/")
    monkeypatch.setenv("LITELLM_API_KEY", "k")
    monkeypatch.setenv("GRAPHIFY_CUSTOMER_ID", "org:pipeline")
    seen = {}

    def fake(req, timeout):
        seen["url"] = req.full_url
        seen["body"] = json.loads(req.data.decode("utf-8"))
        seen["auth"] = req.get_header("Authorization")
        return _answer({"choices": [{"message": {"content": "ok"}}],
                        "usage": {"prompt_tokens": 3, "completion_tokens": 1}})

    monkeypatch.setattr(gw.urllib.request, "urlopen", fake)
    usage = {}
    assert gw.call_gateway("hi", usage_out=usage) == "ok"
    assert seen["url"] == "https://gateway.example/chat/completions"
    assert seen["body"]["user"] == "org:pipeline"
    assert seen["auth"] == "Bearer k"
    assert usage == {"input": 3, "output": 1}


def _http_error(code):
    return urllib.error.HTTPError("https://gateway.example", code, "x", {}, None)


def test_4xx_is_a_permanent_rejection(monkeypatch):
    monkeypatch.setenv("LITELLM_BASE_URL", "https://gateway.example")
    monkeypatch.setenv("LITELLM_API_KEY", "k")
    monkeypatch.setenv("GRAPHIFY_CUSTOMER_ID", "org:pipeline")

    def fake(req, timeout):
        raise _http_error(401)

    monkeypatch.setattr(gw.urllib.request, "urlopen", fake)
    with pytest.raises(gw.GatewayRejectedError):
        gw.call_gateway("hi")


@pytest.mark.parametrize("code", [429, 500, 503])
def test_429_and_5xx_are_transient(monkeypatch, code):
    monkeypatch.setenv("LITELLM_BASE_URL", "https://gateway.example")
    monkeypatch.setenv("LITELLM_API_KEY", "k")
    monkeypatch.setenv("GRAPHIFY_CUSTOMER_ID", "org:pipeline")

    def fake(req, timeout):
        raise _http_error(code)

    monkeypatch.setattr(gw.urllib.request, "urlopen", fake)
    with pytest.raises(RuntimeError) as info:
        gw.call_gateway("hi")
    assert not isinstance(info.value, gw.GatewayPermanentError)


def test_retry_loop_fails_at_once_on_a_config_error(monkeypatch):
    calls = []
    sleeps = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

    def fake_call(prompt, backend, max_tokens, usage_out):
        calls.append(1)
        raise gw.GatewayConfigError("missing")

    import graphify.llm as llm
    monkeypatch.setattr(llm, "_call_llm", fake_call)
    with pytest.raises(gw.GatewayConfigError):
        dl._ask("prompt", "gateway", {})
    assert len(calls) == 1 and sleeps == []


def test_retry_loop_fails_at_once_on_a_missing_module(monkeypatch):
    calls = []

    def fake_call(prompt, backend, max_tokens, usage_out):
        calls.append(1)
        raise ModuleNotFoundError("No module named 'requests'")

    import graphify.llm as llm
    monkeypatch.setattr(llm, "_call_llm", fake_call)
    with pytest.raises(ModuleNotFoundError):
        dl._ask("prompt", "gateway", {})
    assert len(calls) == 1


def test_retry_loop_retries_a_transient_error(monkeypatch):
    calls = []
    sleeps = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

    def fake_call(prompt, backend, max_tokens, usage_out):
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("gateway answered 503; transient")
        return '{"ok": "yes"}'

    import graphify.llm as llm
    monkeypatch.setattr(llm, "_call_llm", fake_call)
    assert dl._ask("prompt", "gateway", {}) == {"ok": "yes"}
    assert len(calls) == 3 and len(sleeps) == 2

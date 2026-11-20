"""Tests for the per-run language overrides in graphify/rcfile.py (GRAPHIFY_LANGUAGE_OVERRIDES).

The variable is merged over the repo's own ``.graphifyrc``, and wins on a clash. A bad value is reported and ignored,
so the scan keeps going. Only language aliases are used, so no extractor module is imported.

Run from the repository root: ``python -m pytest tests``.
"""

from __future__ import annotations




import pytest



from graphify import rcfile


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(rcfile.ENV_LANGUAGE_OVERRIDES, raising=False)
    rcfile._warned_roots.clear()
    rcfile.set_language_overrides(None)
    yield
    rcfile.set_language_overrides(None)


def test_parse_env_overrides_maps_extensions_and_languages():
    assert rcfile.parse_env_overrides("tpl=ts, inc=php ,") == {
        ".tpl": ".ts",
        ".inc": ".php",
    }


@pytest.mark.parametrize("raw", ["tpl", "=ts", "tpl=", "tpl=klingon"])
def test_parse_env_overrides_refuses_bad_pairs(raw):
    with pytest.raises(ValueError):
        rcfile.parse_env_overrides(raw)


def test_env_alone_sets_overrides_when_repo_has_no_rc(monkeypatch, tmp_path):
    monkeypatch.setenv(rcfile.ENV_LANGUAGE_OVERRIDES, "tpl=ts")
    assert rcfile.activate_language_overrides(tmp_path) == {".tpl": ".ts"}
    assert rcfile.effective_suffix("templates/page.tpl") == ".ts"


def test_env_wins_over_repo_rc_on_a_clash(monkeypatch, tmp_path):
    (tmp_path / ".graphifyrc").write_text(
        "language.inc=php\nlanguage.tpl=sql\n", encoding="utf-8"
    )
    monkeypatch.setenv(rcfile.ENV_LANGUAGE_OVERRIDES, "inc=pascal")
    active = rcfile.activate_language_overrides(tmp_path)
    assert active == {".inc": ".pas", ".tpl": ".sql"}


def test_repo_rc_is_unchanged_without_the_env(tmp_path):
    (tmp_path / ".graphifyrc").write_text("language.inc=php\n", encoding="utf-8")
    assert rcfile.activate_language_overrides(tmp_path) == {".inc": ".php"}


def test_bad_env_is_reported_and_ignored(monkeypatch, tmp_path, capsys):
    (tmp_path / ".graphifyrc").write_text("language.inc=php\n", encoding="utf-8")
    monkeypatch.setenv(rcfile.ENV_LANGUAGE_OVERRIDES, "tpl=klingon")
    assert rcfile.activate_language_overrides(tmp_path) == {".inc": ".php"}
    assert "GRAPHIFY_LANGUAGE_OVERRIDES" in capsys.readouterr().err


def test_bad_env_warns_once_per_value(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv(rcfile.ENV_LANGUAGE_OVERRIDES, "tpl=klingon")
    rcfile.activate_language_overrides(tmp_path)
    rcfile.activate_language_overrides(tmp_path)
    assert capsys.readouterr().err.count("GRAPHIFY_LANGUAGE_OVERRIDES") == 1

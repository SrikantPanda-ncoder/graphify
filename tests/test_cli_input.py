"""The run script's input (ncoder/graphify_cli.py, resolve_checkout): a local folder, or a GitHub address cloned into the
output folder. git is replaced, so no network is used.

Run from the repository root: ``python -m pytest tests``.
"""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("graphify_cli", ROOT / "ncoder" / "graphify_cli.py")
cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cli)

GITHUB = "https://github.com/nCoderAI/ncoder-ai.git"


def test_local_folder_is_used_as_given(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    assert cli.resolve_checkout(str(repo), tmp_path / "out") == repo.resolve()


def test_github_address_is_cloned_shallow_into_out_source(tmp_path, monkeypatch):
    calls = []

    def fake_run(args, check, env):
        calls.append(args)
        Path(args[-1]).mkdir(parents=True)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    out = tmp_path / "out"
    out.mkdir()
    assert cli.resolve_checkout(GITHUB, out) == out / "source"
    assert calls == [["git", "clone", "--depth", "1", GITHUB, str(out / "source")]]


def test_an_old_source_folder_is_replaced(tmp_path, monkeypatch):
    out = tmp_path / "out"
    (out / "source").mkdir(parents=True)
    (out / "source" / "old.txt").write_text("from an earlier run")
    monkeypatch.setattr(cli.subprocess, "run", lambda args, check, env: Path(args[-1]).mkdir())
    cli.resolve_checkout(GITHUB, out)
    assert not (out / "source" / "old.txt").exists()

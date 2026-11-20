"""The run script takes a local folder (ncoder/graphify_cli.py). A GitHub address is refused with a message, and a
missing folder is reported; neither starts a build. No network is used.

Run from the repository root: ``python -m pytest tests``.
"""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("graphify_cli", ROOT / "ncoder" / "graphify_cli.py")
cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cli)


def test_a_github_address_is_refused(tmp_path, capsys):
    code = cli.main(["graphify_cli.py", "https://github.com/nCoderAI/ncoder-ai.git", str(tmp_path / "out")])
    assert code == 2
    assert "local folder" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_a_missing_folder_is_reported(tmp_path, capsys):
    code = cli.main(["graphify_cli.py", str(tmp_path / "nowhere"), str(tmp_path / "out")])
    assert code == 2
    assert "checkout not found" in capsys.readouterr().err


def test_too_few_arguments_print_usage(capsys):
    assert cli.main(["graphify_cli.py", "only-one"]) == 2
    assert "usage:" in capsys.readouterr().err

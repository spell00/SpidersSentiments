import os
import sys
import runpy
import pathlib
import types
import builtins
import pytest


def run_update_datasets_with_args(args):
    script = pathlib.Path(__file__).resolve().parents[2] / "update_datasets.py"
    argv = [str(script)] + args
    # Isolate argv during run
    old_argv = sys.argv[:]
    sys.argv = argv
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = old_argv


def test_update_datasets_prints_setup_without_key(capsys, monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    run_update_datasets_with_args(["--upload-all-sql"])  # should not crash, prints setup instructions
    out = capsys.readouterr().out
    assert "To set up LangSmith integration" in out


def test_update_datasets_calls_uploaders_with_key(monkeypatch, tmp_path, capsys):
    # Provide fake API key but stub out the upload methods to avoid external side effects
    monkeypatch.setenv("LANGSMITH_API_KEY", "dummy")

    from spider_guardian.langsmith import config as ls_cfg
    # Replace methods on the shared instance
    calls = {"replies": 0, "streamed": 0, "flagged": 0}

    def fake_replies(db_path, dataset_name="spider-replies-dataset", max_examples=None):
        calls["replies"] += 1
        return {"upserted": 1}

    def fake_streamed(db_path, dataset_name="spider-streamed-dataset", max_examples=None):
        calls["streamed"] += 1
        return {"upserted": 1}

    def fake_flagged(db_path, dataset_name="spider-flagged-dataset", max_examples=None):
        calls["flagged"] += 1
        return {"upserted": 1}

    monkeypatch.setattr(ls_cfg.langsmith_integration, "upload_replies_from_sql", fake_replies, raising=False)
    monkeypatch.setattr(ls_cfg.langsmith_integration, "upload_streamed_from_sql", fake_streamed, raising=False)
    monkeypatch.setattr(ls_cfg.langsmith_integration, "upload_flagged_from_sql", fake_flagged, raising=False)

    # Run script pointing to a temp DB path (will not be used due to stubs)
    run_update_datasets_with_args([
        "--sql-db", str(tmp_path / "fake.sqlite"),
        "--upload-all-sql",
        "--max-examples", "2",
    ])

    assert calls["replies"] == 1
    assert calls["streamed"] == 1
    assert calls["flagged"] == 1

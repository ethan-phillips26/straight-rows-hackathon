"""Tests for scripts/run_all.py wiring, with detect/attribute faked so it runs without the dataset."""
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "starter" / "data"


def load_run_all():
    spec = importlib.util.spec_from_file_location("run_all", ROOT / "scripts" / "run_all.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def run_all():
    pytest.importorskip("straightrows.detect")       # skip until detect.py exists
    pytest.importorskip("straightrows.attribute")    # skip until attribute.py exists
    return load_run_all()


def test_eval_window_constants(run_all):
    assert run_all.EVAL_START == pd.Timestamp("2026-05-06T00:00Z")
    assert run_all.EVAL_END == pd.Timestamp("2026-05-07T00:00Z")


@pytest.mark.skipif(not (DATA / "farm").exists(), reason="starter/data not present")
def test_end_to_end_valid_and_deterministic(run_all, tmp_path, monkeypatch):
    outs = []
    for name in ("a.json", "b.json"):
        p = tmp_path / name
        monkeypatch.setattr(sys, "argv", ["run_all", "--data", str(DATA), "--out", str(p),
                                          "--github-handle", "someuser",
                                          "--repo-url", "https://github.com/someuser/straight-rows"])
        run_all.main()
        outs.append(p.read_bytes())
    assert outs[0] == outs[1]
    assert json.loads(outs[0])["detections"]

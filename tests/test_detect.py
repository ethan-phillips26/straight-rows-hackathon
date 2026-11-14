"""Tests for straightrows/detect.py. Synthetic tests need no dataset; the dev regression skips without it."""
from pathlib import Path

import pandas as pd
import pytest

from straightrows import detect as D
from straightrows import features as F
from tests.test_features import synthetic_fleet

DATA = Path(__file__).resolve().parents[1] / "starter" / "data"
needs_data = pytest.mark.skipif(not (DATA / "farm").exists(), reason="starter/data not present")


def test_find_runs_merges_small_gaps_and_drops_short_runs():
    t = pd.to_datetime(["2026-05-04T10:00Z", "2026-05-04T10:01Z", "2026-05-04T10:03Z", "2026-05-04T11:00Z"])
    runs = D.find_runs(t, D.MINUTE, 2 * D.MINUTE, 3 * D.MINUTE)
    assert runs == [(t[0], t[2] + D.MINUTE)]  # 10:00-10:04; the lone 11:00 minute is too short


def test_snap_to_grid_only_within_tolerance():
    snap = lambda s: D.snap_to_grid(pd.Timestamp(s, tz="UTC"))
    assert snap("2026-05-04T13:14:00") == pd.Timestamp("2026-05-04T13:12Z")   # 2 min off -> grid


def test_quiet_fleet_has_no_episodes():
    sec = F.second_features(synthetic_fleet(n=3600))
    minute = F.minute_features(sec)
    assert D.detect_spoofing(sec) == [] and D.detect_equipment(sec) == []
    assert D.detect_jamming(minute.assign(s4=0.05)) == []


def test_spoof_found_on_one_tractor_with_back_extrapolated_onset():
    # 6 m step on T2 for 30 min of a 4 h run (a real episode is a small part of the day, and the lockstep
    # baseline is the most common spacing, so a fault covering most of the run would become "normal")
    tele = synthetic_fleet(n=14400)
    m = (tele.tractor_id == "T2") & (tele.ts >= "2026-05-04T11:10Z") & (tele.ts < "2026-05-04T11:40Z")
    tele.loc[m, "lon"] += 6.0 / F.M_PER_DEG_LON
    eps = D.detect_spoofing(F.second_features(tele))
    assert len(eps) == 1 and eps[0]["tractors"] == ["T2"]
    assert abs((eps[0]["start"] - pd.Timestamp("2026-05-04T11:10Z")).total_seconds()) <= 120
    assert abs((eps[0]["end"] - pd.Timestamp("2026-05-04T11:40Z")).total_seconds()) <= 120


def test_single_tractor_cn0_drop_is_jamming_not_storm():
    tele = synthetic_fleet(n=14400)
    m = (tele.tractor_id == "T2") & (tele.ts >= "2026-05-04T11:20Z") & (tele.ts < "2026-05-04T11:50Z")
    tele.loc[m, "mean_cn0_dbhz"] -= 10
    minute = F.minute_features(F.second_features(tele)).assign(s4=0.05)
    assert D.detect_storms(minute) == []
    eps = D.detect_jamming(minute)
    assert len(eps) == 1 and eps[0]["tractors"] == ["T2"]


def test_fleet_wide_drop_with_high_s4_is_a_storm():
    tele = synthetic_fleet(n=14400)  # long run so the quiet baseline (a median) is not the storm level
    m = (tele.ts >= "2026-05-04T11:20Z") & (tele.ts < "2026-05-04T12:00Z")
    tele.loc[m, "mean_cn0_dbhz"] -= 9
    minute = F.minute_features(F.second_features(tele)).assign(s4=0.05)
    minute.loc[(minute.ts >= "2026-05-04T11:20Z") & (minute.ts < "2026-05-04T12:00Z"), "s4"] = 0.5
    eps = D.detect_storms(minute)
    assert len(eps) == 1 and eps[0]["tractors"] == F.TRACTORS
    assert D.detect_jamming(minute) == []   # the storm level is subtracted, so no extra jamming


@needs_data
def test_dev_regression_all_twelve_episodes_found():
    """Every dev episode matched with the right class/tractors and no false positives (40/40 at time of writing)."""
    import json, sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import eval_dev
    from straightrows.features import build_features
    from straightrows.load import load_sda, load_telemetry
    sec, minute, passes = build_features(load_telemetry(DATA), load_sda(DATA))
    df = D.detect_all(sec, minute, passes)
    df = df[(df.start >= eval_dev.DEV_START) & (df.start < eval_dev.DEV_END)]
    dets = [{"start": r.start.isoformat(), "end": r.end.isoformat(), "tractors": r.tractors, "class": r.kind,
             "action": D.CLASS_ACTION[r.kind], "confidence": 0.5, "evidence": []} for r in df.itertuples()]
    res = eval_dev.score({"detections": dets}, json.loads((DATA / "labels" / "train_episodes.json").read_text()))
    assert res["diagnostic_score"] >= 38 and not res["false_positives"]

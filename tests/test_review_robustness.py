"""Review tests on the real dev data (skipped without starter/data): do the dev results survive small changes?

Written by the independent reviewer (docs/FABLE_REVIEW.md). These PASSED at review time; they are guards.
Only the labeled dev days (05-04, 05-05) are used.
"""
import json
import sys
from pathlib import Path

import pytest

from straightrows import detect as D
from straightrows.features import build_features
from straightrows.load import load_sda, load_telemetry

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "starter" / "data"
pytestmark = pytest.mark.skipif(not (DATA / "farm").exists(), reason="starter/data not present")

sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture(scope="module")
def raw():
    return load_telemetry(DATA), load_sda(DATA)


def dev_score(detections):
    import eval_dev
    df = detections[(detections.start >= eval_dev.DEV_START) & (detections.start < eval_dev.DEV_END)]
    payload = [{"start": r.start.isoformat(), "end": r.end.isoformat(), "tractors": r.tractors, "class": r.kind,
                "action": D.CLASS_ACTION[r.kind], "confidence": 0.5, "evidence": []} for r in df.itertuples()]
    return eval_dev.score({"detections": payload}, json.loads((DATA / "labels" / "train_episodes.json").read_text()))


def exact(result):
    """Stricter than the diagnostic: right class, exact tractor set, onset within 5 min, no false positives."""
    ok = all(e.get("predicted_class") == e["truth_class"] and e["tractor_iou"] == 1 and abs(e["onset_error_s"]) <= 300
             for e in result["episodes"])
    return ok and not result["false_positives"]


def test_baseline_from_the_storm_day_alone_gives_the_same_detections(raw):
    """05-05 is the worst day for a median baseline: exactly 50 % of working seconds are inside a storm.

    The baseline moves from 44.7 to ~43.1 dB-Hz, and every 05-05 episode is still found exactly.
    """
    tele, sda = raw
    day = tele[(tele.ts >= "2026-05-05") & (tele.ts < "2026-05-06")]
    sec, minute, passes = build_features(day, sda)
    assert 42.5 < sec.cn0_baseline.min() < 44.0          # documents how far the median is dragged
    found = D.detect_all(sec, minute, passes)
    found = found[found.kind != "MULTIPATH"]
    got = sorted((r.kind, r.start.strftime("%H:%M"), r.end.strftime("%H:%M"), ",".join(r.tractors)) for r in found.itertuples())
    assert got == [("EQUIPMENT_FAULT", "02:00", "02:48", "T1"), ("JAMMING", "20:00", "20:36", "T1,T2,T3"),
                   ("SPACE_WEATHER", "11:00", "16:00", "T1,T2,T3"), ("SPACE_WEATHER", "19:00", "22:00", "T1,T2,T3"),
                   ("SPOOFING", "00:30", "01:18", "T1"), ("SPOOFING", "21:30", "22:06", "T3")]


@pytest.mark.parametrize("offset_db", [-3.0, 3.0])
def test_storm_level_line_tolerates_3_db(raw, monkeypatch, offset_db):
    """The storm line (42.7 - 17.7*S4) is fitted on two storms. Its residual on dev is under 0.6 dB, and the
    dev result holds for an intercept error of -3.75 to +3.75 dB (beyond that the storm is taken for jamming).
    (At review time the low side was -6 dB; the storm level check added for review item 2 tightened it.)"""
    tele, sda = raw
    dev = tele[tele.ts < "2026-05-06"]
    sec, minute, passes = build_features(dev, sda)
    monkeypatch.setattr(D, "STORM_CN0_AT_S4_0", D.STORM_CN0_AT_S4_0 + offset_db)
    assert exact(dev_score(D.detect_all(sec, minute, passes)))


def test_snapping_is_not_what_makes_the_boundaries_right(raw, monkeypatch):
    """With grid snapping switched off, every dev start is still within 5 s of the label."""
    tele, sda = raw
    dev = tele[tele.ts < "2026-05-06"]
    sec, minute, passes = build_features(dev, sda)
    monkeypatch.setattr(D, "_snap", lambda e: e)
    result = dev_score(D.detect_all(sec, minute, passes))
    assert exact(result) and max(abs(e["onset_error_s"]) for e in result["episodes"]) <= 5

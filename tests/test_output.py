"""Tests for straightrows/output.py. Synthetic only; no dataset needed."""
import json

import pandas as pd
import pytest

from straightrows.output import CLASS_ACTION, MAX_DETECTIONS, Detection, build_submission, write_submission

HANDLE, URL = "someuser", "https://github.com/someuser/straight-rows"


def det(start="2026-05-06T10:00:00Z", end="2026-05-06T11:00:00Z", tractors=("T2", "T1"),
        cls="JAMMING", conf=0.81234, evidence=("C/N0 dropped 12 dB on T1 at 10:00Z",)):
    return Detection(pd.Timestamp(start), pd.Timestamp(end), list(tractors), cls, conf, list(evidence))


def test_class_action_pairs_match_challenge():
    assert CLASS_ACTION == {
        "SPACE_WEATHER": "HALT_OR_DEGRADED_MODE", "JAMMING": "HALT_AND_REPORT",
        "SPOOFING": "HALT_CROSSCHECK_AND_REPORT", "EQUIPMENT_FAULT": "SERVICE_MACHINE_OTHERS_CONTINUE",
        "MULTIPATH": "CONTINUE_FALSE_ALARM"}


def test_basic_serialisation():
    sub = build_submission([det()], HANDLE, URL)
    d = sub["detections"][0]
    assert d["start"] == "2026-05-06T10:00:00+00:00"        # explicit +00:00, not Z
    assert d["tractors"] == ["T1", "T2"]                    # sorted
    assert d["action"] == "HALT_AND_REPORT"
    assert d["confidence"] == 0.812
    assert d["class"] == "JAMMING"


def test_non_utc_timestamps_converted():
    d = det(start="2026-05-05T19:00:00-05:00", end="2026-05-05T20:00:00-05:00")   # = 00:00Z-01:00Z
    assert build_submission([d], HANDLE, URL)["detections"][0]["start"] == "2026-05-06T00:00:00+00:00"


def test_naive_timestamp_rejected():
    d = det(start="2026-05-06T10:00:00", end="2026-05-06T11:00:00")
    with pytest.raises(ValueError, match="timezone"):
        build_submission([d], HANDLE, URL)


def test_sorted_and_deterministic():
    a, b = det(start="2026-05-06T12:00:00Z", end="2026-05-06T13:00:00Z"), det()
    s1 = build_submission([a, b], HANDLE, URL)
    s2 = build_submission([b, a], HANDLE, URL)
    assert s1 == s2
    assert [d["start"] for d in s1["detections"]] == sorted(d["start"] for d in s1["detections"])


@pytest.mark.parametrize("bad", [
    det(start="2026-05-05T23:00:00Z", end="2026-05-06T01:00:00Z"),   # starts before window
    det(start="2026-05-06T23:00:00Z", end="2026-05-07T01:00:00Z"),   # ends after window
    det(start="2026-05-06T10:00:00Z", end="2026-05-06T10:00:00Z"),   # zero length
    det(tractors=("T4",)),
    det(tractors=()),
    det(conf=1.5),
    det(conf=float("nan")),
    det(evidence=()),
    det(evidence=("  ",)),
])
def test_validator_problems_raise(bad):
    with pytest.raises(ValueError, match="fails validation"):
        build_submission([bad], HANDLE, URL)


def test_unknown_class_rejected():
    with pytest.raises(ValueError, match="unknown class"):
        build_submission([det(cls="SOLAR_FLARE")], HANDLE, URL)


def test_detection_cap():
    many = [det() for _ in range(MAX_DETECTIONS + 1)]
    with pytest.raises(ValueError, match="limit"):
        build_submission(many, HANDLE, URL)
    assert len(build_submission(many[:MAX_DETECTIONS], HANDLE, URL)["detections"]) == MAX_DETECTIONS


def test_empty_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        build_submission([], HANDLE, URL)


def test_repo_url_must_match_handle_and_no_placeholders():
    with pytest.raises(ValueError):
        build_submission([det()], HANDLE, "https://github.com/other/repo")
    with pytest.raises(ValueError):
        build_submission([det()], "your-github-username", "https://github.com/your-github-username/x")


def test_privacy_only_handle_and_optional_name():
    sub = build_submission([det()], HANDLE, URL, display_name="A Name")
    assert set(sub) == {"participant", "repository_url", "detections"}
    assert set(sub["participant"]) == {"github_handle", "display_name"}
    assert set(build_submission([det()], HANDLE, URL)["participant"]) == {"github_handle"}


def test_write_is_byte_identical_and_validates(tmp_path):
    p1, p2 = tmp_path / "a.json", tmp_path / "b.json"
    write_submission([det()], p1, HANDLE, URL)
    write_submission([det()], p2, HANDLE, URL)
    assert p1.read_bytes() == p2.read_bytes()
    assert p1.read_text().endswith("\n")
    assert json.loads(p1.read_text())["detections"][0]["class"] == "JAMMING"

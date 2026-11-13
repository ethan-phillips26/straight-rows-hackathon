"""Tests for the dev scoring harness using tiny hand-made labels (no dataset needed)."""
import pytest

from straightrows.devscore import format_report, labels_to_key, score_dev

LABELS = [
    {"id": "D1", "class": "JAMMING", "start": "2026-05-04T12:00:00+00:00", "end": "2026-05-04T13:00:00+00:00",
     "tractors": ["T1", "T2"], "recommended_action": "HALT_AND_REPORT"},
    {"id": "D2", "class": "EQUIPMENT_FAULT", "start": "2026-05-04T15:00:00+00:00", "end": "2026-05-04T15:30:00+00:00",
     "tractors": ["T3"], "recommended_action": "SERVICE_MACHINE_OTHERS_CONTINUE"},
]


def detection(cls, start, end, tractors, action):
    return {"start": start, "end": end, "tractors": tractors, "class": cls, "action": action,
            "confidence": 0.8, "evidence": ["x"]}


def test_perfect_match_scores_full_credit_for_those_classes():
    dets = [
        detection("JAMMING", "2026-05-04T12:02:00+00:00", "2026-05-04T13:00:00+00:00", ["T1", "T2"], "HALT_AND_REPORT"),
        detection("EQUIPMENT_FAULT", "2026-05-04T15:00:00+00:00", "2026-05-04T15:30:00+00:00", ["T3"],
                  "SERVICE_MACHINE_OTHERS_CONTINUE"),
    ]
    result = score_dev(dets, LABELS)
    assert result["class_metrics"]["JAMMING"]["f1"] == 1.0
    assert result["class_metrics"]["EQUIPMENT_FAULT"]["f1"] == 1.0
    assert result["mean_timing_credit"] == 1.0  # both starts within 5 min
    assert result["action_agreement"] == 2


def test_missed_episode_and_false_positive():
    dets = [detection("SPOOFING", "2026-05-04T20:00:00+00:00", "2026-05-04T21:00:00+00:00", ["T1"],
                      "HALT_CROSSCHECK_AND_REPORT")]
    result = score_dev(dets, LABELS)
    assert [e["matched"] for e in result["episodes"]] == [False, False]
    assert len(result["false_positives"]) == 1


def test_format_report_mentions_every_episode_and_total():
    text = format_report(score_dev([], LABELS))
    assert "D1" in text and "D2" in text and "MISSED" in text and "DIAGNOSTIC (dev)" in text


def test_labels_accept_wrapped_and_alias_keys():
    aliased = {"episodes": [{"episode_id": "X", "cause": "MULTIPATH", "start_utc": "2026-05-04T10:00:00+00:00",
                             "end_utc": "2026-05-04T10:03:00+00:00", "affected_tractors": ["T1"],
                             "action": "CONTINUE_FALSE_ALARM"}]}
    key = labels_to_key(aliased)
    assert key["episodes"][0]["class"] == "MULTIPATH" and key["episodes"][0]["id"] == "X"


def test_unknown_label_keys_fail_loudly():
    with pytest.raises(KeyError):
        labels_to_key([{"foo": 1}])

"""Evidence strings: built from synthetic tables, so these run without the dataset."""
import numpy as np
import pandas as pd

from straightrows import evidence

T0 = pd.Timestamp("2026-05-06T10:00:00Z")


def _minute(drops=None, n=60):
    """60 working minutes for T1-T3. ``drops`` maps tractor -> C/N0 drop (dB); everything else is quiet."""
    drops = drops or {}
    rows = []
    for tr in ["T1", "T2", "T3"]:
        for i in range(n):
            rows.append(dict(tractor_id=tr, ts=T0 + pd.Timedelta(minutes=i), cn0_drop=drops.get(tr, 0.0), sats=12.0,
                             fix_ok_frac=1.0, link_down_frac=0.0, min_low_frac=0.0, cn0_min=40.0,
                             lon_dev_abs=0.03, lat=46.90, lon=-97.20))
    return pd.DataFrame(rows)


def _sda(s4=0.05, rfi=True):
    ts = lambda s: pd.Timestamp(s, tz="UTC")
    return {
        "kp": pd.DataFrame({"ts": [ts("2026-05-06T09:00")], "kp": [2.0]}),
        "dst": pd.DataFrame({"ts": [ts("2026-05-06T09:00")], "dst_nT": [-10.0]}),
        "s4": pd.DataFrame({"ts": [T0 + pd.Timedelta(minutes=15 * i) for i in range(-1, 5)], "s4": [s4] * 6}),
        "advisories": pd.DataFrame({"id": [], "ts": pd.to_datetime([], utc=True), "type": [], "svn": [], "text": []}),
        "rfi": pd.DataFrame({"id": ["RFI-9"], "ts": [T0 + pd.Timedelta(minutes=20)], "band": ["L1"], "confidence": [0.5],
                             "centerLat": [46.90], "centerLon": [-97.20], "semiMajor_km": [2.0], "semiMinor_km": [1.0],
                             "text": ["test report"]}) if rfi else
                 pd.DataFrame(columns=["id", "ts", "band", "confidence", "centerLat", "centerLon", "semiMajor_km",
                                       "semiMinor_km", "text"]).astype({"ts": "datetime64[ns, UTC]"}),
    }


def _ep(kind, tractors, stats, start=0, end=40):
    return {"kind": kind, "start": T0 + pd.Timedelta(minutes=start), "end": T0 + pd.Timedelta(minutes=end),
            "tractors": tractors, "stats": stats}


def _text(lines):
    return "\n".join(lines)


def test_every_class_gives_nonempty_clean_strings():
    eps = [
        _ep("SPACE_WEATHER", ["T1", "T2", "T3"], dict(fleet_drop_db=9.0, s4_max=0.5, s4_median=0.5)),
        _ep("JAMMING", ["T1", "T2"], dict(mean_excess_db={"T1": 10.0, "T2": 9.0, "T3": 1.0}, s4_median=0.05, no_fix_frac=0.2)),
        _ep("SPOOFING", ["T3"], dict(max_offset_m=5.0, ramp_m_per_min=0.15, first_cross=T0 + pd.Timedelta(minutes=3), xte_max_m=5.1)),
        _ep("EQUIPMENT_FAULT", ["T2"], dict(subtype="link", flag_frac=0.3)),
        _ep("EQUIPMENT_FAULT", ["T2"], dict(subtype="antenna", flag_frac=0.2)),
        _ep("MULTIPATH", ["T1"], dict(min_drop_db=8.0, n_tractors_dipped=3, xte_in=0.1), start=5, end=5),
    ]
    eps[-1]["end"] = eps[-1]["start"] + pd.Timedelta(seconds=17)
    for lines in evidence.add_evidence(eps, _minute(), _sda()):
        assert len(lines) >= 3
        t = _text(lines)
        assert all(isinstance(x, str) and x.strip() for x in lines)
        assert "nan" not in t.lower() and "@" not in t       # no missing values, no email-like text


def test_jamming_near_threshold_tractor_is_flagged_and_unaffected_listed():
    ep = _ep("JAMMING", ["T1", "T2"], dict(mean_excess_db={"T1": 12.0, "T2": 5.0, "T3": 2.5}, s4_median=0.05, no_fix_frac=0.1))
    t = _text(evidence.add_evidence([ep], _minute(), _sda())[0])
    assert "Weakest inclusion: T2 5.0 dB" in t
    assert "Not reported: T3 2.5 dB" in t
    assert "quiet -> the drop is not space weather" in t


def test_spoof_inside_storm_does_not_claim_normal_receiver():
    storm = _ep("SPACE_WEATHER", ["T1", "T2", "T3"], dict(fleet_drop_db=9.0, s4_max=0.5, s4_median=0.5), 0, 60)
    spoof = _ep("SPOOFING", ["T3"], dict(max_offset_m=4.0, ramp_m_per_min=0.1, first_cross=T0 + pd.Timedelta(minutes=10), xte_max_m=4.0))
    minute = _minute({"T1": 9.0, "T2": 9.0, "T3": 9.0})
    s_lines, p_lines = evidence.add_evidence([storm, spoof], minute, _sda(s4=0.5))
    p = _text(p_lines)
    assert "storm does not explain it" in p and "S4 is elevated" in p     # one tractor only, so not the storm
    assert "-9.0 dB vs its quiet baseline, neighbours -9.0 dB" in p       # compared against neighbours, not "normal"
    s = _text(s_lines)
    assert "T3 left out because it has its own SPOOFING detection" in s   # storm must not claim "no spoofing" over T3
    assert "Overlaps a separate SPOOFING detection on T3" in s and "Overlaps a separate SPACE_WEATHER" in p


def test_rfi_used_only_for_jamming_and_timed_relative_to_episode():
    jam = _ep("JAMMING", ["T1"], dict(mean_excess_db={"T1": 10.0, "T2": 0.0, "T3": 0.0}, s4_median=0.05, no_fix_frac=0.0))
    eq = _ep("EQUIPMENT_FAULT", ["T2"], dict(subtype="link", flag_frac=0.3))
    j, e = [_text(x) for x in evidence.add_evidence([jam, eq], _minute(), _sda())]
    assert "RFI-9" in j and "(during the episode)" in j and "consistent with interference" in j
    assert "not used as the cause" in e
    # no reports at all -> says so instead of inventing one
    assert "No RFI report within 60 min" in _text(evidence.add_evidence([jam], _minute(), _sda(rfi=False))[0])


def test_multipath_notes_when_not_all_tractors_dipped():
    ep = _ep("MULTIPATH", ["T2"], dict(min_drop_db=6.0, n_tractors_dipped=2, xte_in=0.9))
    ep["end"] = ep["start"] + pd.Timedelta(seconds=17)
    t = _text(evidence.add_evidence([ep], _minute(), _sda())[0])
    assert "only 2 of 3 tractors dipped" in t and "ABOVE the quiet range" in t


def test_deterministic():
    ep = _ep("SPOOFING", ["T3"], dict(max_offset_m=5.0, ramp_m_per_min=0.15, first_cross=T0 + pd.Timedelta(minutes=3), xte_max_m=5.1))
    a = evidence.add_evidence([ep], _minute(), _sda())
    b = evidence.add_evidence([ep], _minute(), _sda())
    assert a == b

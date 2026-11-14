"""Guards for the fixes made after docs/REVIEW.md (items 1, 3, 9, 10 and attribute.py). Synthetic, no dataset."""
import numpy as np
import pandas as pd

from straightrows import attribute, detect as D, features as F
from tests.test_evidence import _ep, _minute, _sda, T0
from tests.test_features import synthetic_fleet
from tests.test_review_unseen_variants import T, between


def _eps(*rows):
    return pd.DataFrame([{"kind": "JAMMING", "start": T(a), "end": T(b), "tractors": F.TRACTORS, "stats": {}} for a, b in rows])


# ---- item 1: clip to the window instead of filtering on start --------------------------------------
def test_clip_keeps_the_part_of_an_episode_that_crosses_midnight():
    window = (T("2026-05-06T00:00"), T("2026-05-07T00:00"))
    df = _eps(("2026-05-05T23:48", "2026-05-06T00:24"), ("2026-05-05T20:00", "2026-05-05T20:36"),
              ("2026-05-06T23:40", "2026-05-07T00:00"), ("2026-05-06T10:00", "2026-05-06T10:36"))
    out = D.clip_to_window(df, window)
    assert [(str(r.start)[:16], str(r.end)[:16]) for r in out.itertuples()] == [
        ("2026-05-06 00:00", "2026-05-06 00:24"), ("2026-05-06 23:40", "2026-05-07 00:00"), ("2026-05-06 10:00", "2026-05-06 10:36")]


# ---- item 10: empty inputs --------------------------------------------------------------------------
def test_no_headland_passes_and_no_episodes_do_not_crash():
    sec = synthetic_fleet(n=600)                       # ten minutes, never reaches the north headland
    sec = F.second_features(sec)
    assert not sec.headland.any()
    passes = F.headland_passes(sec)
    assert passes.empty and "min_drop_db" in passes.columns
    assert D.detect_multipath(passes) == []
    df = D.detect_all(sec, F.minute_features(sec).assign(s4=0.05), passes)
    assert df.empty and list(df.columns) == D.EPISODE_COLUMNS


# ---- item 9: every pass is emitted, even a shallow one ---------------------------------------------
def test_multipath_emits_a_pass_with_a_tiny_dip():
    passes = pd.DataFrame([dict(tractor_id="T1", start=T0, end=T0 + pd.Timedelta(seconds=17), min_drop_db=0.4,
                                n_tractors_dipped=3, xte_in=0.1)])
    eps = D.detect_multipath(passes)
    assert len(eps) == 1 and eps[0]["stats"]["min_drop_db"] == 0.4


# ---- item 3: the XTE sanity check -------------------------------------------------------------------
def test_spoof_offset_without_matching_xte_is_kept_but_flagged():
    tele = synthetic_fleet(n=14400)
    m = between(tele, "2026-05-04T12:00", "2026-05-04T12:30", ["T3"])
    tele.loc[m, "lon"] += 6.0 / F.M_PER_DEG_LON          # position moved, guidance monitor sees nothing
    eps = D.detect_spoofing(F.second_features(tele))
    assert len(eps) == 1 and eps[0]["tractors"] == ["T3"] and eps[0]["stats"]["xte_mismatch"] is True


def test_normal_spoof_is_not_flagged():
    tele = synthetic_fleet(n=14400)
    m = between(tele, "2026-05-04T12:00", "2026-05-04T12:30", ["T3"])
    tele.loc[m, "lon"] += 6.0 / F.M_PER_DEG_LON
    tele.loc[m, "cross_track_error_m"] = 6.0
    eps = D.detect_spoofing(F.second_features(tele))
    assert len(eps) == 1 and eps[0]["tractors"] == ["T3"] and eps[0]["stats"]["xte_mismatch"] is False


# ---- attribute.py -----------------------------------------------------------------------------------
def _conf(ep, others=(), sda=None):
    return attribute.confidence(ep, sda or _sda(), list(others))


def test_confidence_is_lowered_for_a_weak_jam_tractor_and_raised_by_rfi():
    strong = _ep("JAMMING", ["T1"], dict(mean_excess_db={"T1": 12.0, "T2": 0.0, "T3": 0.0}, s4_median=0.05, no_fix_frac=0.0))
    weak = _ep("JAMMING", ["T1", "T2"], dict(mean_excess_db={"T1": 12.0, "T2": 5.4, "T3": 0.0}, s4_median=0.05, no_fix_frac=0.0))
    (c_strong, why_strong), (c_weak, why_weak) = _conf(strong, sda=_sda(rfi=False)), _conf(weak, sda=_sda(rfi=False))
    assert c_weak == c_strong - 0.10 and "cut-off" in why_weak[0]
    assert _conf(strong)[0] == c_strong + 0.05          # same episode with an RFI report nearby


def test_confidence_stays_in_range_and_varies():
    eps = [_ep("SPACE_WEATHER", F.TRACTORS, dict(fleet_drop_db=9, s4_max=0.5, s4_median=0.5)),
           _ep("SPOOFING", ["T3"], dict(max_offset_m=1.0, ramp_m_per_min=0.1, first_cross=T0, xte_max_m=0.3, xte_mismatch=True)),
           _ep("MULTIPATH", ["T1"], dict(min_drop_db=1.0, n_tractors_dipped=1, xte_in=0.9))]
    values = [_conf(e)[0] for e in eps]
    assert all(0.05 <= v <= 0.95 for v in values) and len(set(values)) == 3
    assert values[1] == 0.9 - 0.40 - 0.10 and values[2] == 0.9 - 0.30 - 0.20 - 0.20


def test_attribute_end_to_end_builds_valid_detections():
    eps = pd.DataFrame([_ep("EQUIPMENT_FAULT", ["T2"], dict(subtype="link", flag_frac=0.3))])
    (d,) = attribute.attribute(eps, _minute(), _sda())
    assert d.cls == "EQUIPMENT_FAULT" and d.action == "SERVICE_MACHINE_OTHERS_CONTINUE"
    assert d.evidence[-1].startswith(f"Confidence {d.confidence:.2f}: base 0.80")


def test_multipath_in_a_storm_is_not_penalised_for_the_storms_xte():
    storm = _ep("SPACE_WEATHER", F.TRACTORS, dict(fleet_drop_db=9, s4_max=0.5, s4_median=0.5), 0, 60)
    dip = _ep("MULTIPATH", ["T1"], dict(min_drop_db=8.0, n_tractors_dipped=3, xte_in=0.6), 5, 5)
    dip["end"] = dip["start"] + pd.Timedelta(seconds=17)
    assert attribute.confidence(dip, _sda(s4=0.5), [storm])[0] == 0.90                 # explained by the storm
    assert attribute.confidence(dip, _sda(s4=0.05), [])[0] == 0.90 - 0.20              # nothing explains the XTE
    text = " ".join(attribute.attribute(pd.DataFrame([storm, dip]), _minute(), _sda(s4=0.5))[1].evidence)
    assert "reflect the overlapping SPACE_WEATHER detection" in text and "no storm" not in text


# ---- second review round (docs/LOG.md, hour 11): side effects of the first fixes ---------------------
def test_spoof_with_noisy_xte_on_every_tractor_still_blames_only_the_spoofed_one():
    """A jam makes |XTE| large on all three; the spoofed tractor's XTE is then ABOVE its offset, which is not a mismatch."""
    tele = synthetic_fleet(n=14400)
    m = between(tele, "2026-05-04T12:00", "2026-05-04T12:30", ["T2"])
    tele.loc[m, "lon"] += 4.0 / F.M_PER_DEG_LON
    tele.loc[between(tele, "2026-05-04T12:00", "2026-05-04T12:30"), "cross_track_error_m"] = 3.0   # jam noise, all three
    tele.loc[m, "cross_track_error_m"] = 7.0                                                        # noise + the spoof
    eps = D.detect_spoofing(F.second_features(tele))
    assert len(eps) == 1 and eps[0]["tractors"] == ["T2"] and eps[0]["stats"]["xte_mismatch"] is False


def test_two_tractors_spoofed_by_different_offsets_are_each_reported_once():
    tele = synthetic_fleet(n=14400)
    for tractor, metres in [("T1", 6.0), ("T2", 10.0)]:
        m = between(tele, "2026-05-04T12:00", "2026-05-04T12:30", [tractor])
        tele.loc[m, "lon"] += metres / F.M_PER_DEG_LON
        tele.loc[m, "cross_track_error_m"] = metres
    eps = D.detect_spoofing(F.second_features(tele))
    assert sorted(tuple(e["tractors"]) for e in eps) == [("T1",), ("T2",)]


def test_storm_survives_a_non_uniform_jam_that_covers_most_of_it():
    """3 h storm (uniform, on the S4 line) with a 2 h jam of 9 / 5 / 0 dB on top: both must be reported."""
    tele = synthetic_fleet(n=36000)          # 10 h, so the 3 h storm is a minority and the median baseline stays quiet
    s4, a, b = 0.45, "2026-05-04T11:30", "2026-05-04T14:30"
    storm_level = D.STORM_CN0_AT_S4_0 - D.STORM_CN0_PER_S4 * s4
    tele.loc[between(tele, a, b), "mean_cn0_dbhz"] = storm_level
    for tractor, drop in [("T1", 9.0), ("T2", 5.0)]:
        tele.loc[between(tele, "2026-05-04T12:00", "2026-05-04T14:00", [tractor]), "mean_cn0_dbhz"] -= drop
    tele["lat"] = 46.900                                   # keep the whole run out of the headland zone
    minute = F.minute_features(F.second_features(tele)).assign(s4=0.05)
    minute.loc[(minute.ts >= T(a)) & (minute.ts < T(b)), "s4"] = s4
    storms, jams = D.detect_storms(minute), D.detect_jamming(minute)
    assert [(e["start"], e["end"]) for e in storms] == [(T(a), T(b))]
    assert len(jams) == 1 and jams[0]["tractors"] == ["T1", "T2"]
    assert (jams[0]["start"], jams[0]["end"]) == (T("2026-05-04T12:00"), T("2026-05-04T14:00"))


def test_antenna_fault_starting_just_before_a_jam_keeps_its_start():
    """Fault 11:24-12:48, jam 11:36-12:12: only 12 min of fault before the jam, less than EQUIP_MIN_MIN."""
    tele = synthetic_fleet(n=14400)
    fault = between(tele, "2026-05-04T11:24", "2026-05-04T12:48", ["T1"])
    tele.loc[fault & (tele.ts.dt.second % 5 == 0), "min_cn0_dbhz"] -= 8.0
    tele.loc[between(tele, "2026-05-04T11:36", "2026-05-04T12:12"), ["mean_cn0_dbhz", "min_cn0_dbhz"]] -= 10.0
    sec = F.second_features(tele)
    jam = D.detect_jamming(F.minute_features(sec).assign(s4=0.05))
    eps = D.detect_equipment(sec, jam)
    assert [(e["start"], e["end"], e["tractors"]) for e in eps] == [(T("2026-05-04T11:24"), T("2026-05-04T12:48"), ["T1"])]


def test_a_few_stray_dropout_minutes_around_a_jam_are_not_a_fault():
    """6 min of dropouts before and 6 min after a jam: 12 min outside the jam is under EQUIP_MIN_MIN, so nothing."""
    tele = synthetic_fleet(n=14400)
    stray = between(tele, "2026-05-04T11:30", "2026-05-04T11:36", ["T1"]) | between(tele, "2026-05-04T12:12", "2026-05-04T12:18", ["T1"])
    tele.loc[stray & (tele.ts.dt.second % 5 == 0), "min_cn0_dbhz"] -= 8.0
    tele.loc[between(tele, "2026-05-04T11:36", "2026-05-04T12:12"), ["mean_cn0_dbhz", "min_cn0_dbhz"]] -= 10.0
    sec = F.second_features(tele)
    jam = D.detect_jamming(F.minute_features(sec).assign(s4=0.05))
    assert len(jam) == 1 and D.detect_equipment(sec, jam) == []

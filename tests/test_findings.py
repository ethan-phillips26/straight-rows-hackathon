"""Assertions behind docs/FINDINGS.md.

Two groups:
* synthetic tests of the helper functions (always run, no dataset needed);
* real-data tests of each verified claim (skipped cleanly when starter/data is missing).

The real-data tests check the *claims*, with tolerances, so they document what we believe about the
data and fail loudly if a data refresh or a code change breaks one of them.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("verify_findings", ROOT / "scripts" / "verify_findings.py")
vf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vf)

HAVE_DATA = (vf.DATA / "farm" / "tractor_telemetry.csv.gz").exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="starter/data not present (run starter/scripts/get_data.py)")


# ------------------------------------------------------------------ synthetic (no dataset needed)
def test_runs_finds_true_stretches():
    assert vf.runs([0, 1, 1, 0, 1, 0, 0, 1]) == [(1, 3), (4, 5), (7, 8)]
    assert vf.runs([0, 0]) == []


def test_merge_intervals_bridges_small_gaps_only():
    t = pd.Timestamp("2026-05-04T00:00Z")
    m = pd.Timedelta(minutes=1)
    got = vf.merge_intervals([(t + 5 * m, t + 6 * m), (t, t + m), (t + 2 * m, t + 3 * m)], gap=m)
    assert got == [(t, t + 3 * m), (t + 5 * m, t + 6 * m)]


def test_changepoint_locates_a_step():
    idx = pd.date_range("2026-05-04T10:00Z", periods=3600, freq="1s")
    s = pd.Series(45.0, index=idx)
    s[idx >= pd.Timestamp("2026-05-04T10:30:07Z")] = 35.0
    off, step = vf.changepoint(s, pd.Timestamp("2026-05-04T10:30Z"))
    assert off == 7 and step == pytest.approx(-10.0)


def test_dominant_period_matches_a_sine():
    t = np.arange(2400)
    s = pd.Series(np.sin(2 * np.pi * t / 355.0))
    assert abs(vf.dominant_period_s(s) - 355) <= 2


def test_position_offset_flags_only_the_displaced_tractor():
    # three tractors in lockstep, 0.0034708 deg apart; T2 is pushed 5 m east for 100 s
    idx = pd.date_range("2026-05-04T12:00Z", periods=600, freq="1s")
    rows = []
    for k, tr in enumerate(vf.TRACTORS):
        lon = -97.20 + k * 0.0034708 + np.zeros(len(idx))
        if tr == "T2":
            lon[200:300] += 5.0 / vf.M_PER_DEG_LON
        rows.append(pd.DataFrame({"ts": idx, "tractor_id": tr, "lon": lon, "working": True}))
    tele = vf.add_position_offset(pd.concat(rows, ignore_index=True))
    dev = tele.pivot(index="ts", columns="tractor_id", values="lon_dev_m")
    assert dev.T2.iloc[200:300].between(4.9, 5.1).all()
    assert dev.T2.drop(dev.index[200:300]).abs().max() < 0.01
    assert dev[["T1", "T3"]].abs().max().max() < 0.01  # neighbours are NOT blamed


# ------------------------------------------------------------------ real data
@pytest.fixture(scope="module")
def d():
    tele, sda, labels = vf.load_all()
    geo, apex = vf.geometry(tele)
    passes = vf.headland_passes(tele, apex, labels)
    flags = vf.flag_seconds(tele)
    return {"tele": tele, "sda": sda, "labels": labels, "geo": geo, "apex": apex,
            "passes": passes, "flags": flags}


@needs_data
def test_geometry_lockstep_and_hours(d):
    g = d["geo"]
    assert g["rows"] == 777_600
    assert g["speed_values"] == [0.0, 4.5]
    for tr in vf.TRACTORS:
        assert g["work_start_times"][tr] == ["11:00:00"] and g["work_stop_times"][tr] == ["03:00:00"]
    assert set(g["apex_per_day"].values()) == {162}
    assert g["apex_identical_all3"]                      # all three reach the north edge in the same second
    assert set(g["apex_spacing_s"]) == {355, 356}
    assert g["max_lat_diff_working_m"] < 0.1             # same lat, even during spoofing
    assert g["parked_xte_max"] == 0.0                    # XTE is only produced while working
    assert all(v == 16 for v in g["lon_wraps_while_working"].values())  # normal ~800 m lon jumps exist


@needs_data
def test_quiet_reference_is_clean(d):
    q = vf.quiet_distributions(d["tele"])["away"]
    assert q["fix_values"] == ["RTK_FIXED"] and q["link_down"] == 0
    assert q["sats_values"] == [11, 12, 13]
    assert q["mean_cn0_min"] >= 42.5


@needs_data
def test_flags_hold_out_of_sample(d):
    # The flag thresholds were read off the QUIET window, so testing them there would be circular.
    # Check them on other quiet dev stretches instead: essentially nothing should fire.
    ho = vf.held_out_quiet_flags(d["tele"], d["labels"], d["flags"])
    assert ho["seconds"] > 100_000
    assert sum(ho["flagged"].values()) / ho["seconds"] < 1e-4


@needs_data
def test_label_boundaries_on_six_minute_grid(d):
    g = vf.label_grid(d["labels"])
    assert g["on_6min_grid"] == g["boundaries"] == 20 and g["durations_multiple_of_6"]


@needs_data
def test_storm_is_uniform_staircase_tied_to_s4(d):
    s = vf.storm_s4_summary(vf.storm_slots(d["tele"], d["sda"]))
    assert s["corr(S4, fleet C/N0) over storm slots"] < -0.99
    assert s["max between-tractor spread in storm slots (dB)"] <= 0.5
    assert s["S4 range in storm slots"][0] > s["max S4 in clean slots"]   # 0.317 vs 0.297: thin margin
    assert s["clean slots with S4 > 0.15"] == 10                         # 07:15-09:30 elevated S4, no impact
    lv = vf.storm_levels(d["tele"])
    starts = set(lv.start.dt.strftime("%m-%d %H:%M"))
    assert {"05-05 07:00", "05-05 09:45", "05-05 10:00", "05-05 13:00", "05-05 19:00"} <= starts
    assert lv.working_pct.iloc[0] == 0 and lv.working_pct.iloc[1] == 0  # first two steps while parked


@needs_data
def test_level_shift_boundaries_match_labels(d):
    b = vf.boundary_alignment(d["tele"], d["labels"])
    lvl = b[b.cls.isin(["SPACE_WEATHER", "JAMMING"])]
    assert (lvl.end_offset_s.abs() <= 1).all()
    jam = lvl[lvl.cls == "JAMMING"]
    assert (jam.start_offset_s.abs() <= 10).all()
    assert (b[b.id == "TRN-02"].start_offset_s == 0).all()
    other = b[b.cls.isin(["EQUIPMENT_FAULT"])]
    assert (other.start_offset_s.abs() <= 5).all() and (other.end_offset_s.abs() <= 5).all()
    spoof = b[b.cls == "SPOOFING"]
    assert spoof.start_offset_s.between(0, 60).all() and (spoof.end_offset_s == 0).all()


@needs_data
def test_jamming_is_distance_dependent(d):
    jt, fits = vf.jamming(d["tele"], d["sda"], d["labels"])
    for f in fits.values():
        assert 15 <= f["slope_db_per_decade"] <= 25 and f["rmse"] <= 1.1
    t2 = jt[(jt.id == "TRN-04") & (jt.tractor == "T2")].iloc[0]
    assert not t2.labeled and 2.0 <= t2.drop_db <= 4.0                    # unlabeled partial tractor
    labeled = jt[jt.labeled]
    assert (labeled.drop_db >= 5.0).all()
    assert (jt["|lon_dev|max"] < 0.1).all()                               # jamming does not move position
    delays = jt.dropna(subset=["rfi_delay_min"]).groupby("id").rfi_delay_min.first()
    assert delays.between(10, 15).all() and "TRN-05" not in delays.index


@needs_data
def test_spoofing_moves_reported_position_by_xte(d):
    sp = vf.spoofing(d["tele"], d["labels"]).set_index("id")
    assert (sp.lon_dev_vs_xte_maxdiff_m < 0.5).all()
    assert sp.ramp_m_per_min.between(0.1, 0.3).all()
    assert (sp.plateau_m > sp.ramp_end_xte).all()
    assert (sp.xte_first_5s_after_end < 0.1).all()
    for ep in ["TRN-07", "TRN-08"]:                     # no signal-health change vs the other tractors
        assert abs(sp.loc[ep, "cn0_minus_others"]) < 0.1 and abs(sp.loc[ep, "sats_minus_others"]) < 0.1
    assert (sp.link_down == 0).all()


@needs_data
def test_equipment_fault_signatures(d):
    eq = vf.equipment(d["tele"], d["labels"]).set_index("id")
    a = eq.loc["TRN-09"]
    assert a["min<35% (away from headland)"] > 15 and a["min<35% others"] < 0.1
    assert a["sats<11%"] > 10 and a["fixed%"] == 100 and a["link_down%"] == 0
    link = eq.loc["TRN-10"]
    assert 25 <= link["link_down%"] <= 35 and link["fix_when_link_down"] == ["RTK_FLOAT"]
    assert link["fix_when_link_up"] == ["RTK_FIXED"] and link["min<35% (away from headland)"] == 0


@needs_data
def test_multipath_every_pass_and_labels_equal_dip(d):
    p = d["passes"]
    dev = p[p.apex < vf.EVAL_START]
    assert len(dev) == 972 and (dev.dip_s == 17).all()
    assert dev.min_drop_db.min() >= 3.0 and abs(dev.mean_drop_db.median()) < 0.2
    mp = vf.multipath(dev, d["labels"])
    assert (mp["labels"].start_err_s == 0).all() and (mp["labels"].end_err_s == 0).all()


@needs_data
def test_only_known_degradation_outside_labels(d):
    runs = vf.outside_labels(vf.unusual_runs(d["tele"], d["flags"], vf.DEV_START, vf.EVAL_START), d["labels"])
    out = runs[runs.uncovered_min >= 1]
    got = sorted((r.tractor, r.start.strftime("%m-%d %H:%M"), r.uncovered_min) for r in out.itertuples())
    expected = sorted([(tr, "05-05 07:00", 15.0) for tr in vf.TRACTORS]       # parked storm step
                      + [(tr, "05-05 09:45", 75.0) for tr in vf.TRACTORS]     # parked storm before work start
                      + [("T2", "05-04 21:00", 36.0)])                         # TRN-04 unlabeled tractor
    assert got == expected

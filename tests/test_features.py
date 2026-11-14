"""Tests for straightrows/features.py. Synthetic tests need no dataset; real-data tests skip without it."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from straightrows import features as F
from straightrows.load import FIX_RANK

DATA = Path(__file__).resolve().parents[1] / "starter" / "data"
needs_data = pytest.mark.skipif(not (DATA / "farm").exists(), reason="starter/data not present")


def synthetic_fleet(n=1200, spoof=None, jam=None):
    """3 tractors in lockstep (lon spacing 0.0035 deg), 1 Hz, quiet. Optional faults for T2."""
    ts = pd.date_range("2026-05-04T11:00Z", periods=n, freq="s")
    rows = []
    for i, tr in enumerate(F.TRACTORS):
        lat = 46.900 + 1e-6 * np.arange(n)  # stays south of the headland zone
        d = pd.DataFrame({
            "ts": ts, "tractor_id": tr, "lat": lat, "lon": -97.20 + 0.0035 * i,
            "speed_mps": 4.5, "cross_track_error_m": 0.01, "num_sats": 12,
            "mean_cn0_dbhz": 45.0, "min_cn0_dbhz": 40.0, "fix_state": "RTK_FIXED", "base_link_ok": True,
        })
        d["fix_rank"] = d.fix_state.map(FIX_RANK).astype("int8")
        d["working"] = True
        rows.append(d)
    tele = pd.concat(rows, ignore_index=True)
    if spoof is not None:  # displace T2's reported lon east by `spoof` metres from second 600
        m = (tele.tractor_id == "T2") & (tele.ts >= ts[600])
        tele.loc[m, "lon"] += spoof / F.M_PER_DEG_LON
    if jam is not None:  # extra C/N0 drop on T2 only, for 300 s (a minority of the day, like real episodes)
        m = (tele.tractor_id == "T2") & (tele.ts >= ts[600]) & (tele.ts < ts[900])
        tele.loc[m, "mean_cn0_dbhz"] -= jam
    return tele.sort_values(["tractor_id", "ts"]).reset_index(drop=True)


def test_lockstep_offset_zero_when_quiet():
    out = F.add_lockstep_offset(synthetic_fleet())
    assert out.lon_dev_m.abs().max() < 1e-6


def test_lockstep_offset_finds_the_spoofed_tractor_only():
    out = F.add_lockstep_offset(synthetic_fleet(spoof=6.0))
    late = out[out.ts >= pd.Timestamp("2026-05-04T11:11Z")]
    med = late.groupby("tractor_id").lon_dev_m.median()
    assert med["T2"] == pytest.approx(6.0, abs=0.01)
    assert abs(med["T1"]) < 0.01 and abs(med["T3"]) < 0.01


def test_parked_rows_have_no_offset():
    tele = synthetic_fleet()
    tele.loc[tele.ts < pd.Timestamp("2026-05-04T11:05Z"), ["speed_mps", "working"]] = [0.0, False]
    out = F.add_lockstep_offset(tele)
    assert out[~out.working].lon_dev_m.isna().all()


def test_fleet_median_residual_isolates_one_tractor():
    sec = F.second_features(synthetic_fleet(jam=8.0))
    late = sec[(sec.ts >= pd.Timestamp("2026-05-04T11:10Z")) & (sec.ts < pd.Timestamp("2026-05-04T11:15Z"))].groupby("tractor_id")
    assert late.mean_cn0_dbhz_resid.mean()["T2"] == pytest.approx(-8.0)
    assert late.mean_cn0_dbhz_resid.mean()["T1"] == pytest.approx(0.0)  # neighbours unaffected


def test_cn0_drop_is_relative_to_own_baseline():
    sec = F.second_features(synthetic_fleet(jam=8.0))
    t2 = sec[(sec.tractor_id == "T2") & (sec.ts >= pd.Timestamp("2026-05-04T11:10Z")) & (sec.ts < pd.Timestamp("2026-05-04T11:15Z"))]
    assert t2.cn0_drop.mean() == pytest.approx(8.0)  # baseline is the median of mostly-quiet time


def test_minute_features_shape_and_sda():
    tele = synthetic_fleet()
    sec = F.second_features(tele)
    sda = {"s4": pd.DataFrame({"ts": pd.to_datetime(["2026-05-04T11:00Z", "2026-05-04T11:15Z"], utc=True), "s4": [0.05, 0.6]}),
           "kp": pd.DataFrame({"ts": pd.to_datetime(["2026-05-04T09:00Z"], utc=True), "kp": [2.0]}),
           "dst": pd.DataFrame({"ts": pd.to_datetime(["2026-05-04T10:00Z"], utc=True), "dst_nT": [-10]}),
           "advisories": pd.DataFrame({"ts": pd.to_datetime(["2026-05-04T11:10Z"], utc=True), "type": ["SPACE_WEATHER"]})}
    mn = F.minute_features(sec, sda)
    assert len(mn) == 3 * 20 and (mn.n == 60).all()
    row = mn[(mn.tractor_id == "T1")].set_index("ts")
    assert row.loc[pd.Timestamp("2026-05-04T11:14Z"), "s4"] == 0.05   # slot value applies from its own ts
    assert row.loc[pd.Timestamp("2026-05-04T11:15Z"), "s4"] == 0.6
    assert not row.loc[pd.Timestamp("2026-05-04T11:09Z"), "active_sw_advisory"]
    assert row.loc[pd.Timestamp("2026-05-04T11:10Z"), "active_sw_advisory"]


def test_headland_passes_finds_runs_and_depth():
    tele = synthetic_fleet(n=400)
    # put all three tractors into the headland zone for 17 s starting at second 200, with a dip
    m = (tele.ts >= pd.Timestamp("2026-05-04T11:03:20Z")) & (tele.ts < pd.Timestamp("2026-05-04T11:03:37Z"))
    tele.loc[m, "lat"] = F.HEADLAND_LAT + 0.0001
    tele.loc[~m, "lat"] = 46.9
    tele.loc[m, "min_cn0_dbhz"] = 32.0
    sec = F.second_features(tele)
    hp = F.headland_passes(sec)
    assert len(hp) == 3 and (hp.dip_s == 17).all() and (hp.n_tractors_dipped == 3).all()
    assert hp.min_drop_db.tolist() == pytest.approx([8.0] * 3)
    assert (hp.end - hp.start == pd.Timedelta(seconds=17)).all()


@needs_data
def test_real_data_invariants():
    from straightrows.load import load_sda, load_telemetry
    sec, mn, hp = F.build_features(load_telemetry(DATA), load_sda(DATA))
    assert len(sec) == 777_600
    assert len(hp) == 1458 and (hp.dip_s == 17).all() and (hp.n_tractors_dipped == 3).all()  # FINDINGS section 0.5
    assert not mn.isna().any().any()
    # Headland seconds are excluded from the minute table, so multipath cannot show up as a low min C/N0 there.
    assert mn.n.max() <= 60
    # deterministic
    sec2, mn2, _ = F.build_features(load_telemetry(DATA), load_sda(DATA))
    pd.testing.assert_frame_equal(mn, mn2)

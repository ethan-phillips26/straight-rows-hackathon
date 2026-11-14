"""Jammer location: recover a known synthetic source and refuse to report when the data cannot support one.

Synthetic data only (no real dataset needed): 3 tractors drive rows over a 40 min window and C/N0 follows
C/N0 = 45 - 20*log10(1 km / d) around a fixed source plus noise.
"""
import numpy as np
import pandas as pd

from straightrows import locate

START = pd.Timestamp("2026-05-06T02:00:00Z")
END = START + pd.Timedelta(minutes=40)


def _tele(src_lat, src_lon, noise=0.3, slope=20.0, seed=1):
    rng = np.random.default_rng(seed)
    t = pd.date_range(START, END, freq="1s", inclusive="left")
    rows = []
    for k, tr in enumerate(["T1", "T2", "T3"]):
        # boustrophedon: sweep north-south along a row, then step east (inside the non-headland band)
        phase = (np.arange(len(t)) * 4.5) % 1600.0
        north = np.where(phase < 800, phase, 1600 - phase)
        x = -400 + 40 * k + (np.arange(len(t)) * 4.5 // 1600) * 30
        y = -400 + north * 0.9
        lat, lon = locate.to_latlon(x, y)
        sx, sy = locate.to_xy(src_lat, src_lon)
        d = np.hypot(x - sx, y - sy) + locate.NEAR_FIELD_M
        cn0 = 45 + slope * np.log10(d / 1000) + rng.normal(0, noise, len(t))
        rows.append(pd.DataFrame({"ts": t, "tractor_id": tr, "lat": lat, "lon": lon, "mean_cn0_dbhz": cn0,
                                  "working": True}))
    return pd.concat(rows, ignore_index=True)


def test_recovers_known_source_within_its_error():
    src = locate.to_latlon(900.0, 700.0)          # ~1.1 km from the tractors
    r = locate.locate_jammer(_tele(*src), START, END)
    assert r["reliable"], r["notes"]
    err_m = locate.distance_km(r["lat"], r["lon"], *src) * 1000
    assert err_m <= max(r["semi_major_km"] * 1000, 50), (err_m, r)
    assert 15 < r["slope_db_per_decade"] < 25


def test_heavy_noise_is_flagged_not_reported():
    src = locate.to_latlon(900.0, 700.0)
    r = locate.locate_jammer(_tele(*src, noise=6.0), START, END)
    assert r is None or not r["reliable"]


def test_wrong_physics_is_flagged():
    """C/N0 unrelated to position (e.g. a storm: uniform drop) must not produce a confident location."""
    t = _tele(46.9, -97.19)
    t["mean_cn0_dbhz"] = np.random.default_rng(2).normal(30, 4, len(t))
    r = locate.locate_jammer(t, START, END)
    assert r is None or not r["reliable"]


def test_too_little_data_returns_none():
    assert locate.locate_jammer(_tele(46.9, -97.19).iloc[:50], START, END) is None


def test_bootstrap_ellipse_is_a_deterministic_noise_only_check():
    """The bootstrap measures sampling noise, not model bias, so it need not cover the truth (the main ellipse does,
    see the first test); it must be well-formed, no wider than the main ellipse, and reproducible."""
    src = locate.to_latlon(900.0, 700.0)
    r = locate.locate_jammer(_tele(*src), START, END)
    assert r["reliable"] and r["boot_major_km"] is not None
    assert 0 < r["boot_minor_km"] <= r["boot_major_km"] <= r["semi_major_km"]
    assert r["boot_centre_shift_m"] < 100
    assert locate.locate_jammer(_tele(*src), START, END)["boot_major_km"] == r["boot_major_km"]


def test_bootstrap_skipped_when_fit_is_not_reliable():
    r = locate.locate_jammer(_tele(*locate.to_latlon(900.0, 700.0), noise=6.0), START, END)
    assert r is None or r["boot_major_km"] is None

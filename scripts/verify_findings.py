#!/usr/bin/env python3
"""Phase 1: verify the claims in docs/PLAN.md and measure how each cause changes the data.

    python scripts/verify_findings.py            # prints every number quoted in docs/FINDINGS.md
    python scripts/verify_findings.py --plots    # also writes figures to reports/findings/

This is ANALYSIS, not a detector. Thresholds used here come from the quiet dev data (see QUIET_*)
and only serve to describe the data. Nothing is tuned on the evaluation window: section 10 only
prints shapes and timing for 2026-05-06.

Each section is a function that returns plain dicts/DataFrames, so tests/test_findings.py can
assert the same numbers this script prints. Deterministic: no randomness, no wall clock.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from straightrows.load import load_labels, load_sda, load_telemetry  # noqa: E402

DATA = ROOT / "starter" / "data"
TRACTORS = ["T1", "T2", "T3"]
DEV_START = pd.Timestamp("2026-05-04T00:00Z")
EVAL_START = pd.Timestamp("2026-05-06T00:00Z")
EVAL_END = pd.Timestamp("2026-05-07T00:00Z")

# North-headland multipath zone: the min-C/N0 dip happens exactly while reported lat is above this value
# (~38 m south of the field's north edge at 46.90720). Located from quiet-day lat bins (dip starts between
# 46.906850 and 46.906855) and consistent with both multipath labels to the second (section 7).
HEADLAND_LAT = 46.906854
M_PER_DEG_LAT = 111_320.0
M_PER_DEG_LON = M_PER_DEG_LAT * np.cos(np.radians(46.903))  # ~76 km per degree at the farm

# A long stretch of working time on 05-04 with no label nearby and quiet SDA (S4 < 0.08, Kp ~2).
QUIET = (pd.Timestamp("2026-05-04T17:15Z"), pd.Timestamp("2026-05-04T20:55Z"))

# "Unusual second" flags for section 9. Each threshold sits just outside what the QUIET window
# ever shows (see quiet_distributions()); they describe the data, they are not detector thresholds.
FLAGS = {
    "cn0_low": "mean_cn0_dbhz < 42.5 (quiet min 42.9)",
    "min_low": "min_cn0_dbhz < 35 away from the headland (quiet 0.1% quantile 36.9)",
    "sats_low": "num_sats < 11 (quiet always 11-13)",
    "fix_bad": "fix_state != RTK_FIXED (quiet always RTK_FIXED)",
    "link_down": "base_link_ok False (quiet always True)",
    "xte_high": "|XTE| > 0.5 m while working (quiet max 0.45 m incl. headland)",
    "pos_off": "reported lon offset vs lockstep neighbours > 0.5 m while working (quiet max ~0.05 m)",
}


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------
def runs(mask):
    """Start/stop index pairs (stop exclusive) of consecutive True values in a boolean array."""
    m = np.asarray(mask, dtype=int)
    d = np.diff(np.r_[0, m, 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def merge_intervals(intervals, gap):
    """Merge (start, end) Timestamp intervals separated by <= gap. Input need not be sorted."""
    out = []
    for s, e in sorted(intervals):
        if out and s - out[-1][1] <= gap:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def ts(s):
    return pd.Timestamp(s, tz="UTC") if pd.Timestamp(s).tzinfo is None else pd.Timestamp(s)


def fmt(t):
    return t.strftime("%m-%d %H:%M:%S")


def window(tele, tractor, start, end, working_only=True):
    g = tele[(tele.tractor_id == tractor) & (tele.ts >= start) & (tele.ts < end)]
    return g[g.working] if working_only else g


def load_all(data=DATA):
    tele = load_telemetry(data)
    tele["axte"] = tele.cross_track_error_m.abs()
    tele["headland"] = tele.lat > HEADLAND_LAT
    tele = add_position_offset(tele)
    labels = load_labels(data).rename(columns={"class": "cls"})
    return tele, load_sda(data), labels


def add_position_offset(tele):
    """lon_dev_m: how far (east-west, metres) a tractor's REPORTED position is from where lockstep says.

    All three tractors drive identical patterns at a fixed east-west spacing, so the lon difference
    between two tractors only ever takes two values (normal spacing, or spacing +- field width when one
    has wrapped back to the west edge and the other has not). Any other value means one reported
    position moved sideways. For tractor X we take, of its two pairs, the deviation with the smaller
    magnitude: if X is displaced both pairs show it; if a neighbour is displaced only one does.
    Parked rows get NaN (parking spots are not in formation).
    """
    w = tele[tele.working]
    lon = w.pivot(index="ts", columns="tractor_id", values="lon")
    dev_pairs = {}
    for a in TRACTORS:
        for b in TRACTORS:
            if a == b:
                continue
            off = lon[a] - lon[b]
            # normal spacing = most common offset; the "wrapped" spacing differs by about one field
            # width (~0.009 deg), so only accept a second mode that far away (a displaced tractor is
            # at most metres = 1e-4 deg off and can never be mistaken for it)
            counts = off.round(5).value_counts()
            modes = [m for i, m in enumerate(counts.index[:5]) if i == 0 or abs(m - counts.index[0]) > 0.005][:2]
            modes = np.array([off[(off - m).abs() < 2e-5].median() for m in modes])  # refine to exact
            nearest = modes[np.abs(off.values[:, None] - modes[None, :]).argmin(axis=1)]
            dev_pairs[(a, b)] = (off - nearest) * M_PER_DEG_LON
    dev = {}
    for a in TRACTORS:
        p1, p2 = [dev_pairs[(a, b)] for b in TRACTORS if b != a]
        dev[a] = p1.where(p1.abs() <= p2.abs(), p2)
    dev = pd.DataFrame(dev).stack().rename("lon_dev_m").reset_index()
    dev.columns = ["ts", "tractor_id", "lon_dev_m"]
    return tele.merge(dev, on=["ts", "tractor_id"], how="left")


def label_mask(tele, labels, pad=pd.Timedelta(0)):
    """Boolean Series: row (tractor, second) is inside a labeled episode that names that tractor."""
    m = pd.Series(False, index=tele.index)
    for ep in labels.itertuples():
        m |= (tele.tractor_id.isin(ep.tractors) & (tele.ts >= ep.start - pad) & (tele.ts < ep.end + pad))
    return m


# --------------------------------------------------------------------------------------------
# 1. fleet geometry & working hours
# --------------------------------------------------------------------------------------------
def geometry(tele):
    out = {"rows": len(tele), "rows_per_tractor": tele.groupby("tractor_id").size().to_dict(),
           "speed_values": sorted(tele.speed_mps.unique().tolist())}
    starts, stops = {}, {}
    for tr, g in tele.groupby("tractor_id"):
        d = g.working.astype(int).diff()
        starts[tr] = sorted(set(g.ts[d == 1].dt.strftime("%H:%M:%S")))
        stops[tr] = sorted(set(g.ts[d == -1].dt.strftime("%H:%M:%S")))
    out["work_start_times"], out["work_stop_times"] = starts, stops
    # turn apexes at the north edge (local max of lat while working)
    apex = {}
    for tr, g in tele.groupby("tractor_id"):
        lat, w = g.lat.values, g.working.values
        m = (lat[1:-1] >= lat[:-2]) & (lat[1:-1] > lat[2:]) & w[1:-1] & (lat[1:-1] > 46.906)
        apex[tr] = g.ts.values[np.where(m)[0] + 1]
    out["apex_per_tractor"] = {tr: len(v) for tr, v in apex.items()}
    out["apex_per_day"] = pd.Series(1, index=pd.to_datetime(apex["T1"], utc=True)).groupby(
        lambda x: x.date()).size().astype(int).to_dict()
    gaps = np.diff(apex["T1"]).astype("timedelta64[s]").astype(int)
    out["apex_spacing_s"] = pd.Series(gaps[gaps < 1000]).value_counts().to_dict()
    out["apex_identical_all3"] = bool(set(apex["T1"]) == set(apex["T2"]) == set(apex["T3"]))
    t1 = tele[tele.tractor_id == "T1"]
    out["apex_lat_range"] = (float(t1.lat[t1.ts.isin(pd.to_datetime(apex["T1"], utc=True))].min()),
                             float(tele.lat.max()))
    w = tele[tele.working].pivot(index="ts", columns="tractor_id", values="lat")
    out["max_lat_diff_working_m"] = round(float(max((w.T1 - w.T2).abs().max(), (w.T1 - w.T3).abs().max())
                                                * M_PER_DEG_LAT), 3)
    lon = tele[tele.working].pivot(index="ts", columns="tractor_id", values="lon")
    out["lon_spacing_T2_minus_T1_m"] = round(float((lon.T2 - lon.T1).round(5).mode()[0] * M_PER_DEG_LON), 1)
    out["row_length_m"] = round(float((tele.lat.max() - tele[tele.working].lat.min()) * M_PER_DEG_LAT))
    # east-edge -> west-edge wraps of reported lon (normal operation, not spoofing)
    wraps, wrap_gap = {}, {}
    for tr, g in tele[tele.working].groupby("tractor_id"):
        same_day = g.ts.diff() == pd.Timedelta(seconds=1)
        w_ts = g.ts[(g.lon.diff().abs() > 0.001) & same_day]
        wraps[tr] = len(w_ts)
        wrap_gap[tr] = sorted(set(w_ts.diff().dropna().dt.total_seconds().astype(int)) & set(range(9000, 12000)))
    out["lon_wraps_while_working"] = wraps
    out["lon_wrap_spacing_s (same work day)"] = wrap_gap
    out["field_width_lon_m"] = round(float((tele[tele.working].lon.max() - tele[tele.working].lon.min()) * M_PER_DEG_LON))
    out["parked_xte_max"] = float(tele.loc[~tele.working, "axte"].max())
    q = tele[tele.working & tele.lon_dev_m.notna() & (tele.ts >= QUIET[0]) & (tele.ts < QUIET[1])]
    out["quiet_lon_dev_abs_max_m"] = round(float(q.lon_dev_m.abs().max()), 3)
    return out, apex


def quiet_distributions(tele):
    q = tele[(tele.ts >= QUIET[0]) & (tele.ts < QUIET[1]) & tele.working]
    res = {}
    for name, g in [("away", q[~q.headland]), ("headland", q[q.headland])]:
        res[name] = {
            "n": len(g),
            "mean_cn0_min": g.mean_cn0_dbhz.min(), "mean_cn0_median": g.mean_cn0_dbhz.median(),
            "min_cn0_q001": round(g.min_cn0_dbhz.quantile(0.001), 2), "min_cn0_median": g.min_cn0_dbhz.median(),
            "sats_values": sorted(g.num_sats.unique().tolist()),
            "fix_values": sorted(g.fix_state.unique().tolist()),
            "link_down": int((~g.base_link_ok).sum()),
            "axte_median": round(g.axte.median(), 3), "axte_max": g.axte.max(),
        }
    return res


# --------------------------------------------------------------------------------------------
# 2. per-episode before / during / after table
# --------------------------------------------------------------------------------------------
def stats(g):
    if g.empty:
        return {}
    away = g[~g.headland]
    return {
        "n": len(g),
        "cn0": round(g.mean_cn0_dbhz.median(), 1),
        "cn0_p05": round(g.mean_cn0_dbhz.quantile(0.05), 1),
        "min": round(g.min_cn0_dbhz.median(), 1),
        "min<35%": round(100 * (away.min_cn0_dbhz < 35).mean(), 1),
        "sats": round(g.num_sats.mean(), 1),
        "sats<11%": round(100 * (g.num_sats < 11).mean(), 1),
        "fixed%": round(100 * (g.fix_state == "RTK_FIXED").mean(), 1),
        "nofix%": round(100 * (g.fix_state == "NO_FIX").mean(), 1),
        "link_dn%": round(100 * (~g.base_link_ok).mean(), 1),
        "|xte|med": round(g.axte.median(), 3),
        "|xte|p95": round(g.axte.quantile(0.95), 2),
        "|lon_dev|p95": round(g.lon_dev_m.abs().quantile(0.95), 2),
    }


def episode_table(tele, labels, pad=pd.Timedelta(minutes=30)):
    rows = []
    for ep in labels.itertuples():
        p = pad if ep.end - ep.start > pd.Timedelta(minutes=5) else pd.Timedelta(minutes=2)
        for tr in TRACTORS:
            for phase, a, b in [("before", ep.start - p, ep.start), ("during", ep.start, ep.end),
                                ("after", ep.end, ep.end + p)]:
                rows.append({"id": ep.id, "cls": ep.cls, "tractor": tr, "labeled": tr in ep.tractors,
                             "phase": phase, **stats(window(tele, tr, a, b))})
    return pd.DataFrame(rows)


def changepoint(series, centre, half=pd.Timedelta(minutes=10), k=60):
    """Second within centre +- half where |mean(next k s) - mean(previous k s)| is largest.

    Returns (offset_seconds_from_centre, step_size). A simple, explainable step locator.
    """
    s = series.loc[centre - half - pd.Timedelta(seconds=k): centre + half + pd.Timedelta(seconds=k)]
    if len(s) < 2 * k + 1:
        return None, None
    v = s.values.astype(float)
    c = np.r_[0, np.cumsum(v)]
    i = np.arange(k, len(v) - k)
    before = (c[i] - c[i - k]) / k
    after = (c[i + k] - c[i]) / k
    step = after - before
    j = i[np.argmax(np.abs(step))]
    return int((s.index[j] - centre).total_seconds()), round(float(step[np.argmax(np.abs(step))]), 2)


def boundary_alignment(tele, labels):
    """How far the data's change is from the label start/end, per labeled tractor (long episodes).

    Level shifts (storm, jamming): changepoint of mean C/N0 (largest 60 s before/after step).
    Ramps and sparse faults (spoofing, equipment): first and last second showing the symptom inside
    label +- 10 min. Spoof symptom: reported position off lockstep by > 0.1 m. Equipment symptom:
    link down, or min C/N0 < 35 away from headland, or < 11 sats.
    """
    rows = []
    for ep in labels.itertuples():
        if ep.cls == "MULTIPATH":
            continue
        for tr in ep.tractors:
            g = tele[tele.tractor_id == tr].set_index("ts")
            if ep.cls in ("SPACE_WEATHER", "JAMMING"):
                s_off, s_step = changepoint(g.mean_cn0_dbhz, ep.start)
                e_off, e_step = changepoint(g.mean_cn0_dbhz, ep.end)
                rows.append({"id": ep.id, "cls": ep.cls, "tractor": tr, "method": "C/N0 changepoint",
                             "start_offset_s": s_off, "start_step_db": s_step,
                             "end_offset_s": e_off, "end_step_db": e_step})
                continue
            if ep.cls == "SPOOFING":
                sym = g.lon_dev_m.abs() > 0.1
            else:
                sym = (~g.base_link_ok) | ((g.min_cn0_dbhz < 35) & ~g.headland) | (g.num_sats < 11)
            pad = pd.Timedelta(minutes=10)
            hit = sym.loc[ep.start - pad: ep.end + pad]
            hit = hit[hit].index
            rows.append({"id": ep.id, "cls": ep.cls, "tractor": tr, "method": "first/last symptom",
                         "start_offset_s": int((hit[0] - ep.start).total_seconds()),
                         "end_offset_s": int((hit[-1] + pd.Timedelta(seconds=1) - ep.end).total_seconds())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# 3. SPACE_WEATHER
# --------------------------------------------------------------------------------------------
def storm_slots(tele, sda):
    """Fleet 15-min medians next to the S4 value for the same slot, for 05-04 and 05-05 (dev)."""
    t = tele[tele.ts < EVAL_START].set_index("ts")
    fleet = t.resample("15min")[["mean_cn0_dbhz", "min_cn0_dbhz", "num_sats", "fix_rank"]].median()
    spread = t.groupby("tractor_id").resample("15min").mean_cn0_dbhz.median().unstack(0)
    fleet["cn0_spread_between_tractors"] = (spread.max(axis=1) - spread.min(axis=1)).round(2)
    s4 = sda["s4"].set_index("ts").s4
    fleet["s4"] = s4.reindex(fleet.index)
    fleet["working"] = t.groupby(pd.Grouper(freq="15min")).working.mean() > 0.5
    return fleet


def storm_levels(tele):
    """Contiguous fleet-wide degraded stretches (fleet median mean C/N0 below 42.5 for >= 5 min).

    Excludes jamming by requiring all three tractors within 1 dB of each other (storms are uniform,
    jamming depends on distance to the source)."""
    t = tele[tele.ts < EVAL_START].set_index("ts")
    m = t.groupby("tractor_id").resample("1min").mean_cn0_dbhz.median().unstack(0)
    uniform = (m.max(axis=1) - m.min(axis=1)) < 1.0
    low = (m.median(axis=1) < 42.5) & uniform
    out = []
    lvl = m.median(axis=1).round(0)
    for s, e in runs(low.values):
        # split one degraded stretch into constant-level pieces (the storm is a staircase):
        # a new piece starts when the 1-min level moves >= 2 dB away from the current piece's first minute
        start = s
        for i in range(s + 1, e + 1):
            if i == e or abs(lvl.iloc[i] - lvl.iloc[start]) >= 2:
                if i - start >= 5:
                    g = t[(t.index >= m.index[start]) & (t.index < m.index[i - 1] + pd.Timedelta(minutes=1))]
                    out.append({"start": m.index[start], "end": m.index[i - 1] + pd.Timedelta(minutes=1),
                                "working_pct": round(100 * g.working.mean()),
                                "cn0": round(g.mean_cn0_dbhz.median(), 1), "min": round(g.min_cn0_dbhz.median(), 1),
                                "sats": round(g.num_sats.median(), 1),
                                "fix": g.fix_state.value_counts().index[0],
                                "|xte|med_work": round(g[g.working].axte.median(), 2) if g.working.any() else None,
                                "|xte|p95_work": round(g[g.working].axte.quantile(.95), 2) if g.working.any() else None,
                                "lon_dev_p95": round(g[g.working].lon_dev_m.abs().quantile(.95), 3)
                                if g.working.any() else None})
                start = i
    return pd.DataFrame(out)


def storm_s4_summary(slots):
    """S4 vs fleet C/N0 per 15-min slot. 'Storm slot' = degraded AND uniform across tractors (< 1 dB
    spread), which separates storm from jamming (jamming hits tractors differently by distance).
    Slots containing a jamming episode inside the storm (TRN-05) are not uniform-shifted from the storm
    level but are uniform, so they are dropped by excluding slots below 25 dB fleet C/N0."""
    deg = slots.mean_cn0_dbhz < 43
    uniform = slots.cn0_spread_between_tractors < 1.0
    storm = deg & uniform & (slots.mean_cn0_dbhz > 25)
    jam = deg & ~uniform
    return {
        "storm slots (degraded, uniform)": int(storm.sum()),
        "S4 range in storm slots": (round(slots.s4[storm].min(), 3), round(slots.s4[storm].max(), 3)),
        "S4 range in jamming slots (non-uniform)": (round(slots.s4[jam].min(), 3), round(slots.s4[jam].max(), 3)),
        "max S4 in clean slots": round(slots.s4[~deg].max(), 3),
        "max S4 in clean slots outside 05-05 07:15-09:45": round(slots.s4[~deg & ~((slots.index >= "2026-05-05T07:15Z")
                                                                             & (slots.index < "2026-05-05T09:45Z"))].max(), 3),
        "clean slots with S4 > 0.15": int(((slots.s4 > 0.15) & ~deg).sum()),
        "corr(S4, fleet C/N0) over storm slots": round(float(np.corrcoef(slots.s4[storm],
                                                                          slots.mean_cn0_dbhz[storm])[0, 1]), 3),
        "max between-tractor spread in storm slots (dB)": round(slots.cn0_spread_between_tractors[storm].max(), 2),
        "linear fit fleet C/N0 = a + b*S4 over storm slots (a, b)": tuple(round(float(c), 1) for c in
                                                                         np.polyfit(slots.s4[storm],
                                                                                    slots.mean_cn0_dbhz[storm], 1)[::-1]),
        "storm slot onsets = S4 rise in the same slot": bool((slots.s4[storm] > 0.3).all()),
    }


def storm_blocks_vs_kp(tele, sda):
    """Contiguous fleet-uniform degraded blocks (from storm_levels) and the Kp slots around them."""
    lv = storm_levels(tele)
    blocks = merge_intervals(list(zip(lv.start, lv.end)), pd.Timedelta(minutes=40))  # bridge TRN-05 gap
    kp = sda["kp"].set_index("ts").kp
    rows = []
    for s, e in blocks:
        rows.append({"start": s, "end": e,
                     "kp_slot_at_start": f"{kp[kp.index <= s].index[-1]:%HZ}={kp[kp.index <= s].iloc[-1]}",
                     "kp_slot_before_end": f"{kp[kp.index < e].index[-1]:%HZ}={kp[kp.index < e].iloc[-1]}",
                     "kp_at_end": f"{kp[kp.index <= e].index[-1]:%HZ}={kp[kp.index <= e].iloc[-1]}"})
    return pd.DataFrame(rows)


def label_grid(labels):
    """Do non-multipath label boundaries fall on whole minutes that are multiples of 6?"""
    lab = labels[labels.cls != "MULTIPATH"]
    bounds = pd.concat([lab.start, lab.end])
    on_grid = (bounds.dt.second == 0) & (bounds.dt.minute % 6 == 0)
    durations = ((lab.end - lab.start).dt.total_seconds() / 60).astype(int)
    return {"boundaries": len(bounds), "on_6min_grid": int(on_grid.sum()),
            "durations_min": sorted(durations.tolist()),
            "durations_multiple_of_6": bool((durations % 6 == 0).all())}


def kp_at(sda, t):
    k = sda["kp"][sda["kp"].ts <= t]
    return float(k.kp.iloc[-1]) if len(k) else None


# --------------------------------------------------------------------------------------------
# 4. JAMMING
# --------------------------------------------------------------------------------------------
def jam_distance_fit(tele, start, end):
    """Grid-search a single fixed source: mean C/N0 = a - b*log10(distance). 10-s means, all tractors.

    If C/N0 is explained by distance to one point, the oscillation is the tractors driving toward and
    away from it. Returns best location, slope (dB/decade; free-space path loss is 20) and RMSE.
    """
    w = tele[(tele.ts >= start) & (tele.ts < end)].set_index("ts")
    y = w.groupby("tractor_id")[["lat", "lon", "mean_cn0_dbhz"]].resample("10s").mean().reset_index()
    best = None
    for la in np.linspace(46.88, 46.93, 51):
        for lo in np.linspace(-97.24, -97.16, 81):
            d = np.hypot((y.lat - la) * M_PER_DEG_LAT, (y.lon - lo) * M_PER_DEG_LON) + 30  # +30 m avoids log(0)
            X = np.c_[np.ones(len(d)), np.log10(d)]
            coef, *_ = np.linalg.lstsq(X, y.mean_cn0_dbhz.values, rcond=None)
            rmse = float(np.sqrt(((X @ coef - y.mean_cn0_dbhz.values) ** 2).mean()))
            if best is None or rmse < best["rmse"]:
                best = {"lat": round(la, 4), "lon": round(lo, 4), "slope_db_per_decade": round(coef[1], 1),
                        "rmse": round(rmse, 2), "cn0_std": round(float(y.mean_cn0_dbhz.std()), 2)}
    return best


def dominant_period_s(series_1s):
    """Period (s) of the strongest oscillation in a 1 Hz series between 2 and 20 min (via autocorrelation)."""
    v = series_1s.values - series_1s.values.mean()
    ac = np.correlate(v, v, mode="full")[len(v) - 1:]
    lo, hi = 120, min(1200, len(v) // 2)
    return int(lo + np.argmax(ac[lo:hi]))


def jamming(tele, sda, labels):
    rows = []
    for ep in labels[labels.cls == "JAMMING"].itertuples():
        pre = (ep.start - pd.Timedelta(minutes=10), ep.start)
        rfi = sda["rfi"][(sda["rfi"].ts >= ep.start) & (sda["rfi"].ts < ep.end + pd.Timedelta(minutes=40))]
        for tr in TRACTORS:
            b = window(tele, tr, *pre)
            d = window(tele, tr, ep.start, ep.end)
            r = {"id": ep.id, "tractor": tr, "labeled": tr in ep.tractors,
                 "cn0_before": round(b.mean_cn0_dbhz.median(), 1), "cn0_during": round(d.mean_cn0_dbhz.median(), 1),
                 "drop_db": round(b.mean_cn0_dbhz.median() - d.mean_cn0_dbhz.median(), 1),
                 "cn0_range": f"{d.mean_cn0_dbhz.quantile(.05):.1f}-{d.mean_cn0_dbhz.quantile(.95):.1f}",
                 "sats_min/med": f"{d.num_sats.min()}/{d.num_sats.median():.0f}",
                 "nofix%": round(100 * (d.fix_state == "NO_FIX").mean(), 1),
                 "fixed%": round(100 * (d.fix_state == "RTK_FIXED").mean(), 1),
                 "|xte|p95": round(d.axte.quantile(.95), 2),
                 "|lon_dev|max": round(d.lon_dev_m.abs().max(), 2),
                 "period_s": dominant_period_s(d.set_index("ts").mean_cn0_dbhz),
                 "rfi_delay_min": round((rfi.ts.iloc[0] - ep.start).total_seconds() / 60, 1) if len(rfi) else None}
            rows.append(r)
    fits = {ep.id: jam_distance_fit(tele, ep.start, ep.end) for ep in labels[labels.cls == "JAMMING"].itertuples()}
    out = pd.DataFrame(rows)
    # mean distance of each tractor from the fitted source during the episode
    dist = []
    for r in out.itertuples():
        ep = labels.set_index("id").loc[r.id]
        g = window(tele, r.tractor, ep.start, ep.end)
        f = fits[r.id]
        dist.append(round(float(np.hypot((g.lat - f["lat"]) * M_PER_DEG_LAT, (g.lon - f["lon"]) * M_PER_DEG_LON).mean())))
    out["mean_dist_to_fit_m"] = dist
    return out, fits


def distance_km(lat1, lon1, lat2, lon2):
    return float(np.hypot((lat1 - lat2) * M_PER_DEG_LAT, (lon1 - lon2) * M_PER_DEG_LON) / 1000)


# --------------------------------------------------------------------------------------------
# 5. SPOOFING
# --------------------------------------------------------------------------------------------
def spoofing(tele, labels):
    rows = []
    for ep in labels[labels.cls == "SPOOFING"].itertuples():
        tr = ep.tractors[0]
        g = tele[(tele.tractor_id == tr) & (tele.ts >= ep.start) & (tele.ts < ep.end)].set_index("ts")
        x = g.cross_track_error_m
        # plateau = first second the XTE jumps by > 1 m in one second (ramp -> plateau)
        jump = x.diff()
        plateau_start = jump[jump > 1.0].index[0]
        ramp = x.loc[:plateau_start - pd.Timedelta(seconds=1)]
        secs = (ramp.index - ep.start).total_seconds().values
        slope, intercept = np.polyfit(secs, ramp.values, 1)
        others = tele[(tele.tractor_id != tr) & (tele.ts >= ep.start) & (tele.ts < ep.end)]
        lon_ok = g.lon_dev_m.notna()
        after = tele[(tele.tractor_id == tr) & (tele.ts >= ep.end) & (tele.ts < ep.end + pd.Timedelta(seconds=5))]
        first_dev = g.index[(g.lon_dev_m.abs() > 0.1).fillna(False).values]
        rows.append({
            "id": ep.id, "tractor": tr,
            "ramp_m_per_min": round(slope * 60, 3), "ramp_intercept_m": round(intercept, 3),
            "ramp_min": round((plateau_start - ep.start).total_seconds() / 60, 1),
            "ramp_end_xte": round(float(ramp.iloc[-30:].mean()), 2),
            "plateau_m": round(float(x.loc[plateau_start:].median()), 2),
            "plateau_min": round((ep.end - plateau_start).total_seconds() / 60, 1),
            "xte_sign": "+" if x.median() > 0 else "-",
            "xte_first_5s_after_end": round(float(after.axte.max()), 3),
            "lon_dev_vs_xte_maxdiff_m": round(float((g.lon_dev_m[lon_ok] - x[lon_ok]).abs().max()), 3),
            "lon_dev_seconds_compared": int(lon_ok.sum()),
            "first_lon_dev>0.1m_s_after_start": int((first_dev[0] - ep.start).total_seconds()) if len(first_dev) else None,
            "cn0_minus_others": round(g.mean_cn0_dbhz.mean() - others.mean_cn0_dbhz.mean(), 2),
            "min_cn0_minus_others": round(g.min_cn0_dbhz.mean() - others.min_cn0_dbhz.mean(), 2),
            "sats_minus_others": round(g.num_sats.mean() - others.num_sats.mean(), 2),
            "fixed%": round(100 * (g.fix_state == "RTK_FIXED").mean(), 1),
            "fixed%_others": round(100 * (others.fix_state == "RTK_FIXED").mean(), 1),
            "link_down": int((~g.base_link_ok).sum()),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# 6. EQUIPMENT_FAULT
# --------------------------------------------------------------------------------------------
def equipment(tele, labels):
    rows = []
    for ep in labels[labels.cls == "EQUIPMENT_FAULT"].itertuples():
        tr = ep.tractors[0]
        g = tele[(tele.tractor_id == tr) & (tele.ts >= ep.start) & (tele.ts < ep.end)]
        # exclude any overlapping spoof window on the same tractor for the XTE numbers
        sp = labels[(labels.cls == "SPOOFING") & labels.tractors.apply(lambda x: tr in x)]
        clean = g
        for s in sp.itertuples():
            clean = clean[(clean.ts < s.start) | (clean.ts >= s.end)]
        others = tele[(tele.tractor_id != tr) & (tele.ts >= ep.start) & (tele.ts < ep.end)]
        own_before = window(tele, tr, ep.start - pd.Timedelta(hours=1), ep.start)
        away = g[~g.headland]
        drop = (away.min_cn0_dbhz < 35)
        link_runs = runs(~g.base_link_ok.values)
        drop_runs = runs(drop.values)
        rows.append({
            "id": ep.id, "tractor": tr,
            "min<35% (away from headland)": round(100 * drop.mean(), 1),
            "min<35% others": round(100 * (others[~others.headland].min_cn0_dbhz < 35).mean(), 2),
            "min<35% own hour before": round(100 * (own_before[~own_before.headland].min_cn0_dbhz < 35).mean(), 2),
            "dropout_runs": len(drop_runs),
            "dropout_run_len_max_s": max((e - s for s, e in drop_runs), default=0),
            "sats<11%": round(100 * (g.num_sats < 11).mean(), 1),
            "sats<11% others": round(100 * (others.num_sats < 11).mean(), 1),
            "mean_cn0_p05": round(g.mean_cn0_dbhz.quantile(.05), 1),
            "mean_cn0_p05_others": round(others.mean_cn0_dbhz.quantile(.05), 1),
            "link_down%": round(100 * (~g.base_link_ok).mean(), 1),
            "link_runs": len(link_runs),
            "link_run_len_median_s": float(np.median([e - s for s, e in link_runs])) if link_runs else 0,
            "link_run_len_max_s": max((e - s for s, e in link_runs), default=0),
            "fix_when_link_down": sorted(g[~g.base_link_ok].fix_state.unique().tolist()),
            "fix_when_link_up": sorted(g[g.base_link_ok].fix_state.unique().tolist()),
            "fixed%": round(100 * (g.fix_state == "RTK_FIXED").mean(), 1),
            "|xte|med (no spoof)": round(clean.axte.median(), 3),
            "|xte|p95 (no spoof)": round(clean.axte.quantile(.95), 2),
            "|xte|p95 link down": round(g[~g.base_link_ok].axte.quantile(.95), 2) if (~g.base_link_ok).any() else None,
            "first_symptom_s_after_start": _first_symptom(g, ep.start),
            "last_symptom_s_before_end": _last_symptom(g, ep.end),
        })
    return pd.DataFrame(rows)


def _symptom(g):
    return (~g.base_link_ok) | ((g.min_cn0_dbhz < 35) & ~g.headland) | (g.num_sats < 11)


def _first_symptom(g, start):
    s = g.ts[_symptom(g).values]
    return int((s.iloc[0] - start).total_seconds()) if len(s) else None


def _last_symptom(g, end):
    s = g.ts[_symptom(g).values]
    return int((end - s.iloc[-1]).total_seconds()) if len(s) else None


# --------------------------------------------------------------------------------------------
# 7. MULTIPATH
# --------------------------------------------------------------------------------------------
def headland_passes(tele, apex, labels):
    """One row per (tractor, north-edge pass): dip depth of min C/N0 and context."""
    rows = []
    jam = labels[labels.cls == "JAMMING"]
    storm = labels[labels.cls == "SPACE_WEATHER"]
    for tr in TRACTORS:
        g = tele[tele.tractor_id == tr].reset_index(drop=True)
        pos = pd.Index(g.ts)
        for a in pd.to_datetime(apex[tr], utc=True):
            i = pos.get_loc(a)
            win = g.iloc[i - 40:i + 41]
            inn, out = win[win.headland], win[~win.headland]
            ctx = ("jam" if ((jam.start <= a) & (jam.end > a)).any()
                   else "storm" if ((storm.start <= a) & (storm.end > a)).any() else "other")
            rows.append({"tractor": tr, "apex": a, "dip_start": inn.ts.min(), "dip_end": inn.ts.max() + pd.Timedelta(seconds=1),
                         "dip_s": len(inn), "min_drop_db": out.min_cn0_dbhz.median() - inn.min_cn0_dbhz.median(),
                         "mean_drop_db": out.mean_cn0_dbhz.median() - inn.mean_cn0_dbhz.median(),
                         "xte_in": inn.axte.median(), "xte_out": out.axte.median(), "context": ctx})
    return pd.DataFrame(rows)


def multipath(passes, labels):
    out = {"passes": len(passes), "dip_s_counts": passes.dip_s.value_counts().to_dict(),
           "min_drop_by_context": passes.groupby("context").min_drop_db.describe()[["count", "min", "50%", "max"]].round(1),
           "mean_drop_median": round(passes.mean_drop_db.median(), 2),
           "xte_in_median": round(passes.xte_in.median(), 3), "xte_out_median": round(passes.xte_out.median(), 3)}
    lab = []
    for ep in labels[labels.cls == "MULTIPATH"].itertuples():
        tr = ep.tractors[0]
        p = passes[(passes.tractor == tr) & (passes.dip_start <= ep.end) & (passes.dip_end >= ep.start)].iloc[0]
        same_day = passes[(passes.tractor == tr) & (passes.apex.dt.date == p.apex.date()) & (passes.context == p.context)]
        prev_eps = labels[(labels.cls != "MULTIPATH")]
        # nearest boundary (start or end) of another labeled episode before this pass
        bounds = pd.concat([prev_eps.start, prev_eps.end])
        bounds = bounds[bounds <= ep.start]
        lab.append({"id": ep.id, "tractor": tr, "label": f"{fmt(ep.start)}-{ep.end:%H:%M:%S}",
                    "dip": f"{fmt(p.dip_start)}-{p.dip_end:%H:%M:%S}",
                    "start_err_s": int((p.dip_start - ep.start).total_seconds()),
                    "end_err_s": int((p.dip_end - ep.end).total_seconds()),
                    "context": p.context, "dip_db": round(p.min_drop_db, 1),
                    "dip_rank_same_day_ctx": f"{int((same_day.min_drop_db > p.min_drop_db).sum()) + 1}/{len(same_day)}",
                    "passes_same_tractor_earlier_that_day": int(((passes.tractor == tr) & (passes.apex < p.apex)
                                                                 & (passes.apex.dt.date == p.apex.date())).sum()),
                    "passes_between_last_boundary_and_this": int(((passes.tractor == tr) & (passes.apex < p.apex)
                                                                  & (passes.dip_start >= bounds.max())).sum()),
                    "min_since_last_episode_boundary": round((ep.start - bounds.max()).total_seconds() / 60, 1)})
    out["labels"] = pd.DataFrame(lab)
    return out


# --------------------------------------------------------------------------------------------
# 8/9. unusual minutes (dev: outside labels; eval: shapes only)
# --------------------------------------------------------------------------------------------
def flag_seconds(tele):
    f = pd.DataFrame(index=tele.index)
    f["cn0_low"] = tele.mean_cn0_dbhz < 42.5
    f["min_low"] = (tele.min_cn0_dbhz < 35) & ~tele.headland
    f["sats_low"] = tele.num_sats < 11
    f["fix_bad"] = tele.fix_state != "RTK_FIXED"
    f["link_down"] = ~tele.base_link_ok
    f["xte_high"] = tele.working & (tele.axte > 0.5)
    f["pos_off"] = tele.working & (tele.lon_dev_m.abs() > 0.5)
    return f


# Quiet working stretches in dev OTHER than the QUIET window the flag thresholds were read from.
# Checking the flags here is an out-of-sample test that the thresholds do not fire on ordinary data.
HELD_OUT_QUIET = [("2026-05-04T11:00Z", "2026-05-04T12:00Z"), ("2026-05-04T14:02Z", "2026-05-04T16:28Z"),
                  ("2026-05-04T21:38Z", "2026-05-05T00:28Z"), ("2026-05-05T16:00Z", "2026-05-05T19:00Z"),
                  ("2026-05-05T22:08Z", "2026-05-06T00:00Z")]


def held_out_quiet_flags(tele, labels, flags):
    """Count flagged seconds in HELD_OUT_QUIET (working, >= 2 min away from any label on that tractor)."""
    near_label = label_mask(tele, labels, pad=pd.Timedelta(minutes=2))
    m = pd.Series(False, index=tele.index)
    for a, b in HELD_OUT_QUIET:
        m |= (tele.ts >= ts(a)) & (tele.ts < ts(b))
    m &= tele.working & ~near_label
    return {"seconds": int(m.sum()), "flagged": {k: int(v) for k, v in flags[m].sum().items()}}


def unusual_runs(tele, flags, start, end, min_flagged=3, gap=pd.Timedelta(minutes=2)):
    """Per tractor: minutes with >= min_flagged flagged seconds, merged across gaps <= `gap`.

    Returns one row per run with which flags fired, whether the tractor was working, and the
    run's fleet context. Minutes rather than seconds so isolated noise seconds do not form runs.
    """
    sel = (tele.ts >= start) & (tele.ts < end)
    t, f = tele[sel], flags[sel]
    rows = []
    for tr in TRACTORS:
        m = t.tractor_id == tr
        g = f[m].set_index(t.ts[m])
        any_flag = g.any(axis=1)
        per_min = any_flag.resample("1min").sum()
        bad = per_min[per_min >= min_flagged].index
        ivs = merge_intervals([(b, b + pd.Timedelta(minutes=1)) for b in bad], gap)
        for s, e in ivs:
            seg = g.loc[s:e - pd.Timedelta(seconds=1)]
            tt = t[m].set_index("ts").loc[s:e - pd.Timedelta(seconds=1)]
            rows.append({"tractor": tr, "start": s, "end": e, "minutes": int((e - s).total_seconds() // 60),
                         "working_pct": round(100 * tt.working.mean()),
                         "flags": ",".join(c for c in g.columns if seg[c].mean() > 0.05),
                         "cn0_med": round(tt.mean_cn0_dbhz.median(), 1),
                         "axte_p95": round(tt.axte.quantile(.95), 2),
                         "lon_dev_max": round(tt.lon_dev_m.abs().max(), 2)})
    return pd.DataFrame(rows)


def outside_labels(runs_df, labels, tolerance=pd.Timedelta(minutes=1)):
    """Annotate each run with the labeled episode(s) on the same tractor that overlap it, and the
    minutes of the run NOT covered by any such label."""
    out = []
    for r in runs_df.to_dict("records"):
        cover = labels[labels.tractors.apply(lambda x: r["tractor"] in x)
                       & (labels.start < r["end"] + tolerance) & (labels.end > r["start"] - tolerance)]
        uncovered = r["end"] - r["start"]
        if len(cover):
            covered = merge_intervals([(max(c.start, r["start"]), min(c.end, r["end"])) for c in cover.itertuples()],
                                      pd.Timedelta(0))
            uncovered -= sum((e - s for s, e in covered), pd.Timedelta(0))
        out.append({**r, "labels": ",".join(cover.id), "uncovered_min": round(uncovered.total_seconds() / 60, 1)})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------------------------
def section(title):
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--plots", action="store_true", help="write figures to reports/findings/")
    args = ap.parse_args()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 500)

    tele, sda, labels = load_all(args.data)
    dev = tele[tele.ts < EVAL_START]

    section("1. Fleet geometry and working hours")
    geo, apex = geometry(tele)
    for k, v in geo.items():
        print(f"{k:28s} {v}")
    q = quiet_distributions(tele)
    print("\nQuiet reference window", fmt(QUIET[0]), "->", fmt(QUIET[1]), "(working seconds):")
    print(pd.DataFrame(q).to_string())

    section("2. Per-episode before / during / after (working seconds; * = labeled tractor)")
    et = episode_table(tele, labels)
    et["tractor"] = et.tractor + np.where(et.labeled, "*", " ")
    print(et.drop(columns="labeled").to_string(index=False))
    print("\nLabel boundary grid:", label_grid(labels))
    section("2b. Boundary alignment: data step vs label (seconds; + = data later than label)")
    print(boundary_alignment(tele, labels).to_string(index=False))

    section("3. SPACE_WEATHER")
    print("Kp (3-hourly):", {f"{r.ts:%m-%d %HZ}": r.kp for r in sda["kp"].itertuples() if r.ts < EVAL_START})
    slots = storm_slots(tele, sda)
    print("\nFleet 15-min medians vs S4 for every slot where S4 > 0.10 or fleet C/N0 < 43:")
    sel = (slots.s4 > 0.10) | (slots.mean_cn0_dbhz < 43)
    print(slots[sel].round(3).to_string())
    s = storm_s4_summary(slots)
    for k, v in s.items():
        print(f"{k:55s} {v}")
    lv = storm_levels(tele)
    print("\nConstant-level storm stretches (fleet-uniform, >=5 min):")
    print(lv.to_string(index=False))
    print("\nStorm blocks vs Kp (3-hourly slots):")
    print(storm_blocks_vs_kp(tele, sda).to_string(index=False))
    for ep in labels[labels.cls == "SPACE_WEATHER"].itertuples():
        print(f"{ep.id}: label {fmt(ep.start)}->{ep.end:%H:%M}; Kp at start {kp_at(sda, ep.start)}, "
              f"Kp slot before end {kp_at(sda, ep.end - pd.Timedelta(seconds=1))}, Kp at end {kp_at(sda, ep.end)}")

    section("4. JAMMING")
    jt, fits = jamming(tele, sda, labels)
    print(jt.to_string(index=False))
    print("\nSingle fixed-source fit, C/N0 = a - b*log10(distance):")
    rfi = sda["rfi"].set_index("id")
    for k, v in fits.items():
        near = {r: round(distance_km(v["lat"], v["lon"], rfi.loc[r, "centerLat"], rfi.loc[r, "centerLon"]), 2)
                for r in rfi.index}
        print(f"  {k}: {v}  km to RFI centres: {near}")
    print("\nRFI reports:")
    print(sda["rfi"][["id", "ts", "confidence", "centerLat", "centerLon", "semiMajor_km", "semiMinor_km"]].to_string(index=False))

    section("5. SPOOFING")
    print(spoofing(tele, labels).T.to_string())

    section("6. EQUIPMENT_FAULT")
    print(equipment(tele, labels).T.to_string())

    section("7. MULTIPATH (north-headland passes)")
    passes = headland_passes(tele, apex, labels)
    mp = multipath(passes[passes.apex < EVAL_START], labels)
    print("passes (dev):", mp["passes"], " dip length (s):", mp["dip_s_counts"])
    print("min C/N0 drop by context (dB):\n", mp["min_drop_by_context"].to_string())
    print("median mean-C/N0 change in dip:", mp["mean_drop_median"], "dB; |XTE| median in dip",
          mp["xte_in_median"], "vs out", mp["xte_out_median"])
    print(mp["labels"].to_string(index=False))

    section("8. Unusual minutes in DEV outside the labels")
    for k, v in FLAGS.items():
        print(f"  {k:10s} {v}")
    flags = flag_seconds(tele)
    qmask = (tele.ts >= QUIET[0]) & (tele.ts < QUIET[1])
    print("Flagged-second rate in QUIET window (circular: thresholds were set from it):",
          flags[qmask].mean().round(5).to_dict())
    ho = held_out_quiet_flags(tele, labels, flags)
    print(f"Held-out quiet working seconds (not used to set thresholds): {ho['seconds']}, "
          f"flagged: {ho['flagged']}")
    dr = outside_labels(unusual_runs(tele, flags, DEV_START, EVAL_START), labels)
    dr["start"], dr["end"] = dr.start.map(fmt), dr.end.dt.strftime("%H:%M")
    print("\nAll runs (dev):")
    print(dr.to_string(index=False))
    print("\nRuns with >= 1 min not covered by a label on that tractor:")
    print(dr[dr.uncovered_min >= 1].to_string(index=False))

    section("9. EVAL window shapes (unlabeled, read-only; no thresholds derived here)")
    print("Kp:", {f"{r.ts:%HZ}": r.kp for r in sda["kp"].itertuples() if r.ts >= EVAL_START})
    print("Dst min:", sda["dst"][sda["dst"].ts >= EVAL_START].dst_nT.min(), "nT")
    s4e = sda["s4"][sda["s4"].ts >= EVAL_START].set_index("ts").s4
    print("S4 slots > 0.15:", {f"{k:%H:%M}": v for k, v in s4e[s4e > 0.15].items()})
    print("RFI in eval:", sda["rfi"][sda["rfi"].ts >= EVAL_START][["id", "ts", "confidence"]].to_dict("records"))
    er = unusual_runs(tele, flags, EVAL_START, EVAL_END)
    shape = pd.DataFrame([{"tractor": r.tractor, "start": fmt(r.start), "end": f"{r.end:%H:%M}",
                           **stats(window(tele, r.tractor, r.start, r.end))} for r in er.itertuples()])
    er["start"], er["end"] = er.start.map(fmt), er.end.dt.strftime("%H:%M")
    print(er.to_string(index=False))
    print("\nSame runs, per-second shape statistics (compare with section 2 tables):")
    print(shape.to_string(index=False))
    print("\nLon wraps inside degraded runs (signal level before -> after the wrap, 5-min medians):")
    for name, a, b in [("dev TRN-03", "2026-05-04T16:30Z", "2026-05-04T17:12Z"),
                       ("eval 02:12", "2026-05-06T02:12Z", "2026-05-06T02:54Z"),
                       ("eval 18:30", "2026-05-06T18:30Z", "2026-05-06T19:00Z")]:
        for tr in TRACTORS:
            g = window(tele, tr, ts(a), ts(b))
            jumps = g.ts[(g.lon.diff().abs() > 0.001).values]
            for j in jumps:
                pre = g[(g.ts >= j - pd.Timedelta(minutes=5)) & (g.ts < j)].mean_cn0_dbhz.median()
                post = g[(g.ts >= j) & (g.ts < j + pd.Timedelta(minutes=5))].mean_cn0_dbhz.median()
                low = g[g.ts < j].mean_cn0_dbhz.min()
                print(f"  {name} {tr}: wrap at {fmt(j)}  C/N0 {pre:.1f} -> {post:.1f} (min before wrap {low:.1f})")
    for a, b in [("2026-05-06T02:00Z", "2026-05-06T03:00Z"), ("2026-05-06T18:20Z", "2026-05-06T19:10Z")]:
        jt = tele[(tele.ts >= a) & (tele.ts < b)]
        per = {tr: dominant_period_s(jt[(jt.tractor_id == tr) & (jt.mean_cn0_dbhz < 42.5)].set_index("ts")
                                     .mean_cn0_dbhz) for tr in TRACTORS}
        print(f"oscillation period of mean C/N0 while degraded, {a[11:16]}-{b[11:16]}: {per}")
    ep_ = passes[passes.apex >= EVAL_START]
    print(f"\nEval headland passes: {len(ep_)}; dip length {ep_.dip_s.value_counts().to_dict()}; "
          f"min C/N0 drop median {ep_.min_drop_db.median():.1f} dB (min {ep_.min_drop_db.min():.1f})")
    print("Eval spoof-like stretches (pos_off seconds per tractor):",
          flags[tele.ts >= EVAL_START].groupby(tele.tractor_id).pos_off.sum().to_dict())

    if args.plots:
        make_plots(tele, sda, labels, flags, passes)


# --------------------------------------------------------------------------------------------
# plots (optional)
# --------------------------------------------------------------------------------------------
def make_plots(tele, sda, labels, flags, passes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = ROOT / "reports" / "findings"
    out.mkdir(parents=True, exist_ok=True)
    colors = {"SPACE_WEATHER": "tab:purple", "JAMMING": "tab:red", "SPOOFING": "tab:orange",
              "EQUIPMENT_FAULT": "tab:brown", "MULTIPATH": "tab:green"}

    # 1. 72 h overview: fleet C/N0 per tractor (1 min), S4, flags timeline, labels
    fig, ax = plt.subplots(4, 1, figsize=(16, 10), sharex=True)
    t = tele.set_index("ts")
    m = t.groupby("tractor_id").resample("1min")[["mean_cn0_dbhz", "axte"]].median()
    for tr in TRACTORS:
        ax[0].plot(m.loc[tr].index, m.loc[tr].mean_cn0_dbhz, lw=0.7, label=tr)
        ax[1].plot(m.loc[tr].index, m.loc[tr].axte, lw=0.7, label=tr)
    ax[2].step(sda["s4"].ts, sda["s4"].s4, where="post", color="k", lw=0.8, label="S4")
    k = sda["kp"]
    ax2 = ax[2].twinx()
    ax2.step(k.ts, k.kp, where="post", color="tab:purple", lw=0.8)
    ax2.set_ylabel("Kp", color="tab:purple")
    fl = flags.any(axis=1).groupby([tele.tractor_id, tele.ts.dt.floor("1min")]).sum().unstack(0)
    for i, tr in enumerate(TRACTORS):
        bad = fl[tr][fl[tr] >= 3].index
        ax[3].scatter(bad, np.full(len(bad), i), s=2, marker="|", color=f"C{i}")
    ax[3].set_yticks(range(3), TRACTORS)
    for ep in labels.itertuples():
        for a in ax[:2]:
            a.axvspan(ep.start, max(ep.end, ep.start + pd.Timedelta(minutes=3)), color=colors[ep.cls], alpha=0.15)
        for tr in ep.tractors:
            i = TRACTORS.index(tr)
            ax[3].plot([ep.start, max(ep.end, ep.start + pd.Timedelta(minutes=3))], [i + 0.3, i + 0.3],
                       color=colors[ep.cls], lw=4)
    for r in sda["rfi"].itertuples():
        for a in ax:
            a.axvline(r.ts, color="red", ls=":", lw=0.8)
    for a in ax:
        a.axvline(EVAL_START, color="k", lw=1.2)
    ax[0].set_ylabel("mean C/N0 (1-min med)")
    ax[1].set_ylabel("|XTE| m (1-min med)")
    ax[1].set_yscale("symlog", linthresh=0.1)
    ax[2].set_ylabel("S4")
    ax[3].set_ylabel("flagged minutes\n(bars = labels)")
    ax[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("72 h overview. Coloured spans = dev labels (purple storm, red jam, orange spoof, brown equipment, "
                 "green multipath); red dotted = RFI report; black line = start of eval window (unlabeled)")
    fig.tight_layout()
    fig.savefig(out / "overview_72h.png", dpi=90)
    plt.close(fig)

    # 2. spoofing: XTE and reported-position offset overlay
    sp = labels[labels.cls == "SPOOFING"]
    fig, ax = plt.subplots(1, len(sp), figsize=(16, 4))
    for a, ep in zip(ax, sp.itertuples()):
        tr = ep.tractors[0]
        g = tele[(tele.tractor_id == tr) & (tele.ts >= ep.start - pd.Timedelta(minutes=5))
                 & (tele.ts < ep.end + pd.Timedelta(minutes=5))]
        a.plot(g.ts, g.cross_track_error_m, lw=0.6, label="XTE (m)")
        a.plot(g.ts, g.lon_dev_m, lw=0.8, ls="--", label="reported lon offset vs lockstep (m)")
        a.axvspan(ep.start, ep.end, color="orange", alpha=0.15)
        a.set_title(f"{ep.id} {tr}")
        a.tick_params(axis="x", labelrotation=30, labelsize=7)
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "spoofing_xte_vs_position.png", dpi=90)
    plt.close(fig)

    # 3. jamming: C/N0 vs distance to fitted source
    jm = labels[labels.cls == "JAMMING"]
    fig, ax = plt.subplots(1, len(jm), figsize=(15, 4))
    for a, ep in zip(ax, jm.itertuples()):
        fit = jam_distance_fit(tele, ep.start, ep.end)
        w = tele[(tele.ts >= ep.start) & (tele.ts < ep.end)]
        for i, tr in enumerate(TRACTORS):
            g = w[w.tractor_id == tr]
            d = np.hypot((g.lat - fit["lat"]) * M_PER_DEG_LAT, (g.lon - fit["lon"]) * M_PER_DEG_LON) + 30
            a.scatter(d, g.mean_cn0_dbhz, s=1, alpha=0.3, color=f"C{i}", label=tr)
        a.set_xscale("log")
        a.set_title(f"{ep.id}: slope {fit['slope_db_per_decade']} dB/dec, rmse {fit['rmse']}")
        a.set_xlabel("distance to fitted source (m)")
    ax[0].set_ylabel("mean C/N0")
    ax[0].legend(markerscale=8)
    fig.tight_layout()
    fig.savefig(out / "jamming_cn0_vs_distance.png", dpi=90)
    plt.close(fig)

    # 4. storm: S4 vs fleet C/N0 per 15-min slot
    s = storm_slots(tele, sda)
    fig, a = plt.subplots(figsize=(6, 4))
    a.scatter(s.s4, s.mean_cn0_dbhz, s=12, c=np.where(s.working, "tab:blue", "tab:gray"))
    a.set_xlabel("S4 (15-min slot)")
    a.set_ylabel("fleet median mean C/N0 in slot")
    a.set_title("Dev: storm level vs S4 (blue = working, grey = parked)")
    fig.tight_layout()
    fig.savefig(out / "storm_s4_vs_cn0.png", dpi=90)
    plt.close(fig)
    print(f"\nPlots written to {out}")


if __name__ == "__main__":
    main()

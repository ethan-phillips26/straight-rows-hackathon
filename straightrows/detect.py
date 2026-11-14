"""Episode detectors: one small function per cause, all reading the tables from ``features.py``.

Each detector returns a list of episodes (dicts with ``kind``, ``start``, ``end``, ``tractors`` and a
``stats`` dict of the numbers that triggered it). ``kind`` is already the challenge class name; deciding
confidence and writing evidence strings is the job of ``attribute.py`` / ``evidence.py``.

Design rules (docs/FINDINGS.md section 6):
* Prefer relative, physical features: drop vs the tractor's own quiet baseline, offset from the
  lockstep formation, uniform-vs-single-tractor patterns.
* Put each threshold in the middle of a wide gap (quiet extreme vs smallest real effect), and say which
  two numbers it sits between. Nothing is tuned on the evaluation window.
* A cause is *detected* here from farm data. The SDA feeds (S4) are only used where FINDINGS says they
  separate two causes that look alike on the farm (storm vs jamming).
"""
import numpy as np
import pandas as pd

from .features import TRACTORS

MINUTE = pd.Timedelta(minutes=1)
SECOND = pd.Timedelta(seconds=1)

# ---- tunables, each with the gap it sits in -------------------------------------------------------
# Storm: fleet-median C/N0 drop vs quiet baseline. Smallest storm step is 7.5 dB; quiet never exceeds ~2 dB.
STORM_MIN_DROP_DB = 4.0
# Storm needs S4 elevated too. Quiet S4 <= 0.094, every impacted slot >= 0.317; 0.2 is inside the gap and
# is what separates a uniform storm drop from a (quiet-S4) jamming drop that happens to hit all 3 tractors.
STORM_MIN_S4 = 0.2
STORM_MIN_MIN = 30          # shortest storm slot run we trust (dev storms are 180 and 300 min)
# Fleet C/N0 during a storm is linear in S4 (FINDINGS 0.4, r = -0.998). Used only to subtract the storm
# level so jamming inside a storm (TRN-05) shows up as an EXTRA drop. WEAK: fitted on two storms.
STORM_CN0_AT_S4_0 = 42.7
STORM_CN0_PER_S4 = 17.7

# Jamming: extra C/N0 drop beyond the storm level. Labeled tractors dropped >= 5.7 dB, the one unlabeled
# tractor 3.0 dB, so 4.35 (their midpoint) decides who is "affected". Minute flags use the looser 4.0.
JAM_MINUTE_DROP_DB = 4.0
JAM_TRACTOR_DROP_DB = 4.35
JAM_MIN_MIN = 24            # dev jamming lasts 36-42 min
JAM_MAX_GAP_MIN = 7         # the drop oscillates with the ~6 min row-pair cadence; bridge one cycle

# Spoofing: reported position off the lockstep line. Storm/jamming/quiet stay within 0.05 m, the spoof
# reaches metres, so 0.5 m is a decade above the noise and well under the smallest plateau (4 m).
SPOOF_MIN_DEV_M = 0.5
SPOOF_MIN_MIN = 10
SPOOF_MAX_GAP_S = 60
SPOOF_RAMP_FIT_S = 300      # fit the ramp over the first 5 min after the threshold crossing

# Equipment, link type: quiet has zero link-down seconds. A minute counts with >= 3 s (5 %) down.
LINK_MIN_FRAC = 0.05
# Equipment, antenna type: per-second min-C/N0 dropouts below the tractor's own 61 s running median.
ANT_DIP_DB = 6.0
ANT_MIN_FRAC = 0.05
EQUIP_MIN_MIN = 18
EQUIP_MAX_GAP_MIN = 5

# Multipath: min-C/N0 dip while in the north-headland zone. Measured dips are 3.5-10.9 dB.
MP_MIN_DROP_DB = 2.0

GRID_MIN = 6                # labels start/end on a 6-minute grid (FINDINGS 0.1)
GRID_TOL = pd.Timedelta(minutes=2)

CLASS_ACTION = {
    "SPACE_WEATHER": "HALT_OR_DEGRADED_MODE",
    "JAMMING": "HALT_AND_REPORT",
    "SPOOFING": "HALT_CROSSCHECK_AND_REPORT",
    "EQUIPMENT_FAULT": "SERVICE_MACHINE_OTHERS_CONTINUE",
    "MULTIPATH": "CONTINUE_FALSE_ALARM",
}


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def find_runs(times, step, max_gap, min_dur):
    """Group sorted timestamps into runs, joining neighbours at most ``max_gap`` apart.

    Returns (start, end) pairs with ``end`` exclusive (last timestamp + ``step``), dropping runs
    shorter than ``min_dur``.
    """
    times = pd.DatetimeIndex(times).sort_values()
    if len(times) == 0:
        return []
    breaks = np.flatnonzero(np.diff(times.values) > (max_gap + step).to_timedelta64()) + 1
    runs = []
    for chunk in np.split(np.arange(len(times)), breaks):
        start, end = times[chunk[0]], times[chunk[-1]] + step
        if end - start >= min_dur:
            runs.append((start, end))
    return runs


def snap_to_grid(t, tol=GRID_TOL):
    """Move ``t`` to the nearest 6-minute grid point if it is within ``tol``, else just round to the minute.

    Every non-multipath dev label boundary sits on that grid (20/20). Inside ``tol`` snapping can only
    help; outside it we leave the measured time alone, so a wrong grid assumption costs at most ``tol``.
    """
    t = t.round("min")
    step = GRID_MIN * 60
    secs = t.value // 10**9
    nearest = pd.Timestamp(round(secs / step) * step, unit="s", tz="UTC")
    return nearest if abs(nearest - t) <= tol else t


def _episode(kind, start, end, tractors, **stats):
    return {"kind": kind, "start": start, "end": end, "tractors": sorted(tractors), "stats": stats}


# ----------------------------------------------------------------------------------------------
# storm: the whole fleet drops together while S4 is elevated
# ----------------------------------------------------------------------------------------------
def storm_flags(minute):
    """Per-minute table (ts, fleet_drop, s4, storm) for working minutes."""
    fleet = minute.groupby("ts").agg(fleet_drop=("cn0_drop", "median"), s4=("s4", "first")).reset_index()
    fleet["storm"] = (fleet.fleet_drop >= STORM_MIN_DROP_DB) & (fleet.s4 >= STORM_MIN_S4)
    return fleet


def detect_storms(minute):
    fleet = storm_flags(minute)
    runs = find_runs(fleet.ts[fleet.storm], MINUTE, 2 * MINUTE, STORM_MIN_MIN * MINUTE)
    out = []
    for start, end in runs:
        m = fleet[(fleet.ts >= start) & (fleet.ts < end)]
        out.append(_episode("SPACE_WEATHER", start, end, TRACTORS,
                            fleet_drop_db=float(m.fleet_drop.median()), s4_max=float(m.s4.max()),
                            s4_median=float(m.s4.median())))
    return [_snap(e) for e in out]


# ----------------------------------------------------------------------------------------------
# jamming: a drop beyond what the storm (if any) explains, differing between tractors
# ----------------------------------------------------------------------------------------------
def add_excess_drop(minute):
    """``excess_drop`` = C/N0 we expect for this minute minus the tractor's C/N0.

    Expected is the tractor's quiet baseline, except inside a storm minute where it is the S4-driven
    storm level. So a storm alone gives ~0 and jamming (alone or inside a storm) gives its own drop.
    """
    m = minute.copy()
    storm = storm_flags(m)[["ts", "storm"]]
    m = m.merge(storm, on="ts", how="left")
    baseline = m.cn0 + m.cn0_drop                      # cn0_drop = baseline - cn0
    storm_level = STORM_CN0_AT_S4_0 - STORM_CN0_PER_S4 * m.s4
    expected = np.where(m.storm, np.minimum(baseline, storm_level), baseline)
    m["excess_drop"] = expected - m.cn0
    return m


def detect_jamming(minute):
    m = add_excess_drop(minute)
    hit = m[m.excess_drop >= JAM_MINUTE_DROP_DB]
    runs = find_runs(hit.ts.unique(), MINUTE, JAM_MAX_GAP_MIN * MINUTE, JAM_MIN_MIN * MINUTE)
    out = []
    for start, end in runs:
        w = m[(m.ts >= start) & (m.ts < end)]
        per_tractor = w.groupby("tractor_id").excess_drop.mean()
        tractors = per_tractor.index[per_tractor >= JAM_TRACTOR_DROP_DB].tolist()
        if not tractors:
            continue
        out.append(_episode("JAMMING", start, end, tractors,
                            mean_excess_db={k: float(v) for k, v in per_tractor.items()},
                            s4_median=float(w.s4.median()),
                            no_fix_frac=float(w[w.tractor_id.isin(tractors)].no_fix_frac.mean())))
    return [_snap(e) for e in out]


# ----------------------------------------------------------------------------------------------
# spoofing: one tractor's reported position leaves the lockstep line
# ----------------------------------------------------------------------------------------------
def detect_spoofing(sec):
    out = []
    for tr, g in sec.groupby("tractor_id", sort=True):
        hit = g[g.lon_dev_abs_m >= SPOOF_MIN_DEV_M]
        for start, end in find_runs(hit.ts, SECOND, SPOOF_MAX_GAP_S * SECOND, SPOOF_MIN_MIN * MINUTE):
            # The ramp starts at 0 so the threshold is only crossed minutes after onset (FINDINGS P6):
            # back-extrapolate the straight line |dev| = slope * (t - t0) fitted on the first minutes.
            ramp = g[(g.ts >= start) & (g.ts < start + SPOOF_RAMP_FIT_S * SECOND)]
            x = (ramp.ts - start).dt.total_seconds().to_numpy()
            slope, intercept = np.polyfit(x, ramp.lon_dev_abs_m.to_numpy(), 1)
            # A real ramp is ~SPOOF_MIN_DEV_M at the crossing. If the first flagged value is far above that,
            # the offset jumped (a step), there is no ramp to extend, and slope ~ 0 would give a huge lead.
            is_ramp = slope > 0 and ramp.lon_dev_abs_m.iloc[0] <= 2 * SPOOF_MIN_DEV_M
            lead = min(max(intercept / slope, 0.0), 600.0) if is_ramp else 0.0  # line hits 0 at -intercept/slope s from the crossing
            onset = start - pd.Timedelta(seconds=lead)
            whole = g[(g.ts >= start) & (g.ts < end)]
            out.append(_episode("SPOOFING", onset, end, [tr],
                                max_offset_m=float(whole.lon_dev_abs_m.max()),
                                ramp_m_per_min=float(slope * 60), first_cross=start,
                                xte_max_m=float(whole.axte.max())))
    return [_snap(e) for e in out]


# ----------------------------------------------------------------------------------------------
# equipment: one machine's own receiver / link misbehaves
# ----------------------------------------------------------------------------------------------
def add_dropout_flag(sec, jam=()):
    """1 Hz flag ``dip``: min C/N0 is >= ANT_DIP_DB below the tractor's own 61 s running median.

    Running median because a storm/jamming level shift is slow (minutes) and is absorbed by it, while an
    antenna fault's 1-5 s dropouts are not. Headland seconds are skipped (multipath lives there). Seconds
    inside a detected jamming episode (for the jammed tractors) are skipped too: a collapsing, oscillating
    C/N0 is noisy second to second, and that noise is the jamming, not a second fault (dev TRN-03 T1).
    """
    s = sec.copy()
    base = s.groupby("tractor_id").min_cn0_dbhz.transform(lambda x: x.rolling(61, center=True, min_periods=31).median())
    s["dip"] = ((base - s.min_cn0_dbhz) >= ANT_DIP_DB) & ~s.headland & s.working
    for e in jam:
        s.loc[s.tractor_id.isin(e["tractors"]) & (s.ts >= e["start"]) & (s.ts < e["end"]), "dip"] = False
    return s


def _flag_minutes(sec, col):
    """Fraction of working, non-headland seconds with ``col`` true, per (tractor, minute)."""
    g = sec[sec.working & ~sec.headland]
    return g.groupby(["tractor_id", g.ts.dt.floor("min")])[col].mean().rename("frac").reset_index().rename(columns={"ts": "minute"})


def detect_equipment(sec, jam=()):
    s = add_dropout_flag(sec, jam)
    out = []
    for subtype, col, thresh in [("link", "link_down", LINK_MIN_FRAC), ("antenna", "dip", ANT_MIN_FRAC)]:
        frac = _flag_minutes(s, col)
        for tr, g in frac.groupby("tractor_id", sort=True):
            hit = g[g.frac >= thresh]
            for start, end in find_runs(hit.minute, MINUTE, EQUIP_MAX_GAP_MIN * MINUTE, EQUIP_MIN_MIN * MINUTE):
                w = g[(g.minute >= start) & (g.minute < end)]
                out.append(_episode("EQUIPMENT_FAULT", start, end, [tr], subtype=subtype,
                                    flag_frac=float(w.frac.mean())))
    return [_snap(e) for e in out]


# ----------------------------------------------------------------------------------------------
# multipath: one detection per tractor per north-headland pass (option A in docs/LOG.md)
# ----------------------------------------------------------------------------------------------
def detect_multipath(passes):
    p = passes[passes.min_drop_db >= MP_MIN_DROP_DB]
    return [_episode("MULTIPATH", r.start, r.end, [r.tractor_id], min_drop_db=float(r.min_drop_db),
                     n_tractors_dipped=int(r.n_tractors_dipped), xte_in=float(r.xte_in))
            for r in p.itertuples()]


# ----------------------------------------------------------------------------------------------
def _snap(e):
    e["start"], e["end"] = snap_to_grid(e["start"]), snap_to_grid(e["end"])
    return e


def detect_all(sec, minute, passes):
    """Run every detector; returns a DataFrame sorted by start time (deterministic order)."""
    jam = detect_jamming(minute)
    eps = (detect_storms(minute) + jam + detect_spoofing(sec)
           + detect_equipment(sec, jam) + detect_multipath(passes))
    df = pd.DataFrame(eps)
    return df.sort_values(["start", "kind", "end"]).reset_index(drop=True)

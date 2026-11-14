"""Shared features for every detector (see docs/FINDINGS.md for why each one exists).

Three layers, each a plain DataFrame so detectors stay simple:

* ``second_features``  1 Hz rows (tractor, second): the raw columns plus
    - ``axte``          |cross-track error|
    - ``headland``      True while reported lat is in the north-headland multipath zone
    - ``lon_dev_m``     reported east-west position vs where lockstep says it should be (spoof signal)
    - ``fleet_*``       median of the three tractors at the same second
    - ``*_resid``       own value minus the fleet median (isolates ONE bad tractor)
    - ``cn0_drop``      tractor's own quiet baseline minus its mean C/N0 (isolates ANY drop)
* ``minute_features``  the same, aggregated to 1 minute for the slow causes (storm, jamming,
    spoofing, equipment). Headland seconds are left out so multipath cannot contaminate them.
* ``headland_passes``  one row per (tractor, north-edge pass) for multipath, which a 1-minute
    table would erase (the dip is 17 s long).

Nothing here uses labels, and nothing is tuned: constants are geometry or physical units.
Deterministic, vectorised, no wall-clock or randomness.
"""
import numpy as np
import pandas as pd

TRACTORS = ["T1", "T2", "T3"]

# Multipath zone: min C/N0 dips exactly while reported lat is above this value (~38 m from the north
# edge). Located from label-free quiet-day lat bins; see FINDINGS.md section 0 item 5 and section 6.
HEADLAND_LAT = 46.906854

M_PER_DEG_LAT = 111_320.0
M_PER_DEG_LON = M_PER_DEG_LAT * np.cos(np.radians(46.903))  # about 76 km per degree at the farm


# ----------------------------------------------------------------------------------------------
# lockstep position offset
# ----------------------------------------------------------------------------------------------
def add_lockstep_offset(tele):
    """Add ``lon_dev_m``: metres the REPORTED lon is off from where the formation says it should be.

    The three tractors drive identical patterns at a fixed east-west spacing, so the lon difference of
    any pair only takes two values: the normal spacing, or the spacing +- one field width when one
    tractor has wrapped to the west edge a second before the other. Anything else means one reported
    position moved sideways, which is what spoofing does (FINDINGS section 0 item 2).

    For tractor X we look at its two pairs and keep the smaller deviation: a displaced X shows up in
    both pairs, but a displaced neighbour only shows up in one. Parked rows get NaN (parking spots are
    not in formation). Needs all three tractors present at the same timestamps.
    """
    work = tele[tele.working]
    lon = work.pivot(index="ts", columns="tractor_id", values="lon")
    pair_dev = {}
    for a in TRACTORS:
        for b in TRACTORS:
            if a == b:
                continue
            off = lon[a] - lon[b]
            # Most common offset = normal spacing. The "wrapped" spacing differs by ~0.009 deg (one field
            # width); a spoofed tractor is only metres (~1e-4 deg) off, so it can never be mistaken for it.
            # value_counts ranks the (rounded) offsets by frequency: the top one is the normal spacing. Keep the next
            # of the top 5 that is > 0.005 deg (~380 m) away as the "wrapped" spacing; 2e-5 deg (~1.5 m) is the
            # tolerance for "same offset" when refining each mode to its exact median.
            counts = off.round(5).value_counts()
            modes = [m for i, m in enumerate(counts.index[:5]) if i == 0 or abs(m - counts.index[0]) > 0.005][:2]
            modes = np.array([off[(off - m).abs() < 2e-5].median() for m in modes])  # refine to the exact value
            nearest = modes[np.abs(off.values[:, None] - modes[None, :]).argmin(axis=1)]
            pair_dev[(a, b)] = (off - nearest) * M_PER_DEG_LON
    dev = {}
    for a in TRACTORS:
        p1, p2 = [pair_dev[(a, b)] for b in TRACTORS if b != a]
        dev[a] = p1.where(p1.abs() <= p2.abs(), p2)
    dev = pd.DataFrame(dev).stack().rename("lon_dev_m").reset_index()
    dev.columns = ["ts", "tractor_id", "lon_dev_m"]
    return tele.merge(dev, on=["ts", "tractor_id"], how="left")


# ----------------------------------------------------------------------------------------------
# 1 Hz features
# ----------------------------------------------------------------------------------------------
FLEET_COLS = ["mean_cn0_dbhz", "min_cn0_dbhz", "num_sats", "fix_rank", "axte"]


def second_features(tele):
    """Per-second table with fleet medians and residuals. ``tele`` comes from ``load.load_telemetry``."""
    df = tele.copy()
    df["axte"] = df.cross_track_error_m.abs()
    df["headland"] = df.lat > HEADLAND_LAT
    df = add_lockstep_offset(df)
    df["lon_dev_abs_m"] = df.lon_dev_m.abs()

    # Fleet median across the 3 tractors at each second. With three tractors the median is the middle
    # one, so a single bad tractor cannot drag it (a mean would be pulled toward the outlier).
    for col in FLEET_COLS:
        df[f"fleet_{col}"] = df.groupby("ts")[col].transform("median")
        df[f"{col}_resid"] = df[col] - df[f"fleet_{col}"]

    # Own quiet baseline: median mean-C/N0 of working, non-headland seconds. The median ignores the
    # degraded stretches because most working time is quiet (FINDINGS section 0 item 1).
    ok = df.working & ~df.headland
    base = df[ok].groupby("tractor_id").mean_cn0_dbhz.median()
    df["cn0_baseline"] = df.tractor_id.map(base)
    df["cn0_drop"] = df.cn0_baseline - df.mean_cn0_dbhz
    df["fleet_cn0_drop"] = df.groupby("ts")["cn0_drop"].transform("median")
    df["link_down"] = ~df.base_link_ok
    df["sats_low"] = df.num_sats < 11  # quiet is always 11-13 (FINDINGS section 0 item 1)
    return df.sort_values(["tractor_id", "ts"]).reset_index(drop=True)


# ----------------------------------------------------------------------------------------------
# 1-minute features
# ----------------------------------------------------------------------------------------------
def minute_features(sec, sda=None):
    """One row per (tractor, minute) for working, non-headland seconds.

    Levels are means (a storm is a step, so the mean is clean); extremes use min/max/quantiles so short
    dropouts (antenna fault, link loss) are not averaged away. Adds S4 / Kp / Dst when ``sda`` is given.
    """
    g = sec[sec.working & ~sec.headland].copy()
    g["minute"] = g.ts.dt.floor("min")
    g["fix_ok"] = g.fix_rank == 4
    g["min_low"] = g.min_cn0_dbhz < 35  # quiet 0.1% quantile is 36.9 (FINDINGS section 8)
    keys = ["tractor_id", "minute"]
    out = g.groupby(keys).agg(
        n=("ts", "size"),
        cn0=("mean_cn0_dbhz", "mean"),
        cn0_min=("min_cn0_dbhz", "min"),
        cn0_drop=("cn0_drop", "mean"),
        cn0_resid=("mean_cn0_dbhz_resid", "mean"),
        fleet_cn0=("fleet_mean_cn0_dbhz", "mean"),
        sats=("num_sats", "mean"),
        sats_resid=("num_sats_resid", "mean"),
        fix_rank=("fix_rank", "mean"),
        fix_ok_frac=("fix_ok", "mean"),
        no_fix_frac=("fix_rank", lambda s: (s == 0).mean()),
        link_down_frac=("link_down", "mean"),
        min_low_frac=("min_low", "mean"),
        sats_low_frac=("sats_low", "mean"),
        xte_med=("axte", "median"),
        xte_p95=("axte", lambda s: s.quantile(0.95)),
        xte_resid=("axte_resid", "mean"),
        lon_dev=("lon_dev_m", "mean"),
        lon_dev_abs=("lon_dev_abs_m", "max"),
        lat=("lat", "mean"),
        lon=("lon", "mean"),
    ).reset_index().rename(columns={"minute": "ts"})
    out["ts"] = pd.to_datetime(out.ts, utc=True)
    if sda is not None:
        out = add_sda(out, sda)
    return out.sort_values(["tractor_id", "ts"]).reset_index(drop=True)


def add_sda(df, sda):
    """Attach the SDA value that applies at each row's ``ts`` (latest report at or before it).

    S4 is a 15-minute series, Kp 3-hourly and Dst hourly, each valid from its own timestamp.
    ``active_sw_advisory`` is True once a SPACE_WEATHER advisory has been issued (it says nothing
    about when the effect ends, so use it as evidence only).
    """
    df = df.sort_values("ts")
    for name, feed, col in [("s4", sda["s4"], "s4"), ("kp", sda["kp"], "kp"), ("dst", sda["dst"], "dst_nT")]:
        right = feed[["ts", col]].rename(columns={col: name}).sort_values("ts")
        df = pd.merge_asof(df, right, on="ts", direction="backward")
    adv = sda["advisories"]
    first_sw = adv.loc[adv.type == "SPACE_WEATHER", "ts"].min()
    df["active_sw_advisory"] = df.ts >= first_sw if pd.notna(first_sw) else False
    return df


# ----------------------------------------------------------------------------------------------
# multipath: one row per north-edge pass
# ----------------------------------------------------------------------------------------------
def headland_passes(sec, context_s=40):
    """One row per (tractor, headland pass): the dip window and how deep the dip is.

    A pass is a run of consecutive ``headland`` seconds. ``min_drop_db`` compares the median of
    min-C/N0 in the ``context_s`` seconds either side of the pass (outside the zone) with the median
    inside it, so it stays meaningful during storms and jamming. ``n_tractors_dipped`` counts how many
    tractors have a pass starting in the same second; for real multipath it is 3 (all tractors reach the
    headland together), which is the evidence that this is geometry rather than one bad machine.
    """
    rows = []
    for tr, g in sec.groupby("tractor_id", sort=True):
        g = g.reset_index(drop=True)
        h = g.headland.to_numpy()
        # run boundaries: positions where the flag changes. Padding with 0 at both ends makes every run start with
        # a 0->1 step and end with a 1->0 step, so the edges alternate: even entries are starts, odd are ends.
        edge = np.flatnonzero(np.diff(np.r_[0, h.astype(int), 0]))
        for i0, i1 in zip(edge[::2], edge[1::2]):  # [i0, i1) is one pass
            inside = g.iloc[i0:i1]
            around = pd.concat([g.iloc[max(i0 - context_s, 0):i0], g.iloc[i1:i1 + context_s]])
            rows.append({
                "tractor_id": tr,
                "start": inside.ts.iloc[0],
                "end": inside.ts.iloc[-1] + pd.Timedelta(seconds=1),  # exclusive end, like the labels
                "dip_s": len(inside),
                "min_cn0_in": inside.min_cn0_dbhz.median(),
                "min_cn0_out": around.min_cn0_dbhz.median(),
                "min_drop_db": around.min_cn0_dbhz.median() - inside.min_cn0_dbhz.median(),
                "mean_drop_db": around.mean_cn0_dbhz.median() - inside.mean_cn0_dbhz.median(),
                "xte_in": inside.axte.median(),
                "xte_out": around.axte.median(),
                "fleet_cn0_drop": inside.fleet_cn0_drop.median(),
            })
    columns = ["tractor_id", "start", "end", "dip_s", "min_cn0_in", "min_cn0_out", "min_drop_db", "mean_drop_db",
               "xte_in", "xte_out", "fleet_cn0_drop"]
    out = pd.DataFrame(rows, columns=columns).sort_values(["start", "tractor_id"]).reset_index(drop=True)
    out["n_tractors_dipped"] = out.groupby("start").tractor_id.transform("nunique")
    return out


# ----------------------------------------------------------------------------------------------
def build_features(tele, sda):
    """Everything the detectors need: (per-second table, per-minute table, headland passes)."""
    sec = second_features(tele)
    return sec, minute_features(sec, sda), headland_passes(sec)

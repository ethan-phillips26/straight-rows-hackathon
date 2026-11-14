"""Locate a jammer from the C/N0 the tractors lose as they drive around it, with an error region.

Physics: a fixed jammer at distance d lowers a receiver's C/N0 roughly like free-space path loss, so C/N0
rises with distance from it,
    mean C/N0 = a + b * log10(d),     b ~ 20 dB/decade   (docs/FINDINGS.md section 3: b = 19-23 on dev).
The three tractors drive rows, so over a 40 minute episode they sample many different distances to the
same point. Each candidate source position (x, y) is scored by how well that law fits all the C/N0 data
(a and b are solved exactly for each candidate, so only the 2 position unknowns are searched).

Error has two parts, added together (conservative):
  * statistical: the fit's sum of squared residuals RSS(x, y) over the grid gives a 95 % region (every position
    whose RSS is within the chi-square(2 dof) = 5.99 band of the best one), reported as an ellipse;
  * sensitivity: the fit is repeated on each half of the episode and with each tractor left out; the largest
    shift of the best position is added to both semi-axes. This catches what the noise model cannot
    (a tractor whose data is odd, a source that changed during the episode).
A poor fit or an open-ended region is flagged ``reliable = False`` rather than reported as a location.

A third, independent error estimate is a block bootstrap (``_bootstrap``): resample each tractor's 60 s blocks
of C/N0 with replacement, refit, and take the spread of the refitted positions as a 95 % ellipse. It makes no
noise-model assumption beyond "blocks are independent", so it is a check on the chi-square band, not a replacement.

Only numpy/pandas; deterministic (fixed grids, bootstrap RNG seeded).
"""
import numpy as np
import pandas as pd

from .features import HEADLAND_LAT, M_PER_DEG_LAT, M_PER_DEG_LON

LAT0, LON0 = 46.903, -97.200                        # farm centre: origin of the local metre grid

NEAR_FIELD_M = 30.0        # added to d so a source right on a row cannot give log10(0)
SLOPE_RANGE = (10.0, 40.0)  # plausible dB/decade; free space is 20 (anything else = the model is wrong)
BIN_S = 10                 # average 1 Hz C/N0 to 10 s bins (1 s values are noisy)
BLOCK_S = 60               # C/N0 errors are correlated over about a minute -> effective sample = one per minute
CHI2_95_2DOF = 5.99

# Trust gates: below these the result is reported as "no reliable location".
MAX_RMSE_DB = 2.0          # dev fits are <= 1 dB (FINDINGS section 3); the bad eval fits had 3-4 dB
MIN_COVERAGE_M = 300.0     # tractors must spread over at least this much ground, or distances barely change
MAX_RADIUS_M = 3000.0      # a region wider than this is not a location
N_BOOT = 100               # bootstrap refits per episode
BOOT_SEED = 20260506       # fixed so the submission is reproducible
BOOT_HALF_M, BOOT_STEP_M = 750.0, 25.0    # refits search a local grid around the best fit (cheap; 61 x 61 cells)
MIN_ERROR_M = 25.0         # reporting floor: 10 m grid cell plus the unknown near-field offset of the model


def to_xy(lat, lon):
    """Degrees -> metres east/north of the farm centre."""
    return (np.asarray(lon) - LON0) * M_PER_DEG_LON, (np.asarray(lat) - LAT0) * M_PER_DEG_LAT


def to_latlon(x, y):
    return LAT0 + np.asarray(y) / M_PER_DEG_LAT, LON0 + np.asarray(x) / M_PER_DEG_LON


def _samples(w):
    """10 s mean C/N0 and position per tractor, for the working, non-headland seconds of ``w``.

    Headland seconds are left out because the 17 s multipath dip there is not jamming. Bins where the
    tractor jumped (the row-end wrap of longitude) are dropped, since an averaged position would be nowhere.
    """
    w = w[w.working & (w.lat <= HEADLAND_LAT)].copy()
    w["bin"] = w.ts.dt.floor(f"{BIN_S}s")
    x, y = to_xy(w.lat, w.lon)
    w["x"], w["y"] = x, y
    g = w.groupby(["tractor_id", "bin"]).agg(x=("x", "mean"), y=("y", "mean"), c=("mean_cn0_dbhz", "mean"),
                                             xr=("x", np.ptp), yr=("y", np.ptp), n=("x", "size")).reset_index()
    g = g[(g.n >= BIN_S // 2) & (g.xr < 100) & (g.yr < 100)]
    return g


def _rss_surface(sx, sy, c, gx, gy):
    """RSS, intercept and slope of  c = a + b*log10(d)  for every candidate source (gx, gy), vectorised.

    For a fixed position the model is linear in (a, b), so regression has a closed form: no iterations.
    """
    L = np.log10(np.hypot(sx[None, :] - gx[:, None], sy[None, :] - gy[:, None]) + NEAR_FIELD_M)
    Lm, cm = L.mean(1, keepdims=True), c.mean()
    Ld = L - Lm
    sxx = (Ld ** 2).sum(1)
    sxy = (Ld * (c - cm)[None, :]).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = sxy / sxx                                   # b: dB gained per decade of distance
    rss = ((c - cm) ** 2).sum() - slope * sxy
    ok = (slope >= SLOPE_RANGE[0]) & (slope <= SLOPE_RANGE[1])
    rss = np.where(ok & np.isfinite(rss), rss, np.inf)
    return rss, cm - slope * Lm[:, 0], slope


def _grid(cx, cy, half, step):
    ax = np.arange(-half, half + step / 2, step)
    gx, gy = np.meshgrid(cx + ax, cy + ax)
    return gx.ravel(), gy.ravel()


def _fit(g, duration_s):
    """Best source position for samples ``g`` plus its statistical 95 % region. None if nothing fits."""
    sx, sy, c = g.x.to_numpy(), g.y.to_numpy(), g.c.to_numpy()
    # Coarse search over +-6 km (100 m cells), then a fine one (10 m) around the best coarse cell.
    gx, gy = _grid(0.0, 0.0, 6000.0, 100.0)
    rss, _, _ = _rss_surface(sx, sy, c, gx, gy)
    if not np.isfinite(rss).any():
        return None
    i = int(np.argmin(rss))
    gx, gy = _grid(gx[i], gy[i], 600.0, 10.0)
    rss, a, b = _rss_surface(sx, sy, c, gx, gy)
    if not np.isfinite(rss).any():
        return None
    i = int(np.argmin(rss))
    n = len(c)

    # Noise variance comes from the fit itself. Neighbouring 10 s bins are not independent, so the band is
    # widened by n / n_eff, where n_eff is one independent point per BLOCK_S per tractor.
    n_eff = max(8.0, duration_s / BLOCK_S * g.tractor_id.nunique())
    band = CHI2_95_2DOF * (rss[i] / n) * (n / n_eff)
    inside = rss <= rss[i] + band
    off = np.c_[gx[inside] - gx[i], gy[inside] - gy[i]]
    if len(off) > 2:
        _, evecs = np.linalg.eigh(np.cov(off.T))
    else:
        evecs = np.eye(2)
    # The region is a thin sliver of grid cells: take its extent (not its std) along each principal axis.
    half_ax = np.maximum(np.abs(off @ evecs).max(0), 10.0)       # at least one grid cell
    major = evecs[:, int(np.argmax(half_ax))]
    edge = bool(np.abs(off).max() >= 600.0 - 10.0)               # region reaches the edge of the fine grid
    return {"x": float(gx[i]), "y": float(gy[i]), "major_m": float(half_ax.max()), "minor_m": float(half_ax.min()),
            "bearing": float(np.degrees(np.arctan2(major[0], major[1])) % 180), "rmse": float(np.sqrt(rss[i] / n)),
            "slope": float(b[i]), "n": n, "edge": edge}


def _bootstrap(g, best):
    """Block-bootstrap 95 % ellipse of the source position.

    Each replicate redraws, per tractor, as many 60 s blocks as it has, with replacement, and refits the
    position on a local grid around ``best``. The 95 % ellipse is sqrt(chi2_95(2 dof)) standard deviations along the
    principal axes of the replicate positions (exact for a Gaussian cloud). Returns None if fewer than half the
    refits converge (the data do not pin a position down).
    """
    rng = np.random.default_rng(BOOT_SEED)
    blk = ((g.bin - g.bin.min()).dt.total_seconds() // BLOCK_S).astype(int).to_numpy()
    sx, sy, c, tr = g.x.to_numpy(), g.y.to_numpy(), g.c.to_numpy(), g.tractor_id.to_numpy()
    groups = []                                           # per tractor: {block id -> sample indices}
    for t in np.unique(tr):
        idx = np.flatnonzero(tr == t)
        groups.append({b: idx[blk[idx] == b] for b in np.unique(blk[idx])})
    gx, gy = _grid(best["x"], best["y"], BOOT_HALF_M, BOOT_STEP_M)
    pts = []
    for _ in range(N_BOOT):
        take = []
        for grp in groups:
            keys = list(grp)
            take += [grp[keys[k]] for k in rng.integers(0, len(keys), len(keys))]
        i = np.concatenate(take)
        rss, _, _ = _rss_surface(sx[i], sy[i], c[i], gx, gy)
        if np.isfinite(rss).any():
            j = int(np.argmin(rss))
            pts.append((gx[j], gy[j]))
    if len(pts) < N_BOOT // 2:
        return None
    pts = np.array(pts)
    evals, evecs = np.linalg.eigh(np.cov(pts.T))
    k = np.sqrt(CHI2_95_2DOF * np.maximum(evals, 0.0))   # semi-axes, ascending
    major = evecs[:, 1]
    return {"major_m": float(max(k[1], BOOT_STEP_M)), "minor_m": float(max(k[0], BOOT_STEP_M)),
            "bearing": float(np.degrees(np.arctan2(major[0], major[1])) % 180),
            "centre_shift_m": float(np.hypot(*(pts.mean(0) - [best["x"], best["y"]]))),
            "edge": bool(np.abs(pts - [best["x"], best["y"]]).max() >= BOOT_HALF_M - BOOT_STEP_M), "n": len(pts)}


def locate_jammer(tele, start, end):
    """Fit one fixed source to the telemetry between ``start`` and ``end`` (UTC timestamps).

    ``tele``: 1 Hz telemetry (``load.load_telemetry``). All three tractors are used, including ones the jammer
    barely reached: a small drop far away is exactly the distance information the fit needs.

    Returns a dict, or ``None`` when there is too little data to try:
      lat, lon                      best position
      semi_major_km, semi_minor_km  95 % error ellipse (statistical + sensitivity), long axis bearing in degrees
      stat_major_km, sensitivity_m  the two parts of that error
      boot_major_km, boot_minor_km, boot_bearing_deg   independent block-bootstrap 95 % ellipse (None if it failed)
      rmse_db, slope_db_per_decade  fit quality (dev: rmse <= 1, slope 19-23)
      reliable, notes               False + reasons if a trust gate failed (poor fit, little coverage, open region)
    """
    g = _samples(tele[(tele.ts >= start) & (tele.ts < end)])
    if len(g) < 60:
        return None
    dur = (end - start).total_seconds()
    best = _fit(g, dur)
    if best is None:
        return None

    # Sensitivity: refit on each half of the episode and with each tractor left out.
    mid = start + (end - start) / 2
    subsets = [(g[g.bin < mid], dur / 2), (g[g.bin >= mid], dur / 2)]
    subsets += [(g[g.tractor_id != t], dur) for t in sorted(g.tractor_id.unique())]
    shifts = []
    for sub, d in subsets:
        if len(sub) >= 60:
            f = _fit(sub, d)
            shifts.append(np.hypot(f["x"] - best["x"], f["y"] - best["y"]) if f else np.inf)
    sens = max(shifts) if shifts else np.inf

    coverage = float(np.hypot(np.ptp(g.x), np.ptp(g.y)))
    major_m, minor_m = (max(v + sens, MIN_ERROR_M) for v in (best["major_m"], best["minor_m"]))
    notes = []
    if best["rmse"] > MAX_RMSE_DB:
        notes.append(f"fit RMSE {best['rmse']:.1f} dB exceeds {MAX_RMSE_DB} dB")
    if coverage < MIN_COVERAGE_M:
        notes.append("tractors covered too little ground")
    if best["edge"] or not np.isfinite(major_m) or major_m > MAX_RADIUS_M:
        notes.append("error region not bounded")
    boot = None if notes else _bootstrap(g, best)     # only worth 10 s for a fit we would report
    lat, lon = to_latlon(best["x"], best["y"])
    return {"lat": float(lat), "lon": float(lon),
            "semi_major_km": major_m / 1000, "semi_minor_km": minor_m / 1000,
            "major_axis_bearing_deg": best["bearing"], "stat_major_km": best["major_m"] / 1000,
            "sensitivity_m": float(sens), "rmse_db": best["rmse"], "slope_db_per_decade": best["slope"],
            "boot_major_km": boot and boot["major_m"] / 1000, "boot_minor_km": boot and boot["minor_m"] / 1000,
            "boot_bearing_deg": boot and boot["bearing"], "boot_centre_shift_m": boot and boot["centre_shift_m"],
            "n_samples": best["n"], "reliable": not notes, "notes": notes}


def distance_km(lat1, lon1, lat2, lon2):
    """Straight-line distance between two points (small-area approximation, fine at farm scale)."""
    return float(np.hypot((lat1 - lat2) * M_PER_DEG_LAT, (lon1 - lon2) * M_PER_DEG_LON) / 1000)

"""Write the checkable evidence strings for each detection.

Every number in a string is measured here from the data, never typed in, so the text stays true if a
detector threshold or an episode boundary moves. Each string says what was measured (tractor, UTC time,
value) and, where it matters, which alternative cause that rules out (e.g. "S4 quiet -> not space weather").

Entry point: ``add_evidence(episodes, minute, sda)``. ``episodes`` is the list of dicts from
``detect.detect_all`` (keys kind, start, end, tractors, stats); it returns one list of strings per episode.
Only the 1-minute table and the SDA feeds are used, so this stays cheap even for ~490 multipath episodes.
"""
import numpy as np
import pandas as pd

from . import locate
from .detect import JAM_TRACTOR_DROP_DB, STORM_MIN_S4
from .features import M_PER_DEG_LAT, M_PER_DEG_LON, TRACTORS

QUIET_XTE_M = 0.45          # quiet-time |XTE| never exceeds this (README, Data findings)
RFI_WINDOW = pd.Timedelta(minutes=60)   # a report this close to an episode counts as "near" it


# ----------------------------------------------------------------------------------------------
# small formatting + lookup helpers
# ----------------------------------------------------------------------------------------------
def _t(ts, seconds=False):
    return pd.Timestamp(ts).tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%SZ" if seconds else "%Y-%m-%d %H:%MZ")


def _span(ep, seconds=False):
    return f"{_t(ep['start'], seconds)} to {_t(ep['end'], seconds)}"


def _latest(feed, col, ts):
    """Value of the latest SDA report at or before ``ts`` (each report holds until the next one)."""
    i = feed.ts.searchsorted(pd.Timestamp(ts), side="right") - 1
    return float(feed[col].iloc[i]) if i >= 0 else float("nan")


def s4_window(sda, start, end):
    """(median, max) of the S4 slots that overlap [start, end). A slot is valid for 15 min from its ts."""
    s4 = sda["s4"]
    w = s4[(s4.ts + pd.Timedelta(minutes=15) > start) & (s4.ts < end)].s4
    return (float(w.median()), float(w.max())) if len(w) else (float("nan"), float("nan"))


def _window(minute, start, end):
    return minute[(minute.ts >= start) & (minute.ts < end)]


def _per_tractor(w, col, how="mean"):
    return getattr(w.groupby("tractor_id")[col], how)()


def _list(series, unit="", fmt=".1f"):
    """'T1 12.7 dB, T2 12.3 dB' from a per-tractor Series."""
    return ", ".join(f"{k} {v:{fmt}}{unit}" for k, v in series.items())


# ----------------------------------------------------------------------------------------------
# helpers that look at the OTHER detections (overlapping causes must not contradict each other)
# ----------------------------------------------------------------------------------------------
def overlapping(ep, others, kinds):
    """Other-cause episodes of the given kinds that overlap ``ep`` in time."""
    return [o for o in others if o is not ep and o["kind"] in kinds and o["start"] < ep["end"] and ep["start"] < o["end"]]


def _without(w, ep, others, kinds):
    """Drop tractors that carry their own overlapping detection of ``kinds``; return (table, their names)."""
    names = sorted({t for o in overlapping(ep, others, kinds) for t in o["tractors"]})
    return w[~w.tractor_id.isin(names)], names


def covering_causes(ep, others):
    """Kinds of other detections that overlap ``ep`` in time and include its tractor (storm covers all three)."""
    return sorted({o["kind"] for o in overlapping(ep, others, {o["kind"] for o in others}) if set(ep["tractors"]) & set(o["tractors"])})


def _excluded_note(names, kind):
    return f"; {','.join(names)} left out because it has its own {kind} detection" if names else ""


def _delta(w):
    """Mean C/N0 change vs the tractors' own quiet baselines, signed (negative = signal lost)."""
    return -float(w.cn0_drop.mean())


def _overlap_lines(ep, others):
    """Name overlapping other-cause episodes: separate detections of one operational event."""
    return [f"Overlaps a separate {o['kind']} detection on {','.join(o['tractors'])} ({_span(o)}): "
            f"reported as its own cause, same operational period."
            for o in overlapping(ep, others, {o["kind"] for o in others})]


def _farm_position(w):
    return float(w.lat.mean()), float(w.lon.mean())


# ----------------------------------------------------------------------------------------------
# SDA lines shared by several classes
# ----------------------------------------------------------------------------------------------
def _advisory_line(sda, start, space_weather_is_cause):
    adv = sda["advisories"]
    adv = adv[(adv.type == "SPACE_WEATHER") & (adv.ts <= start)]
    if adv.empty:
        return f"No SPACE_WEATHER advisory had been issued by {_t(start)}."
    a = adv.iloc[-1]
    if space_weather_is_cause:
        return f"Advisory {a['id']} (issued {_t(a.ts)}) reports storming in progress: {a.text}"
    return (f"Advisory {a['id']} (issued {_t(a.ts)}) says storming was in progress, but an advisory does not "
            f"say when it ends, so it is not used as the cause here; the S4 value decides.")


def s4_one_machine(ep, sda):
    """S4 sentence for a cause that hits ONE tractor: quiet S4 rules out a storm; elevated S4 means a storm
    is present but it hits all three alike, so it cannot explain a one-tractor effect."""
    med, mx = s4_window(sda, ep["start"], ep["end"])
    if med < STORM_MIN_S4:
        return f"S4 median {med:.2f} (max {mx:.2f}) is quiet -> not space weather."
    return (f"S4 is elevated (median {med:.2f}, max {mx:.2f}) so a storm is present, but a storm hits all 3 tractors "
            f"alike and this effect is on {','.join(ep['tractors'])} only -> the storm does not explain it.")


def rfi_near(sda, ep):
    """RFI reports within an hour of the episode (empty frame if none)."""
    r = sda["rfi"]
    return r[(r.ts >= ep["start"] - RFI_WINDOW) & (r.ts <= ep["end"] + RFI_WINDOW)]


def _rfi_line(sda, ep, farm_lat, farm_lon, consistent):
    """Nearest RFI report within an hour of the episode: when, where, and whether it is used as the cause."""
    r = sda["rfi"]
    near = rfi_near(sda, ep)
    if near.empty:
        return (f"No RFI report within {int(RFI_WINDOW.total_seconds() // 60)} min of "
                f"{_span(ep)} ({len(r)} reports in the feed).")
    x = near.assign(gap=(near.ts - ep["start"]).abs()).sort_values("gap").iloc[0]
    if ep["start"] <= x.ts < ep["end"]:
        when = "during the episode"
    elif x.ts < ep["start"]:
        when = f"{(ep['start'] - x.ts).total_seconds() / 60:.0f} min before it starts"
    else:
        when = f"{(x.ts - ep['end']).total_seconds() / 60:.0f} min after it ends"
    km = float(np.hypot((x.centerLon - farm_lon) * M_PER_DEG_LON, (x.centerLat - farm_lat) * M_PER_DEG_LAT) / 1000)
    verdict = ("consistent with interference on this episode" if consistent
               else "not used as the cause: the measurements above do not look like interference")
    return (f"RFI report {x['id']} at {_t(x.ts, True)} ({when}): band {x.band}, confidence {x.confidence:.2f}, "
            f"centre {km:.1f} km from the farm (ellipse {x.semiMajor_km:.1f} x {x.semiMinor_km:.1f} km), "
            f"\"{x.text}\" -> {verdict}.")


# ----------------------------------------------------------------------------------------------
# one function per class
# ----------------------------------------------------------------------------------------------
def _space_weather(ep, minute, sda, others):
    w = _window(minute, ep["start"], ep["end"])
    drop = _per_tractor(w, "cn0_drop")
    s4_med, s4_max = s4_window(sda, ep["start"], ep["end"])
    lat, lon = _farm_position(w)
    pos, pos_skip = _without(w, ep, others, {"SPOOFING"})
    link, link_skip = _without(w, ep, others, {"EQUIPMENT_FAULT"})
    return [
        f"All 3 tractors lost C/N0 together over {_span(ep)}: mean drop vs own quiet baseline "
        f"{_list(drop, ' dB')} (spread between tractors {drop.max() - drop.min():.1f} dB) -> a regional effect, not one machine.",
        f"S4 scintillation index median {s4_med:.2f}, max {s4_max:.2f} in the window (quiet days stay below 0.1); "
        f"Kp {_latest(sda['kp'], 'kp', ep['start']):.1f}, Dst {_latest(sda['dst'], 'dst_nT', ep['start']):.0f} nT at {_t(ep['start'])}.",
        _advisory_line(sda, ep["start"], True),
        f"Reported position stays on the formation line (max offset {pos.lon_dev_abs.max():.2f} m"
        f"{_excluded_note(pos_skip, 'SPOOFING')}) and the base link is up (max {link.link_down_frac.max():.0%} link-down per "
        f"minute{_excluded_note(link_skip, 'EQUIPMENT_FAULT')}) -> the storm itself does not move positions or drop links.",
        _rfi_line(sda, ep, lat, lon, consistent=False),
    ]


def _jamming(ep, minute, sda, others):
    w = _window(minute, ep["start"], ep["end"])
    hit = ep["tractors"]
    excess = pd.Series(ep["stats"]["mean_excess_db"])
    s4_med, s4_max = s4_window(sda, ep["start"], ep["end"])
    lat, lon = _farm_position(w)
    wh = w[w.tractor_id.isin(hit)]
    pos, pos_skip = _without(wh, ep, others, {"SPOOFING"})
    lines = [
        f"C/N0 loss beyond the quiet baseline (or the storm level, if S4 is elevated) averaged over {_span(ep)}: "
        f"{_list(excess, ' dB')}; tractors at or above {JAM_TRACTOR_DROP_DB} dB are reported: {','.join(hit)}.",
        f"On {','.join(hit)}: mean satellites {wh.sats.mean():.1f} (quiet 11-13), receiver had NO_FIX in "
        f"{ep['stats']['no_fix_frac']:.0%} of working seconds, RTK-fixed in {wh.fix_ok_frac.mean():.0%}.",
    ]
    # Be honest about tractors near the decision line: they are the weakest part of the claim.
    weak = [f"{k} {v:.1f} dB" for k, v in excess.items() if k in hit and v < JAM_TRACTOR_DROP_DB + 2]
    skipped = [f"{k} {v:.1f} dB" for k, v in excess.items() if k not in hit and v >= 2]
    if weak:
        lines.append(f"Weakest inclusion: {', '.join(weak)}, within 2 dB of the {JAM_TRACTOR_DROP_DB} dB cut-off.")
    if skipped:
        lines.append(f"Not reported: {', '.join(skipped)}, below the {JAM_TRACTOR_DROP_DB} dB cut-off.")
    if s4_med < STORM_MIN_S4:
        lines.append(f"S4 median {s4_med:.2f} (max {s4_max:.2f}) is quiet -> the drop is not space weather. "
                     + _advisory_line(sda, ep["start"], False))
    else:
        lines.append(f"S4 is elevated (median {s4_med:.2f}, max {s4_max:.2f}), so a storm is also present; the drop above is "
                     f"measured beyond the S4-predicted storm level, so the storm alone does not explain it.")
    lines.append(f"Reported position stays on the formation line (max offset {pos.lon_dev_abs.max():.2f} m"
                 f"{_excluded_note(pos_skip, 'SPOOFING')}) -> the loss is signal, not a displaced position.")
    lines.append(_rfi_line(sda, ep, lat, lon, consistent=True))
    loc = ep.get("location")
    if isinstance(loc, dict):
        lines.append(_location_line(loc, sda, ep))
    return lines


def _location_line(loc, sda, ep):
    """Where the fitted jammer is, how sure that is, and how it compares with an RFI report (if any)."""
    where = (f"{loc['lat']:.4f}N {abs(loc['lon']):.4f}W, 95% error ellipse {loc['semi_major_km'] * 1000:.0f} x "
             f"{loc['semi_minor_km'] * 1000:.0f} m (long axis bearing {loc['major_axis_bearing_deg']:.0f} deg)")
    fit = (f"fit of C/N0 = a + b*log10(distance) to {loc['n_samples']} 10-s samples from all 3 tractors: "
           f"RMSE {loc['rmse_db']:.2f} dB, b = {loc['slope_db_per_decade']:.1f} dB/decade (free space 20)")
    if not loc["reliable"]:
        return (f"Jammer location NOT reported: {fit}; {', '.join(loc['notes'])}.")
    near = rfi_near(sda, ep)
    if near.empty:
        cross = "no RFI report to cross-check"
    else:
        x = near.assign(gap=(near.ts - ep["start"]).abs()).sort_values("gap").iloc[0]
        km = locate.distance_km(loc["lat"], loc["lon"], x.centerLat, x.centerLon)
        inside = ("inside" if km <= x.semiMinor_km else "within the long axis of" if km <= x.semiMajor_km else "outside")
        cross = (f"{km:.1f} km from the {x['id']} centre, {inside} its {x.semiMajor_km:.1f} x {x.semiMinor_km:.1f} km ellipse "
                 f"(orientation not given); the fit is far more precise than the report")
    boot = ""
    if loc.get("boot_major_km") is not None:
        boot = (f" An independent block bootstrap (100 refits, resampling 60-s blocks per tractor) gives a 95% ellipse of "
                f"{loc['boot_major_km'] * 1000:.0f} x {loc['boot_minor_km'] * 1000:.0f} m (bearing "
                f"{loc['boot_bearing_deg']:.0f} deg), "
                + ("narrower than" if loc['boot_major_km'] <= loc['semi_major_km'] else "wider than")
                + " the ellipse above (the bootstrap captures sampling noise only; the ellipse above also carries the "
                "sensitivity shift).")
    return (f"Fitted jammer position {where}; {fit}. The ellipse is the fit's precision under the path-loss model "
            f"(statistical band {loc['stat_major_km'] * 1000:.0f} m + {loc['sensitivity_m']:.0f} m shift when refitting each half "
            f"of the episode / dropping each tractor), not a surveyed truth;{boot} {cross}.")


def _spoofing(ep, minute, sda, others):
    trs = ep["tractors"]
    who = ",".join(trs)
    s = ep["stats"]
    w = _window(minute, ep["start"], ep["end"])
    mine, rest = w[w.tractor_id.isin(trs)], w[~w.tractor_id.isin(trs)]
    lat, lon = _farm_position(w)
    lines = [
        f"{who}: reported longitude drifted off the lockstep formation line, crossing 0.5 m at {_t(s['first_cross'], True)} "
        f"at {s['ramp_m_per_min']:.2f} m/min and reaching {s['max_offset_m']:.1f} m; onset back-extrapolated to {_t(ep['start'])}.",
    ]
    if s.get("xte_mismatch"):
        lines.append(f"WARNING: the guidance monitor does NOT show this offset (peak cross-track error {s['xte_max_m']:.2f} m vs peak "
                     f"offset {s['max_offset_m']:.2f} m), so the lockstep offset may be a relative artifact; low confidence.")
    else:
        lines.append(f"Peak cross-track error {s['xte_max_m']:.2f} m vs peak position offset {s['max_offset_m']:.2f} m: the guidance "
                     f"monitor on {who} saw the displacement the position report shows.")
    if rest.empty:
        lines.append(f"{who}: C/N0 {_delta(mine):+.1f} dB vs its quiet baseline, satellites {mine.sats.mean():.1f}, RTK-fixed "
                     f"{mine.fix_ok_frac.mean():.0%}; no unaffected neighbour to compare against.")
        lines.append(s4_one_machine(ep, sda))
    else:
        gap = _delta(mine) - _delta(rest)
        jam_note = ("no more C/N0 loss than its neighbours, so not jamming (jamming adds loss beyond what neighbours see)"
                    if abs(gap) < 2 else f"it lost {-gap:.1f} dB more than its neighbours, so a second effect is not excluded")
        lines.append(f"{who}: C/N0 {_delta(mine):+.1f} dB vs its quiet baseline, neighbours {_delta(rest):+.1f} dB; satellites "
                     f"{mine.sats.mean():.1f} (neighbours {rest.sats.mean():.1f}); RTK-fixed {mine.fix_ok_frac.mean():.0%} "
                     f"(neighbours {rest.fix_ok_frac.mean():.0%}). Jamming check: {jam_note}.")
        lines.append(f"Other tractors stay in formation (max offset {rest.lon_dev_abs.max():.2f} m) -> not a regional effect. "
                     + s4_one_machine(ep, sda))
    lines.append(_rfi_line(sda, ep, lat, lon, consistent=False))
    return lines


def _equipment(ep, minute, sda, others):
    tr = ep["tractors"][0]
    sub = ep["stats"]["subtype"]
    w = _window(minute, ep["start"], ep["end"])
    mine, rest = w[w.tractor_id == tr], w[w.tractor_id != tr]
    lat, lon = _farm_position(w)
    mates = ",".join(sorted(set(TRACTORS) - {tr}))
    if sub == "link":
        first = (f"{tr}'s base-station link was down in {mine.link_down_frac.mean():.0%} of working seconds over "
                 f"{_span(ep)} (quiet: 0%); neighbours {_list(_per_tractor(rest, 'link_down_frac') * 100, '%', '.0f')}.")
        second = (f"{tr}'s C/N0 is {_delta(mine):+.1f} dB vs its quiet baseline, so any signal loss is not what "
                  f"marks this episode -> a correction-link fault, not jamming.")
        neighbours = f"Neighbours {mates}: link-down {rest.link_down_frac.mean():.0%}"
    else:
        first = (f"{tr}'s lowest-channel C/N0 dropped >= 6 dB below its own running median in "
                 f"{ep['stats']['flag_frac']:.0%} of working seconds over {_span(ep)} (lowest value {mine.cn0_min.min():.1f} dB-Hz; "
                 f"quiet floor ~36.9).")
        second = (f"{tr}: mean satellites {mine.sats.mean():.1f}, C/N0 {_delta(mine):+.1f} dB vs baseline: short per-channel "
                  f"dropouts rather than a sustained signal loss -> antenna/receiver fault.")
        neighbours = f"Neighbours {mates}: {rest.min_low_frac.mean():.1%} of seconds with min C/N0 below 35 dB-Hz"
    return [
        first,
        second,
        f"{neighbours} (C/N0 {_delta(rest):+.1f} dB vs baseline) -> the fault is local to {tr}; the others can continue. "
        + s4_one_machine(ep, sda),
        _rfi_line(sda, ep, lat, lon, consistent=False),
    ]


def _multipath(ep, minute, sda, others):
    tr, s = ep["tractors"][0], ep["stats"]
    s4 = _latest(sda["s4"], "s4", ep["start"])
    n = s["n_tractors_dipped"]
    xte_note = "within" if s["xte_in"] <= QUIET_XTE_M else "ABOVE"
    where = ("all 3 tractors dipped in the same second -> tied to the location (north headland), not to this machine"
             if n == 3 else f"only {n} of 3 tractors dipped in this second, which weakens the geometry explanation")
    depth = "could not be measured (no data around the pass)" if pd.isna(s["min_drop_db"]) else f"fell {s['min_drop_db']:.1f} dB"
    covering = covering_causes(ep, others)
    if covering:   # XTE and S4 here belong to the other cause, so they say nothing about the dip itself
        third = (f"Cross-track error {s['xte_in']:.2f} m and S4 {s4:.2f} during the dip reflect the overlapping "
                 f"{', '.join(covering)} detection, not the dip; the dip matches the headland pattern, so it is reported as "
                 f"multipath and the other cause separately.")
    else:
        third = (f"Cross-track error during the dip {s['xte_in']:.2f} m, {xte_note} the quiet range (<= {QUIET_XTE_M} m); S4 {s4:.2f} at "
                 f"{_t(ep['start'])} -> " + ("guidance is not degraded and there is no storm: a false alarm."
                                             if s["xte_in"] <= QUIET_XTE_M and s4 < STORM_MIN_S4
                                             else "guidance or S4 is elevated with no other detection to explain it; treat with caution."))
    return [
        f"{tr}: minimum C/N0 {depth} for {(ep['end'] - ep['start']).total_seconds():.0f} s "
        f"({_span(ep, True)}) while its reported position was in the north-headland zone.",
        f"{where}; the same dip recurs at every north-headland pass.",
        third,
    ]


WRITERS = {"SPACE_WEATHER": _space_weather, "JAMMING": _jamming, "SPOOFING": _spoofing,
           "EQUIPMENT_FAULT": _equipment, "MULTIPATH": _multipath}


def add_evidence(episodes, minute, sda):
    """Return one list of evidence strings per episode (same order as ``episodes``)."""
    others = [e for e in episodes if e["kind"] != "MULTIPATH"]
    return [WRITERS[ep["kind"]](ep, minute, sda, others) + _overlap_lines(ep, others) for ep in episodes]

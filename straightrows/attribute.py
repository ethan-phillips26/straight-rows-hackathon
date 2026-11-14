"""Turn detector episodes into ``Detection`` records: class -> action, a calibrated confidence, and evidence.

Confidence is EXPERT-SET, not fitted: twelve dev episodes cannot train a calibrator. Each class starts from a base
that says how much its rule rests on (2-3 dev examples each), and is moved by named, checkable conditions that
make the call weaker or stronger. Every adjustment is written into the last evidence line, so a judge can see why
a number is what it is. Wrong-but-confident is penalised, so the conditions only lower confidence except where an
independent feed agrees (an RFI report for jamming).
"""
import pandas as pd

from .detect import JAM_TRACTOR_DROP_DB, STORM_MIN_S4
from .evidence import add_evidence, covering_causes, overlapping, rfi_near, s4_window
from .output import Detection

BASE = {"SPACE_WEATHER": 0.90, "JAMMING": 0.85, "SPOOFING": 0.90, "EQUIPMENT_FAULT": 0.80, "MULTIPATH": 0.90}
CAP, FLOOR = 0.95, 0.05

# Thresholds used by the adjustments (each says what it is compared with).
STORM_CLEAR_S4 = 0.3        # dev storms have S4 >= 0.45; below this a storm call is marginal
WEAK_MARGIN_DB = 2.0        # an affected tractor this close to the jamming cut-off is the weakest part of the call
SMALL_SPOOF_M = 2.0         # dev spoof plateaus are 4-12 m; a smaller offset is close to noise-level evidence
LOW_DUTY = 0.10             # equipment flag rate: dev faults are ~22-30 %; a link fault at 4 % was found 18 min late
SHALLOW_DIP_DB = 3.0        # measured multipath dips are 3.5-10.9 dB
QUIET_XTE_M = 0.45


def _has_sw_advisory(sda, start):
    adv = sda["advisories"]
    return bool(((adv.type == "SPACE_WEATHER") & (adv.ts <= start)).any())


def confidence(ep, sda, others):
    """Return (confidence, [reasons]) for one episode. Reasons are plain sentences for the evidence list."""
    kind, s = ep["kind"], ep["stats"]
    c, why = BASE[kind], []

    def adjust(delta, reason):
        nonlocal c
        c += delta
        why.append(f"{delta:+.2f} {reason}")

    if kind == "SPACE_WEATHER":
        if not _has_sw_advisory(sda, ep["start"]):
            adjust(-0.10, "no SPACE_WEATHER advisory issued by the start (the feeds do not corroborate)")
        if s4_window(sda, ep["start"], ep["end"])[0] < STORM_CLEAR_S4:
            adjust(-0.10, f"S4 median below {STORM_CLEAR_S4}, a marginal storm")
    elif kind == "JAMMING":
        excess = s["mean_excess_db"]
        weak = [t for t in ep["tractors"] if excess[t] < JAM_TRACTOR_DROP_DB + WEAK_MARGIN_DB]
        unclear = [t for t, v in excess.items() if t not in ep["tractors"] and v >= WEAK_MARGIN_DB]
        if weak:
            adjust(-0.10, f"{','.join(weak)} only just above the {JAM_TRACTOR_DROP_DB} dB cut-off (one dev example supports the cut-off)")
        if unclear:
            adjust(-0.05, f"{','.join(unclear)} lost {WEAK_MARGIN_DB}+ dB but is not reported")
        if s4_window(sda, ep["start"], ep["end"])[0] >= STORM_MIN_S4:
            adjust(-0.10, "inside a storm: the drop is measured beyond a storm level fitted on two storms")
        if not rfi_near(sda, ep).empty:
            adjust(+0.05, "an RFI report within 60 min of the episode agrees")
    elif kind == "SPOOFING":
        if s.get("xte_mismatch"):
            adjust(-0.40, "the tractor's own guidance monitor does not show the offset")
        if s["max_offset_m"] < SMALL_SPOOF_M:
            adjust(-0.10, f"peak offset under {SMALL_SPOOF_M} m")
        if overlapping(ep, others, {"SPACE_WEATHER"}):
            adjust(-0.05, "overlaps a storm (one dev example of that overlap)")
    elif kind == "EQUIPMENT_FAULT":
        if s["flag_frac"] < LOW_DUTY:
            adjust(-0.10, f"fault active in under {LOW_DUTY:.0%} of working seconds")
        if overlapping(ep, others, {"SPOOFING", "JAMMING", "SPACE_WEATHER"}):
            adjust(-0.05, "overlaps another cause")
    elif kind == "MULTIPATH":
        if s["n_tractors_dipped"] < 3:
            adjust(-0.30, f"only {s['n_tractors_dipped']} of 3 tractors dipped (geometry predicts all 3)")
        if pd.isna(s["min_drop_db"]) or s["min_drop_db"] < SHALLOW_DIP_DB:
            adjust(-0.20, f"dip depth unmeasured or under {SHALLOW_DIP_DB} dB")
        # XTE during a storm or spoof is legitimately high; only penalise it when no other detection explains it.
        if s["xte_in"] > QUIET_XTE_M and not covering_causes(ep, others):
            adjust(-0.20, f"cross-track error during the dip above the quiet {QUIET_XTE_M} m with no other cause detected")
    return min(max(c, FLOOR), CAP), why


def attribute(episodes, minute, sda):
    """``episodes``: DataFrame (or list of dicts) from ``detect.detect_all``. Returns a list of ``Detection``."""
    eps = episodes.to_dict("records") if isinstance(episodes, pd.DataFrame) else list(episodes)
    others = [e for e in eps if e["kind"] != "MULTIPATH"]
    detections = []
    for ep, lines in zip(eps, add_evidence(eps, minute, sda)):
        conf, why = confidence(ep, sda, others)
        adjustments = "; ".join(why) if why else "no adjustments"
        lines = lines + [f"Confidence {conf:.2f}: base {BASE[ep['kind']]:.2f} for {ep['kind']} (expert-set; each rule rests on "
                         f"2-3 dev episodes, not a fitted model); {adjustments}."]
        detections.append(Detection(start=ep["start"], end=ep["end"], tractors=list(ep["tractors"]), cls=ep["kind"],
                                    confidence=conf, evidence=lines))
    return detections

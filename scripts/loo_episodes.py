#!/usr/bin/env python3
"""Leave-one-episode-out check of the detector's gap thresholds (docs/FINDINGS.md section 6: "12 dev episodes").

python scripts/loo_episodes.py

Most detector constants sit in a gap: a quiet extreme (what healthy data never exceeds) below, the smallest real
effect above. This script asks: if one labeled episode had never existed, would the rule that put each constant in
its gap still catch it?

The rule (one rule for every constant, so nothing is hand-fitted per episode):
    threshold = geometric mean of (quiet extreme, smallest effect among the episodes still allowed to vote)
Geometric, not arithmetic, because these gaps span a decade or more (quiet lon offset 0.05 m vs spoof 4.5 m).
Steps:
  1. CONTROL: derive the constants from all 12 episodes. If the rule itself cannot give 12/12, a later failure
     would say nothing about overfitting, so this runs first.
  2. For each non-multipath episode, hide it, re-derive ONLY its own class's constants from the other episodes of
     that class, rerun the detectors on dev, and judge the hidden episode strictly (right class, exact tractor
     set, onset within 5 min) plus the dev-wide false-positive count.
Multipath is skipped: 972 identical passes, the two labels are the same event (FINDINGS 0.5).
Not covered (not gap-type): the storm line (42.7, 17.7), grid and gap-merging constants. sweep_thresholds.py covers those.
"""
import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import eval_dev  # noqa: E402
from straightrows import detect as D  # noqa: E402
from straightrows.features import build_features  # noqa: E402
from straightrows.load import load_sda, load_telemetry  # noqa: E402

QUIET = (pd.Timestamp("2026-05-04T17:15Z"), pd.Timestamp("2026-05-04T20:55Z"))   # as in verify_findings.py
ONE_SECOND_PER_MINUTE = 1 / 60      # smallest nonzero per-minute flag fraction; stands in for "quiet = 0"

# constants re-derived per class
CLASS_CONSTANTS = {
    "SPACE_WEATHER": ["STORM_MIN_DROP_DB", "STORM_MIN_S4"],
    "JAMMING": ["JAM_MINUTE_DROP_DB", "JAM_TRACTOR_DROP_DB"],
    "SPOOFING": ["SPOOF_MIN_DEV_M"],
    "EQUIPMENT_FAULT": ["LINK_MIN_FRAC", "ANT_MIN_FRAC"],
}


def geo(lower, upper):
    return math.sqrt(lower * upper)


def quiet_extremes(sec, minute):
    q = minute[(minute.ts >= QUIET[0]) & (minute.ts < QUIET[1])]
    qs = sec[(sec.ts >= QUIET[0]) & (sec.ts < QUIET[1]) & sec.working & ~sec.headland]
    # per-second drops, as FINDINGS 0.1 quotes them (quiet max 0.99 dB); minute means sit below the baseline and give ~0
    return {"drop": float(qs.cn0_drop.max()), "fleet_drop": float(qs.fleet_cn0_drop.max()), "s4": float(q.s4.max()),
            "dev": float(qs.lon_dev_abs_m.max())}


def effects(dets, labels):
    """One row per labeled episode: the number each gap constant is compared with (from the detector's own stats)."""
    out = {}
    for lab in labels:
        if lab["class"] == "MULTIPATH":
            continue
        s, e = pd.Timestamp(lab["start"]), pd.Timestamp(lab["end"])
        hit = dets[(dets.kind == lab["class"]) & (dets.start < e) & (dets.end > s)]
        if hit.empty:
            raise SystemExit(f'{lab["id"]}: baseline does not detect it; fix that before running leave-one-out')
        st = hit.iloc[0].stats
        row = {"class": lab["class"]}
        if lab["class"] == "SPACE_WEATHER":
            row.update(drop=st["fleet_drop_db"], s4=st["s4_median"])
        elif lab["class"] == "JAMMING":
            ex = st["mean_excess_db"]
            row.update(affected=[ex[t] for t in lab["tractors"]], unaffected=[v for t, v in ex.items() if t not in lab["tractors"]])
        elif lab["class"] == "SPOOFING":
            row.update(dev=st["max_offset_m"])
        else:
            row.update(frac=st["flag_frac"])
        out[lab["id"]] = row
    return out


def derive(cls, voters, q):
    """Constants for ``cls`` from the episodes in ``voters`` (dict id -> effect row of that class)."""
    rows = list(voters.values())
    if cls == "SPACE_WEATHER":
        return {"STORM_MIN_DROP_DB": geo(q["fleet_drop"], min(r["drop"] for r in rows)),
                "STORM_MIN_S4": geo(q["s4"], min(r["s4"] for r in rows))}
    if cls == "JAMMING":
        affected = [v for r in rows for v in r["affected"]]
        lower = max([q["drop"]] + [v for r in rows for v in r["unaffected"]])   # loudest tractor that was NOT jammed
        return {"JAM_MINUTE_DROP_DB": geo(q["drop"], min(affected)), "JAM_TRACTOR_DROP_DB": geo(lower, min(affected))}
    if cls == "SPOOFING":
        return {"SPOOF_MIN_DEV_M": geo(q["dev"], min(r["dev"] for r in rows))}
    t = geo(ONE_SECOND_PER_MINUTE, min(r["frac"] for r in rows))
    return {"LINK_MIN_FRAC": t, "ANT_MIN_FRAC": t}


def run(minute, sec, passes, labels, constants):
    saved = {k: getattr(D, k) for k in constants}
    try:
        for k, v in constants.items():
            setattr(D, k, v)
        dets = D.detect_all(sec, minute, passes, (eval_dev.DEV_START, eval_dev.DEV_END))
    finally:
        for k, v in saved.items():
            setattr(D, k, v)
    payload = [{"start": r.start.isoformat(), "end": r.end.isoformat(), "tractors": r.tractors, "class": r.kind,
                "action": D.CLASS_ACTION[r.kind], "confidence": 0.5, "evidence": []} for r in dets.itertuples()]
    result = eval_dev.score({"detections": payload}, {"episodes": labels})
    return {e["episode"]: e for e in result["episodes"]}, len(result["false_positives"])


def strict_ok(e):
    return e.get("predicted_class") == e["truth_class"] and e.get("tractor_iou") == 1 and abs(e.get("onset_error_s", 1e9)) <= 300


def main(data=ROOT / "starter" / "data"):
    tele, sda = load_telemetry(data), load_sda(data)
    labels = json.loads((data / "labels" / "train_episodes.json").read_text())["episodes"]
    sec, minute, passes = build_features(tele[tele.ts < eval_dev.DEV_END], sda)
    q = quiet_extremes(sec, minute)
    print("quiet extremes (05-04 17:15-20:55Z): " + ", ".join(f"{k}={v:.3g}" for k, v in q.items()))

    base_dets = D.detect_all(sec, minute, passes, (eval_dev.DEV_START, eval_dev.DEV_END))
    eff = effects(base_dets, labels)

    def derive_all(skip=None):
        c = {}
        for cls in CLASS_CONSTANTS:
            voters = {i: r for i, r in eff.items() if r["class"] == cls and i != skip}
            if voters:
                c.update(derive(cls, voters, q))
        return c

    print(f"\nCONTROL (rule fitted on all 12):")
    control = derive_all()
    results, fps = run(minute, sec, passes, labels, control)
    ok = [i for i, e in results.items() if strict_ok(e)]
    print(f"  strict {len(ok)}/{len(results)}, {fps} FP")
    for k, v in control.items():
        print(f"  {k:20} rule {v:7.3g}   current {getattr(D, k):7.3g}")
    for i, e in results.items():
        if not strict_ok(e):
            print(f"  control miss {i}: predicted {e.get('predicted_class')}, tractorIoU {e.get('tractor_iou')}, onset err {e.get('onset_error_s')}")
    if len(ok) != len(results) or fps:
        print("  control fails: the rule is too crude to test anything; stopping.")
        return

    print("\nLEAVE-ONE-OUT (hidden episode: strict result, dev false positives, derived vs control constants):")
    flips = 0
    for lab in labels:
        i, cls = lab["id"], lab["class"]
        if cls == "MULTIPATH":
            print(f"  {i} {cls:16} skipped (972 identical passes)")
            continue
        voters = {k: r for k, r in eff.items() if r["class"] == cls and k != i}
        if not voters:
            print(f"  {i} {cls:16} no other episode of this class: nothing to derive from")
            continue
        # only this class's constants change; every other class keeps the control values
        consts = dict(control)
        consts.update(derive(cls, voters, q))
        res, fp = run(minute, sec, passes, labels, consts)
        moved = ", ".join(f"{k} {control[k]:.3g}->{consts[k]:.3g}" for k in CLASS_CONSTANTS[cls])
        bad = [k for k, e in res.items() if not strict_ok(e)]
        flips += i in bad
        print(f"  {i} {cls:16} {'HELD OUT OK ' if i not in bad else 'MISSED      '} FP={fp}  {moved}"
              + (f"  also broke: {[b for b in bad if b != i]}" if [b for b in bad if b != i] else ""))
    print(f"\n{flips} episode(s) lost when held out.")


if __name__ == "__main__":
    main()

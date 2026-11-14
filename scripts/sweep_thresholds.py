#!/usr/bin/env python3
"""How sensitive is the dev result to each detector constant? (docs/REVIEW.md item 5)

python scripts/sweep_thresholds.py            # each constant x0.8 and x1.2, one at a time

Two criteria per run on the dev days (05-04, 05-05):
  score   the organizers' diagnostic (/40). It forgives a wrong tractor set if tractor IoU >= 0.5.
  strict  right class, EXACT tractor set, onset within 5 min, no false positive. This is the one that can see a
          tractor dropped from a jamming episode, which the diagnostic cannot.
A constant is "robust" if both criteria are unchanged at x0.8 and x1.2. Percent is a poor unit for the storm line
(an intercept of 42.7 dB moves 8.5 dB at 20 %), so that constant is also reported as a shift in dB.
Constants are module globals that the detectors read at call time, so setting them on the module is enough.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import eval_dev  # noqa: E402
from straightrows import detect as D  # noqa: E402
from straightrows.features import build_features  # noqa: E402
from straightrows.load import load_sda, load_telemetry  # noqa: E402

CONSTANTS = ["STORM_MIN_DROP_DB", "STORM_MIN_S4", "STORM_MIN_MIN", "STORM_CN0_AT_S4_0", "STORM_CN0_PER_S4",
             "STORM_MAX_SPREAD_DB", "STORM_MAX_LEVEL_ERR_DB", "STORM_MIN_CLEAN_FRAC", "JAM_MINUTE_DROP_DB", "JAM_TRACTOR_DROP_DB", "JAM_MIN_MIN",
             "JAM_MAX_GAP_MIN", "SPOOF_MIN_DEV_M", "SPOOF_MIN_MIN", "SPOOF_MAX_GAP_S", "SPOOF_RAMP_FIT_S", "SPOOF_XTE_TOL_M",
             "LINK_MIN_FRAC", "ANT_DIP_DB", "ANT_MIN_FRAC", "EQUIP_MIN_MIN", "EQUIP_MAX_GAP_MIN"]
STORM_SHIFTS_DB = [-6.0, -3.0, 3.0, 3.5, 4.0]


def run(minute, sec, passes, labels):
    dets = D.detect_all(sec, minute, passes, (eval_dev.DEV_START, eval_dev.DEV_END))
    payload = [{"start": r.start.isoformat(), "end": r.end.isoformat(), "tractors": r.tractors, "class": r.kind,
                "action": D.CLASS_ACTION[r.kind], "confidence": 0.5, "evidence": []} for r in dets.itertuples()]
    result = eval_dev.score({"detections": payload}, labels)
    strict = (all(e.get("predicted_class") == e["truth_class"] and e["tractor_iou"] == 1 and abs(e["onset_error_s"]) <= 300
                  for e in result["episodes"]) and not result["false_positives"])
    return result["diagnostic_score"], strict, len(result["false_positives"])


def main(data=ROOT / "starter" / "data"):
    tele, sda = load_telemetry(data), load_sda(data)
    labels = json.loads((data / "labels" / "train_episodes.json").read_text())
    sec, minute, passes = build_features(tele[tele.ts < eval_dev.DEV_END], sda)
    base = run(minute, sec, passes, labels)
    print(f"baseline: score {base[0]:.2f}/40, strict {'ok' if base[1] else 'FAIL'}, {base[2]} FP\n")
    print(f'{"constant":24} {"value":>8}   {"x0.8":^22} {"x1.2":^22}')
    for name in CONSTANTS:
        original = getattr(D, name)
        cells = []
        for factor in (0.8, 1.2):
            value = type(original)(round(original * factor)) if isinstance(original, int) else original * factor
            setattr(D, name, value)
            score, strict, fps = run(minute, sec, passes, labels)
            cells.append(f"{score:5.2f} {'ok  ' if strict else 'FAIL'} {fps}FP" + ("" if (score, strict) != (base[0], base[1]) else "  same"))
        setattr(D, name, original)
        print(f"{name:24} {original:8.3g}   {cells[0]:22} {cells[1]:22}")
    print("\nstorm line intercept as a shift in dB (measured residual on dev: <= 0.53 dB):")
    for shift in STORM_SHIFTS_DB:
        D.STORM_CN0_AT_S4_0 += shift
        score, strict, fps = run(minute, sec, passes, labels)
        D.STORM_CN0_AT_S4_0 -= shift
        print(f"  {shift:+4.1f} dB: score {score:5.2f}, strict {'ok' if strict else 'FAIL'}, {fps} FP")


if __name__ == "__main__":
    main()

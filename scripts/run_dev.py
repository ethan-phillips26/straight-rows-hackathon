#!/usr/bin/env python3
"""Run the detectors on the dev window (05-04, 05-05) and score them with scripts/eval_dev.py's scorer.

python scripts/run_dev.py            # print detections + dev score

Confidence and evidence are placeholders here (attribute.py / evidence.py come later); this only checks
WHICH windows/tractors/classes we find.
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import eval_dev  # noqa: E402
from straightrows.detect import CLASS_ACTION, detect_all  # noqa: E402
from straightrows.features import build_features  # noqa: E402
from straightrows.load import load_sda, load_telemetry  # noqa: E402


def main(data=ROOT / "starter" / "data"):
    tele, sda = load_telemetry(data), load_sda(data)
    sec, minute, passes = build_features(tele, sda)
    dets = detect_all(sec, minute, passes, (eval_dev.DEV_START, eval_dev.DEV_END))   # clipped, not start-filtered

    non_mp = dets[dets.kind != "MULTIPATH"]
    print(f"{len(dets)} detections ({len(dets) - len(non_mp)} multipath)")
    for r in non_mp.itertuples():
        print(f"{r.kind:16} {r.start:%m-%d %H:%M} -> {r.end:%H:%M} {','.join(r.tractors):9} {r.stats}")

    payload = [{"start": r.start.isoformat(), "end": r.end.isoformat(), "tractors": r.tractors, "class": r.kind,
                "action": CLASS_ACTION[r.kind], "confidence": 0.5, "evidence": []} for r in dets.itertuples()]
    print()
    eval_dev.print_report(eval_dev.score({"detections": payload}, json.loads(
        (data / "labels" / "train_episodes.json").read_text())))


if __name__ == "__main__":
    main()

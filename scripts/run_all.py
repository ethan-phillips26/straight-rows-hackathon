#!/usr/bin/env python3
"""Regenerate submission.json from the dataset: load -> features -> detect -> attribute -> output.

python scripts/run_all.py --data starter/data --out submission.json --github-handle <handle> --repo-url <url>

Deterministic: no RNG, sorted outputs, no clock use. Detections cover the evaluation window only.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from straightrows import attribute, detect, features, load, output  # noqa: E402

EVAL_START = pd.Timestamp("2026-05-06T00:00:00Z")
EVAL_END = pd.Timestamp("2026-05-07T00:00:00Z")


def run(data, window=(EVAL_START, EVAL_END)):
    """Return the list of Detection objects for `window`. Features use the whole dataset (baselines)."""
    tele = load.load_telemetry(data)
    sda = load.load_sda(data)
    sec, minute, passes = features.build_features(tele, sda)
    dets = detect.detect_all(sec, minute, passes, window)      # independent detector per cause class
    return attribute.attribute(dets, sda)                      # class, action, confidence, evidence


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "starter" / "data"))
    ap.add_argument("--out", default="submission.json")
    ap.add_argument("--github-handle", required=True)
    ap.add_argument("--repo-url", required=True)
    ap.add_argument("--display-name", default=None)
    args = ap.parse_args()

    dets = run(args.data)
    sub = output.write_submission(dets, args.out, args.github_handle, args.repo_url, args.display_name)
    counts = {}
    for d in sub["detections"]:
        counts[d["class"]] = counts.get(d["class"], 0) + 1
    print(f"wrote {args.out}: {len(sub['detections'])} detections {counts}")
    print("next: python starter/submission/validate_submission.py", args.out)


if __name__ == "__main__":
    main()

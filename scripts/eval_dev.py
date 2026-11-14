#!/usr/bin/env python3
"""Score detections on the 12 labeled dev episodes (2026-05-04 and 05-05) with the organizers' scorer.

python scripts/eval_dev.py --baseline                   # starter baseline, re-pointed at the dev window
python scripts/eval_dev.py --detections dev_dets.json   # any detections file ({"detections": [...]} or a list)

Why this exists: the official answer key (05-06) is private, and both the baseline and our pipeline
only look at 05-06. This script runs a detector on the dev days instead and scores the result with
`scoring/score_submission.py` (copied unchanged from the organizers). The dev label file is already in
the scorer's answer-key format, so it is passed in as-is. The validator is skipped here because it
rejects anything outside 05-06; the real submission must still pass it.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scoring"))
from score_submission import score  # noqa: E402  (organizers' scorer)

DEV_START = pd.Timestamp("2026-05-04T00:00:00Z")
DEV_END = pd.Timestamp("2026-05-06T00:00:00Z")


def baseline_dev_detections(data):
    """Run the unmodified starter baseline with its window constants swapped to the dev window.

    baseline.detect() reads EVAL_START/EVAL_END as module globals, so overriding them is enough;
    the starter file itself is not edited.
    """
    sys.path.insert(0, str(ROOT / "starter" / "submission"))
    import baseline
    baseline.EVAL_START, baseline.EVAL_END = DEV_START, DEV_END
    return baseline.detect(data)


def print_report(result):
    """Per-episode table (with the two overlap numbers that decide matching), per-class metrics, total."""
    print(f'{"episode":8} {"truth":16} {"predicted":16} {"onset err":>10} {"timeIoU":>8} {"trkIoU":>7}')
    for ep in result["episodes"]:
        predicted = ep.get("predicted_class", "MISSED")
        error = f'{ep["onset_error_s"] / 60:+.1f} min' if "onset_error_s" in ep else "-"
        time_iou = f'{ep["time_iou"]:.2f}' if "time_iou" in ep else "-"
        tractor_iou = f'{ep["tractor_iou"]:.2f}' if "tractor_iou" in ep else "-"
        print(f'{ep["episode"]:8} {ep["truth_class"]:16} {predicted:16} {error:>10} {time_iou:>8} {tractor_iou:>7}')
    print()
    for cls, m in result["class_metrics"].items():
        print(f'{cls:16} P={m["precision"]:.3f} R={m["recall"]:.3f} F1={m["f1"]:.3f}  '
              f'(tp={m["tp"]} fp={m["fp"]} fn={m["fn"]})')
    print()
    print(f'DIAGNOSTIC (dev): {result["diagnostic_score"]:.2f}/40   macro-F1={result["macro_f1"]:.3f}   '
          f'mean onset credit={result["mean_timing_credit"]:.3f}')
    print(f'Action agreement: {result["action_agreement"]}/{result["episode_count"]}; '
          f'benign multipath: {result["benign_multipath_reports"]}; false positives: {len(result["false_positives"])}')
    for d in result["false_positives"]:
        print(f'  FP {d["start"][:16]} -> {d["end"][11:16]} {",".join(d["tractors"]):8} {d["class"]}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--detections", help="JSON file of detections on the dev window")
    source.add_argument("--baseline", action="store_true", help="score the starter baseline on the dev window")
    parser.add_argument("--data", default=str(ROOT / "starter" / "data"))
    parser.add_argument("--labels", default=str(ROOT / "starter" / "data" / "labels" / "train_episodes.json"))
    parser.add_argument("--json", dest="json_out", help="also write the full result dict here")
    args = parser.parse_args()

    if args.baseline:
        detections = baseline_dev_detections(args.data)
    else:
        detections = json.loads(Path(args.detections).read_text())
        if isinstance(detections, dict):
            detections = detections["detections"]
    labels = json.loads(Path(args.labels).read_text())

    result = score({"detections": detections}, labels)
    print_report(result)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

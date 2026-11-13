#!/usr/bin/env python3
"""Score detections on the 12 labeled dev episodes (hours 0-48) with the organizers' matcher.

python scripts/eval_dev.py --detections dev_detections.json
python scripts/eval_dev.py --detections dev_detections.json --labels starter/data/labels/train_episodes.json

`--detections` is a JSON file shaped like submission.json ({"detections": [...]}) or a bare list.
TODO (kickoff): when the pipeline exists, run it on the dev window here instead of reading a file.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from straightrows.devscore import format_report, score_dev  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--detections", required=True)
    parser.add_argument("--labels", default="starter/data/labels/train_episodes.json")
    parser.add_argument("--json", dest="json_out", help="also write the full result dict here")
    args = parser.parse_args()

    detections = json.loads(Path(args.detections).read_text())
    if isinstance(detections, dict):
        detections = detections["detections"]
    labels = json.loads(Path(args.labels).read_text())

    result = score_dev(detections, labels)
    print(format_report(result))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

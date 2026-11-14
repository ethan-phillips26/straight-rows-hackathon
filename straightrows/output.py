"""Turn detections into submission.json and refuse to write one the validator would reject.

`Detection` is the one record every stage passes along: detect.py creates them, attribute.py fills in
class / action / confidence / evidence, and this module serialises them.
"""
import json
from dataclasses import dataclass, field
from importlib import util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# Fixed by the challenge (docs/CHALLENGE.md). Never invent other pairs.
CLASS_ACTION = {
    "SPACE_WEATHER": "HALT_OR_DEGRADED_MODE",
    "JAMMING": "HALT_AND_REPORT",
    "SPOOFING": "HALT_CROSSCHECK_AND_REPORT",
    "EQUIPMENT_FAULT": "SERVICE_MACHINE_OTHERS_CONTINUE",
    "MULTIPATH": "CONTINUE_FALSE_ALARM",
}
# The validator's limit (starter/submission/validate_submission.py). Eval emits 493, so there is room for 7 more.
MAX_DETECTIONS = 500


@dataclass
class Detection:
    start: pd.Timestamp          # tz-aware
    end: pd.Timestamp            # tz-aware, > start
    tractors: list               # subset of T1/T2/T3
    cls: str                     # key of CLASS_ACTION
    confidence: float            # [0, 1]
    evidence: list = field(default_factory=list)   # measured, checkable strings

    @property
    def action(self):
        return CLASS_ACTION[self.cls]


def _iso(ts):
    """ISO 8601 in UTC with an explicit +00:00 offset (the format the validator and docs use)."""
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        raise ValueError(f"naive timestamp {ts!r}: timestamps must be timezone-aware")
    return ts.tz_convert("UTC").isoformat()


def _to_dict(d):
    if d.cls not in CLASS_ACTION:
        raise ValueError(f"unknown class {d.cls!r}")
    return {
        "start": _iso(d.start),
        "end": _iso(d.end),
        "tractors": sorted(set(d.tractors)),
        "class": d.cls,
        "action": d.action,
        "confidence": round(float(d.confidence), 3),
        "evidence": [str(e) for e in d.evidence],
    }


def _load_validator():
    path = ROOT / "starter" / "submission" / "validate_submission.py"
    spec = util.spec_from_file_location("validate_submission", path)
    mod = util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.validate


def build_submission(detections, github_handle, repo_url, display_name=None):
    """Return the submission dict. Output is deterministic: detections sorted by time, then class.

    Privacy: only the GitHub handle (and optional display name) identifies the participant.
    Raises ValueError listing every problem if the result would fail the organizers' validator.
    """
    if len(detections) > MAX_DETECTIONS:
        raise ValueError(f"{len(detections)} detections; the validator's limit is {MAX_DETECTIONS}. Almost all are "
                         f"one-per-pass MULTIPATH detections (docs/LOG.md, option A): check for a new cause class "
                         f"firing everywhere, or drop the lowest-confidence MULTIPATH passes.")
    rows = sorted((_to_dict(d) for d in detections),
                  key=lambda r: (r["start"], r["end"], r["class"], r["tractors"]))
    participant = {"github_handle": github_handle}
    if display_name:
        participant["display_name"] = display_name
    sub = {"participant": participant, "repository_url": repo_url, "detections": rows}
    problems = _load_validator()(sub)
    if problems:
        raise ValueError("submission fails validation:\n  - " + "\n  - ".join(problems))
    return sub


def write_submission(detections, path, github_handle, repo_url, display_name=None):
    """Validate, then write. Stable key order and trailing newline so reruns are byte-identical."""
    sub = build_submission(detections, github_handle, repo_url, display_name)
    Path(path).write_text(json.dumps(sub, indent=2) + "\n")
    return sub

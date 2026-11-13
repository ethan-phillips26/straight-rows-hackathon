"""Dev-window scoring: run the organizers' matcher (scoring/score_submission.py) on our detections
against the 12 labeled dev episodes, bypassing the validator (which only accepts the eval window).

The label file's exact key names are unknown until the dataset is released, so `labels_to_key`
accepts a few likely spellings and fails loudly if it can't find a field. Adjust it on Friday.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scoring"))
import score_submission as organizer  # noqa: E402  (organizers' scorer, copied unchanged)

# Likely spellings for each field in train_episodes.json -> our key names. VERIFY at kickoff.
ALIASES = {
    "id": ("id", "episode_id", "episode"),
    "class": ("class", "cause", "truth_class", "label"),
    "start": ("start", "start_utc", "start_time"),
    "end": ("end", "end_utc", "end_time"),
    "tractors": ("tractors", "affected_tractors", "tractor_ids"),
    "recommended_action": ("recommended_action", "action"),
}


def _pick(raw, name):
    for alias in ALIASES[name]:
        if alias in raw:
            return raw[alias]
    raise KeyError(f"label has none of {ALIASES[name]} (keys: {sorted(raw)}). Update ALIASES.")


def labels_to_key(labels):
    """Convert the dev labels (a list, or {"episodes": [...]}) to the scorer's answer-key format."""
    raw_episodes = labels["episodes"] if isinstance(labels, dict) else labels
    episodes = [{name: _pick(raw, name) for name in ALIASES} for raw in raw_episodes]
    return {"episodes": episodes}


def score_dev(detections, labels):
    """Score a list of detection dicts against dev labels. Returns the organizers' result dict."""
    return organizer.score({"detections": detections}, labels_to_key(labels))


def format_report(result):
    """Human-readable table: one row per episode, per-class P/R/F1, then the total."""
    lines = [f'{"episode":10} {"truth":16} {"predicted":16} {"onset err":>10} {"timeIoU":>8} {"trkIoU":>7}']
    for ep in result["episodes"]:
        predicted = ep.get("predicted_class", "MISSED")
        error = f'{ep["onset_error_s"] / 60:+.1f} min' if "onset_error_s" in ep else "-"
        time_iou = f'{ep["time_iou"]:.2f}' if "time_iou" in ep else "-"
        tractor_iou = f'{ep["tractor_iou"]:.2f}' if "tractor_iou" in ep else "-"
        lines.append(f'{str(ep["episode"]):10} {ep["truth_class"]:16} {predicted:16} {error:>10} {time_iou:>8} {tractor_iou:>7}')
    lines.append("")
    for cls, m in result["class_metrics"].items():
        lines.append(f'{cls:16} P={m["precision"]:.3f} R={m["recall"]:.3f} F1={m["f1"]:.3f}  (tp={m["tp"]} fp={m["fp"]} fn={m["fn"]})')
    lines.append("")
    lines.append(f'DIAGNOSTIC (dev): {result["diagnostic_score"]:.2f}/40   macro-F1={result["macro_f1"]:.3f}   '
                 f'mean onset credit={result["mean_timing_credit"]:.3f}')
    lines.append(f'Action agreement: {result["action_agreement"]}/{result["episode_count"]}; '
                 f'false positives: {len(result["false_positives"])}; benign multipath: {result["benign_multipath_reports"]}')
    return "\n".join(lines)

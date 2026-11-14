"""Load the farm telemetry and the four SDA feeds into tidy pandas DataFrames.

Schemas were checked against the released dataset (see docs/CHALLENGE.md, "Verified schemas").
"""
import json
from pathlib import Path

import pandas as pd

# Ordinal quality of the GNSS fix: higher is better. Used to spot fix degradation numerically.
FIX_RANK = {"NO_FIX": 0, "AUTONOMOUS": 1, "DGPS": 2, "RTK_FLOAT": 3, "RTK_FIXED": 4}


def load_telemetry(data):
    """1 Hz telemetry, one row per (tractor, second), sorted by tractor then time."""
    tele = pd.read_csv(Path(data) / "farm" / "tractor_telemetry.csv.gz")
    tele["ts"] = pd.to_datetime(tele["ts"], utc=True)
    tele["base_link_ok"] = tele["base_link_ok"].astype(str).str.lower() == "true"
    tele["fix_rank"] = tele["fix_state"].map(FIX_RANK).astype("int8")
    tele["working"] = tele["speed_mps"] > 1.0  # 4.5 m/s working, 0 parked
    return tele.sort_values(["tractor_id", "ts"]).reset_index(drop=True)


def _read(data, name):
    return json.loads((Path(data) / "sda" / name).read_text())


def _frame(rows):
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, format="ISO8601")
    return df.sort_values("ts").reset_index(drop=True)


def load_sda(data):
    """Return dict of DataFrames: kp, dst, s4, advisories, rfi."""
    sgi = _read(data, "sgi_indices.json")
    return {
        "kp": _frame(sgi["kp"]),                       # ts, kp           (3-hourly, value applies from ts)
        "dst": _frame(sgi["dst"]),                     # ts, dst_nT       (hourly)
        "s4": _frame(_read(data, "scintillation_series.json")["series"]),  # ts, s4 (15-min)
        "advisories": _frame(_read(data, "constellation_advisories.json")["advisories"]),
        "rfi": _frame(_read(data, "rfi_reports.json")["reports"]),
    }


def load_labels(data):
    """Dev labels as a DataFrame with parsed start/end."""
    episodes = json.loads((Path(data) / "labels" / "train_episodes.json").read_text())["episodes"]
    df = pd.DataFrame(episodes)
    df["start"] = pd.to_datetime(df["start"], utc=True)
    df["end"] = pd.to_datetime(df["end"], utc=True)
    return df

"""Scratch comparison (not part of `make submission`): spoof detection from XTE + signal health instead of the lockstep offset.

Prints the current offset-based spoof episodes and the episodes a fleet-relative XTE rule finds on all 72 h.
Run from the repo root: python scripts/xte_spoof_check.py. Results are recorded in docs/SPOOF_OFFSET.md section 6.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from straightrows import detect as D, features as F, load  # noqa: E402

tele = load.load_telemetry("starter/data")
sda = load.load_sda("starter/data")
sec, minute, passes = F.build_features(tele, sda)

print("CURRENT (offset-based):")
for e in sorted(D.detect_spoofing(sec), key=lambda e: e["start"]):
    print(" ", e["start"], e["end"], e["tractors"])

print("\nRULE B: |XTE| above the fleet median by >= 0.5 m (60 s rolling median), own C/N0, fix and sats at fleet level, >= 10 min")
for tr, g in sec.groupby("tractor_id", sort=True):
    g = g[g.working].sort_values("ts")
    excess = g.axte_resid.rolling(60, center=True, min_periods=30).median()
    healthy = (g.mean_cn0_dbhz_resid > -2) & (g.fix_rank_resid.abs() < 0.5) & (g.num_sats_resid > -2) & g.base_link_ok
    hit = g[(excess >= 0.5) & healthy]
    for s, e in D.find_runs(hit.ts, D.SECOND, 60 * D.SECOND, 10 * D.MINUTE):
        w = g[(g.ts >= s) & (g.ts < e)]
        print(" ", s, e, tr, f"xte_max {w.axte.max():.1f}")

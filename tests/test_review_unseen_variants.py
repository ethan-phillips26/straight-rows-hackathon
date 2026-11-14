"""Review tests: variants of each cause that the 12 dev episodes never showed.

Written by the independent reviewer (docs/FABLE_REVIEW.md). All synthetic, no dataset needed.
A FAILING test here is a review finding, not a broken test: do not weaken it to make it pass.
Each docstring says what a correct detector should report and what detect.py reported at review time.
"""
import numpy as np
import pandas as pd

from straightrows import detect as D
from straightrows import features as F
from tests.test_features import synthetic_fleet


def T(s):
    return pd.Timestamp(s, tz="UTC")


def between(tele, a, b, tractors=F.TRACTORS):
    return (tele.ts >= T(a)) & (tele.ts < T(b)) & tele.tractor_id.isin(tractors)


def test_two_tractors_spoofed_by_the_same_offset_are_the_ones_reported():
    """One spoofer displaces T1 AND T2 by the same 6 m for 30 min. Truth: SPOOFING on T1 and T2.

    At review time: one SPOOFING detection on T3, the only healthy tractor. The lockstep offset keeps
    "the smaller of my two pair deviations", so T1 and T2 cancel each other and T3 looks displaced from both.
    """
    tele = synthetic_fleet(n=14400)
    m = between(tele, "2026-05-04T12:00", "2026-05-04T12:30", ["T1", "T2"])
    tele.loc[m, "lon"] += 6.0 / F.M_PER_DEG_LON
    tele.loc[m, "cross_track_error_m"] = 6.0          # the guidance monitor does see it on T1 and T2
    eps = D.detect_spoofing(F.second_features(tele))
    reported = sorted({t for e in eps for t in e["tractors"]})
    assert reported == ["T1", "T2"], f"reported {reported}"


def test_three_tractor_jam_while_s4_is_elevated_but_harmless_is_not_a_storm():
    """Non-uniform drop (10 / 7 / 6 dB) on a day when S4 = 0.28, a level FINDINGS says has no farm impact.

    Truth: JAMMING on all three (tractors disagree by 4 dB; a storm is uniform within 0.2 dB).
    At review time: SPACE_WEATHER (median drop >= 4 dB and S4 >= 0.2 is all the storm rule asks for), and
    the jamming is hidden because the S4-predicted storm level is subtracted from it.
    """
    tele = synthetic_fleet(n=14400)
    for tractor, drop in [("T1", 10.0), ("T2", 7.0), ("T3", 6.0)]:
        tele.loc[between(tele, "2026-05-04T12:00", "2026-05-04T12:36", [tractor]), "mean_cn0_dbhz"] -= drop
    minute = F.minute_features(F.second_features(tele)).assign(s4=0.28)
    assert D.detect_storms(minute) == [], "a 4 dB tractor-to-tractor spread was reported as a storm"
    jam = D.detect_jamming(minute)
    assert len(jam) == 1 and jam[0]["tractors"] == F.TRACTORS


def test_antenna_fault_that_spans_a_jam_stays_one_detection():
    """T1 has antenna dropouts 11:06-12:48 (every 5th second, -8 dB on min C/N0); all three are jammed 11:36-12:12.

    Truth: one EQUIPMENT_FAULT on T1 for 11:06-12:48, plus the jam.
    At review time: the fault is cut in two (11:06-11:36 and 12:12-12:48) because dip seconds inside a
    detected jam are discarded and the 36 min hole is wider than EQUIP_MAX_GAP_MIN. With the scorer only the
    second half reaches time IoU 0.30, so the episode matches with a 66 min onset error (zero onset credit)
    and the first half is a false positive.
    (Times stay before 12:54 because synthetic_fleet's latitude drifts into the headland zone after that.)
    """
    tele = synthetic_fleet(n=14400)
    fault = between(tele, "2026-05-04T11:06", "2026-05-04T12:48", ["T1"])
    every_fifth = fault & (tele.ts.dt.second % 5 == 0)
    tele.loc[every_fifth, "min_cn0_dbhz"] -= 8.0
    jam_rows = between(tele, "2026-05-04T11:36", "2026-05-04T12:12")
    tele.loc[jam_rows, ["mean_cn0_dbhz", "min_cn0_dbhz"]] -= 10.0
    sec = F.second_features(tele)
    jam = D.detect_jamming(F.minute_features(sec).assign(s4=0.05))
    assert len(jam) == 1 and jam[0]["tractors"] == F.TRACTORS       # precondition: the jam itself is found
    equipment = [e for e in D.detect_equipment(sec, jam) if e["stats"]["subtype"] == "antenna"]
    assert len(equipment) == 1, f"{len(equipment)} fragments: {[(str(e['start'])[11:16], str(e['end'])[11:16]) for e in equipment]}"
    assert equipment[0]["start"] <= T("2026-05-04T11:12") and equipment[0]["end"] >= T("2026-05-04T12:42")

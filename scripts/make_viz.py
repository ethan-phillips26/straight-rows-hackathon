"""Build a self-contained replay viewer (viz/out/<name>.html) from telemetry + events.

Usage:
  python scripts/make_viz.py --data starter/data --events labels --pick TRN-03,TRN-06,TRN-05
  python scripts/make_viz.py --data starter/data --events submission.json --pick 1,2,5 --name final

--events labels            : dev labels (train_episodes.json)
--events <submission.json> : your detections (class/action/confidence/evidence); ids become D1, D2, ...
--pick                     : comma list of event ids; default = all events (can get big, so pick a few)
--residuals <file.json>    : optional override, see load_external_residuals() below.

Residual definitions (simple on purpose, so you can defend them):
  own   = signal - that tractor's median over all working seconds in the dataset
  fleet = signal - median of the 3 tractors at the same second (cancels fleet-wide effects,
          so it isolates single-tractor events like spoofing / equipment faults)
Scale for the +/-3 sigma band is the MAD of the own-residual on working seconds (robust to events).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from straightrows.features import add_lockstep_offset
from straightrows.load import load_labels, load_sda, load_telemetry

PAD = pd.Timedelta("20min")
MAX_POINTS = 5000  # per tractor per clip; longer clips are decimated


def load_events(spec, data):
    if spec == "labels":
        df = load_labels(data)
        return [
            dict(id=r.id, start=r.start, end=r.end, tractors=list(r.tractors), cls=r["class"],
                 action=r.recommended_action, confidence=None, evidence=[])
            for _, r in df.sort_values("start").iterrows()
        ]
    sub = json.loads(Path(spec).read_text())
    out = []
    for i, d in enumerate(sub["detections"], 1):
        out.append(dict(id=f"D{i}", start=pd.Timestamp(d["start"]), end=pd.Timestamp(d["end"]),
                        tractors=d["tractors"], cls=d["class"], action=d["action"],
                        confidence=d.get("confidence"), evidence=d.get("evidence", [])))
    return out


def baselines(tele):
    """Per-tractor medians and a robust sigma for each signal, from working seconds."""
    w = tele[tele["working"]].copy()
    w["axte"] = w["cross_track_error_m"].abs()
    b = {}
    for tid, g in w.groupby("tractor_id"):
        b[tid] = {}
        for col in ["mean_cn0_dbhz", "axte", "num_sats"]:
            med = float(g[col].median())
            mad = float((g[col] - med).abs().median()) * 1.4826
            b[tid][col] = (med, max(mad, 1e-3))
    return b


def residual_cols(tele, b):
    tele = tele.copy()
    tele["axte"] = tele["cross_track_error_m"].abs()
    for col, key in [("mean_cn0_dbhz", "cn0"), ("axte", "xte"), ("num_sats", "sats")]:
        med = tele["tractor_id"].map(lambda t: b[t][col][0])
        tele[f"r_{key}_own"] = tele[col] - med
        fleet = tele.groupby("ts")[col].transform("median")
        tele[f"r_{key}_fleet"] = tele[col] - fleet
    return tele


def r(a, n=2):
    return [None if (v is None or (isinstance(v, float) and math.isnan(v))) else round(float(v), n) for v in a]


def build_clip(ev, tele, sda, b, origin):
    t0, t1 = ev["start"] - PAD, ev["end"] + PAD
    w = tele[(tele.ts >= t0) & (tele.ts <= t1)]
    secs = (t1 - t0).total_seconds()
    step = max(1, math.ceil(secs / MAX_POINTS))
    lat0, lon0 = origin
    kx, ky = 111320 * math.cos(math.radians(lat0)), 110574
    tr = {}
    for tid, g in w.groupby("tractor_id"):
        g = g.iloc[::step]
        tr[tid] = dict(
            x=r((g.lon - lon0) * kx, 1), y=r((g.lat - lat0) * ky, 1),
            xte=r(g.cross_track_error_m, 3), londev=r(g.lon_dev_m, 2), cn0=r(g.mean_cn0_dbhz, 1), mincn0=r(g.min_cn0_dbhz, 1),
            sats=g.num_sats.astype(int).tolist(), fix=g.fix_rank.astype(int).tolist(),
            link=g.base_link_ok.astype(int).tolist(), speed=r(g.speed_mps, 1),
            r_cn0_own=r(g.r_cn0_own, 2), r_cn0_fleet=r(g.r_cn0_fleet, 2),
            r_xte_own=r(g.r_xte_own, 3), r_xte_fleet=r(g.r_xte_fleet, 3),
            r_sats_own=r(g.r_sats_own, 1), r_sats_fleet=r(g.r_sats_fleet, 1),
        )
    ts0 = w.ts.min()

    def win(df, col, lo=t0 - pd.Timedelta("3h")):  # wider window: SDA feeds are coarse
        d = df[(df.ts >= lo) & (df.ts <= t1)]
        return [[(x - ts0).total_seconds(), round(float(v), 3)] for x, v in zip(d.ts, d[col])]

    rfi = sda["rfi"][(sda["rfi"].ts >= t0) & (sda["rfi"].ts <= t1)]
    adv = sda["advisories"][(sda["advisories"].ts >= t0 - pd.Timedelta("6h")) & (sda["advisories"].ts <= t1)]
    return dict(
        id=ev["id"], cls=ev["cls"], action=ev["action"], confidence=ev["confidence"], evidence=ev["evidence"],
        tractors=ev["tractors"], t0=ts0.isoformat(), step=step, n=len(next(iter(tr.values()))["x"]),
        ev_start=(ev["start"] - ts0).total_seconds(), ev_end=(ev["end"] - ts0).total_seconds(),
        trk=tr, s4=win(sda["s4"], "s4"), kp=win(sda["kp"], "kp"),
        rfi=[dict(t=(x.ts - ts0).total_seconds(), text=x.text, conf=float(x.confidence)) for x in rfi.itertuples()],
        adv=[dict(t=(x.ts - ts0).total_seconds(), type=x.type, text=x.text) for x in adv.itertuples()],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="starter/data")
    ap.add_argument("--events", default="labels")
    ap.add_argument("--pick", default="")
    ap.add_argument("--name", default="demo")
    ap.add_argument("--outdir", default="viz/out")
    a = ap.parse_args()

    tele = load_telemetry(a.data)
    sda = load_sda(a.data)
    b = baselines(tele)
    tele = add_lockstep_offset(tele)  # lon_dev_m: reported east-west offset from the lockstep formation line
    tele = residual_cols(tele, b)
    events = load_events(a.events, a.data)
    if a.pick:
        want = [p.strip() for p in a.pick.split(",")]
        want = [w if not w.isdigit() or a.events == "labels" else f"D{w}" for w in want]
        events = [e for e in events if e["id"] in want]
        events.sort(key=lambda e: want.index(e["id"]))
    origin = (float(tele.lat.mean()), float(tele.lon.mean()))
    bundle = dict(
        source=a.events, sigma={t: {k: v[1] for k, v in d.items()} for t, d in b.items()},
        baseline={t: {k: v[0] for k, v in d.items()} for t, d in b.items()},
        clips=[build_clip(e, tele, sda, b, origin) for e in events],
    )
    tmpl = Path(__file__).resolve().parent.parent / "viz" / "viewer_template.html"
    html = tmpl.read_text().replace("/*__BUNDLE__*/null", json.dumps(bundle, separators=(",", ":")))
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.name}.html").write_text(html)
    (out / f"{a.name}.json").write_text(json.dumps(bundle, separators=(",", ":")))  # loadable via the viewer's file button
    print(f"wrote {out / (a.name + '.html')}  ({len(html) / 1e6:.1f} MB, {len(events)} clips)")


if __name__ == "__main__":
    main()

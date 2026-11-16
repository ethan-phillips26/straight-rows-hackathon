"""Build a self-contained data explorer (viz/out/explorer.html): the whole 72 h of 1 Hz telemetry, SDA feeds and events.

Usage:
  python scripts/make_explorer.py --data starter/data --detections submission.json

Unlike make_viz.py (event clips), nothing is cut or decimated here: every second of every tractor is embedded, so the
browser can zoom from the 72 h overview down to single samples. To keep one file small the signals are stored as
integers (value * scale, see SIGNALS), concatenated, gzipped and base64-encoded; the page inflates them with the
browser's built-in DecompressionStream. No network, no libraries.

Layers shown in the page:
  labels      = dev ground truth (train_episodes.json, hours 0-48 only)
  detections  = your submission.json (evaluation window, plus anything else in the file), plus our detections on the
                labelled dev days (ids DV1..), produced by the same pipeline
  Dev check   = a view that scores those dev detections against the labels with the organizers' scorer and lists, per label,
                whether it was matched, the onset error and the overlap numbers
"""
import argparse
import base64
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_viz import baselines, font_faces  # deck fonts as base64; per-tractor quiet baselines for the residual views
import run_all                                   # the same pipeline that writes submission.json, run on the dev window here
from straightrows import locate, output
from straightrows.features import add_lockstep_offset
from straightrows.load import load_labels, load_sda, load_telemetry

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scoring"))
from score_submission import score  # the organizers' scorer, unchanged

T0 = pd.Timestamp("2026-05-04T00:00:00Z")
DEV_END = T0 + pd.Timedelta(hours=48)
N = 72 * 3600
MISSING = -32768  # int16 sentinel for "no sample at this second"

# (key, column, scale): stored value = round(column * scale). The page divides by the same scale.
SIGNALS = [
    ("cn0", "mean_cn0_dbhz", 100),
    ("mincn0", "min_cn0_dbhz", 100),
    ("xte", "cross_track_error_m", 1000),  # max |xte| in the data is ~27 m, so int16 (+/-32.7 m) holds it
    ("sats", "num_sats", 1),
    ("fix", "fix_rank", 1),
    ("link", "base_link_ok", 1),
    ("speed", "speed_mps", 100),
    ("working", "working", 1),
    ("x", "x_m", 10),        # reported east of the field origin, 0.1 m; extents are about +/-400 m
    ("y", "y_m", 10),        # reported north of the field origin, 0.1 m; extents are about +/-550 m
    ("londev", "lon_dev_m", 100),  # reported east-west offset from the lockstep formation line (spoofing signature)
]


def explorer_fonts():
    """make_viz.font_faces() (Hanken Grotesk, IBM Plex Mono) plus Newsreader, embedded the same way. A missing file is
    skipped and the template's font stacks fall back to Georgia / system sans."""
    f = Path(__file__).resolve().parent.parent / "demo" / "fonts" / "Newsreader.ttf"
    css = font_faces()
    if f.exists():
        b64 = base64.b64encode(f.read_bytes()).decode()
        css += (f"\n@font-face{{font-family:'Newsreader';src:url(data:font/ttf;base64,{b64}) format('truetype');"
                f"font-weight:400 700;font-style:normal;font-display:swap}}")
    return css


def desired_tracks(tele, pitch=12.0):
    """Desired-track (DTK) lanes: the tractors drive parallel north-south passes `pitch` m apart on one shared grid. The dataset has no
    planned-path file, so recover the grid from working samples with a small cross-track error: its phase is the circular mean of
    x mod pitch, its extent the lanes those samples span. Returns x0 (first lane, m east of origin), pitch, count and the y extent."""
    ok = tele[(tele.speed_mps > 0) & (tele.cross_track_error_m < 0.5)]
    ang = 2 * math.pi * ok.x_m.to_numpy() / pitch
    phase = (math.atan2(np.sin(ang).mean(), np.cos(ang).mean()) % (2 * math.pi)) * pitch / (2 * math.pi)
    k0, k1 = math.floor((ok.x_m.min() - phase) / pitch + .5), math.floor((ok.x_m.max() - phase) / pitch + .5)
    return dict(x0=round(phase + k0 * pitch, 2), pitch=pitch, n=int(k1 - k0 + 1), y0=round(float(ok.y_m.min()), 1), y1=round(float(ok.y_m.max()), 1))


def pack_telemetry(tele):
    tractors = sorted(tele.tractor_id.unique())
    blocks = []
    for tid in tractors:
        g = tele[tele.tractor_id == tid]
        idx = ((g.ts - T0).dt.total_seconds()).astype(int).to_numpy()
        for _, col, scale in SIGNALS:
            a = np.full(N, MISSING, dtype="<i2")
            vals = g[col].astype(float).to_numpy()
            ok = np.isfinite(vals)
            a[idx[ok]] = np.round(vals[ok] * scale).astype("<i2")
            blocks.append(a.tobytes())
    raw = b"".join(blocks)
    return tractors, base64.b64encode(gzip.compress(raw, 9, mtime=0)).decode(), len(raw)


def secs(ts):
    return float((pd.Timestamp(ts) - T0).total_seconds())


def events_from(labels_df, detections_path, dev_rows=()):
    ev = []
    for _, r in labels_df.sort_values("start").iterrows():
        ev.append(dict(id=r.id, src="label", t0=secs(r.start), t1=secs(r.end), tractors=list(r.tractors),
                       cls=r["class"], action=r.recommended_action, conf=None, evidence=[]))
    if detections_path and Path(detections_path).exists():
        sub = json.loads(Path(detections_path).read_text())
        for i, d in enumerate(sub["detections"], 1):
            ev.append(dict(id=f"D{i}", src="detection", t0=secs(d["start"]), t1=secs(d["end"]), tractors=d["tractors"],
                           cls=d["class"], action=d["action"], conf=d.get("confidence"), evidence=d.get("evidence", [])))
    for i, d in enumerate(dev_rows, 1):   # our detections on the labelled dev days, ids DV1.. (the submission only covers 05-06)
        ev.append(dict(id=f"DV{i}", src="detection", t0=secs(d["start"]), t1=secs(d["end"]), tractors=d["tractors"],
                       cls=d["class"], action=d["action"], conf=d.get("confidence"), evidence=d.get("evidence", [])))
    return ev


def dev_detections(data):
    """Run the real pipeline on hours 0-48 (features still use the whole dataset) and return submission-style rows."""
    return [output._to_dict(d) for d in run_all.run(data, (T0, DEV_END))]


def compare_with_labels(rows, key, events):
    """Score the dev detections against the dev labels with the organizers' scorer; keep what the page shows.

    The scorer says which detection matched each label only by its start/end/class, so map that back to our DV ids."""
    res = score({"detections": rows}, key)
    ids = {(e["t0"], e["t1"], e["cls"], tuple(e["tractors"])): e["id"] for e in events if e["id"].startswith("DV")}
    episodes = []
    for r in res["episodes"]:
        det = ids.get((secs(r["start"]), secs(r["end"]), r["predicted_class"], tuple(r["tractors"]))) if r["matched"] else None
        episodes.append(dict(id=r["episode"], truth=r["truth_class"], pred=r.get("predicted_class"), matched=r["matched"],
                             correct=r["correct"], onset_s=r.get("onset_error_s"), time_iou=r.get("time_iou"),
                             tractor_iou=r.get("tractor_iou"), action_ok=r["action_agreement"], det=det))
    fps = [dict(det=ids.get((secs(d["start"]), secs(d["end"]), d["class"], tuple(d["tractors"]))), cls=d["class"]) for d in res["false_positives"]]
    return dict(score=round(res["diagnostic_score"], 2), max=res["maximum"], macro_f1=round(res["macro_f1"], 3),
                timing=round(res["mean_timing_credit"], 3), n=res["episode_count"], correct=sum(r["correct"] for r in res["episodes"]),
                action=res["action_agreement"], benign_multipath=res["benign_multipath_reports"], fps=fps, episodes=episodes)


def add_jammer_fits(events, tele, lat0, lon0):
    """Attach the fitted jammer position + 95 % error ellipse (straightrows/locate.py) to every JAMMING event,
    in the page's metre frame (x east, y north of the field origin). Labels and detections are fitted separately."""
    kx, ky = 111320 * math.cos(math.radians(lat0)), 110574
    for e in events:
        e["jam"] = None
        if e["cls"] != "JAMMING":
            continue
        f = locate.locate_jammer(tele, T0 + pd.Timedelta(seconds=e["t0"]), T0 + pd.Timedelta(seconds=e["t1"]))
        if f:
            e["jam"] = dict(x=round((f["lon"] - lon0) * kx, 1), y=round((f["lat"] - lat0) * ky, 1), lat=f["lat"], lon=f["lon"],
                            a_m=round(f["semi_major_km"] * 1000), b_m=round(f["semi_minor_km"] * 1000),
                            bearing=round(f["major_axis_bearing_deg"]), rmse=round(f["rmse_db"], 2),
                            slope=round(f["slope_db_per_decade"], 1), reliable=f["reliable"], notes=f["notes"])


def sda_bundle(sda):
    def series(df, col, nd):
        return [[secs(t), round(float(v), nd)] for t, v in zip(df.ts, df[col])]
    return dict(
        kp=series(sda["kp"], "kp", 2), dst=series(sda["dst"], "dst_nT", 1), s4=series(sda["s4"], "s4", 3),
        rfi=[dict(t=secs(x.ts), id=x.id, band=x.band, text=x.text, conf=float(x.confidence), lat=float(x.centerLat),
                  lon=float(x.centerLon), a_km=float(x.semiMajor_km), b_km=float(x.semiMinor_km)) for x in sda["rfi"].itertuples()],
        adv=[dict(t=secs(x.ts), id=x.id, type=x.type, svn=x.svn, text=x.text) for x in sda["advisories"].itertuples()],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="starter/data")
    ap.add_argument("--detections", default="submission.json", help="submission.json to show as the detections layer")
    ap.add_argument("--name", default="explorer")
    ap.add_argument("--outdir", default="viz/out")
    a = ap.parse_args()

    tele = load_telemetry(a.data)
    b = baselines(tele)  # medians and robust sigma over working seconds, same definition as the replay viewer
    tele = add_lockstep_offset(tele)
    lat0, lon0 = float(tele.lat.mean()), float(tele.lon.mean())
    tele["x_m"] = (tele.lon - lon0) * 111320 * math.cos(math.radians(lat0))
    tele["y_m"] = (tele.lat - lat0) * 110574
    dev_rows = dev_detections(a.data)
    events = events_from(load_labels(a.data), a.detections, dev_rows)
    key = json.loads((Path(a.data) / "labels" / "train_episodes.json").read_text())
    compare = compare_with_labels(dev_rows, key, events)
    add_jammer_fits(events, tele, lat0, lon0)
    tractors, blob, raw_bytes = pack_telemetry(tele)
    fix_names = {int(k): v for k, v in tele.drop_duplicates("fix_rank").set_index("fix_rank").fix_state.items()}
    bundle = dict(
        t0=T0.isoformat(), n=N, tractors=tractors, signals=[dict(key=k, scale=s) for k, _, s in SIGNALS],
        missing=MISSING, fix_names=fix_names,
        baseline={t: {k: v[0] for k, v in d.items()} for t, d in b.items()},
        sigma={t: {k: v[1] for k, v in d.items()} for t, d in b.items()}, dev_end=48 * 3600.0, sda=sda_bundle(load_sda(a.data)),
        events=events, compare=compare, origin=[lat0, lon0], lanes=desired_tracks(tele),
    )
    tmpl = Path(__file__).resolve().parent.parent / "viz" / "explorer_template.html"
    html = tmpl.read_text()
    html = html.replace("/*__BUNDLE__*/null", json.dumps(bundle, separators=(",", ":")))
    html = html.replace("/*__BLOB__*/", blob)
    html = html.replace("/*__FONTS__*/", explorer_fonts())
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.name}.html").write_text(html)
    print(f"wrote {out / (a.name + '.html')}  ({len(html) / 1e6:.1f} MB; telemetry {raw_bytes / 1e6:.1f} MB raw -> "
          f"{len(blob) / 1e6:.1f} MB gz+b64; {len(bundle['events'])} events)")


if __name__ == "__main__":
    main()

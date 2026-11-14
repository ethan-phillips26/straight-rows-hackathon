#!/usr/bin/env python3
"""Exploration: what does each labeled dev episode look like in the data?

python scripts/explore_dev.py            # prints per-episode tables, writes plots to reports/dev_episodes/

For every dev episode and every tractor, compares the episode window with the 30 minutes before it
(working seconds only), and lists the SDA context (Kp, Dst, S4, RFI reports). Not part of the pipeline.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from straightrows.load import load_labels, load_sda, load_telemetry  # noqa: E402

DATA = ROOT / "starter" / "data"
OUT = ROOT / "reports" / "dev_episodes"
PRE = pd.Timedelta(minutes=30)


def window_stats(df):
    w = df[df.working]
    if w.empty:
        return {"work%": 0}
    return {
        "work%": round(100 * len(w) / len(df)),
        "cn0": round(w.mean_cn0_dbhz.mean(), 1),
        "cn0min": round(w.min_cn0_dbhz.mean(), 1),
        "sats": round(w.num_sats.mean(), 1),
        "|xte|med": round(w.cross_track_error_m.abs().median(), 3),
        "|xte|p95": round(w.cross_track_error_m.abs().quantile(0.95), 2),
        "fixed%": round(100 * (w.fix_state == "RTK_FIXED").mean()),
        "linkdown%": round(100 * (~w.base_link_ok).mean()),
        "nofix%": round(100 * (w.fix_state == "NO_FIX").mean()),
    }


def as_of(series_df, col, t):
    rows = series_df[series_df.ts <= t]
    return rows[col].iloc[-1] if len(rows) else None


def main():
    tele = load_telemetry(DATA)
    sda = load_sda(DATA)
    labels = load_labels(DATA).rename(columns={"class": "cls"})  # "class" is a Python keyword
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)

    for ep in labels.itertuples():
        long_ep = (ep.end - ep.start) > pd.Timedelta(minutes=5)
        pre = PRE if long_ep else pd.Timedelta(minutes=2)
        lo, hi = ep.start - pre, ep.end + pre
        rows = []
        for tr, g in tele[(tele.ts >= lo) & (tele.ts < hi)].groupby("tractor_id"):
            before = window_stats(g[g.ts < ep.start])
            during = window_stats(g[(g.ts >= ep.start) & (g.ts < ep.end)])
            rows.append({"tractor": tr + ("*" if tr in ep.tractors else ""), "when": "before", **before})
            rows.append({"tractor": tr + ("*" if tr in ep.tractors else ""), "when": "during", **during})
        s4 = sda["s4"][(sda["s4"].ts >= lo) & (sda["s4"].ts < hi)].s4
        rfi = sda["rfi"][(sda["rfi"].ts >= ep.start - pd.Timedelta(minutes=10))
                         & (sda["rfi"].ts <= ep.end + pd.Timedelta(minutes=40))]
        print(f"\n=== {ep.id} {ep.cls} {ep.start:%m-%d %H:%M:%S}->{ep.end:%H:%M:%S} "
              f"tractors={ep.tractors} | Kp={as_of(sda['kp'], 'kp', ep.start)} "
              f"Dst={as_of(sda['dst'], 'dst_nT', ep.start)} S4 {s4.min():.2f}-{s4.max():.2f} "
              f"RFI={[f'{r.id}@{r.ts:%H:%M}' for r in rfi.itertuples()]}")
        print(pd.DataFrame(rows).to_string(index=False))
        plot_episode(ep, tele[(tele.ts >= lo) & (tele.ts < hi)], sda, lo, hi)
    print(f"\nPlots: {OUT}")


def plot_episode(ep, win, sda, lo, hi):
    fig, axes = plt.subplots(5, 1, figsize=(11, 10), sharex=True)
    resample = "10s" if (hi - lo) > pd.Timedelta(minutes=20) else "1s"
    for tr, g in win.groupby("tractor_id"):
        g = g.set_index("ts")
        r = g[["mean_cn0_dbhz", "min_cn0_dbhz", "num_sats", "fix_rank"]].resample(resample).mean()
        xte = g["cross_track_error_m"].abs().resample(resample).max()
        lw = 1.6 if tr in ep.tractors else 0.8
        axes[0].plot(r.index, r.mean_cn0_dbhz, lw=lw, label=tr)
        axes[1].plot(r.index, r.min_cn0_dbhz, lw=lw, label=tr)
        axes[2].plot(xte.index, xte, lw=lw, label=tr)
        axes[3].plot(r.index, r.num_sats, lw=lw, label=f"{tr} sats")
        axes[3].plot(r.index, r.fix_rank, lw=lw, ls="--", label=f"{tr} fix")
    s4 = sda["s4"][(sda["s4"].ts >= lo) & (sda["s4"].ts < hi)]
    axes[4].step(s4.ts, s4.s4, where="post", label="S4")
    for ax, name in zip(axes, ["mean C/N0", "min C/N0", "|XTE| max (m)", "sats / fix rank", "S4"]):
        ax.axvspan(ep.start, ep.end, color="orange", alpha=0.15)
        ax.set_ylabel(name, fontsize=8)
        ax.legend(fontsize=7, loc="upper right", ncol=3)
    for r in sda["rfi"][(sda["rfi"].ts >= lo) & (sda["rfi"].ts < hi)].itertuples():
        for ax in axes:
            ax.axvline(r.ts, color="red", ls=":")
    fig.suptitle(f"{ep.id} {ep.cls} {sorted(ep.tractors)} (shaded = label; red dotted = RFI report)")
    fig.tight_layout()
    fig.savefig(OUT / f"{ep.id}.png", dpi=80)
    plt.close(fig)


if __name__ == "__main__":
    main()

"""Sidereal-day test for multipath: do headland dips recur 23h56m04s or 24h00m later?

GPS orbits repeat in sidereal time, so a satellite-geometry (reflector) effect at a fixed spot should
recur ~236 s EARLIER each solar day. A tractor-path effect (machine passes the same headland on a schedule
that restarts at 11:00:00 every day) recurs at exactly 24 h. We measure which one the data follows.

For each pass on day d+1 we find the nearest pass on day d once shifted by each candidate lag and
report the residual. The passes are ~355 s apart, so a 236 s shift is clearly distinguishable from 0.
"""
import argparse
import numpy as np
import pandas as pd
from straightrows.load import load_telemetry, load_sda
from straightrows.features import build_features

SIDEREAL_S = 86164.0905  # one sidereal day
SOLAR_S = 86400.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="starter/data")
    a = ap.parse_args()
    tele = load_telemetry(a.data)
    sec, _, passes = build_features(tele, load_sda(a.data))
    p = passes[passes.tractor_id == passes.tractor_id.iloc[0]]
    p = p.assign(day=p.start.dt.floor("D"))
    print("passes per UTC day (one tractor):", p.groupby("day").size().to_dict())
    first = p.groupby("day").start.min()
    print("first pass of each day (UTC):", [str(x.time()) for x in first])
    print("lag between consecutive days' first passes:", [(b - a_).total_seconds() for a_, b in zip(first[:-1], first[1:])])
    days = sorted(p.day.unique())
    print(f"\n{'pair':<24}{'lag':<10}{'median |resid| s':>18}{'share <=2 s':>14}")
    for d0, d1 in zip(days[:-1], days[1:]):
        s0 = p[p.day == d0].start
        s1 = p[p.day == d1].start
        for name, lag in (("24h solar", SOLAR_S), ("23h56m04s sidereal", SIDEREAL_S)):
            # residual = distance from each day-d1 pass to the nearest day-d pass moved forward by the lag
            t1 = s1.astype("int64").to_numpy() / 1e9
            t0 = s0.astype("int64").to_numpy() / 1e9
            res = np.array([np.min(np.abs(t0 + lag - x)) for x in t1])
            print(f"{str(d0.date())}->{str(d1.date()):<12}{name:<22}{np.median(res):>10.1f}{(res <= 2).mean():>14.2f}")


if __name__ == "__main__":
    main()

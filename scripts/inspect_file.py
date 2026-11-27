"""Print the structure of data files (rest / discharge / charge segments) and plot them.

Usage:  python scripts/inspect_file.py --kind HPPC --temp 25
        python scripts/inspect_file.py --file "data/raw/.../some.mat"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from soc_ekf.data import classify, discharged_ah, find_files, load  # noqa: E402
from soc_ekf.model import runs  # noqa: E402


def describe(path: Path, cap: float) -> None:
    rec = load(path)
    df = rec.df
    t, i, v = df["t_s"].to_numpy(), df["i"].to_numpy(), df["v"].to_numpy()
    dt = np.diff(t)
    inc = 0.5 * (i[1:] + i[:-1]) * np.diff(t)
    inc[np.diff(t) > 120] = 0.0  # no integration across logging gaps
    q = np.concatenate([[0.0], np.cumsum(inc)]) / 3600.0
    print(f"\n=== {path.name}  [{rec.kind}, {rec.temp_c_nominal}, {rec.condition}]")
    print(f"samples {len(t)}, {t[-1]/3600:.2f} h, dt median {np.median(dt):.2f} s, "
          f"dt p99 {np.percentile(dt, 99):.1f} s, dt max {dt.max():.0f} s, raw_sign {rec.raw_sign:+d}")
    print(f"V {v.min():.3f}..{v.max():.3f}, I {i.min():.2f}..{i.max():.2f} A, net discharged {q[-1]:.3f} Ah, "
          f"min-SoC point at {t[np.argmax(q)]/3600:.2f} h ({q.max():.3f} Ah removed)")
    q_tst = discharged_ah(df, True, rec.raw_sign)
    print(f"tester Ah counter: net {q_tst[-1]:.3f} Ah removed (logged current integrates to {q[-1]:.3f} Ah)")
    gi = np.flatnonzero(dt > 120)
    print(f"logging gaps > 2 min: {len(gi)}")
    for k in gi:
        print(f"   gap at {t[k]/3600:6.2f} h, {dt[k]/60:6.1f} min, tester Ah moved {q_tst[k+1]-q_tst[k]:+.3f}, "
              f"V {v[k]:.3f} -> {v[k+1]:.3f}, I {i[k]:.2f} -> {i[k+1]:.2f} A")
    state = np.where(i > 0.05, 1, np.where(i < -0.05, -1, 0))
    segs = []
    for s_val in (1, -1, 0):
        segs += [(a, b, s_val) for a, b in runs(state == s_val)]
    segs.sort()
    print(f"{'start_h':>8} {'dur_min':>8} {'state':>6} {'I_mean':>7} {'Ah':>7} {'V_start':>7} {'V_end':>7}   (segments >= 2 min; short pulses counted)")
    short = 0
    for a, b, s_val in segs:
        dur = (t[b - 1] - t[a]) / 60 if b - 1 > a else 0.0
        if dur < 2:
            short += 1
            continue
        if short:
            print(f"{'':8} {'':8} {'':>6}   ... {short} short segments (pulses)")
            short = 0
        name = {1: "DIS", -1: "CHG", 0: "rest"}[s_val]
        print(f"{t[a]/3600:8.2f} {dur:8.1f} {name:>6} {i[a:b].mean():7.3f} {q[b-1]-q[a]:7.3f} {v[a]:7.3f} {v[b-1]:7.3f}")
    if short:
        print(f"{'':8} {'':8} {'':>6}   ... {short} short segments (pulses)")

    fig, ax = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    ax[0].plot(t / 3600, v, lw=0.5); ax[0].set_ylabel("V"); ax[0].set_title(path.name)
    ax[1].plot(t / 3600, i, lw=0.5, color="tab:orange"); ax[1].set_ylabel("I [A] (+dis)")
    ax[2].plot(t / 3600, 100 * (1 - q / cap), color="tab:green"); ax[2].set_ylabel("SoC [%]"); ax[2].set_xlabel("h")
    for x in ax:
        x.grid(alpha=0.3)
    fig.tight_layout()
    out = ROOT / "plots" / f"inspect_{rec.kind}_{path.stem[:40].replace(' ', '_')}.png"
    fig.savefig(out, dpi=110)
    print(f"plot: {out.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind")
    ap.add_argument("--temp", type=float)
    ap.add_argument("--file")
    ap.add_argument("--cap", type=float, default=2.9)
    ap.add_argument("--root", default=str(ROOT / "data" / "raw"))
    a = ap.parse_args()
    (ROOT / "plots").mkdir(exist_ok=True)
    if a.file:
        files = [Path(a.file)]
    else:
        files = [p for p in find_files(a.root)
                 if (a.kind is None or classify(p)[0] == a.kind) and (a.temp is None or classify(p)[1] == a.temp)]
    if not files:
        sys.exit("no matching files")
    for p in files:
        describe(p, a.cap)


if __name__ == "__main__":
    main()

"""Plot voltage, current and reference SoC for UDDS / US06 / HWFET at one temperature.

Usage:  python scripts/plot_cycles.py [--temp 25] [--root data/raw]
Writes plots/stage1_drive_cycles_<T>C.png and plots/stage1_<KIND>_<T>C.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from soc_ekf.data import classify, find_files, load, reference_soc  # noqa: E402

KINDS = ("UDDS", "US06", "HWFET")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--temp", type=float, default=25.0)
    ap.add_argument("--root", default=str(ROOT / "data" / "raw"))
    a = ap.parse_args()
    out_dir = ROOT / "plots"
    out_dir.mkdir(exist_ok=True)

    picked = {}
    for p in find_files(a.root):
        k, T, cond = classify(p)
        if k in KINDS and T == a.temp and cond == "const" and k not in picked:
            picked[k] = load(p)
    if not picked:
        sys.exit(f"No UDDS/US06/HWFET files found at {a.temp:g} degC under {a.root}")

    fig, axes = plt.subplots(3, len(picked), figsize=(5 * len(picked), 8), sharex="col", squeeze=False)
    for col, k in enumerate(KINDS):
        if k not in picked:
            continue
        rec = picked[k]
        t_min = rec.df["t_s"] / 60
        soc = reference_soc(rec)
        ax = axes[:, list(picked).index(k)]
        ax[0].plot(t_min, rec.df["v"], lw=0.6)
        ax[0].set_ylabel("Voltage [V]")
        ax[0].set_title(f"{k} @ {a.temp:g}°C")
        ax[1].plot(t_min, rec.df["i"], lw=0.4, color="tab:orange")
        ax[1].set_ylabel("Current [A] (+ = discharge)")
        ax[2].plot(t_min, 100 * soc, color="tab:green")
        ax[2].set_ylabel("Reference SoC [%]")
        ax[2].set_xlabel("Time [min]")
        for x in ax:
            x.grid(alpha=0.3)
    fig.tight_layout()
    out = out_dir / f"stage1_drive_cycles_{a.temp:g}C.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")
    for k, rec in picked.items():
        print(f"  {k:6} {rec.path.name}  ({len(rec.df)} samples, {rec.df['t_s'].iloc[-1]/3600:.2f} h)")


if __name__ == "__main__":
    main()

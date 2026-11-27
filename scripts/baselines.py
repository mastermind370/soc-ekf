"""Stage 3: baseline SoC estimators on the held-out 25 degC drive cycles.

Usage:  python scripts/baselines.py [--temp 25] [--wrong-start 0.30]
Needs results/model_params.json (stage 2). Writes results/baselines.json and plots/stage3_*.png

Methods (what a BMS without a model-based filter would do):
  cc_true_start     coulomb counting from the correct SoC, BMS uses the 2.9 Ah datasheet capacity
  cc_wrong_start    same, but started --wrong-start too low (e.g. BMS woke up thinking 70% at 100%)
  ocv_lookup        SoC = OCV^-1(V_terminal)
  ocv_lookup_ir     SoC = OCV^-1(V_terminal + R0*I)
Ground truth: lab coulomb counting with the measured C/20 capacity from a known full start.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from soc_ekf.baselines import coulomb_count, ocv_lookup, ocv_lookup_ir, soc_errors  # noqa: E402
from soc_ekf.cycles import load_cycles  # noqa: E402
from soc_ekf.data import NOMINAL_CAPACITY_AH  # noqa: E402
from soc_ekf.model import load_params  # noqa: E402

METHODS = ("cc_true_start", "cc_wrong_start", "ocv_lookup", "ocv_lookup_ir")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--temp", type=float, default=25.0)
    ap.add_argument("--wrong-start", type=float, default=0.30)
    ap.add_argument("--root", default=str(ROOT / "data" / "raw"))
    a = ap.parse_args()
    pfile = ROOT / "results" / "model_params.json"
    if not pfile.exists():
        sys.exit("results/model_params.json missing: run python scripts/fit_model.py first")
    ocv, p = load_params(pfile)
    Q = p.capacity_ah
    cycles = load_cycles(a.root, a.temp, ocv, Q)
    if not cycles:
        sys.exit(f"no drive cycles at {a.temp:g} degC")
    print(f"truth capacity {Q:.4f} Ah (C/20); BMS capacity {NOMINAL_CAPACITY_AH} Ah (datasheet); "
          f"wrong start = true - {100*a.wrong_start:.0f}%\n")

    rows, per_kind = {}, defaultdict(lambda: defaultdict(list))
    print(f"{'cycle':6} {'file':34} " + " ".join(f"{m:>20}" for m in METHODS))
    print(f"{'':41} " + " ".join(f"{'RMSE / max [%]':>20}" for _ in METHODS))
    for c in cycles:
        s0 = float(c.soc_ref[0])
        est = {
            "cc_true_start": coulomb_count(c.i, c.dt, NOMINAL_CAPACITY_AH, s0),
            "cc_wrong_start": coulomb_count(c.i, c.dt, NOMINAL_CAPACITY_AH, s0 - a.wrong_start),
            "ocv_lookup": ocv_lookup(c.v, ocv),
            "ocv_lookup_ir": ocv_lookup_ir(c.v, c.i, ocv, p),
        }
        r = {m: soc_errors(est[m], c.soc_ref, c.valid) for m in METHODS}
        r["hours"] = float(c.t[-1] / 3600)
        rows[f"{c.kind}:{c.name}"] = r
        for m in METHODS:
            per_kind[c.kind][m].append(r[m]["rmse_pct"])
        print(f"{c.kind:6} {c.name[:34]:34} " +
              " ".join(f"{r[m]['rmse_pct']:9.2f} / {r[m]['max_abs_pct']:7.2f}" for m in METHODS))

        fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
        tm = c.t / 60
        ax[0].plot(tm, 100 * c.soc_ref, "k", lw=1.5, label="truth")
        for m in METHODS:
            ax[0].plot(tm, 100 * est[m], lw=0.7, label=m)
            ax[1].plot(tm, 100 * (est[m] - c.soc_ref), lw=0.7, label=m)
        ax[0].set(ylabel="SoC [%]", title=f"Stage 3 baselines — {c.kind} @ {a.temp:g}°C"); ax[0].legend(fontsize=8)
        ax[1].set(ylabel="error [% SoC]", xlabel="min"); ax[1].axhline(0, color="k", lw=0.5)
        for x in ax:
            x.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(ROOT / "plots" / f"stage3_{c.kind}_{c.name[:20].replace(' ', '_')}.png", dpi=120)
        plt.close(fig)

    summary = {k: {m: float(np.mean(v)) for m, v in d.items()} for k, d in per_kind.items()}
    print("\nmean SoC RMSE per cycle type [%]:")
    for k, d in summary.items():
        print(f"  {k:6} " + "  ".join(f"{m}={v:.2f}" for m, v in d.items()))
    out = {"temp_c": a.temp, "truth_capacity_ah": Q, "bms_capacity_ah": NOMINAL_CAPACITY_AH,
           "wrong_start": a.wrong_start, "per_file": rows, "rmse_by_cycle_pct": summary}
    (ROOT / "results" / "baselines.json").write_text(json.dumps(out, indent=2))
    print("\nwrote results/baselines.json and plots/stage3_*.png")


if __name__ == "__main__":
    main()

"""Stage 2: capacity + OCV from C/20, 2-RC fit on HPPC, validation on 25 degC drive cycles.

Usage:  python scripts/fit_model.py [--temp 25] [--root data/raw]
Writes results/model_params.json, results/model_fit.json, plots/stage2_*.png

Logging gaps: the Panasonic HPPC files leave out the 10%-SoC discharges between pulse groups
(no samples there). Each gap gets the constant current that removes the right charge: from the
tester's Ah counter if it saw the gap, otherwise fitted. Samples in, and 2 min after, a gap
are left out of fits and error metrics.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from soc_ekf.data import load  # noqa: E402
from soc_ekf.fitting import KNOTS, c20_capacity_ocv, choose_hppc, fit_hppc, pick  # noqa: E402
from soc_ekf.model import (  # noqa: E402
    fill_gaps, initial_soc, prepare, save_params, simulate, soc_zoh, voltage_errors_mv,
)

DRIVE = ("UDDS", "US06", "HWFET", "LA92")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--temp", type=float, default=25.0)
    ap.add_argument("--root", default=str(ROOT / "data" / "raw"))
    a = ap.parse_args()
    T = a.temp
    (ROOT / "plots").mkdir(exist_ok=True)

    # ---------------------------------------------------------- 1. capacity + OCV from C/20
    c20 = c20_capacity_ocv(pick(a.root, "C20", T))
    if c20 is None:
        sys.exit("No C/20 discharge segment found. Paste me: ls -R data/raw | grep -i -E 'c20|ocv'")
    Q, ocv, ocv_info = c20
    src = ocv_info["sources"]
    print(f"[C/20] capacity = {Q:.4f} Ah  ({ocv_info['method']}; dis from {src.get('dis')}, chg from {src.get('chg')})")
    print(f"[C/20] OCV spline fit RMSE = {ocv_info['ocv_spline_rmse_mv']:.2f} mV")

    # ---------------------------------------------------------- 2. HPPC fit
    rec = choose_hppc(pick(a.root, "HPPC", T))
    if rec is None:
        sys.exit("No HPPC file found. Paste me: ls -R data/raw | grep -i -E 'pulse|hppc'")
    H = fit_hppc(rec, ocv, Q)
    fits, gap_dq, gap_mode = H["fits"], H["gap_dq"], H["gap_mode"]
    dt, t, i, v, soc, valid, dq_tst = H["dt"], H["t"], H["i"], H["v"], H["soc"], H["valid"], H["dq_tester"]
    gaps = list(range(H["n_gaps"]))
    p_soc = fits["soc_dependent"][0]
    for k in ("r0", "r1", "r2"):
        print(f"        {k} [mOhm] at SoC {list(KNOTS)}: " + " ".join(f"{1e3*x:.1f}" for x in getattr(p_soc, k)))

    # ---------------------------------------------------------- 3. validate on held-out drive cycles
    val, traces = {}, {}
    for kind in DRIVE:
        for path in pick(a.root, kind, T):
            r = load(path)
            d = prepare(r.df, r.raw_sign, dt=0.1)
            ii = fill_gaps(d["i"], d["gaps"], d["dq_tester"], 0.1)
            vi, ok = d["v"], d["valid"]
            s0 = initial_soc(vi[0], ii[0], ocv)
            si = soc_zoh(ii, 0.1, Q, s0)
            row = {"file": path.name, "soc0": s0, "soc_end": float(si[-1]), "hours": float(d["t"][-1] / 3600),
                   "n_gaps": len(d["gaps"])}
            for name, (p, _) in fits.items():
                vm = simulate(0.1, ii, si, p, ocv)[0]
                row[name] = voltage_errors_mv(vm, vi, valid=ok)
                row[name + "_soc_ge_10pct"] = voltage_errors_mv(vm, vi, si, 0.10, ok)
                if name == "soc_dependent":
                    traces[path.name] = (kind, d["t"], vi, vm)
            val[f"{kind}:{path.name}"] = row
            print(f"[VAL]  {kind:6s} {path.name[:42]:42s} gaps={row['n_gaps']} SoC {s0:.2f}->{si[-1]:.2f}"
                  f"  RMSE const={row['constant']['rmse_mv']:6.1f}  soc-dep={row['soc_dependent']['rmse_mv']:6.1f} mV"
                  f"  (SoC>=10%: {row['soc_dependent_soc_ge_10pct']['rmse_mv']:.1f})")

    best = min(fits, key=lambda n: np.mean([r[n]["rmse_mv"] for r in val.values()]) if val else fits[n][1])
    print(f"\nselected model for the EKF: {best}")

    save_params(ROOT / "results" / "model_params.json", ocv, fits[best][0],
                {"temp_c": T, "model": best, "ocv_info": ocv_info})
    out = {
        "temp_c": T, "capacity_ah": Q, "ocv": ocv_info, "hppc_file": rec.path.name,
        "hppc_grid_dt_s": dt, "hppc_gap_mode": gap_mode, "hppc_n_gaps": len(gaps),
        "hppc_gap_dq_ah": gap_dq["soc_dependent"] if gap_dq["soc_dependent"] is not None else dq_tst.tolist(),
        "hppc_soc_range": [float(soc.min()), float(soc.max())],
        "hppc_fit_rmse_mv": {n: f[1] for n, f in fits.items()},
        "params": {n: f[0].to_dict() for n, f in fits.items()},
        "selected": best, "validation": val,
    }
    (ROOT / "results" / "model_fit.json").write_text(json.dumps(out, indent=2))
    print("wrote results/model_params.json, results/model_fit.json")

    # ---------------------------------------------------------- plots
    g = np.linspace(0, 1, 501)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(100 * g, ocv(g))
    ax[0].set(xlabel="SoC [%]", ylabel="OCV [V]", title=f"OCV from C/20 @ {T:g}°C (Q = {Q:.3f} Ah)")
    ax[1].plot(100 * g, ocv.deriv(g))
    ax[1].set(xlabel="SoC [%]", ylabel="dOCV/dSoC [V per unit SoC]", title="OCV slope (EKF sensitivity)")
    for x in ax:
        x.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(ROOT / "plots" / f"stage2_ocv_{T:g}C.png", dpi=130)

    vm = simulate(dt, i, soc, p_soc, ocv)[0]
    vm_plot = np.where(valid, vm, np.nan)
    fig, ax = plt.subplots(2, 1, figsize=(11, 6))
    ax[0].plot(t / 3600, np.where(valid, v, np.nan), lw=0.6, label="measured")
    ax[0].plot(t / 3600, vm_plot, lw=0.6, label="2-RC model")
    ax[0].set(ylabel="V", title=f"HPPC fit @ {T:g}°C (RMSE {fits['soc_dependent'][1]:.1f} mV, gaps blanked)")
    ax[0].legend()
    big = np.flatnonzero(np.abs(np.diff(i)) > 3)
    k = int(big[len(big) // 2]) if len(big) else 0  # a pulse mid-way through the test
    w = slice(max(0, k - int(60 / dt)), min(len(t), k + int(1200 / dt)))
    ax[1].plot(t[w] - t[w][0], v[w], label="measured"); ax[1].plot(t[w] - t[w][0], vm[w], "--", label="model")
    ax[1].set(xlabel="s", ylabel="V", title=f"zoom on one pulse (SoC ≈ {100*soc[k]:.0f}%)"); ax[1].legend()
    for x in ax:
        x.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(ROOT / "plots" / f"stage2_hppc_fit_{T:g}C.png", dpi=130)

    if traces:
        fig, axes = plt.subplots(len(traces), 1, figsize=(11, 2.6 * len(traces)), squeeze=False)
        for ax, (name, (kind, ti, vi, vmi)) in zip(axes[:, 0], traces.items()):
            e = val[f"{kind}:{name}"]["soc_dependent"]["rmse_mv"]
            ax.plot(ti / 60, vi, lw=0.4, label="measured"); ax.plot(ti / 60, vmi, lw=0.4, label="model")
            ax.set(ylabel="V", title=f"{kind} {name[:30]} (held out) — RMSE {e:.1f} mV"); ax.grid(alpha=0.3)
        axes[0, 0].legend(); axes[-1, 0].set_xlabel("min")
        fig.tight_layout(); fig.savefig(ROOT / "plots" / f"stage2_validation_{T:g}C.png", dpi=130)
    print("wrote plots/stage2_ocv, stage2_hppc_fit, stage2_validation")


if __name__ == "__main__":
    main()

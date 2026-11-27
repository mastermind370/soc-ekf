"""Stage 2: capacity + OCV from C/20, 2-RC fit on HPPC, validation on 25 degC drive cycles.

Usage:  python scripts/fit_model.py [--temp 25] [--root data/raw]
Writes results/model_params.json, results/model_fit.json, plots/stage2_*.png
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from soc_ekf.data import classify, find_files, load, resample  # noqa: E402
from soc_ekf.model import (  # noqa: E402
    capacity_and_ocv, fit_ecm, initial_soc, low_rate_segments, save_params,
    simulate, soc_coulomb, voltage_errors_mv,
)

KNOTS = (0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)
DRIVE = ("UDDS", "US06", "HWFET", "LA92")


def pick(root, kind, temp):
    out = []
    for p in find_files(root):
        k, T, cond = classify(p)
        if k == kind and T == temp and cond == "const" and "end" not in str(p).lower():
            out.append(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--temp", type=float, default=25.0)
    ap.add_argument("--root", default=str(ROOT / "data" / "raw"))
    a = ap.parse_args()
    T = a.temp
    (ROOT / "plots").mkdir(exist_ok=True)

    # ---------------------------------------------------------- 1. capacity + OCV from C/20
    segs, src = {}, {}
    for p in pick(a.root, "C20", T):
        df = load(p).df
        s = low_rate_segments(df["t_s"].to_numpy(), df["i"].to_numpy(), df["v"].to_numpy())
        for k, seg in s.items():
            if k not in segs or seg["ah"] > segs[k]["ah"]:
                segs[k], src[k] = seg, p.name
    if "dis" not in segs:
        sys.exit("No C/20 discharge segment found. Paste me: ls -R data/raw | grep -i -E 'c20|ocv'")
    Q, ocv, ocv_info = capacity_and_ocv(segs)
    print(f"[C/20] capacity = {Q:.4f} Ah  ({ocv_info['method']}; dis from {src.get('dis')}, chg from {src.get('chg')})")
    print(f"[C/20] OCV spline fit RMSE = {ocv_info['ocv_spline_rmse_mv']:.2f} mV")

    # ---------------------------------------------------------- 2. HPPC fit
    hppc = [load(p) for p in pick(a.root, "HPPC", T)]
    if not hppc:
        sys.exit("No HPPC file found. Paste me: ls -R data/raw | grep -i -E 'pulse|hppc'")
    rec = max(hppc, key=lambda r: r.df["i"].clip(lower=0).sum())  # the full 100%->0% sweep
    dt = 0.1 if rec.df["t_s"].iloc[-1] < 40_000 else 0.25
    h = resample(rec.df, dt)
    t, i, v = h["t_s"].to_numpy(), h["i"].to_numpy(), h["v"].to_numpy()
    soc0 = initial_soc(v[0], i[0], ocv)
    soc = soc_coulomb(t, i, Q, soc0)
    print(f"[HPPC] {rec.path.name}: {t[-1]/3600:.2f} h, SoC0 = {soc0:.3f}, SoC end = {soc[-1]:.3f}")

    fits = {}
    for name, knots in (("constant", (0.0, 1.0)), ("soc_dependent", KNOTS)):
        t0 = time.perf_counter()
        init = fits["constant"][0] if name == "soc_dependent" else None
        p, rmse = fit_ecm(dt, i, v, soc, ocv, Q, knots, init=init)
        fits[name] = (p, rmse)
        print(f"[HPPC] {name:13s} fit RMSE = {rmse:6.2f} mV   tau1 = {p.tau1:6.2f} s  tau2 = {p.tau2:7.1f} s"
              f"   ({time.perf_counter()-t0:.1f} s)")
    p_soc = fits["soc_dependent"][0]
    for k in ("r0", "r1", "r2"):
        print(f"        {k} [mOhm] at SoC {list(KNOTS)}: " + " ".join(f"{1e3*x:.1f}" for x in getattr(p_soc, k)))

    # ---------------------------------------------------------- 3. validate on held-out drive cycles
    val, traces = {}, {}
    for kind in DRIVE:
        for path in pick(a.root, kind, T):
            r = load(path)
            d = resample(r.df, 0.1)
            ti, ii, vi = d["t_s"].to_numpy(), d["i"].to_numpy(), d["v"].to_numpy()
            s0 = initial_soc(vi[0], ii[0], ocv)
            si = soc_coulomb(ti, ii, Q, s0)
            row = {"file": path.name, "soc0": s0, "soc_end": float(si[-1]), "hours": float(ti[-1] / 3600)}
            for name, (p, _) in fits.items():
                vm = simulate(0.1, ii, si, p, ocv)[0]
                row[name] = voltage_errors_mv(vm, vi)
                row[name + "_soc_ge_10pct"] = voltage_errors_mv(vm, vi, si, 0.10)
                if name == "soc_dependent":
                    traces[path.name] = (kind, ti, vi, vm, si)
            val[f"{kind}:{path.name}"] = row
            print(f"[VAL]  {kind:6s} {path.name[:45]:45s} SoC0={s0:.3f}  RMSE const={row['constant']['rmse_mv']:6.1f} mV"
                  f"  soc-dep={row['soc_dependent']['rmse_mv']:6.1f} mV  (SoC>=10%: {row['soc_dependent_soc_ge_10pct']['rmse_mv']:.1f} mV)")

    best = min(fits, key=lambda n: np.mean([r[n]["rmse_mv"] for r in val.values()]) if val else fits[n][1])
    print(f"\nselected model for the EKF: {best}")

    save_params(ROOT / "results" / "model_params.json", ocv, fits[best][0],
                {"temp_c": T, "model": best, "ocv_info": ocv_info})
    out = {
        "temp_c": T,
        "capacity_ah": Q,
        "ocv": ocv_info,
        "hppc_file": rec.path.name,
        "hppc_fit_rmse_mv": {n: f[1] for n, f in fits.items()},
        "params": {n: f[0].to_dict() for n, f in fits.items()},
        "selected": best,
        "validation": val,
    }
    (ROOT / "results" / "model_fit.json").write_text(json.dumps(out, indent=2))
    print("wrote results/model_params.json, results/model_fit.json")

    # ---------------------------------------------------------- plots
    g = np.linspace(0, 1, 501)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(100 * g, ocv(g))
    ax[0].set(xlabel="SoC [%]", ylabel="OCV [V]", title=f"OCV from C/20 @ {T:g}°C (Q = {Q:.3f} Ah)")
    ax[1].plot(100 * g, 1e3 * ocv.deriv(g))
    ax[1].set(xlabel="SoC [%]", ylabel="dOCV/dSoC [mV per unit SoC]", title="OCV slope (EKF sensitivity)")
    for x in ax:
        x.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(ROOT / "plots" / f"stage2_ocv_{T:g}C.png", dpi=130)

    vm = simulate(dt, i, soc, p_soc, ocv)[0]
    fig, ax = plt.subplots(2, 1, figsize=(11, 6))
    ax[0].plot(t / 3600, v, lw=0.6, label="measured"); ax[0].plot(t / 3600, vm, lw=0.6, label="2-RC model")
    ax[0].set(ylabel="V", title=f"HPPC fit @ {T:g}°C (RMSE {fits['soc_dependent'][1]:.1f} mV)"); ax[0].legend()
    k = np.argmax(np.abs(np.diff(i)) > 3) if (np.abs(np.diff(i)) > 3).any() else 0
    w = slice(max(0, k - int(30 / dt)), min(len(t), k + int(400 / dt)))
    ax[1].plot(t[w] - t[w][0], v[w], label="measured"); ax[1].plot(t[w] - t[w][0], vm[w], "--", label="model")
    ax[1].set(xlabel="s", ylabel="V", title="zoom on first pulse"); ax[1].legend()
    for x in ax:
        x.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(ROOT / "plots" / f"stage2_hppc_fit_{T:g}C.png", dpi=130)

    if traces:
        fig, axes = plt.subplots(len(traces), 1, figsize=(11, 2.6 * len(traces)), squeeze=False)
        for ax, (name, (kind, ti, vi, vmi, si)) in zip(axes[:, 0], traces.items()):
            e = 1e3 * np.sqrt(np.mean((vmi - vi) ** 2))
            ax.plot(ti / 60, vi, lw=0.4, label="measured"); ax.plot(ti / 60, vmi, lw=0.4, label="model")
            ax.set(ylabel="V", title=f"{kind} (held out) — RMSE {e:.1f} mV"); ax.grid(alpha=0.3)
        axes[0, 0].legend(); axes[-1, 0].set_xlabel("min")
        fig.tight_layout(); fig.savefig(ROOT / "plots" / f"stage2_validation_{T:g}C.png", dpi=130)
    print("wrote plots/stage2_ocv, stage2_hppc_fit, stage2_validation")


if __name__ == "__main__":
    main()

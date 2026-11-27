"""Stage 5: run the fixed-point C EKF on the held-out drive cycles and compare it with the Python float EKF.

Usage:  sh c/build.sh && python scripts/compare_c.py [--temp 25]
Needs results/model_params.json (stage 2) and results/ekf_params.json (stage 4).
Writes results/c_port.json, results/c/*.bin (git-ignored), plots/stage5_*.png

Reports per cycle: max / RMS |SoC_fixed - SoC_python| (correct and 30%-wrong start), the fixed-point RMSE vs
truth next to the float RMSE, ns per update on this machine (fixed and double), and an op-count based
cycle estimate for a Cortex-M4.
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from soc_ekf.baselines import soc_errors  # noqa: E402
from soc_ekf.cycles import load_cycles  # noqa: E402
from soc_ekf.ekf import EKFTuning, Tables, decimate, run_ekf  # noqa: E402
from soc_ekf.model import load_params  # noqa: E402

TEST_KINDS = ("UDDS", "US06", "HWFET")
# Cortex-M4 cost per counted op [cycles], (low, high). int64 multiply = 1-3 MUL/UMLAL, int64 add = ADDS+ADC,
# variable int64 shift = 2-4, int64/int64 division = libgcc __aeabi_ldivmod (data dependent), table load = LDR.
M4_COST = {"mul": (1, 4), "add": (2, 2), "shift": (2, 4), "div": (50, 150), "load": (1, 3), "branch": (1, 3)}
CALL_OVERHEAD = (1.1, 1.4)  # function calls, register spills


def write_model(path: Path, tb: Tables, tune: EKFTuning, dt: float):
    with open(path, "wb") as f:
        for tab in (tb.ocv, tb.docv, tb.r0, tb.dr0, tb.r1, tb.r2):
            f.write(struct.pack(f"<{len(tab)}d", *tab))
        f.write(struct.pack("<9d", tb.tau1, tb.tau2, tb.Q, dt, tune.sigma_v, tune.sigma_soc_proc,
                            tune.sigma_rc_proc, tune.sigma_soc0, tune.sigma_rc0))


def write_trace(path: Path, ia, inow, v, s0_true, s0_wrong):
    with open(path, "wb") as f:
        f.write(struct.pack("<q2d", len(v), s0_true, s0_wrong))
        f.write(np.column_stack([ia, inow, v]).astype("<f8").tobytes())


def run_bin(binary: Path, model: Path, trace: Path, out: Path):
    r = subprocess.run([str(binary), str(model), str(trace), str(out)], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"{binary.name} failed: {r.stderr}")
    return r.stdout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--temp", type=float, default=25.0)
    ap.add_argument("--wrong-start", type=float, default=0.30)
    ap.add_argument("--bin-dir", default=str(ROOT / "c" / "build"))
    ap.add_argument("--root", default=str(ROOT / "data" / "raw"))
    a = ap.parse_args()
    bdir = Path(a.bin_dir)
    for b in ("ekf_trace", "ekf_trace_ops"):
        if not (bdir / b).exists():
            sys.exit(f"{bdir / b} missing: run  sh c/build.sh  first")
    ocv, p = load_params(ROOT / "results" / "model_params.json")
    ep = json.loads((ROOT / "results" / "ekf_params.json").read_text())
    dt = ep.pop("dt_s")
    tune = EKFTuning(**ep)
    tb = Tables(ocv, p)
    work = ROOT / "results" / "c"
    work.mkdir(parents=True, exist_ok=True)
    write_model(work / "model.bin", tb, tune, dt)

    rows, ns_q, ns_f, ops = {}, [], [], None
    print(f"{'cycle':6} {'steps':>6} {'max|Q-py|':>10} {'rms|Q-py|':>10} {'max wrong':>10} {'max|C64-py|':>12} "
          f"{'RMSE py':>8} {'RMSE Q':>7} {'ns Q':>6} {'ns f64':>7}")
    for c in load_cycles(a.root, a.temp, ocv, p.capacity_ah, TEST_KINDS):
        t, ia, inow, v, sref, ok = decimate(c.t, c.i, c.v, c.soc_ref, c.valid, c.dt, dt)
        s0 = float(sref[0])
        tag = f"{c.kind}_{abs(hash(c.name)) % 10000:04d}"
        write_trace(work / f"{tag}.bin", ia, inow, v, s0, s0 - a.wrong_start)
        out_txt = run_bin(bdir / "ekf_trace", work / "model.bin", work / f"{tag}.bin", work / f"{tag}_out.bin")
        m = re.search(r"ns_per_update_q ([\d.]+) ns_per_update_f ([\d.]+)", out_txt)
        nq, nf = float(m.group(1)), float(m.group(2))
        ns_q.append(nq); ns_f.append(nf)
        if ops is None:
            txt = run_bin(bdir / "ekf_trace_ops", work / "model.bin", work / f"{tag}.bin", work / "ops_out.bin")
            ops = {k: float(x) for k, x in re.findall(r"(\w+)=([\d.]+)", txt)}
        res = np.fromfile(work / f"{tag}_out.bin", dtype="<f8").reshape(-1, 4)
        q_true, q_wrong, f_true, f_wrong = res.T
        py_true, _ = run_ekf(tb, dt, ia, inow, v, s0, tune)
        py_wrong, _ = run_ekf(tb, dt, ia, inow, v, s0 - a.wrong_start, tune)
        d_true = 100 * np.abs(q_true - py_true)
        d_wrong = 100 * np.abs(q_wrong - py_wrong)
        d_c64 = 100 * max(np.abs(f_true - py_true).max(), np.abs(f_wrong - py_wrong).max())
        e_py = soc_errors(py_true, sref, ok)
        e_q = soc_errors(q_true, sref, ok)
        e_qw = soc_errors(q_wrong, sref, ok)
        rows[f"{c.kind}:{c.name}"] = {
            "steps": int(len(v)),
            "fixed_vs_float_max_pct": float(d_true.max()), "fixed_vs_float_rms_pct": float(np.sqrt(np.mean(d_true ** 2))),
            "fixed_vs_float_wrong_start_max_pct": float(d_wrong.max()),
            "c_double_vs_python_max_pct": float(d_c64),
            "rmse_vs_truth_float_pct": e_py["rmse_pct"], "rmse_vs_truth_fixed_pct": e_q["rmse_pct"],
            "rmse_vs_truth_fixed_wrong_start_pct": e_qw["rmse_pct"],
            "ns_per_update_fixed": nq, "ns_per_update_double": nf,
        }
        print(f"{c.kind:6} {len(v):6d} {d_true.max():10.5f} {np.sqrt(np.mean(d_true**2)):10.5f} {d_wrong.max():10.5f} "
              f"{d_c64:12.2e} {e_py['rmse_pct']:8.3f} {e_q['rmse_pct']:7.3f} {nq:6.1f} {nf:7.1f}")

        fig, ax = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
        tm = t / 60
        ax[0].plot(tm, 100 * sref, "k", lw=1.4, label="truth")
        ax[0].plot(tm, 100 * py_true, lw=0.8, label="Python float EKF")
        ax[0].plot(tm, 100 * q_true, "--", lw=0.8, label="C fixed-point EKF")
        ax[0].set(ylabel="SoC [%]", title=f"Stage 5 — {c.kind}: fixed-point C vs float Python"); ax[0].legend(fontsize=8)
        ax[1].plot(tm, 100 * (q_true - py_true), lw=0.7, label="correct start")
        ax[1].plot(tm, 100 * (q_wrong - py_wrong), lw=0.7, label=f"{100*a.wrong_start:.0f}% wrong start")
        ax[1].set(ylabel="fixed − float [% SoC]", xlabel="min"); ax[1].legend(fontsize=8)
        for x in ax:
            x.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(ROOT / "plots" / f"stage5_{c.kind}_{tag[-4:]}.png", dpi=120); plt.close(fig)

    if not rows:
        sys.exit("no test cycles found")
    lo = sum(ops[k] * M4_COST[k][0] for k in M4_COST) * CALL_OVERHEAD[0]
    hi = sum(ops[k] * M4_COST[k][1] for k in M4_COST) * CALL_OVERHEAD[1]
    summary = {
        "fixed_vs_float_max_pct": max(r["fixed_vs_float_max_pct"] for r in rows.values()),
        "fixed_vs_float_rms_pct": float(np.sqrt(np.mean([r["fixed_vs_float_rms_pct"] ** 2 for r in rows.values()]))),
        "fixed_vs_float_wrong_start_max_pct": max(r["fixed_vs_float_wrong_start_max_pct"] for r in rows.values()),
        "ns_per_update_fixed_median": float(np.median(ns_q)),
        "ns_per_update_double_median": float(np.median(ns_f)),
        "ops_per_update": ops,
        "cortex_m4_cycles_estimate": [round(lo), round(hi)],
        "cortex_m4_us_at_168mhz": [round(lo / 168, 2), round(hi / 168, 2)],
        "cortex_m4_cost_model": {k: list(v) for k, v in M4_COST.items()},
        "ekf_rate_hz": 1 / dt,
    }
    print(f"\nops per update: " + ", ".join(f"{k} {v:.0f}" for k, v in ops.items()))
    print(f"Cortex-M4 estimate: {lo:.0f}-{hi:.0f} cycles per update "
          f"({lo/168:.1f}-{hi/168:.1f} us at 168 MHz; the EKF runs at {1/dt:g} Hz)")
    print(f"this machine: fixed {summary['ns_per_update_fixed_median']:.1f} ns, double "
          f"{summary['ns_per_update_double_median']:.1f} ns per update (median)")
    print("summary:", json.dumps({k: v for k, v in summary.items() if "cost" not in k}, indent=1))
    (ROOT / "results" / "c_port.json").write_text(json.dumps({"per_file": rows, "summary": summary}, indent=2))
    print("wrote results/c_port.json, plots/stage5_*.png")


if __name__ == "__main__":
    main()

"""Stage 4: tune the EKF on LA92 + mixed cycles, test on held-out UDDS / US06 / HWFET at 25 degC.

Usage:  python scripts/run_ekf.py [--temp 25] [--dt 1.0] [--wrong-start 0.30]
Needs results/model_params.json. Writes results/ekf.json, results/ekf_params.json, plots/stage4_*.png

Per test cycle: SoC RMSE / max error from the correct start and from a start that is --wrong-start low,
time to converge (|error| stays < 5% and < 2%), and robustness to sensor corruption:
  gain_+5% / gain_-5%  current sensor reads 5% high / low (coulomb counting shown for comparison)
  white_noise          Gaussian noise, std = 5% of the cycle's RMS current and 5 mV on voltage
"""
from __future__ import annotations

import argparse
import itertools
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

from soc_ekf.baselines import coulomb_count, soc_errors  # noqa: E402
from soc_ekf.cycles import load_cycles  # noqa: E402
from soc_ekf.data import NOMINAL_CAPACITY_AH  # noqa: E402
from soc_ekf.ekf import EKFTuning, Tables, convergence_time, decimate, run_ekf  # noqa: E402
from soc_ekf.model import load_params  # noqa: E402

TEST_KINDS = ("UDDS", "US06", "HWFET")
TUNE_KINDS = ("LA92", "MIX", "NN")
GRID = {
    "sigma_v": [0.005, 0.01, 0.02, 0.04],
    "sigma_soc_proc": [1e-6, 1e-5, 1e-4],
    "sigma_rc_proc": [1e-4, 1e-3, 1e-2],
}


def prep(c, dt):
    t, ia, inow, v, sref, ok = decimate(c.t, c.i, c.v, c.soc_ref, c.valid, c.dt, dt)
    return {"name": c.name, "kind": c.kind, "t": t, "ia": ia, "inow": inow, "v": v, "sref": sref, "ok": ok}


def score(tb, dt, recs, tune, wrong):
    """Tuning objective: mean SoC RMSE over correct-start and wrong-start runs."""
    errs = []
    for r in recs:
        for s0 in (r["sref"][0], r["sref"][0] - wrong):
            est, _ = run_ekf(tb, dt, r["ia"], r["inow"], r["v"], s0, tune)
            errs.append(soc_errors(est, r["sref"], r["ok"])["rmse_pct"])
    return float(np.mean(errs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--temp", type=float, default=25.0)
    ap.add_argument("--dt", type=float, default=1.0, help="EKF update period [s]")
    ap.add_argument("--wrong-start", type=float, default=0.30)
    ap.add_argument("--root", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    pfile = ROOT / "results" / "model_params.json"
    if not pfile.exists():
        sys.exit("results/model_params.json missing: run python scripts/fit_model.py first")
    ocv, p = load_params(pfile)
    tb = Tables(ocv, p)
    dt, W = a.dt, a.wrong_start

    test = [prep(c, dt) for c in load_cycles(a.root, a.temp, ocv, p.capacity_ah, TEST_KINDS)]
    tune_set = [prep(c, dt) for c in load_cycles(a.root, a.temp, ocv, p.capacity_ah, TUNE_KINDS)]
    if not test:
        sys.exit("no UDDS/US06/HWFET cycles found")
    if not tune_set:
        print("WARNING: no LA92/MIX/NN cycles to tune on; tuning on UDDS (not held out)")
        tune_set = [r for r in test if r["kind"] == "UDDS"]
    print(f"EKF @ {1/dt:g} Hz; tuning on {[r['kind'] for r in tune_set]}, testing on {[r['kind'] for r in test]}")

    # ------------------------------------------------------------- tune (grid search)
    t0 = time.perf_counter()
    best = None
    for sv, sq, sr in itertools.product(GRID["sigma_v"], GRID["sigma_soc_proc"], GRID["sigma_rc_proc"]):
        tune = EKFTuning(sigma_v=sv, sigma_soc_proc=sq, sigma_rc_proc=sr)
        sc = score(tb, dt, tune_set, tune, W)
        if best is None or sc < best[0]:
            best = (sc, tune)
    tune = best[1]
    print(f"tuned in {time.perf_counter()-t0:.0f} s: sigma_v={tune.sigma_v} V, sigma_soc_proc={tune.sigma_soc_proc}, "
          f"sigma_rc_proc={tune.sigma_rc_proc} (tuning-set RMSE {best[0]:.2f}%)")

    # ------------------------------------------------------------- test
    rng = np.random.default_rng(a.seed)
    results = {}
    hdr = (f"{'cycle':6} {'RMSE true':>9} {'max':>6} {'RMSE wrong':>10} {'max':>6} {'t<5%':>7} {'t<2%':>7} "
           f"{'gain+5%':>8} {'gain-5%':>8} {'noise':>6} {'CC+5%':>6} {'CC-5%':>6}")
    print("\n" + hdr + "\n" + "-" * len(hdr))
    n_steps, t_loop = 0, 0.0
    for r in test:
        s_true = r["sref"][0]
        tt = time.perf_counter()
        est_t, sig_t = run_ekf(tb, dt, r["ia"], r["inow"], r["v"], s_true, tune)
        t_loop += time.perf_counter() - tt
        n_steps += len(r["v"])
        est_w, sig_w = run_ekf(tb, dt, r["ia"], r["inow"], r["v"], s_true - W, tune)
        e_true = soc_errors(est_t, r["sref"], r["ok"])
        e_wrong = soc_errors(est_w, r["sref"], r["ok"])
        err_w = 100 * (est_w - r["sref"])
        conv5 = convergence_time(r["t"], err_w, r["ok"], 5.0)
        conv2 = convergence_time(r["t"], err_w, r["ok"], 2.0)

        rob = {}
        for name, g in (("gain_plus5pct", 1.05), ("gain_minus5pct", 0.95)):
            est, _ = run_ekf(tb, dt, g * r["ia"], g * r["inow"], r["v"], s_true, tune)
            cc = coulomb_count(g * r["ia"], dt, NOMINAL_CAPACITY_AH, s_true)
            rob[name] = {"ekf": soc_errors(est, r["sref"], r["ok"]), "cc": soc_errors(cc, r["sref"], r["ok"])}
        irms = float(np.sqrt(np.mean(r["ia"] ** 2)))
        ni = rng.normal(0, 0.05 * irms, len(r["v"]))
        nv = rng.normal(0, 0.005, len(r["v"]))
        est_n, _ = run_ekf(tb, dt, r["ia"] + ni, r["inow"] + ni, r["v"] + nv, s_true, tune)
        rob["white_noise"] = {"ekf": soc_errors(est_n, r["sref"], r["ok"]),
                              "sigma_i_a": 0.05 * irms, "sigma_v_v": 0.005}

        key = f"{r['kind']}:{r['name']}"
        results[key] = {"kind": r["kind"], "hours": float(r["t"][-1] / 3600), "true_start": e_true,
                        "wrong_start": e_wrong, "wrong_start_offset": -W,
                        "converge_s_5pct": conv5, "converge_s_2pct": conv2, "robustness": rob,
                        "final_sigma_soc_pct": float(100 * sig_t[-1])}
        f = lambda x: "never" if x is None else f"{x:.0f}s"  # noqa: E731
        print(f"{r['kind']:6} {e_true['rmse_pct']:9.2f} {e_true['max_abs_pct']:6.2f} {e_wrong['rmse_pct']:10.2f} "
              f"{e_wrong['max_abs_pct']:6.2f} {f(conv5):>7} {f(conv2):>7} "
              f"{rob['gain_plus5pct']['ekf']['rmse_pct']:8.2f} {rob['gain_minus5pct']['ekf']['rmse_pct']:8.2f} "
              f"{rob['white_noise']['ekf']['rmse_pct']:6.2f} {rob['gain_plus5pct']['cc']['rmse_pct']:6.2f} "
              f"{rob['gain_minus5pct']['cc']['rmse_pct']:6.2f}")

        fig, ax = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
        tm = r["t"] / 60
        cc_w = coulomb_count(r["ia"], dt, NOMINAL_CAPACITY_AH, s_true - W)
        ax[0].plot(tm, 100 * r["sref"], "k", lw=1.6, label="truth")
        ax[0].plot(tm, 100 * est_t, lw=0.9, label="EKF, correct start")
        ax[0].plot(tm, 100 * est_w, lw=0.9, label=f"EKF, start {100*W:.0f}% low")
        ax[0].plot(tm, 100 * cc_w, "--", lw=0.9, label=f"coulomb count, start {100*W:.0f}% low")
        ax[0].set(ylabel="SoC [%]", title=f"Stage 4 EKF — {r['kind']} @ {a.temp:g}°C (held out)")
        ax[0].legend(fontsize=8)
        ax[1].plot(tm, 100 * (est_t - r["sref"]), lw=0.8, label="EKF correct start")
        ax[1].plot(tm, err_w, lw=0.8, label="EKF wrong start")
        ax[1].fill_between(tm, -300 * sig_t, 300 * sig_t, alpha=0.15, label="±3σ (correct start)")
        ax[1].set(ylabel="error [% SoC]", xlabel="min", ylim=(-10, 10))
        ax[1].axhline(0, color="k", lw=0.5); ax[1].legend(fontsize=8)
        for x in ax:
            x.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(ROOT / "plots" / f"stage4_{r['kind']}_{r['name'][:20].replace(' ', '_')}.png", dpi=120)
        plt.close(fig)

    by_kind = {}
    for k in TEST_KINDS:
        rows = [v for v in results.values() if v["kind"] == k]
        if rows:
            by_kind[k] = {
                "rmse_true_start_pct": float(np.mean([r["true_start"]["rmse_pct"] for r in rows])),
                "rmse_wrong_start_pct": float(np.mean([r["wrong_start"]["rmse_pct"] for r in rows])),
                "converge_s_2pct": [r["converge_s_2pct"] for r in rows],
                "rmse_noise_pct": float(np.mean([r["robustness"]["white_noise"]["ekf"]["rmse_pct"] for r in rows])),
            }
    py_us = 1e6 * t_loop / max(n_steps, 1)
    print(f"\nPython EKF: {py_us:.1f} us per update (reference only; the C port is timed in stage 5)")
    out = {"temp_c": a.temp, "dt_s": dt, "tuning": tune.to_dict(), "tuning_set": [r["name"] for r in tune_set],
           "tuning_objective_rmse_pct": best[0], "per_file": results, "summary": by_kind,
           "python_us_per_update": py_us}
    (ROOT / "results" / "ekf.json").write_text(json.dumps(out, indent=2))
    (ROOT / "results" / "ekf_params.json").write_text(json.dumps({"dt_s": dt, **tune.to_dict()}, indent=2))
    print("summary:", json.dumps(by_kind, indent=1))
    print("wrote results/ekf.json, results/ekf_params.json, plots/stage4_*.png")


if __name__ == "__main__":
    main()

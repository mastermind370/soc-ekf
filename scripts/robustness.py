"""Stage 6: robustness of the EKF to temperature, cell aging and current-sensor bias.

Usage:  python scripts/robustness.py
Needs results/model_params.json (stage 2) and results/ekf_params.json (stage 4).
Writes results/robustness.json, results/model_params_<T>C.json, plots/stage6_*.png

Truth SoC everywhere = lab coulomb counting with the 25 degC C/20 capacity, from a full start.
Test cycles: UDDS, US06, HWFET (never used for fitting or tuning). EKF noise tuning is the 25 degC one.

1. Temperature: at each temperature with data, the EKF with
     model_25C   the 25 degC model used as-is (what you get without temperature compensation)
     model_T     R0/R1/R2/tau refitted on that temperature's own HPPC test (25 degC OCV and capacity)
   vs coulomb counting from the correct start (2.9 Ah datasheet capacity).
   Cycles whose chamber temperature drifts ("Trise") are run too, with model_25C and the start-temperature model.
2. Aging: the end-of-tests C/20 check measures the faded capacity. Without aged drive-cycle data, the effect is
   emulated on the 25 degC cycles: the EKF (and coulomb counting) believe the cell is bigger than it is by the
   measured fresh/aged ratio, which is what an un-updated BMS does on an aged cell.
3. Current-sensor bias: constant offsets added to the measured current, EKF vs coulomb counting.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import replace
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
from soc_ekf.ekf import EKFTuning, Tables, decimate, run_ekf  # noqa: E402
from soc_ekf.fitting import available_temps, c20_capacity_ocv, choose_hppc, fit_hppc, pick  # noqa: E402
from soc_ekf.model import load_params, save_params  # noqa: E402

TEST_KINDS = ("UDDS", "US06", "HWFET")
BIAS_MA = (-100, -50, -20, 20, 50, 100)


def prep(c, dt):
    t, ia, inow, v, sref, ok = decimate(c.t, c.i, c.v, c.soc_ref, c.valid, c.dt, dt)
    temp = c.temp[:: int(round(dt / c.dt))][: len(t)] if np.isfinite(c.temp).any() else None
    return {"kind": c.kind, "name": c.name, "ia": ia, "inow": inow, "v": v, "sref": sref, "ok": ok, "temp": temp}


def ekf_err(tb, tune, dt, r, soc_off=0.0, bias=0.0):
    est, _ = run_ekf(tb, dt, r["ia"] + bias, r["inow"] + bias, r["v"], r["sref"][0] + soc_off, tune)
    return soc_errors(est, r["sref"], r["ok"])


def cc_err(dt, r, bias=0.0, cap=NOMINAL_CAPACITY_AH):
    return soc_errors(coulomb_count(r["ia"] + bias, dt, cap, r["sref"][0]), r["sref"], r["ok"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--wrong-start", type=float, default=0.30)
    ap.add_argument("--quiet-fit", action="store_true", help="hide the HPPC fit log")
    a = ap.parse_args()
    for f in ("model_params.json", "ekf_params.json"):
        if not (ROOT / "results" / f).exists():
            sys.exit(f"results/{f} missing: run stages 2 and 4 first")
    ocv, p25 = load_params(ROOT / "results" / "model_params.json")
    selected = json.loads((ROOT / "results" / "model_params.json").read_text()).get("model", "soc_dependent")
    ep = json.loads((ROOT / "results" / "ekf_params.json").read_text())
    dt = ep.pop("dt_s")
    tune = EKFTuning(**ep)
    Q = p25.capacity_ah
    tb25 = Tables(ocv, p25)
    W = a.wrong_start
    log = (lambda *x, **k: None) if a.quiet_fit else (lambda s: print("    " + s))
    out = {"truth_capacity_ah": Q, "ekf_tuning": tune.to_dict(), "dt_s": dt}

    # ===================================================================== 1. temperature
    print("=== 1. temperature")
    temps = available_temps(a.root)
    temp_rows, temp_models = {}, {}
    for T in temps:
        cyc = [prep(c, dt) for c in load_cycles(a.root, T, ocv, Q, TEST_KINDS)]
        if not cyc:
            continue
        if T == 25:
            pT, fit_info = p25, {"hppc_rmse_mv": None, "warnings": []}
        else:
            rec = choose_hppc(pick(a.root, "HPPC", T))
            if rec is None:
                print(f"  {T:g} degC: no HPPC file, model_T = model_25C")
                pT, fit_info = p25, {"hppc_rmse_mv": None, "warnings": ["no HPPC"]}
            else:
                print(f"  {T:g} degC: fitting on {rec.path.name}")
                H = fit_hppc(rec, ocv, Q, log=log)
                pT, rmse = H["fits"][selected]
                fit_info = {"hppc_file": rec.path.name, "hppc_rmse_mv": rmse, "warnings": H["warnings"][selected],
                            "gap_mode": H["gap_mode"]}
                save_params(ROOT / "results" / f"model_params_{T:g}C.json", ocv, pT,
                            {"temp_c": T, "model": selected, **fit_info})
        temp_models[T] = pT
        tbT = Tables(ocv, pT)
        per = []
        for r in cyc:
            row = {"kind": r["kind"], "file": r["name"],
                   "soc_end_truth": float(r["sref"][-1]),
                   "model_25C": ekf_err(tb25, tune, dt, r), "model_T": ekf_err(tbT, tune, dt, r),
                   "model_T_wrong_start": ekf_err(tbT, tune, dt, r, -W), "cc": cc_err(dt, r)}
            per.append(row)
        temp_rows[T] = {"fit": fit_info, "params": pT.to_dict(), "cycles": per,
                        "mean_rmse_pct": {m: float(np.mean([x[m]["rmse_pct"] for x in per]))
                                          for m in ("model_25C", "model_T", "model_T_wrong_start", "cc")}}
        mr = temp_rows[T]["mean_rmse_pct"]
        print(f"  {T:>5g} degC  ({len(per)} cycles, truth SoC at cutoff {np.mean([x['soc_end_truth'] for x in per])*100:.0f}%)"
              f"  EKF 25C-model {mr['model_25C']:6.2f}%   EKF T-model {mr['model_T']:6.2f}%"
              f"   (wrong start {mr['model_T_wrong_start']:.2f}%)   CC {mr['cc']:.2f}%")
        for x in per:
            print(f"         {x['kind']:6} 25C-model {x['model_25C']['rmse_pct']:6.2f}  T-model {x['model_T']['rmse_pct']:6.2f}"
                  f"  max {x['model_T']['max_abs_pct']:6.2f}  CC {x['cc']['rmse_pct']:5.2f}")
    out["temperature"] = {f"{T:g}": v for T, v in temp_rows.items()}

    trise = {}
    for T in temps:
        cyc = [prep(c, dt) for c in load_cycles(a.root, T, ocv, Q, TEST_KINDS, condition="trise")]
        if not cyc or T not in temp_models:
            continue
        tbT = Tables(ocv, temp_models[T])
        rows = []
        for r in cyc:
            rows.append({"kind": r["kind"], "file": r["name"],
                         "temp_range_c": [float(np.nanmin(r["temp"])), float(np.nanmax(r["temp"]))] if r["temp"] is not None else None,
                         "model_25C": ekf_err(tb25, tune, dt, r), "model_start_T": ekf_err(tbT, tune, dt, r),
                         "cc": cc_err(dt, r)})
        trise[f"{T:g}"] = rows
        print(f"  Trise from {T:g} degC: " + "; ".join(
            f"{x['kind']} 25C-model {x['model_25C']['rmse_pct']:.2f}% start-T-model {x['model_start_T']['rmse_pct']:.2f}%"
            for x in rows))
    out["temperature_drift"] = trise

    # ===================================================================== 2. aging
    print("\n=== 2. aging")
    test25 = [prep(c, dt) for c in load_cycles(a.root, 25.0, ocv, Q, TEST_KINDS)]
    aged = c20_capacity_ocv(pick(a.root, "C20", 25.0, aged=True))
    if aged is not None:
        Q_aged = aged[0]
        src = f"measured: end-of-tests C/20 = {Q_aged:.4f} Ah ({aged[2]['sources'].get('dis')})"
    else:
        Q_aged = 0.8 * Q
        src = "ASSUMED 20% fade (no end-of-tests C/20 file found)"
    ratio = Q / Q_aged
    print(f"  fresh {Q:.4f} Ah, aged {Q_aged:.4f} Ah -> fade {100*(1-Q_aged/Q):.1f}%  [{src}]")
    tb_old = Tables(ocv, replace(p25, capacity_ah=Q * ratio))  # EKF still believes the fresh capacity
    age_rows = []
    for r in test25:
        age_rows.append({"kind": r["kind"], "file": r["name"],
                         "ekf_matched": ekf_err(tb25, tune, dt, r), "ekf_fresh_capacity": ekf_err(tb_old, tune, dt, r),
                         "ekf_fresh_capacity_wrong_start": ekf_err(tb_old, tune, dt, r, -W),
                         "cc_fresh_capacity": cc_err(dt, r, cap=NOMINAL_CAPACITY_AH * ratio)})
        x = age_rows[-1]
        print(f"  {r['kind']:6} EKF right capacity {x['ekf_matched']['rmse_pct']:5.2f}%  EKF fresh capacity "
              f"{x['ekf_fresh_capacity']['rmse_pct']:5.2f}% (wrong start {x['ekf_fresh_capacity_wrong_start']['rmse_pct']:.2f}%)"
              f"  CC fresh capacity {x['cc_fresh_capacity']['rmse_pct']:5.2f}%")
    out["aging"] = {"fresh_capacity_ah": Q, "aged_capacity_ah": Q_aged, "fade_pct": 100 * (1 - Q_aged / Q),
                    "source": src, "cycles": age_rows}

    # ===================================================================== 3. current-sensor bias
    print("\n=== 3. current-sensor bias")
    bias_rows = defaultdict(dict)
    for b in BIAS_MA:
        e_ekf = [ekf_err(tb25, tune, dt, r, bias=b / 1000) for r in test25]
        e_cc = [cc_err(dt, r, bias=b / 1000) for r in test25]
        for r, x, y in zip(test25, e_ekf, e_cc):
            bias_rows[f"{b:+d}mA"][r["kind"] + ":" + r["name"]] = {"ekf": x, "cc": y}
        print(f"  {b:+5d} mA   EKF RMSE " + " ".join(f"{r['kind']} {x['rmse_pct']:.2f}%" for r, x in zip(test25, e_ekf)) +
              "   |  CC RMSE " + " ".join(f"{r['kind']} {y['rmse_pct']:.2f}%" for r, y in zip(test25, e_cc)))
    out["current_bias"] = bias_rows

    (ROOT / "results" / "robustness.json").write_text(json.dumps(out, indent=2))

    # ===================================================================== plots
    if temp_rows:
        Ts = sorted(temp_rows)
        x = np.arange(len(Ts))
        fig, ax = plt.subplots(figsize=(9, 4))
        for k, (m, lab) in enumerate((("model_25C", "EKF, 25°C model"), ("model_T", "EKF, per-temperature model"),
                                      ("model_T_wrong_start", f"EKF, per-T model, start {100*W:.0f}% low"),
                                      ("cc", "coulomb counting"))):
            ax.bar(x + (k - 1.5) * 0.2, [temp_rows[T]["mean_rmse_pct"][m] for T in Ts], 0.2, label=lab)
        ax.set_xticks(x, [f"{T:g}°C" for T in Ts])
        ax.set(ylabel="SoC RMSE [%] (mean of UDDS/US06/HWFET)", title="Stage 6 — temperature")
        ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
        fig.tight_layout(); fig.savefig(ROOT / "plots" / "stage6_temperature.png", dpi=130); plt.close(fig)
    if test25:
        fig, ax = plt.subplots(figsize=(8, 4))
        for r in test25:
            key = r["kind"] + ":" + r["name"]
            ax.plot(BIAS_MA, [bias_rows[f"{b:+d}mA"][key]["ekf"]["rmse_pct"] for b in BIAS_MA], "o-", label=f"EKF {r['kind']}")
            ax.plot(BIAS_MA, [bias_rows[f"{b:+d}mA"][key]["cc"]["rmse_pct"] for b in BIAS_MA], "x--", label=f"CC {r['kind']}")
        ax.set(xlabel="current-sensor offset [mA]", ylabel="SoC RMSE [%]", title="Stage 6 — current-sensor bias")
        ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(ROOT / "plots" / "stage6_bias.png", dpi=130); plt.close(fig)
    print("\nwrote results/robustness.json, plots/stage6_temperature.png, plots/stage6_bias.png")


if __name__ == "__main__":
    main()

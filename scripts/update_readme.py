"""Fill the <!-- BEGIN:x --> ... <!-- END:x --> blocks of README.md from results/*.json and plots/*.png.

Usage:  python scripts/update_readme.py
Every number in the README comes from a results file written by the stage scripts; a missing file
leaves a "not run yet" note instead of a number.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_KINDS = ("UDDS", "US06", "HWFET")


def load(name):
    p = ROOT / "results" / name
    return json.loads(p.read_text()) if p.exists() else None


def missing(cmd):
    return f"_Not run yet: `{cmd}`._"


def f(x, nd=2, unit=""):
    if x is None:
        return "–"
    return f"{x:.{nd}f}{unit}"


def short(name):
    m = re.search(r"(HWFTa|HWFTb)", name)
    return f" ({m.group(1)[-1]})" if m else ""


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# ------------------------------------------------------------------ blocks
def block_headline():
    ekf, base, cport = load("ekf.json"), load("baselines.json"), load("c_port.json")
    if ekf is None:
        return missing("python scripts/run_ekf.py")
    W = 100 * (base or {}).get("wrong_start", 0.30)
    bl = {}
    if base:
        for k, r in base["per_file"].items():
            bl[k.split(":", 1)[1]] = r
    rows = []
    for k, r in ekf["per_file"].items():
        name = k.split(":", 1)[1]
        b = bl.get(name, {})
        rows.append([
            r["kind"] + short(name),
            f(r["true_start"]["rmse_pct"], 2, "%"),
            f(r["wrong_start"]["rmse_pct"], 2, "%"),
            "never" if r["converge_s_2pct"] is None else f"{r['converge_s_2pct']:.0f} s",
            f(b.get("cc_wrong_start", {}).get("rmse_pct"), 1, "%"),
            f(b.get("ocv_lookup", {}).get("rmse_pct"), 1, "%"),
        ])
    txt = table(["Cycle (held out, 25 °C)", "EKF RMSE", f"EKF RMSE, start {W:.0f}% low",
                 "EKF converges to ±2%", f"Coulomb counting, start {W:.0f}% low", "Voltage lookup"], rows)
    if cport:
        s = cport["summary"]
        lo, hi = s["cortex_m4_cycles_estimate"]
        txt += (f"\n\nFixed-point C port: max difference from the float EKF **{s['fixed_vs_float_max_pct']:.1e} % SoC** "
                f"(RMS {s['fixed_vs_float_rms_pct']:.1e} %), **{s['ns_per_update_fixed_median']:.0f} ns per update** on an "
                f"Apple M4 (double-precision C: {s['ns_per_update_double_median']:.0f} ns); "
                f"estimated **{lo}–{hi} Cortex-M4 cycles** per update (op-count model, not measured on hardware).")
    return txt


def block_data():
    inv = load("data_inventory.json")
    if inv is None:
        return missing("python scripts/inventory.py")
    s = inv["summary"]
    return table(["Cell", "Files", "Temperatures", "Drive-cycle files", "Drive-cycle hours", "Total hours", "Size"],
                 [[s["cell"], s["n_files_loaded"], ", ".join(f"{t:g} °C" for t in s["temperatures_c"]),
                   s["n_drive_cycle_files"], f(s["drive_cycle_hours"], 1), f(s["total_hours"], 1),
                   f"{s['dataset_size_mb']:.0f} MB"]])


def block_model():
    m = load("model_fit.json")
    if m is None:
        return missing("python scripts/fit_model.py")
    sel = m["selected"]
    p = m["params"][sel]
    head = (f"Capacity (C/20, 25 °C): **{m['capacity_ah']:.3f} Ah** · OCV spline fit: {m['ocv']['ocv_spline_rmse_mv']:.2f} mV · "
            f"HPPC fit ({sel.replace('_', '-')} R): {m['hppc_fit_rmse_mv'][sel]:.1f} mV · "
            f"τ1 = {p['tau1']:.1f} s, τ2 = {p['tau2']:.0f} s · HPPC logging gaps: {m.get('hppc_n_gaps', '–')} "
            f"(charge from {m.get('hppc_gap_mode', '–')})")
    rows = []
    for k, r in m["validation"].items():
        kind = k.split(":", 1)[0]
        rows.append([kind + short(r["file"]), f(r["constant"]["rmse_mv"], 1), f(r["soc_dependent"]["rmse_mv"], 1),
                     f(r[sel]["max_abs_mv"], 0)])
    return head + "\n\n" + table(["Held-out cycle", "Voltage RMSE, constant R [mV]", "Voltage RMSE, SoC-dependent R [mV]",
                                  "Max error, selected [mV]"], rows)


def block_baselines():
    b = load("baselines.json")
    if b is None:
        return missing("python scripts/baselines.py")
    methods = ("cc_true_start", "cc_wrong_start", "ocv_lookup", "ocv_lookup_ir")
    rows = []
    for k, r in b["per_file"].items():
        kind, name = k.split(":", 1)
        rows.append([kind + short(name)] + [f"{f(r[m]['rmse_pct'])} / {f(r[m]['max_abs_pct'], 1)}" for m in methods])
    note = (f"RMSE / max error in % SoC. Coulomb counting uses the {b['bms_capacity_ah']} Ah datasheet capacity; "
            f"the wrong start is {100*b['wrong_start']:.0f}% low.")
    return note + "\n\n" + table(["Cycle", "Coulomb count", "Coulomb count, wrong start", "OCV(V) lookup",
                                  "OCV(V + R0·I) lookup"], rows)


def block_ekf():
    e = load("ekf.json")
    if e is None:
        return missing("python scripts/run_ekf.py")
    t = e["tuning"]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from soc_ekf.data import classify
    kinds = ", ".join(sorted({classify(n)[0] for n in e["tuning_set"]}))
    head = (f"EKF at {1/e['dt_s']:g} Hz, tuned by grid search on {kinds} "
            f"only: σ_v = {t['sigma_v']*1e3:g} mV, σ_SoC,proc = {t['sigma_soc_proc']:g} /√s, σ_RC,proc = {t['sigma_rc_proc']*1e3:g} mV/√s.")
    rows = []
    for k, r in e["per_file"].items():
        name = k.split(":", 1)[1]
        rb = r["robustness"]
        rows.append([r["kind"] + short(name), f"{f(r['true_start']['rmse_pct'])} / {f(r['true_start']['max_abs_pct'], 1)}",
                     f"{f(r['wrong_start']['rmse_pct'])} / {f(r['wrong_start']['max_abs_pct'], 1)}",
                     "never" if r["converge_s_5pct"] is None else f"{r['converge_s_5pct']:.0f} s",
                     "never" if r["converge_s_2pct"] is None else f"{r['converge_s_2pct']:.0f} s",
                     f"{f(rb['gain_plus5pct']['ekf']['rmse_pct'])} ({f(rb['gain_plus5pct']['cc']['rmse_pct'])})",
                     f"{f(rb['gain_minus5pct']['ekf']['rmse_pct'])} ({f(rb['gain_minus5pct']['cc']['rmse_pct'])})",
                     f(rb["white_noise"]["ekf"]["rmse_pct"])])
    return head + "\n\n" + table(["Cycle", "RMSE / max, correct start", "RMSE / max, 30% low start", "→ ±5%", "→ ±2%",
                                  "Current +5% (CC)", "Current −5% (CC)", "Noise*"], rows) + \
        "\n\n% SoC. *Gaussian noise, σ = 5% of the cycle's RMS current plus 5 mV on voltage."


def block_cport():
    c = load("c_port.json")
    if c is None:
        return missing("sh c/build.sh && python scripts/compare_c.py")
    rows = []
    for k, r in c["per_file"].items():
        kind, name = k.split(":", 1)
        rows.append([kind + short(name), r["steps"], f"{r['fixed_vs_float_max_pct']:.1e}", f"{r['fixed_vs_float_rms_pct']:.1e}",
                     f"{r['fixed_vs_float_wrong_start_max_pct']:.1e}", f(r["rmse_vs_truth_float_pct"]),
                     f(r["rmse_vs_truth_fixed_pct"]), f(r["ns_per_update_fixed"], 0), f(r["ns_per_update_double"], 0)])
    s = c["summary"]
    ops = s["ops_per_update"]
    lo, hi = s["cortex_m4_cycles_estimate"]
    us_lo, us_hi = s["cortex_m4_us_at_168mhz"]
    tail = (f"Ops per update: {', '.join(f'{v:.0f} {k}' for k, v in ops.items())}. Cortex-M4 estimate: "
            f"**{lo}–{hi} cycles** ({us_lo}–{us_hi} µs at 168 MHz) from per-op cost ranges in `scripts/compare_c.py` — "
            f"an estimate, not a hardware measurement.")
    return table(["Cycle", "Updates", "max \\|fixed − float\\| [% SoC]", "RMS", "max, wrong start", "RMSE float",
                  "RMSE fixed", "ns fixed (M4 Mac)", "ns double"], rows) + "\n\n" + tail


def block_robust():
    r = load("robustness.json")
    if r is None:
        return missing("python scripts/robustness.py")
    parts = []
    if r.get("temperature"):
        rows = []
        for T in sorted(r["temperature"], key=float):
            x = r["temperature"][T]
            mr = x["mean_rmse_pct"]
            soc_end = sum(c["soc_end_truth"] for c in x["cycles"]) / len(x["cycles"])
            rows.append([f"{float(T):g} °C", f(mr["model_25C"]), f(mr["model_T"]), f(mr["model_T_wrong_start"]),
                         f(mr["cc"]), f"{100*soc_end:.0f}%",
                         f(x["fit"].get("hppc_rmse_mv"), 1) if x["fit"].get("hppc_rmse_mv") is not None else "–"])
        parts.append("**Temperature** (mean SoC RMSE over UDDS/US06/HWFET, %)\n\n" + table(
            ["Temp", "EKF, 25 °C model", "EKF, per-temperature model", "…start 30% low", "Coulomb counting",
             "Truth SoC at cutoff", "HPPC fit [mV]"], rows))
    a = r.get("aging")
    if a:
        rows = [[c["kind"] + short(c["file"]), f(c["ekf_matched"]["rmse_pct"]), f(c["ekf_fresh_capacity"]["rmse_pct"]),
                 f(c["cc_fresh_capacity"]["rmse_pct"])] for c in a["cycles"]]
        parts.append(f"**Aging** — capacity {a['fresh_capacity_ah']:.3f} → {a['aged_capacity_ah']:.3f} Ah "
                     f"({a['fade_pct']:.1f}% fade; {a['source']}). Emulated on the 25 °C cycles: the estimators keep the "
                     f"fresh capacity.\n\n" + table(["Cycle", "EKF, right capacity", "EKF, fresh capacity",
                                                     "Coulomb counting, fresh capacity"], rows))
    b = r.get("current_bias")
    if b:
        keys = list(next(iter(b.values())).keys())
        rows = []
        for off, d in b.items():
            rows.append([off] + [f"{f(d[k]['ekf']['rmse_pct'])} ({f(d[k]['cc']['rmse_pct'])})" for k in keys])
        parts.append("**Current-sensor bias** — SoC RMSE %, EKF (coulomb counting)\n\n" +
                     table(["Offset"] + [k.split(":")[0] + short(k) for k in keys], rows))
    return "\n\n".join(parts) if parts else missing("python scripts/robustness.py")


PLOTS = [
    ("stage1_drive_cycles_25C.png", "Drive cycles at 25 °C: voltage, current, reference SoC"),
    ("stage2_ocv_25C.png", "OCV curve and slope from the C/20 test"),
    ("stage2_hppc_fit_25C.png", "2-RC fit on the HPPC test"),
    ("stage2_validation_25C.png", "Model voltage on held-out drive cycles"),
    ("stage4_US06_*.png", "EKF on US06: correct start, 30% low start, coulomb counting"),
    ("stage4_UDDS_*.png", "EKF on UDDS"),
    ("stage5_US06_*.png", "Fixed-point C vs float Python"),
    ("stage6_temperature.png", "SoC RMSE vs temperature"),
    ("stage6_bias.png", "SoC RMSE vs current-sensor bias"),
]


def block_plots():
    out = []
    for pat, cap in PLOTS:
        hits = sorted((ROOT / "plots").glob(pat))
        if hits:
            rel = hits[0].relative_to(ROOT).as_posix()
            out.append(f"**{cap}**\n\n![{cap}]({rel.replace(' ', '%20')})")
    return "\n\n".join(out) if out else "_No plots yet: run the stage scripts._"


BLOCKS = {"headline": block_headline, "data": block_data, "model": block_model, "baselines": block_baselines,
          "ekf": block_ekf, "cport": block_cport, "robust": block_robust, "plots": block_plots}


def render(text: str) -> str:
    for name, fn in BLOCKS.items():
        pat = re.compile(rf"(<!-- BEGIN:{name} -->)(.*?)(<!-- END:{name} -->)", re.S)
        if not pat.search(text):
            print(f"warning: block '{name}' not found in README", file=sys.stderr)
            continue
        body = fn()
        text = pat.sub(lambda m: f"{m.group(1)}\n{body}\n{m.group(3)}", text)
    return text


def main() -> None:
    readme = ROOT / "README.md"
    readme.write_text(render(readme.read_text()))
    filled = [n for n, fn in BLOCKS.items() if not fn().startswith("_Not run yet")]
    print(f"README.md updated; blocks with results: {', '.join(filled) or 'none'}")


if __name__ == "__main__":
    main()

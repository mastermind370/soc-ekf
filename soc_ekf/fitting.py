"""HPPC -> 2-RC parameter fit with logging-gap handling (shared by stage 2 and stage 6)."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .data import NOMINAL_CAPACITY_AH, classify, find_files, load
from .model import (
    OCVTable, capacity_and_ocv, discharge_window_end, fill_gaps, fit_ecm, initial_soc, low_rate_segments,
    ocv_gap_guess, params_at_bounds, prepare, soc_zoh,
)

KNOTS = (0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)
ONE_C = NOMINAL_CAPACITY_AH  # 1C step between HPPC pulse groups [A]


def pick(root, kind: str, temp: float, aged: bool = False) -> list[Path]:
    """Files of one kind at one nominal temperature (constant-temperature tests).
    aged=False excludes the end-of-tests folder; aged=True returns only it."""
    out = []
    for p in find_files(root):
        k, T, cond = classify(p)
        is_end = "end" in str(p).lower()
        if k == kind and cond == "const" and is_end == aged and (aged or T == temp):
            out.append(p)
    return out


def available_temps(root) -> list[float]:
    return sorted({classify(p)[1] for p in find_files(root) if classify(p)[1] is not None and classify(p)[2] == "const"})


def c20_capacity_ocv(paths, branch: str = "mean"):
    segs, src = {}, {}
    for p in paths:
        df = load(p).df
        s = low_rate_segments(df["t_s"].to_numpy(), df["i"].to_numpy(), df["v"].to_numpy())
        for k, seg in s.items():
            if k not in segs or seg["ah"] > segs[k]["ah"]:
                segs[k], src[k] = seg, Path(p).name
    if "dis" not in segs:
        return None
    Q, ocv, info = capacity_and_ocv(segs, branch)
    info["sources"] = src
    return Q, ocv, info


def choose_hppc(paths):
    """Prefer the pulse file itself (not the separate 'dis5_10p' step-discharge files), then the longest."""
    recs = [load(p) for p in paths]
    pulse = [r for r in recs if "dis5_10p" not in r.path.name.lower()
             and any(w in r.path.name.lower() for w in ("pulse", "hppc"))]
    pool = pulse or recs
    return max(pool, key=lambda r: len(r.df)) if pool else None


def fit_hppc(rec, ocv: OCVTable, Q: float, log=print):
    """Fit constant and SoC-dependent 2-RC parameters on one HPPC record. Returns dict with fits + arrays."""
    h = prepare(rec.df, rec.raw_sign)
    dt, t, i, v, gaps, valid = h["dt"], h["t"], h["i"], h["v"], h["gaps"], h["valid"]
    soc0 = initial_soc(v[0], i[0], ocv)
    dq_tst = np.array(h["dq_tester"])
    log(f"[HPPC] {rec.path.name}: {t[-1]/3600:.2f} h on a {dt:g} s grid, SoC0 = {soc0:.3f}")
    log(f"[HPPC] logged current removes {np.sum(i) * dt / 3600:.3f} Ah; {len(gaps)} logging gaps "
        f"({sum(h['gap_s'])/3600:.2f} h) where the tester Ah counter moved {dq_tst.sum():.3f} Ah")
    free = None
    if dq_tst.sum() > 0.05 * Q or not gaps:
        gap_mode = "tester Ah" if gaps else "no gaps"
        i = fill_gaps(i, gaps, dq_tst, dt, ONE_C)
    else:
        gap_mode = "fitted"
        free = gaps
    soc = soc_zoh(i, dt, Q, soc0)
    end = discharge_window_end(i, soc) if free is None else len(t)
    t, i, v, soc, valid = t[:end], i[:end], v[:end], soc[:end], valid[:end]
    log(f"[HPPC] gap charge: {gap_mode}; fitting {valid.mean()*100:.0f}% of samples")

    fits, gap_dq, warnings = {}, {}, {}
    dq0 = ocv_gap_guess(v, gaps, ocv, Q) if free else None
    for name, knots in (("constant", (0.0, 1.0)), ("soc_dependent", KNOTS)):
        t0 = time.perf_counter()
        init = fits["constant"][0] if name == "soc_dependent" else None
        p, rmse, dq = fit_ecm(dt, i, v, soc, ocv, Q, knots, init=init, valid=valid, free_gaps=free,
                              gap_dq0=dq0, gap_rate_a=ONE_C)
        if dq is not None:
            dq0 = dq
        fits[name], gap_dq[name] = (p, rmse), dq
        log(f"[HPPC] {name:13s} fit RMSE = {rmse:6.2f} mV   tau1 = {p.tau1:6.2f} s  tau2 = {p.tau2:7.1f} s"
            f"   ({time.perf_counter()-t0:.1f} s)")
        warnings[name] = params_at_bounds(p)
        if warnings[name]:
            log(f"        WARNING: at bounds -> {', '.join(warnings[name])}")
    if gap_dq["soc_dependent"] is not None:
        dq = np.array(gap_dq["soc_dependent"])
        i = fill_gaps(i, gaps, dq, dt, ONE_C)
        soc = soc_zoh(i, dt, Q, soc0)
        log("        fitted gap charge [Ah]: " + " ".join(f"{x:.3f}" for x in dq) + f"  (sum {dq.sum():.3f})")
    log(f"[HPPC] SoC range covered: {soc.max():.3f} -> {soc.min():.3f}")
    return {"fits": fits, "gap_dq": gap_dq, "gap_mode": gap_mode, "dq_tester": dq_tst, "warnings": warnings,
            "dt": dt, "t": t, "i": i, "v": v, "soc": soc, "valid": valid, "n_gaps": len(gaps)}

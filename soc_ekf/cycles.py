"""Drive-cycle loading with ground-truth SoC, shared by the baselines, the EKF and robustness stages."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import classify, find_files, load
from .model import fill_gaps, initial_soc, prepare, soc_zoh, OCVTable

DRIVE_KINDS = ("UDDS", "US06", "HWFET", "LA92")


@dataclass
class Cycle:
    kind: str
    name: str
    temp_c: float | None
    dt: float
    t: np.ndarray
    i: np.ndarray        # measured current, + = discharge (gaps filled from the tester Ah counter)
    v: np.ndarray        # measured terminal voltage
    soc_ref: np.ndarray  # ground truth: lab-grade coulomb counting from a known full start
    valid: np.ndarray    # False inside logging gaps (+2 min)
    temp: np.ndarray     # battery temperature [degC] (NaN if missing)


def load_cycles(root, temp_c: float, ocv: OCVTable, capacity_ah: float, kinds=DRIVE_KINDS,
                condition: str = "const", dt: float = 0.1) -> list[Cycle]:
    out = []
    for p in find_files(root):
        k, T, cond = classify(p)
        if k not in kinds or T != temp_c or cond != condition:
            continue
        rec = load(p)
        d = prepare(rec.df, rec.raw_sign, dt=dt)
        i = fill_gaps(d["i"], d["gaps"], d["dq_tester"], dt)
        s0 = initial_soc(d["v"][0], i[0], ocv)
        from .data import resample
        temp = resample(rec.df[["t_s", "temp_c"]], dt)["temp_c"].to_numpy()
        out.append(Cycle(k, Path(p).name, T, dt, d["t"], i, d["v"], soc_zoh(i, dt, capacity_ah, s0), d["valid"], temp))
    order = {k: n for n, k in enumerate(kinds)}
    return sorted(out, key=lambda c: (order[c.kind], c.name))

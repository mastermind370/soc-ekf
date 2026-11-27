"""Baseline SoC estimators the EKF has to beat."""
from __future__ import annotations

import numpy as np

from .model import ECMParams, OCVTable, soc_zoh


def coulomb_count(i: np.ndarray, dt: float, capacity_ah: float, soc0: float) -> np.ndarray:
    """Open-loop current integration. Never corrects itself: any start / capacity / sensor error stays."""
    return soc_zoh(i, dt, capacity_ah, soc0)


def ocv_lookup(v: np.ndarray, ocv: OCVTable) -> np.ndarray:
    """SoC = OCV^-1(terminal voltage). Ignores that the voltage sags under load."""
    return np.clip(ocv.inverse(v), 0.0, 1.0)


def ocv_lookup_ir(v: np.ndarray, i: np.ndarray, ocv: OCVTable, p: ECMParams) -> np.ndarray:
    """SoC = OCV^-1(v + R0*i): removes the instantaneous IR drop, not the slow RC polarisation."""
    s = ocv_lookup(v, ocv)
    s = ocv_lookup(v + p.R("r0", s) * i, ocv)  # one refinement with SoC-dependent R0
    return np.clip(ocv.inverse(v + p.R("r0", s) * i), 0.0, 1.0)


def soc_errors(est: np.ndarray, ref: np.ndarray, valid: np.ndarray | None = None) -> dict:
    """Errors in SoC percentage points."""
    m = np.ones(len(ref), bool) if valid is None else valid
    e = 100.0 * (est - ref)[m]
    return {"rmse_pct": float(np.sqrt(np.mean(e ** 2))), "max_abs_pct": float(np.max(np.abs(e))),
            "final_pct": float(e[-1]), "mean_pct": float(np.mean(e))}

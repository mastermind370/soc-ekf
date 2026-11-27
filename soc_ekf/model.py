"""2-RC equivalent-circuit model (Thevenin + 2 RC pairs) and its parameter fit.

    v = OCV(soc) - R0(soc)*i - v1 - v2
    dv_k/dt = -v_k/tau_k + R_k(soc)*i/tau_k      (k = 1, 2)
    i > 0 = discharge.  R0, R1, R2 are piecewise-linear in SoC; tau1, tau2 are constants.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import LSQUnivariateSpline
from scipy.optimize import least_squares
from scipy.signal import lfilter


# ============================================================== OCV table
@dataclass
class OCVTable:
    soc: np.ndarray  # strictly increasing, covers [0, 1]
    ocv: np.ndarray  # non-decreasing

    def __post_init__(self):
        self.soc = np.asarray(self.soc, float)
        self.ocv = np.maximum.accumulate(np.asarray(self.ocv, float))
        self._slope = np.gradient(self.ocv, self.soc)

    def __call__(self, s):
        """OCV with linear extrapolation outside the table (keeps the EKF Jacobian finite)."""
        s = np.asarray(s, float)
        v = np.interp(s, self.soc, self.ocv)
        lo, hi = s < self.soc[0], s > self.soc[-1]
        v = np.where(lo, self.ocv[0] + self._slope[0] * (s - self.soc[0]), v)
        v = np.where(hi, self.ocv[-1] + self._slope[-1] * (s - self.soc[-1]), v)
        return v

    def deriv(self, s):
        return np.interp(np.clip(s, self.soc[0], self.soc[-1]), self.soc, self._slope)

    def inverse(self, v):
        """SoC from a rested voltage (table is monotonic, so interp on (ocv, soc))."""
        o, idx = np.unique(self.ocv, return_index=True)
        return np.interp(v, o, self.soc[idx])

    def to_dict(self):
        return {"soc": self.soc.tolist(), "ocv": self.ocv.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(np.array(d["soc"]), np.array(d["ocv"]))


def fit_ocv_spline(soc: np.ndarray, v: np.ndarray, knot_step: float = 0.05, grid_step: float = 0.005):
    """Bin-average (soc, v) then fit a cubic least-squares spline. Returns (OCVTable, rmse_mV)."""
    edges = np.arange(0.0, 1.0 + grid_step, grid_step)
    centers, means = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (soc >= a) & (soc < b)
        if m.any():
            centers.append(0.5 * (a + b))
            means.append(v[m].mean())
    x, y = np.array(centers), np.array(means)
    knots = np.arange(x[0] + knot_step, x[-1] - knot_step / 2, knot_step)
    knots = np.concatenate([[x[0] + 0.01, x[0] + 0.025], knots[knots > x[0] + 0.03]])  # denser near empty
    spl = LSQUnivariateSpline(x, y, knots, k=3)
    rmse_mv = float(np.sqrt(np.mean((spl(x) - y) ** 2)) * 1e3)
    grid = np.round(np.arange(0.0, 1.0 + 1e-9, grid_step), 6)
    return OCVTable(grid, spl(np.clip(grid, x[0], x[-1]))), rmse_mv


# ============================================================== C/20 -> capacity + OCV
def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, stop) index pairs of contiguous True runs."""
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def cum_ah(t: np.ndarray, i: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(0.5 * (i[1:] + i[:-1]) * np.diff(t))]) / 3600.0


def low_rate_segments(t, i, v, max_abs_a: float = 0.5, min_ah: float = 1.0):
    """Longest low-current discharge and charge segments (C/20 = ~0.145 A for this cell)."""
    out = {}
    for name, mask in (("dis", i > 0.02), ("chg", i < -0.02)):
        best = None
        for a, b in runs(mask):
            if b - a < 10 or np.median(np.abs(i[a:b])) > max_abs_a:
                continue
            q = abs(cum_ah(t[a:b], i[a:b])[-1])
            if q >= min_ah and (best is None or q > best[2]):
                best = (a, b, q)
        if best:
            a, b, q = best
            out[name] = {"t": t[a:b], "i": i[a:b], "v": v[a:b], "ah": q}
    return out


def capacity_and_ocv(segments: dict):
    """Capacity = C/20 discharge Ah. OCV = mean of discharge and charge curves (cancels IR + most hysteresis)."""
    dis = segments["dis"]
    Q = dis["ah"]
    soc_d = 1.0 - cum_ah(dis["t"], dis["i"]) / Q
    soc_all, v_all = [soc_d], [dis["v"]]
    method = "discharge only"
    if "chg" in segments:
        c = segments["chg"]
        q = -cum_ah(c["t"], c["i"])
        soc_c = q / q[-1]
        grid = np.linspace(0, 1, 2001)
        vd = np.interp(grid, soc_d[::-1], dis["v"][::-1])
        vc = np.interp(grid, soc_c, c["v"])
        soc_all, v_all = [grid], [0.5 * (vd + vc)]
        method = "mean of C/20 discharge and charge"
    ocv, rmse = fit_ocv_spline(np.concatenate(soc_all), np.concatenate(v_all))
    return Q, ocv, {"method": method, "ocv_spline_rmse_mv": rmse}


# ============================================================== ECM
@dataclass
class ECMParams:
    knots: list  # SoC knots for R0, R1, R2
    r0: list
    r1: list
    r2: list
    tau1: float
    tau2: float
    capacity_ah: float

    def R(self, which: str, soc):
        return np.interp(np.clip(soc, 0, 1), self.knots, getattr(self, which))

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


def soc_coulomb(t, i, capacity_ah, soc0):
    return soc0 - cum_ah(t, i) / capacity_ah


def simulate(dt: float, i: np.ndarray, soc: np.ndarray, p: ECMParams, ocv: OCVTable):
    """Exact ZOH discretisation on a uniform grid. Returns (v, v1, v2)."""
    out = []
    for r, tau in (("r1", p.tau1), ("r2", p.tau2)):
        a = np.exp(-dt / tau)
        out.append(lfilter([0.0, 1.0 - a], [1.0, -a], p.R(r, soc) * i))
    v1, v2 = out
    v = ocv(soc) - p.R("r0", soc) * i - v1 - v2
    return v, v1, v2


def _unpack(x, nk, knots, Q):
    r0, r1, r2 = np.exp(x[:nk]), np.exp(x[nk:2 * nk]), np.exp(x[2 * nk:3 * nk])
    return ECMParams(list(knots), r0.tolist(), r1.tolist(), r2.tolist(), float(np.exp(x[-2])), float(np.exp(x[-1])), Q)


def fit_ecm(dt, i, v, soc, ocv: OCVTable, capacity_ah: float, knots=(0.0, 1.0), max_points: int = 200_000,
            init: "ECMParams | None" = None, soc_window=(0.02, 1.0)):
    """Least-squares fit of R0/R1/R2 at SoC knots and tau1/tau2 to a measured voltage trace.

    Residuals are taken only where soc_window[0] <= SoC <= soc_window[1] (the model is not valid past
    the cutoff knee). `init` warm-starts from another fit (e.g. the constant-parameter one).
    """
    knots = np.asarray(knots, float)
    nk = len(knots)
    if init is None:
        x0 = np.concatenate([np.log([0.025] * nk), np.log([0.010] * nk), np.log([0.015] * nk), np.log([10.0, 300.0])])
    else:
        x0 = np.concatenate([np.log(init.R(k, knots)) for k in ("r0", "r1", "r2")] + [np.log([init.tau1, init.tau2])])
    lb = np.concatenate([np.log([1e-4] * 3 * nk), np.log([0.5, 40.0])])
    ub = np.concatenate([np.log([0.5] * 3 * nk), np.log([40.0, 5000.0])])
    x0 = np.clip(x0, lb + 1e-6, ub - 1e-6)
    ok = np.flatnonzero((soc >= soc_window[0]) & (soc <= soc_window[1]))
    idx = ok[:: max(1, len(ok) // max_points)]

    def resid(x):
        return (simulate(dt, i, soc, _unpack(x, nk, knots, capacity_ah), ocv)[0] - v)[idx]

    sol = least_squares(resid, x0, bounds=(lb, ub), x_scale=1.0, diff_step=1e-4, max_nfev=400)
    p = _unpack(sol.x, nk, knots, capacity_ah)
    vm = simulate(dt, i, soc, p, ocv)[0]
    return p, float(np.sqrt(np.mean((vm - v)[idx] ** 2)) * 1e3)


def voltage_errors_mv(v_model, v_meas, soc=None, soc_min=None):
    e = (v_model - v_meas) * 1e3
    if soc is not None and soc_min is not None:
        e = e[soc >= soc_min]
    return {"rmse_mv": float(np.sqrt(np.mean(e ** 2))), "max_abs_mv": float(np.max(np.abs(e)))}


# ============================================================== persistence
def save_params(path, ocv: OCVTable, p: ECMParams, extra: dict | None = None):
    d = {"capacity_ah": p.capacity_ah, "ocv": ocv.to_dict(), "ecm": p.to_dict(), **(extra or {})}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(d, indent=2))


def load_params(path):
    d = json.loads(Path(path).read_text())
    return OCVTable.from_dict(d["ocv"]), ECMParams.from_dict(d["ecm"])


def initial_soc(v0: float, i0: float, ocv: OCVTable, full_v_soc: float = 0.97) -> float:
    """Every test starts after a full charge, so SoC0 = 1 unless the starting rest voltage says otherwise."""
    if abs(i0) < 0.05 and v0 < float(ocv(full_v_soc)):
        return float(np.clip(ocv.inverse(v0), 0.0, 1.0))
    return 1.0

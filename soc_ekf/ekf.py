"""Extended Kalman Filter for SoC with a 2-RC model.

State x = [soc, v1, v2]
  predict:  soc+ = soc - I*dt/(3600*Q)
            vk+  = ak*vk + Rk(soc)*(1-ak)*I          ak = exp(-dt/tau_k)
  measure:  V = OCV(soc) - R0(soc)*I_now - v1 - v2
  H = [dOCV/dsoc - dR0/dsoc*I_now, -1, -1]

Written with scalars and uniform lookup tables (no numpy inside the loop) so the C port in stage 5
is a line-by-line translation. Covariance P is symmetric; only 6 entries are kept.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .model import ECMParams, OCVTable


@dataclass
class EKFTuning:
    sigma_v: float = 0.01          # measurement noise std [V] (mostly model error)
    sigma_soc_proc: float = 1e-5   # SoC random walk [1/sqrt(s)] (current-sensor / capacity error)
    sigma_rc_proc: float = 1e-3    # RC voltage random walk [V/sqrt(s)]
    sigma_soc0: float = 0.25       # initial SoC std
    sigma_rc0: float = 0.01        # initial RC voltage std [V]

    def to_dict(self):
        return asdict(self)


class Tables:
    """Uniform-grid tables of OCV, dOCV/dsoc, R0, R1, R2 and dR0/dsoc over soc in [0, 1]."""

    def __init__(self, ocv: OCVTable, p: ECMParams, n: int = 201):
        self.n = n
        self.step = 1.0 / (n - 1)
        g = np.linspace(0.0, 1.0, n)
        self.ocv = ocv(g).tolist()
        self.docv = ocv.deriv(g).tolist()
        self.r0 = p.R("r0", g).tolist()
        self.r1 = p.R("r1", g).tolist()
        self.r2 = p.R("r2", g).tolist()
        self.dr0 = np.gradient(np.array(self.r0), g).tolist()
        self.tau1, self.tau2, self.Q = p.tau1, p.tau2, p.capacity_ah

    def lookup(self, s: float):
        """Return (ocv, docv, r0, dr0, r1, r2) at soc s; OCV extrapolated linearly outside [0, 1]."""
        x = s / self.step
        k = int(x) if x >= 0 else 0
        if k > self.n - 2:
            k = self.n - 2
        f = x - k
        sc = 0.0 if f < 0 else (1.0 if f > 1 else f)  # clamped fraction for R and slopes
        ocv = self.ocv[k] + (self.ocv[k + 1] - self.ocv[k]) * f  # unclamped -> linear extrapolation
        lin = lambda tb: tb[k] + (tb[k + 1] - tb[k]) * sc  # noqa: E731
        return ocv, lin(self.docv), lin(self.r0), lin(self.dr0), lin(self.r1), lin(self.r2)


def run_ekf(tb: Tables, dt: float, i_avg, i_now, v, soc0: float, tune: EKFTuning):
    """Run the EKF over a record. i_avg[k] = mean current over (t[k-1], t[k]]; i_now[k], v[k] at t[k].

    Returns (soc_est, sigma_soc) arrays.
    """
    n = len(v)
    a1, a2 = math.exp(-dt / tb.tau1), math.exp(-dt / tb.tau2)
    q0 = tune.sigma_soc_proc ** 2 * dt
    qr = tune.sigma_rc_proc ** 2 * dt
    r = tune.sigma_v ** 2
    kq = dt / (3600.0 * tb.Q)
    soc, v1, v2 = soc0, 0.0, 0.0
    p00, p01, p02 = tune.sigma_soc0 ** 2, 0.0, 0.0
    p11, p12, p22 = tune.sigma_rc0 ** 2, 0.0, tune.sigma_rc0 ** 2
    out = np.empty(n)
    sig = np.empty(n)
    for k in range(n):
        if k > 0:  # ---------------- predict with the current over the last interval
            I = i_avg[k]
            _, _, _, _, r1, r2 = tb.lookup(soc)
            soc -= kq * I
            v1 = a1 * v1 + r1 * (1.0 - a1) * I
            v2 = a2 * v2 + r2 * (1.0 - a2) * I
            p00 += q0
            p01 *= a1
            p02 *= a2
            p11 = a1 * a1 * p11 + qr
            p12 = a1 * a2 * p12
            p22 = a2 * a2 * p22 + qr
        # -------------------------------- update with the voltage at t[k]
        In = i_now[k]
        ocv, docv, r0, dr0, _, _ = tb.lookup(soc)
        h0 = docv - dr0 * In
        e = v[k] - (ocv - r0 * In - v1 - v2)
        ph0 = p00 * h0 - p01 - p02
        ph1 = p01 * h0 - p11 - p12
        ph2 = p02 * h0 - p12 - p22
        s = h0 * ph0 - ph1 - ph2 + r
        k0, k1, k2 = ph0 / s, ph1 / s, ph2 / s
        soc += k0 * e
        v1 += k1 * e
        v2 += k2 * e
        p00 -= k0 * ph0
        p01 -= k0 * ph1
        p02 -= k0 * ph2
        p11 -= k1 * ph1
        p12 -= k1 * ph2
        p22 -= k2 * ph2
        out[k] = soc
        sig[k] = math.sqrt(p00) if p00 > 0 else 0.0
    return out, sig


def decimate(t, i, v, soc_ref, valid, dt_in: float, dt_out: float):
    """Down-sample a dt_in record to dt_out: interval-mean current (coulomb-exact), point samples otherwise."""
    m = int(round(dt_out / dt_in))
    if m <= 1:
        return t, i, i, v, soc_ref, valid
    n = (len(t) - 1) // m
    idx = np.arange(n + 1) * m
    c = np.concatenate([[0.0], np.cumsum(i)])  # ZOH: i[j] held over [j, j+1)
    i_avg = np.empty(n + 1)
    i_avg[0] = i[0]
    i_avg[1:] = (c[idx[1:]] - c[idx[:-1]]) / m
    ok = np.array([valid[a:b + 1].all() for a, b in zip(np.r_[0, idx[:-1]], idx)])
    return t[idx], i_avg, i[idx], v[idx], soc_ref[idx], ok


def convergence_time(t, err_pct, valid, band_pct: float):
    """First time after which |error| stays within band_pct for the rest of the record (None if never)."""
    bad = (np.abs(err_pct) > band_pct) & valid
    if not bad.any():
        return 0.0
    last = int(np.flatnonzero(bad)[-1])
    return None if last >= len(t) - 1 else float(t[last + 1] - t[0])

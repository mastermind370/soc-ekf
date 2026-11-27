import numpy as np
import pytest

from soc_ekf.baselines import soc_errors
from soc_ekf.ekf import EKFTuning, Tables, convergence_time, decimate, run_ekf
from soc_ekf.model import ECMParams, OCVTable, simulate, soc_zoh


@pytest.fixture
def cell():
    g = np.linspace(0, 1, 201)
    ocv = OCVTable(g, 3.0 + 1.1 * g + 0.15 * np.tanh(20 * (g - 0.05)))
    p = ECMParams([0, 0.5, 1], [0.03, 0.022, 0.02], [0.012] * 3, [0.015] * 3, 9.0, 300.0, 2.9)
    dt = 0.1
    t = np.arange(0, 2.5 * 3600, dt)
    rng = np.random.default_rng(3)
    i = 1.0 + 2.5 * np.sin(t / 31) * np.sin(t / 9.7) + rng.normal(0, 0.3, len(t))
    soc = soc_zoh(i, dt, 2.9, 1.0)
    v = simulate(dt, i, soc, p, ocv)[0] + rng.normal(0, 2e-3, len(t))
    return ocv, p, dt, t, i, v, soc


def test_tables_match_model(cell):
    ocv, p, *_ = cell
    tb = Tables(ocv, p)
    for s in (0.0, 0.123, 0.5, 0.999):
        o, d, r0, _, r1, _ = tb.lookup(s)
        assert o == pytest.approx(float(ocv(s)), abs=2e-4)
        assert r0 == pytest.approx(float(p.R("r0", s)), rel=1e-3)
    assert tb.lookup(1.05)[0] > tb.lookup(1.0)[0]  # extrapolates


def test_decimate_preserves_charge(cell):
    _, _, dt, t, i, v, soc = cell
    td, ia, inow, vd, sd, ok = decimate(t, i, v, soc, np.ones(len(t), bool), dt, 1.0)
    assert np.allclose(np.diff(td), 1.0)
    q_full = np.sum(i[: (len(td) - 1) * 10]) * dt
    assert np.sum(ia[1:]) * 1.0 == pytest.approx(q_full, rel=1e-9)
    assert soc_errors(soc_zoh(ia[1:], 1.0, 2.9, 1.0), sd[:-1])["max_abs_pct"] < 1e-6


def test_ekf_true_start_and_converges_from_wrong_start(cell):
    ocv, p, dt, t, i, v, soc = cell
    td, ia, inow, vd, sd, ok = decimate(t, i, v, soc, np.ones(len(t), bool), dt, 1.0)
    tb = Tables(ocv, p)
    tune = EKFTuning(sigma_v=0.005, sigma_soc_proc=1e-5, sigma_rc_proc=1e-3)
    est, sig = run_ekf(tb, 1.0, ia, inow, vd, 1.0, tune)
    assert soc_errors(est, sd)["rmse_pct"] < 1.0
    est_w, _ = run_ekf(tb, 1.0, ia, inow, vd, 0.7, tune)
    err = 100 * (est_w - sd)
    tc = convergence_time(td, err, np.ones(len(td), bool), 2.0)
    assert tc is not None and tc < 1800
    assert abs(err[-1]) < 1.5
    assert np.all(sig > 0) and sig[-1] < sig[0]


def test_ekf_beats_cc_under_gain_error(cell):
    ocv, p, dt, t, i, v, soc = cell
    td, ia, inow, vd, sd, ok = decimate(t, i, v, soc, np.ones(len(t), bool), dt, 1.0)
    tb = Tables(ocv, p)
    est, _ = run_ekf(tb, 1.0, 1.05 * ia, 1.05 * inow, vd, 1.0, EKFTuning(sigma_v=0.005))
    cc = soc_zoh(1.05 * ia, 1.0, 2.9, 1.0)
    assert soc_errors(est, sd)["rmse_pct"] < soc_errors(cc, sd)["rmse_pct"]


def test_convergence_time():
    t = np.arange(10.0)
    e = np.array([30, 20, 10, 4, 1, 1, 3, 1, 1, 1.0])
    ok = np.ones(10, bool)
    assert convergence_time(t, e, ok, 5.0) == 3.0
    assert convergence_time(t, e, ok, 2.0) == 7.0
    assert convergence_time(t, np.full(10, 9.0), ok, 2.0) is None

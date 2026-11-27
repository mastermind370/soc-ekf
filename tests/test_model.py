import numpy as np
import pytest

from soc_ekf.model import (
    ECMParams, OCVTable, capacity_and_ocv, cum_ah, fit_ecm, initial_soc, load_params,
    low_rate_segments, runs, save_params, simulate, soc_coulomb,
)


def true_ocv(s):
    s = np.asarray(s, float)
    return 3.0 + 1.0 * s + 0.2 * np.tanh(15 * (s - 0.03)) + 0.05 * np.exp(-30 * (1 - s))


@pytest.fixture
def ocv():
    g = np.linspace(0, 1, 201)
    return OCVTable(g, true_ocv(g))


def test_ocv_table_interp_extrap_inverse(ocv):
    assert float(ocv(0.5)) == pytest.approx(float(true_ocv(0.5)), abs=1e-4)
    assert float(ocv(1.05)) > float(ocv(1.0))  # linear extrapolation, not clamped
    assert float(ocv(-0.05)) < float(ocv(0.0))
    for s in (0.1, 0.5, 0.9):
        assert float(ocv.inverse(ocv(s))) == pytest.approx(s, abs=1e-3)
    assert np.all(ocv.deriv(np.linspace(0, 1, 50)) >= 0)


def test_runs():
    assert runs(np.array([0, 1, 1, 0, 1, 0, 1, 1, 1], bool)) == [(1, 3), (4, 5), (6, 9)]


def test_c20_capacity_and_ocv():
    Q, I, dt = 2.8, 0.14, 10.0
    n = int(Q / I * 3600 / dt)
    t = np.arange(n) * dt
    soc_d = 1 - cum_ah(t, np.full(n, I)) / Q
    soc_c = cum_ah(t, np.full(n, I)) / Q
    R = 0.05  # IR + hysteresis shift, cancelled by averaging
    tt = np.concatenate([t, t[-1] + 3600 + t])
    ii = np.concatenate([np.full(n, I), np.full(n, -I)])
    vv = np.concatenate([true_ocv(soc_d) - R * I, true_ocv(soc_c) + R * I])
    segs = low_rate_segments(tt, ii, vv)
    assert set(segs) == {"dis", "chg"}
    Qf, tab, info = capacity_and_ocv(segs)
    assert Qf == pytest.approx(Q, rel=2e-3)
    g = np.linspace(0.05, 0.95, 50)
    assert np.max(np.abs(tab(g) - true_ocv(g))) < 3e-3
    assert info["method"].startswith("mean")


def _pulse_profile(dt=0.1, hours=3.0):
    t = np.arange(0, hours * 3600, dt)
    i = np.zeros_like(t)
    for k, start in enumerate(np.arange(60, t[-1] - 1200, 1200)):
        m = (t >= start) & (t < start + 10)
        i[m] = [1.5, 3.0, 6.0][k % 3]
        m = (t >= start + 40) & (t < start + 50)
        i[m] = -1.5
        m = (t >= start + 100) & (t < start + 460)
        i[m] = 2.9  # 1C step down in SoC
    return t, i


def test_simulate_steady_state_and_fit_recovers(ocv):
    true = ECMParams([0, 1], [0.02, 0.03], [0.012, 0.012], [0.02, 0.015], 8.0, 250.0, 2.9)
    dt = 0.1
    t, i = _pulse_profile(dt)
    soc = soc_coulomb(t, i, 2.9, 1.0)
    v, v1, v2 = simulate(dt, i, soc, true, ocv)
    # after a long constant current, RC voltage -> R*I
    ic = np.full(200_000, 2.0)
    _, a1, a2 = simulate(dt, ic, np.full_like(ic, 0.5), true, ocv)
    assert a1[-1] == pytest.approx(0.012 * 2.0, rel=1e-3) and a2[-1] == pytest.approx(true.R("r2", 0.5) * 2, rel=1e-2)
    noisy = v + np.random.default_rng(1).normal(0, 1e-3, len(v))
    p, rmse, _ = fit_ecm(dt, i, noisy, soc, ocv, 2.9, knots=(0.0, 1.0))
    assert rmse < 1.3  # ~ the 1 mV noise floor
    assert p.tau1 == pytest.approx(8.0, rel=0.15)
    assert p.R("r0", 0.7) == pytest.approx(true.R("r0", 0.7), rel=0.05)


def test_params_roundtrip(tmp_path, ocv):
    p = ECMParams([0, 0.5, 1], [0.02] * 3, [0.01] * 3, [0.01] * 3, 10.0, 300.0, 2.85)
    save_params(tmp_path / "p.json", ocv, p, {"temp_c": 25})
    o2, p2 = load_params(tmp_path / "p.json")
    assert p2 == p and np.allclose(o2.ocv, ocv.ocv)


def test_initial_soc(ocv):
    assert initial_soc(float(ocv(1.0)), 0.0, ocv) == 1.0
    assert initial_soc(float(ocv(0.5)), 0.0, ocv) == pytest.approx(0.5, abs=2e-3)
    assert initial_soc(float(ocv(0.5)), 2.0, ocv) == 1.0  # not at rest -> assume full


def test_discharge_window_end_cuts_recharge():
    i = np.concatenate([np.full(100, 2.0), np.zeros(20), np.full(50, -1.5)])
    t = np.arange(len(i)) * 1.0
    soc = soc_coulomb(t, i, 2.9, 1.0)
    from soc_ekf.model import discharge_window_end
    assert discharge_window_end(i, soc) == 120  # discharge + following rest, recharge removed


def test_params_at_bounds_flags():
    from soc_ekf.model import params_at_bounds
    good = ECMParams([0, 1], [0.02] * 2, [0.01] * 2, [0.01] * 2, 10.0, 300.0, 2.9)
    bad = ECMParams([0, 1], [0.02] * 2, [0.01] * 2, [0.5] * 2, 40.0, 5000.0, 2.9)
    assert params_at_bounds(good) == [] and len(params_at_bounds(bad)) == 4


def test_fit_with_unlogged_gaps_recovers_charge(ocv):
    """HPPC-style file where the SoC-step discharges are not logged: fit the missing charge."""
    from soc_ekf.model import fill_gaps, prepare, soc_zoh
    import pandas as pd
    true = ECMParams([0, 1], [0.022, 0.022], [0.012, 0.012], [0.018, 0.018], 8.0, 250.0, 2.9)
    dt = 1.0
    t, i = _pulse_profile(dt, hours=3.0)
    soc = soc_zoh(i, dt, 2.9, 1.0)
    v = simulate(dt, i, soc, true, ocv)[0]
    keep = ~((i > 2.8) & (i < 3.0))  # drop the 1C step discharges from the "log"
    df = pd.DataFrame({"t_s": t[keep], "v": v[keep], "i": i[keep], "ah_raw": np.nan,
                       "temp_c": np.nan, "chamber_c": np.nan})
    h = prepare(df, -1, dt=1.0)
    assert len(h["gaps"]) == 8 and not h["have_tester_ah"]
    soc_guess = soc_zoh(h["i"], dt, 2.9, 1.0)
    p, rmse, dq = fit_ecm(dt, h["i"], h["v"], soc_guess, ocv, 2.9, valid=h["valid"],
                          free_gaps=h["gaps"], gap_dq0=[0.1] * 8)
    assert rmse < 2.0
    assert np.allclose(dq, 2.9 * 360 / 3600, rtol=0.05)  # 360 s at 2.9 A each
    assert p.tau1 == pytest.approx(8.0, rel=0.2)


def test_choose_hppc_prefers_pulse_file(tmp_path):
    import scipy.io
    from soc_ekf.fitting import choose_hppc
    paths = []
    for name, n in (("x 10degC_dis5_10p.mat", 5000), ("x 10degC_5Pulse_HPPC.mat", 1000)):
        p = tmp_path / "10degC" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        scipy.io.savemat(p, {"meas": {"Time": np.arange(n) * 1.0, "Voltage": np.full(n, 3.7),
                                      "Current": np.zeros(n), "Ah": np.zeros(n)}})
        paths.append(p)
    assert "Pulse" in choose_hppc(paths).path.name  # even though the step-discharge file is longer

import numpy as np
import pytest

from soc_ekf.baselines import coulomb_count, ocv_lookup, ocv_lookup_ir, soc_errors
from soc_ekf.model import ECMParams, OCVTable, simulate, soc_zoh


@pytest.fixture
def setup():
    g = np.linspace(0, 1, 201)
    ocv = OCVTable(g, 3.0 + 1.1 * g + 0.1 * np.tanh(20 * (g - 0.05)))
    p = ECMParams([0, 1], [0.02, 0.02], [0.01, 0.01], [0.01, 0.01], 10.0, 300.0, 2.9)
    dt = 1.0
    t = np.arange(0, 3 * 3600, dt)
    i = 0.8 + 1.5 * np.sin(t / 50.0)
    soc = soc_zoh(i, dt, 2.9, 1.0)
    v = simulate(dt, i, soc, p, ocv)[0]
    return ocv, p, dt, i, v, soc


def test_cc_exact_and_wrong_start_offset_persists(setup):
    _, _, dt, i, _, soc = setup
    assert soc_errors(coulomb_count(i, dt, 2.9, 1.0), soc)["rmse_pct"] < 1e-9
    e = soc_errors(coulomb_count(i, dt, 2.9, 0.7), soc)
    assert e["rmse_pct"] == pytest.approx(30.0) and e["final_pct"] == pytest.approx(-30.0)


def test_cc_capacity_error_grows(setup):
    _, _, dt, i, _, soc = setup
    est = coulomb_count(i, dt, 3.1, 1.0)  # BMS thinks the cell is bigger -> SoC falls too slowly
    e = 100 * (est - soc)
    assert e[-1] > e[len(e) // 2] > 0


def test_ocv_lookup_biased_under_load_ir_helps(setup):
    ocv, p, _, i, v, soc = setup
    plain = soc_errors(ocv_lookup(v, ocv), soc)
    ir = soc_errors(ocv_lookup_ir(v, i, ocv, p), soc)
    assert plain["mean_pct"] < 0  # sag under discharge reads as low SoC
    assert ir["rmse_pct"] < plain["rmse_pct"]


def test_soc_errors_mask():
    ref = np.array([0.5, 0.5, 0.5])
    est = np.array([0.6, 0.5, 0.5])
    assert soc_errors(est, ref, np.array([False, True, True]))["max_abs_pct"] == 0.0

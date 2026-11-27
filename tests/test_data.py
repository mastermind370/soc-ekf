from pathlib import Path

import numpy as np
import pytest
import scipy.io

from soc_ekf.data import (
    classify,
    detect_sign,
    discharged_ah,
    find_files,
    load,
    reference_soc,
    resample,
    summarize,
)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def _fake_cycle(path: Path, discharge_negative: bool = True, n: int = 3000, dt: float = 0.1, r0: float = 0.03):
    """Synthetic drive cycle saved in the same layout as the Panasonic files (struct 'meas')."""
    rng = np.random.default_rng(0)
    t = np.arange(n) * dt
    i_dis = np.clip(2.0 + 3.0 * np.sin(t / 7.0) + rng.normal(0, 0.3, n), -2, 8)  # + = discharge
    q = np.concatenate([[0], np.cumsum(0.5 * (i_dis[1:] + i_dis[:-1]) * dt)]) / 3600
    v = 4.15 - 0.4 * q / 2.9 - r0 * i_dis
    s = -1.0 if discharge_negative else 1.0
    meas = {
        "Time": t, "Voltage": v, "Current": s * i_dis, "Ah": s * q,
        "Battery_Temp_degC": np.full(n, 25.0), "Chamber_Temp_degC": np.full(n, 25.0),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.savemat(path, {"meas": meas})
    return t, i_dis, q


@pytest.mark.parametrize(
    "path,kind,temp,cond",
    [
        ("25degC/Drive cycles/03-21-17_00.29 25degC_UDDS_Pan18650PF.mat", "UDDS", 25, "const"),
        ("25degC/Drive cycles/03-20-17_01.43 25degC_US06_Pan18650PF.mat", "US06", 25, "const"),
        ("-10degC/Drive Cycles/06-10-17_10.26 n10degC_HWFET_Pan18650PF.mat", "HWFET", -10, "const"),
        ("25degC/Drive cycles/03-18-17_02.17 25degC_Cycle_1_Pan18650PF.mat", "MIX", 25, "const"),
        ("25degC/Drive cycles/03-22-17_12.01 25degC_NN_Pan18650PF.mat", "NN", 25, "const"),
        ("25degC/5 pulse disch/03-11-17_08.47 25degC_5Pulse_Disch.mat", "HPPC", 25, "const"),
        ("25degC/C20 OCV and 1C capacity tests/03-09-17_11.37 C20 Dis.mat", "C20", 25, "const"),
        ("-20degC Trise/06-25-17_US06_Pan18650PF.mat", "US06", -20, "trise"),
        ("10degC/Drive cycles/x_10degC_LA92_Pan18650PF.mat", "LA92", 10, "const"),
        ("25degC/Drive cycles/03-19-17_03.25 25degC_HWFTa_Pan18650PF.mat", "HWFET", 25, "const"),
        ("25degC/Drive cycles/03-19-17_09.44 25degC_HWFTb_Pan18650PF.mat", "HWFET", 25, "const"),
    ],
)
def test_classify(path, kind, temp, cond):
    assert classify(path) == (kind, temp, cond)


def test_pan18650pf_is_not_nn():
    assert classify("25degC/x_25degC_UDDS_Pan18650PF.mat")[0] == "UDDS"


@pytest.mark.parametrize("neg", [True, False])
def test_load_roundtrip_and_sign(tmp_path, neg):
    p = tmp_path / "25degC" / "Drive cycles" / "x 25degC_UDDS_Pan18650PF.mat"
    t, i_dis, q = _fake_cycle(p, discharge_negative=neg)
    rec = load(p)
    assert rec.kind == "UDDS" and rec.temp_c_nominal == 25
    assert rec.raw_sign == (-1 if neg else 1)
    np.testing.assert_allclose(rec.df["i"], i_dis, atol=1e-12)  # always + = discharge
    np.testing.assert_allclose(discharged_ah(rec.df, True, rec.raw_sign), q, atol=1e-12)
    np.testing.assert_allclose(discharged_ah(rec.df, False), q, atol=1e-9)


def test_detect_sign_rest_only_falls_back():
    assert detect_sign(np.zeros(100), np.full(100, 3.6)) == -1


def test_reference_soc(tmp_path):
    p = tmp_path / "25degC" / "x 25degC_US06_Pan18650PF.mat"
    _, _, q = _fake_cycle(p)
    rec = load(p)
    soc = reference_soc(rec, capacity_ah=2.9, soc0=1.0)
    assert soc[0] == pytest.approx(1.0)
    assert soc[-1] == pytest.approx(1.0 - q[-1] / 2.9)
    assert np.all(np.diff(soc)[rec.df["i"].to_numpy()[1:] > 0.5] < 0)  # discharging lowers SoC


def test_clean_drops_duplicate_and_unsorted_time(tmp_path):
    p = tmp_path / "25degC" / "x 25degC_UDDS.mat"
    t = np.array([5.0, 5.1, 5.1, 5.3, 5.2])
    meas = {"Time": t, "Voltage": np.full(5, 3.7), "Current": np.zeros(5), "Ah": np.zeros(5)}
    p.parent.mkdir(parents=True)
    scipy.io.savemat(p, {"meas": meas})
    df = load(p).df
    assert df["t_s"].tolist() == pytest.approx([0.0, 0.1, 0.2, 0.3])
    assert df["temp_c"].isna().all()  # missing field -> NaN column


def test_resample_uniform(tmp_path):
    p = tmp_path / "25degC" / "x 25degC_UDDS.mat"
    _fake_cycle(p, n=500)
    df = load(p).df.iloc[::3].reset_index(drop=True)  # make it non-uniform-ish
    r = resample(df, dt=0.1)
    assert np.allclose(np.diff(r["t_s"]), 0.1)
    assert r["v"].iloc[0] == pytest.approx(df["v"].iloc[0])


def test_summarize_keys(tmp_path):
    p = tmp_path / "25degC" / "x 25degC_HWFET.mat"
    _fake_cycle(p)
    s = summarize(load(p))
    assert s["kind"] == "HWFET" and s["n_samples"] == 3000
    assert s["ah_integration_vs_tester_max_err"] < 1e-6


# ---------------------------------------------------------------- real data (skipped if absent)
real = pytest.mark.skipif(not find_files(RAW), reason="real dataset not in data/raw")


@real
def test_real_drive_cycles_present():
    kinds = {classify(p)[0] for p in find_files(RAW)}
    for k in ("UDDS", "US06", "HWFET", "HPPC", "C20"):
        assert k in kinds, f"missing {k} files in data/raw"


@real
def test_real_25c_cycles_physical():
    for p in find_files(RAW):
        k, T, cond = classify(p)
        if k in ("UDDS", "US06", "HWFET") and T == 25 and cond == "const":
            rec = load(p)
            df = rec.df
            assert 2.4 < df["v"].min() and df["v"].max() < 4.35, p.name
            assert summarize(rec)["net_discharged_ah"] > 0, f"{p.name}: drive cycle should net-discharge"
            assert np.all(np.diff(df["t_s"]) > 0)

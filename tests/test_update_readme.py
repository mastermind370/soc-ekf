import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _mod(tmp_root):
    spec = importlib.util.spec_from_file_location("update_readme", ROOT / "scripts" / "update_readme.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.ROOT = tmp_root
    return m


def test_missing_results_give_placeholders(tmp_path):
    (tmp_path / "results").mkdir()
    (tmp_path / "plots").mkdir()
    m = _mod(tmp_path)
    out = m.render((ROOT / "README.md").read_text())
    assert "Not run yet: `python scripts/run_ekf.py`" in out
    assert out.count("<!-- BEGIN:") == out.count("<!-- END:") == 8


def test_numbers_come_from_json_and_render_is_idempotent(tmp_path):
    (tmp_path / "results").mkdir()
    (tmp_path / "plots").mkdir()
    err = lambda r: {"rmse_pct": r, "max_abs_pct": 2 * r, "final_pct": 0.0, "mean_pct": 0.0}  # noqa: E731
    ekf = {"dt_s": 1.0, "tuning": {"sigma_v": 0.02, "sigma_soc_proc": 1e-5, "sigma_rc_proc": 1e-3},
           "tuning_set": ["x_LA92.mat"],
           "per_file": {"UDDS:x 25degC_UDDS.mat": {
               "kind": "UDDS", "true_start": err(0.123), "wrong_start": err(0.456),
               "converge_s_5pct": 3.0, "converge_s_2pct": 42.0,
               "robustness": {"gain_plus5pct": {"ekf": err(1.0), "cc": err(3.0)},
                              "gain_minus5pct": {"ekf": err(1.1), "cc": err(3.1)},
                              "white_noise": {"ekf": err(0.2)}}}}}
    (tmp_path / "results" / "ekf.json").write_text(json.dumps(ekf))
    m = _mod(tmp_path)
    once = m.render((ROOT / "README.md").read_text())
    assert "0.12%" in once and "0.46%" in once and "42 s" in once
    assert m.render(once) == once

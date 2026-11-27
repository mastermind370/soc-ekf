"""Loaders for the Panasonic 18650PF dataset (Kollmeyer, UW-Madison, Mendeley DOI 10.17632/wykht8y7tg.1).

Internal conventions used everywhere in this repo:
  t_s   time in seconds, starting at 0 for each file
  v     terminal voltage [V]
  i     current [A], POSITIVE = DISCHARGE
  temp_c battery surface temperature [degC] (NaN if missing)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io

CELL_NAME = "Panasonic NCR18650PF"
NOMINAL_CAPACITY_AH = 2.9  # datasheet value; the fitted capacity replaces this in stage 2

# Order matters: the first match wins.
_TYPE_PATTERNS: list[tuple[str, str]] = [
    ("C20", r"c20|ocv"),
    ("HPPC", r"hppc|pulse|dis5_10p"),
    ("UDDS", r"udds"),
    ("US06", r"us06"),
    ("HWFET", r"hwfe?t[ab]?"),
    ("LA92", r"la92"),
    ("NN", r"(^|[_\s])nn([_\s.]|$)"),
    ("MIX", r"cycle_?\d"),
    ("EIS", r"eis"),
    ("CHARGE", r"charge"),
]
DRIVE_CYCLES = ("UDDS", "US06", "HWFET", "LA92", "NN", "MIX")


@dataclass
class Record:
    path: Path
    kind: str
    temp_c_nominal: float | None
    condition: str  # "const" or "trise" (temperature drifting during the test)
    df: pd.DataFrame
    raw_sign: int  # i = raw_sign * Current_from_file
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------- classify
def classify(path: str | Path) -> tuple[str, float | None, str]:
    """Return (kind, nominal temperature, condition) from a file path."""
    p = str(path).replace("\\", "/")
    name = p.split("/")[-1].lower()
    full = p.lower()
    kind = "OTHER"
    for k, pat in _TYPE_PATTERNS:
        # match the file name first, then the folder path (C20 / HPPC live in named folders)
        if re.search(pat, name) or (k in ("C20", "HPPC") and re.search(pat, full)):
            kind = k
            break
    m = re.search(r"(-?\d+)\s*degc", full)
    temp = float(m.group(1)) if m else None
    condition = "trise" if "trise" in full else "const"
    return kind, temp, condition


def find_files(root: str | Path) -> list[Path]:
    return sorted(Path(root).rglob("*.mat"))


# --------------------------------------------------------------------- load
_FIELDS = {
    "t_s": ("Time",),
    "v": ("Voltage",),
    "i_raw": ("Current",),
    "ah_raw": ("Ah",),
    "temp_c": ("Battery_Temp_degC", "Temp", "Temperature"),
    "chamber_c": ("Chamber_Temp_degC",),
}


def _from_scipy(path: Path) -> dict[str, np.ndarray]:
    m = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    s = m.get("meas")
    if s is None:  # take the first struct-like variable
        s = next(v for k, v in m.items() if not k.startswith("__") and hasattr(v, "_fieldnames"))
    return {f: np.asarray(getattr(s, f)) for f in s._fieldnames}


def _from_h5py(path: Path) -> dict[str, np.ndarray]:
    import h5py  # MATLAB v7.3 files are HDF5

    with h5py.File(path, "r") as f:
        g = f["meas"] if "meas" in f else f[next(iter(f.keys()))]
        out = {}
        for k in g.keys():
            arr = np.asarray(g[k])
            if arr.dtype.kind in "fiu":
                out[k] = arr.ravel()
        return out


def read_mat(path: str | Path) -> pd.DataFrame:
    """Read one .mat file into a DataFrame with columns t_s, v, i_raw, ah_raw, temp_c, chamber_c."""
    path = Path(path)
    try:
        raw = _from_scipy(path)
    except (NotImplementedError, ValueError):
        raw = _from_h5py(path)
    cols = {}
    for col, names in _FIELDS.items():
        src = next((n for n in names if n in raw), None)
        if src is not None:
            cols[col] = np.asarray(raw[src], dtype=float).ravel()
    for req in ("t_s", "v", "i_raw"):
        if req not in cols:
            raise KeyError(f"{path.name}: missing field for {req}; found {sorted(raw)}")
    n = len(cols["t_s"])
    for col in _FIELDS:
        if col not in cols or len(cols[col]) != n:
            cols[col] = np.full(n, np.nan)
    return pd.DataFrame(cols)


def detect_sign(i_raw: np.ndarray, v: np.ndarray) -> int:
    """Return +1 if positive file current = discharge, else -1.

    Physics: on a current step, dV has the opposite sign to the discharge current (V = OCV - I*R).
    So corr(dI_raw, dV) < 0  ->  positive raw current is discharge.
    """
    di, dv = np.diff(i_raw), np.diff(v)
    mask = np.abs(di) > 0.05  # only real current steps
    if mask.sum() < 3:
        return -1  # rest-only file: fall back to the dataset convention (negative = discharge)
    c = np.corrcoef(di[mask], dv[mask])[0, 1]
    return 1 if c < 0 else -1


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["t_s", "v", "i_raw"]).copy()
    df = df.sort_values("t_s", kind="stable").drop_duplicates("t_s", keep="first")
    df["t_s"] = df["t_s"] - df["t_s"].iloc[0]
    return df.reset_index(drop=True)


def load(path: str | Path, sign: int | None = None) -> Record:
    path = Path(path)
    kind, temp, cond = classify(path)
    df = clean(read_mat(path))
    s = detect_sign(df["i_raw"].to_numpy(), df["v"].to_numpy()) if sign is None else sign
    df["i"] = s * df["i_raw"]
    return Record(path, kind, temp, cond, df[["t_s", "v", "i", "ah_raw", "temp_c", "chamber_c"]], s)


def load_all(root: str | Path, kinds: tuple[str, ...] | None = None) -> list[Record]:
    out = []
    for p in find_files(root):
        k, _, _ = classify(p)
        if kinds is None or k in kinds:
            out.append(load(p))
    return out


# --------------------------------------------------------------------- helpers
def discharged_ah(df: pd.DataFrame, use_tester_ah: bool = True, raw_sign: int = -1) -> np.ndarray:
    """Cumulative discharged charge [Ah] from file start (positive = removed from cell)."""
    if use_tester_ah and np.isfinite(df["ah_raw"]).all():
        return raw_sign * (df["ah_raw"].to_numpy() - df["ah_raw"].iloc[0])
    t, i = df["t_s"].to_numpy(), df["i"].to_numpy()
    q = np.concatenate([[0.0], np.cumsum(0.5 * (i[1:] + i[:-1]) * np.diff(t))])
    return q / 3600.0


def reference_soc(rec: Record, capacity_ah: float = NOMINAL_CAPACITY_AH, soc0: float = 1.0) -> np.ndarray:
    """Ground-truth SoC by lab-grade coulomb counting (tester Ah column), starting from soc0."""
    return soc0 - discharged_ah(rec.df, True, rec.raw_sign) / capacity_ah


def resample(df: pd.DataFrame, dt: float = 0.1) -> pd.DataFrame:
    """Linear interpolation onto a uniform grid (data rate varies inside files)."""
    t = df["t_s"].to_numpy()
    grid = np.arange(0.0, t[-1] + 1e-9, dt)
    out = {"t_s": grid}
    for c in df.columns:
        if c == "t_s":
            continue
        y = df[c].to_numpy(dtype=float)
        out[c] = np.interp(grid, t, y) if np.isfinite(y).all() else np.full(len(grid), np.nan)
    return pd.DataFrame(out)


def summarize(rec: Record) -> dict:
    df = rec.df
    t = df["t_s"].to_numpy()
    q_int = discharged_ah(df, use_tester_ah=False)
    q_tst = discharged_ah(df, use_tester_ah=True, raw_sign=rec.raw_sign)
    ah_err = float(np.nanmax(np.abs(q_int - q_tst))) if np.isfinite(q_tst).all() else None
    return {
        "file": rec.path.name,
        "kind": rec.kind,
        "temp_c": rec.temp_c_nominal,
        "condition": rec.condition,
        "n_samples": int(len(df)),
        "duration_h": float(t[-1] / 3600.0),
        "median_dt_s": float(np.median(np.diff(t))) if len(t) > 1 else None,
        "v_start": float(df["v"].iloc[0]),
        "v_min": float(df["v"].min()),
        "v_max": float(df["v"].max()),
        "i_max_discharge_a": float(df["i"].max()),
        "i_max_charge_a": float(-df["i"].min()),
        "net_discharged_ah": float(q_tst[-1]) if np.isfinite(q_tst[-1]) else float(q_int[-1]),
        "raw_sign": rec.raw_sign,
        "ah_integration_vs_tester_max_err": ah_err,
    }

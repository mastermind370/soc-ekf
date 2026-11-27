"""Scan data/raw, print a table of every .mat file, write results/data_inventory.json.

Usage:  python scripts/inventory.py [--root data/raw]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from soc_ekf.data import CELL_NAME, DRIVE_CYCLES, find_files, load, summarize  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--out", default=str(ROOT / "results" / "data_inventory.json"))
    a = ap.parse_args()

    files = find_files(a.root)
    if not files:
        sys.exit(f"No .mat files under {a.root}. See README 'Get the data'.")

    rows, failed = [], []
    for p in files:
        try:
            rows.append(summarize(load(p)))
        except Exception as e:  # keep going; report at the end
            failed.append({"file": str(p), "error": repr(e)})

    hdr = f"{'kind':7} {'T':>5} {'cond':5} {'hours':>6} {'dt':>5} {'Vstart':>6} {'Vmin':>5} {'Imax':>5} {'Ah':>6} sgn  file"
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda r: (r["temp_c"] if r["temp_c"] is not None else 99, r["kind"])):
        T = f"{r['temp_c']:.0f}" if r["temp_c"] is not None else "?"
        print(f"{r['kind']:7} {T:>5} {r['condition']:5} {r['duration_h']:6.2f} {r['median_dt_s']:5.2f} "
              f"{r['v_start']:6.3f} {r['v_min']:5.2f} {r['i_max_discharge_a']:5.1f} "
              f"{r['net_discharged_ah']:6.3f} {r['raw_sign']:+d}  {r['file']}")

    hours_by_kind = defaultdict(float)
    for r in rows:
        hours_by_kind[r["kind"]] += r["duration_h"]
    temps = sorted({r["temp_c"] for r in rows if r["temp_c"] is not None})
    drive = [r for r in rows if r["kind"] in DRIVE_CYCLES]
    size_mb = sum(p.stat().st_size for p in files) / 1e6
    ah_errs = [r["ah_integration_vs_tester_max_err"] for r in rows if r["ah_integration_vs_tester_max_err"] is not None]

    summary = {
        "cell": CELL_NAME,
        "n_cells": 1,
        "n_files_loaded": len(rows),
        "n_files_failed": len(failed),
        "dataset_size_mb": round(size_mb, 1),
        "temperatures_c": temps,
        "n_drive_cycle_files": len(drive),
        "drive_cycle_hours": round(sum(r["duration_h"] for r in drive), 2),
        "total_hours": round(sum(r["duration_h"] for r in rows), 2),
        "hours_by_kind": {k: round(v, 2) for k, v in sorted(hours_by_kind.items())},
        "max_ah_integration_vs_tester_err": max(ah_errs) if ah_errs else None,
    }
    print("\nSUMMARY")
    print(json.dumps(summary, indent=2))
    if failed:
        print(f"\n{len(failed)} file(s) failed to load:")
        for f in failed:
            print("  ", f["file"], f["error"])

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"summary": summary, "files": rows, "failed": failed}, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()

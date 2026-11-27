"""Builds the C port with plain cc and runs its unit + UBSan tests (skipped if no C compiler)."""
import shutil
import subprocess
from pathlib import Path

import pytest

C = Path(__file__).resolve().parents[1] / "c"


@pytest.mark.skipif(shutil.which("cc") is None, reason="no C compiler")
def test_c_unit_and_ubsan():
    subprocess.run(["sh", str(C / "build.sh")], check=True, capture_output=True)
    for exe in ("test_ekf_q", "test_ekf_q_ubsan"):
        r = subprocess.run([str(C / "build" / exe)], capture_output=True, text=True)
        assert r.returncode == 0 and "ALL PASSED" in r.stdout, r.stdout + r.stderr

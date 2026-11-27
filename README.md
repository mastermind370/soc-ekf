# soc-ekf — EKF state-of-charge estimator + fixed-point C port

Extended Kalman Filter SoC estimation for a Li-ion cell (Panasonic NCR18650PF), with a Q15/Q31 C port.
*Work in progress: stage 1 of 7 (data + loaders).*

## Get the data
Panasonic 18650PF Li-ion Battery Data, P. Kollmeyer, UW-Madison, Mendeley Data, DOI 10.17632/wykht8y7tg.1 (CC-BY-4.0).

1. Open https://data.mendeley.com/datasets/wykht8y7tg/1 and click **Download All**.
2. Unzip it. Move the temperature folders (`25degC`, `10degC`, `0degC`, `-10degC`, `-20degC`, and the `... Trise` ones) into `data/raw/`.
   They can be nested at any depth: the loader searches recursively for `*.mat`.
3. `data/raw/` is git-ignored, so the dataset is never committed.

## Conventions
Current `i` is **positive for discharge**. The loader works out the file's sign from the physics (voltage drops when discharge current steps up) and reports it as `raw_sign`.
Reference SoC is lab-grade coulomb counting from the tester's Ah column, starting at 100% (every test begins after a full charge).

## Run
```bash
source .venv/bin/activate
pytest -q
python scripts/inventory.py        # -> results/data_inventory.json
python scripts/plot_cycles.py      # -> plots/stage1_drive_cycles_25C.png
```

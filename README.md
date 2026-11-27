# soc-ekf — Li-ion state-of-charge EKF with a fixed-point C port

Estimates a lithium-ion cell's state of charge (SoC) from its voltage and current with an Extended Kalman Filter on a
2-RC equivalent-circuit model, then ports the filter to fixed-point C (int32/int64, no floating point in the update)
for a microcontroller BMS. Everything is fitted and tested on real lab data from a Panasonic NCR18650PF cell.

## Results

<!-- BEGIN:headline -->
| Cycle (held out, 25 °C) | EKF RMSE | EKF RMSE, start 30% low | EKF converges to ±2% | Coulomb counting, start 30% low | Voltage lookup |
|---|---|---|---|---|---|
| UDDS | 0.60% | 0.55% | 6 s | 31.5% | 6.5% |
| US06 | 1.34% | 1.37% | 14 s | 31.5% | 18.0% |
| HWFET (a) | 0.43% | 0.44% | 8 s | 31.5% | 12.0% |
| HWFET (b) | 0.66% | 0.68% | never | 31.5% | 12.3% |

Fixed-point C port: max difference from the float EKF **1.3e-05 % SoC** (RMS 4.0e-06 %), **82 ns per update** on an Apple M4 (double-precision C: 44 ns); estimated **651–1936 Cortex-M4 cycles** per update (op-count model, not measured on hardware).
<!-- END:headline -->

All numbers in this README are written by `scripts/update_readme.py` from `results/*.json`.

## How it works

```mermaid
flowchart LR
    subgraph DATA["Panasonic 18650PF lab data (.mat)"]
        C20["C/20 discharge + charge"]
        HPPC["HPPC pulse test"]
        DRV["Drive cycles<br/>UDDS · US06 · HWFET · LA92 · mixed"]
    end
    C20 -->|"capacity Q,<br/>OCV(SoC) spline"| MODEL["2-RC model<br/>R0(SoC), R1, C1, R2, C2"]
    HPPC -->|"least-squares fit<br/>(logging gaps filled)"| MODEL
    MODEL --> EKF
    DRV -->|"I, V at 1 Hz"| EKF["EKF<br/>x = [SoC, v1, v2]<br/>predict: coulomb count + RC decay<br/>update: V vs OCV − R0·I − v1 − v2"]
    DRV -->|"lab coulomb count<br/>(truth)"| EVAL["Evaluation<br/>RMSE, convergence,<br/>noise, bias, temperature, aging"]
    EKF --> EVAL
    EKF -->|"same equations"| CQ["C11 fixed-point EKF<br/>Q1.30 SoC · Q3.28 V · Q6.25 A<br/>block-floating covariance"]
    CQ -->|"vs float"| EVAL
    BASE["Baselines<br/>coulomb counting · OCV lookup"] --> EVAL
```

**Model.** Terminal voltage `V = OCV(SoC) − R0(SoC)·I − v1 − v2`, with each RC pair `dv_k/dt = −v_k/τ_k + R_k(SoC)·I/τ_k`
(I > 0 = discharge). Capacity and OCV come from the C/20 test. Two OCV curves are tried: the mean of the C/20
discharge and charge curves (the hysteresis-free OCV), and the discharge curve alone (the hysteresis branch a cell rests
on after discharging, which is what every test here does). R0, R1, R2 are piecewise-linear in SoC; τ1, τ2 are
constants. They are fitted on the HPPC test only. The OCV curve and the R type are chosen on LA92 and the mixed drive
cycles; UDDS, US06 and HWFET are never used for any fitting, tuning or model choice.

**EKF.** State `[SoC, v1, v2]`, updating at 1 Hz. The predict step is exact discretisation of the model with the
interval-mean current; the update compares measured and predicted voltage, with
`H = [dOCV/dSoC − dR0/dSoC·I, −1, −1]`. The three noise levels are tuned by grid search on LA92 and the mixed cycles;
UDDS, US06 and HWFET are held out for testing.

**Fixed-point C port** (`c/`). C11, int32 storage with int64 intermediates and saturating arithmetic. Formats: SoC
Q1.30, voltages Q3.28, current Q6.25, resistances Q0.31, slopes Q4.27, gain Q5.26. The covariance spans about
1e-12 to 0.06, more than one Q31 format can hold, so it is stored as block floating point (int32 mantissas, one
shared exponent) with a carry accumulator that keeps sub-LSB process noise instead of rounding it to zero.
Tests cover saturation and rounding edges, tracking the double-precision reference, and 100 000 steps of extreme or
garbage input under UBSan (no undefined behaviour, bounded state, recovery after re-initialisation).

## Data

Panasonic 18650PF Li-ion Battery Data, P. Kollmeyer, University of Wisconsin–Madison, Mendeley Data,
[doi:10.17632/wykht8y7tg.1](https://doi.org/10.17632/wykht8y7tg.1), CC-BY-4.0. Not included in this repo.

<!-- BEGIN:data -->
_Not run yet: `python scripts/inventory.py`._
<!-- END:data -->

Ground-truth SoC is lab coulomb counting from the tester's Ah counter, starting at 100% (every test begins after a
full charge), with the measured C/20 capacity. The HPPC files leave out the 10%-SoC discharges between pulse groups
(no samples there); the loader detects those logging gaps and restores the missing charge from the tester's Ah counter.

## Details

### Stage 2 — battery model
<!-- BEGIN:model -->
Capacity (C/20, 25 °C): **2.995 Ah** · OCV: C/20 discharge curve (discharge branch) · OCV spline fit: 0.52 mV · HPPC fit (soc-dependent R): 11.0 mV · τ1 = 28.0 s, τ2 = 5000 s · HPPC logging gaps: 13 (charge from tester Ah)

Model choice on LA92/mixed cycles only (voltage RMSE, mV): discharge OCV + soc-dependent R 41.7, mean OCV + soc-dependent R 111.1, mean OCV + constant R 120.7, discharge OCV + constant R 130.8. Parameters at fit bounds: tau2=5000, r1@0=0.5, r2@0=0.5, r2@0.1=0.5, r2@0.2=0.5, r2@0.4=0.0001, r2@1=0.0001.

| Drive cycle | Voltage RMSE, constant R [mV] | Voltage RMSE, SoC-dependent R [mV] | Max error, selected [mV] |
|---|---|---|---|
| UDDS | 78.7 | 35.4 | 394 |
| US06 | 227.9 | 58.8 | 567 |
| HWFET (a) | 175.9 | 48.3 | 349 |
| HWFET (b) | 173.1 | 46.4 | 376 |
| LA92 | 108.2 | 34.5 | 314 |
| MIX | 140.4 | 48.2 | 465 |
| MIX | 145.2 | 51.6 | 435 |
| MIX | 130.3 | 33.6 | 305 |
| MIX | 140.3 | 48.8 | 413 |
| NN | 120.5 | 33.3 | 389 |
<!-- END:model -->

### Stage 3 — baselines
<!-- BEGIN:baselines -->
RMSE / max error in % SoC. Coulomb counting uses the 2.9 Ah datasheet capacity; the wrong start is 30% low.

| Cycle | Coulomb count | Coulomb count, wrong start | OCV(V) lookup | OCV(V + R0·I) lookup |
|---|---|---|---|---|
| UDDS | 1.69 / 2.9 | 31.47 / 32.9 | 6.53 / 30.0 | 3.45 / 22.0 |
| US06 | 1.69 / 2.8 | 31.47 / 32.8 | 18.00 / 64.3 | 9.39 / 62.0 |
| HWFET (a) | 1.72 / 3.0 | 31.49 / 33.0 | 12.03 / 26.9 | 7.09 / 22.8 |
| HWFET (b) | 1.72 / 3.0 | 31.49 / 33.0 | 12.35 / 27.3 | 7.42 / 25.9 |
| LA92 | 1.64 / 2.8 | 31.42 / 32.8 | 9.74 / 39.5 | 4.97 / 32.5 |
<!-- END:baselines -->

### Stage 4 — EKF
<!-- BEGIN:ekf -->
EKF at 1 Hz, tuned by grid search on LA92, MIX, NN only: σ_v = 80 mV, σ_SoC,proc = 1e-06 /√s, σ_RC,proc = 1 mV/√s.

| Cycle | RMSE / max, correct start | RMSE / max, 30% low start | → ±5% | → ±2% | Current +5% (CC) | Current −5% (CC) | Noise* |
|---|---|---|---|---|---|---|---|
| UDDS | 0.60 / 1.9 | 0.55 / 3.7 | 0 s | 6 s | 1.62 (4.35) | 2.62 (0.98) | 0.57 |
| US06 | 1.34 / 2.5 | 1.37 / 3.3 | 0 s | 14 s | 2.87 (4.33) | 1.71 (0.99) | 1.29 |
| HWFET (a) | 0.43 / 2.1 | 0.44 / 3.7 | 0 s | 8 s | 1.93 (4.43) | 2.14 (1.00) | 0.43 |
| HWFET (b) | 0.66 / 2.7 | 0.68 / 2.9 | 0 s | never | 2.32 (4.43) | 1.86 (1.00) | 0.66 |

% SoC. *Gaussian noise, σ = 5% of the cycle's RMS current plus 5 mV on voltage.
<!-- END:ekf -->

### Stage 5 — fixed-point C port
<!-- BEGIN:cport -->
| Cycle | Updates | max \|fixed − float\| [% SoC] | RMS | max, wrong start | RMSE float | RMSE fixed | ns fixed (M4 Mac) | ns double |
|---|---|---|---|---|---|---|---|---|
| UDDS | 22448 | 1.3e-05 | 6.8e-06 | 1.1e-05 | 0.60 | 0.60 | 82 | 45 |
| US06 | 4819 | 5.2e-06 | 2.8e-06 | 3.0e-06 | 1.34 | 1.34 | 81 | 44 |
| HWFET (a) | 7613 | 3.4e-06 | 1.5e-06 | 3.2e-06 | 0.43 | 0.43 | 82 | 44 |
| HWFET (b) | 7598 | 6.2e-06 | 2.8e-06 | 4.5e-06 | 0.66 | 0.66 | 82 | 44 |

Ops per update: 39 mul, 84 add, 47 shift, 3 div, 24 load, 116 branch. Cortex-M4 estimate: **651–1936 cycles** (3.87–11.52 µs at 168 MHz) from per-op cost ranges in `scripts/compare_c.py` — an estimate, not a hardware measurement.
<!-- END:cport -->

### Stage 6 — robustness
<!-- BEGIN:robust -->
**Temperature** (mean SoC RMSE over UDDS/US06/HWFET, %)

| Temp | EKF, 25 °C model | EKF, per-temperature model | …start 30% low | Coulomb counting | Truth SoC at cutoff | HPPC fit [mV] |
|---|---|---|---|---|---|---|
| -20 °C | 18.66 | 9.65 | 9.75 | 1.00 | 51% | 21.8 |
| -10 °C | 6.94 | 13.81 | 13.84 | 0.93 | 55% | 15.7 |
| 0 °C | 10.30 | 3.68 | 3.75 | 1.49 | 23% | 13.9 |
| 10 °C | 3.30 | 3.58 | 3.56 | 1.44 | 36% | 10.9 |
| 25 °C | 0.76 | 0.76 | 0.76 | 1.70 | 11% | – |

**Aging** — capacity 2.995 → 2.396 Ah (20.0% fade; ASSUMED 20% fade (no end-of-tests C/20 file found)). Emulated on the 25 °C cycles: the estimators keep the fresh capacity.

| Cycle | EKF, right capacity | EKF, fresh capacity | Coulomb counting, fresh capacity |
|---|---|---|---|
| UDDS | 0.60 | 9.85 | 8.96 |
| US06 | 1.34 | 8.19 | 8.97 |
| HWFET (a) | 0.43 | 9.79 | 9.16 |
| HWFET (b) | 0.66 | 9.48 | 9.15 |

**Current-sensor bias** — SoC RMSE %, EKF (coulomb counting)

| Offset | UDDS | US06 | HWFET (a) | HWFET (b) |
|---|---|---|---|---|
| -100mA | 11.11 (10.73) | 1.82 (0.99) | 3.31 (2.50) | 3.00 (2.49) |
| -50mA | 5.44 (4.52) | 1.35 (0.34) | 1.64 (0.39) | 1.36 (0.39) |
| -20mA | 2.38 (0.80) | 1.25 (1.14) | 0.78 (0.87) | 0.65 (0.87) |
| +20mA | 1.40 (4.17) | 1.54 (2.21) | 0.71 (2.56) | 1.10 (2.55) |
| +50mA | 3.86 (7.89) | 1.92 (3.00) | 1.52 (3.82) | 1.92 (3.81) |
| +100mA | 7.72 (14.10) | 2.62 (4.34) | 2.92 (5.92) | 3.32 (5.91) |
<!-- END:robust -->

## Plots

<!-- BEGIN:plots -->
**OCV curve and slope from the C/20 test**

![OCV curve and slope from the C/20 test](plots/stage2_ocv_25C.png)

**2-RC fit on the HPPC test**

![2-RC fit on the HPPC test](plots/stage2_hppc_fit_25C.png)

**Model voltage on held-out drive cycles**

![Model voltage on held-out drive cycles](plots/stage2_validation_25C.png)

**EKF on US06: correct start, 30% low start, coulomb counting**

![EKF on US06: correct start, 30% low start, coulomb counting](plots/stage4_US06_03-20-17_01.43_25deg.png)

**EKF on UDDS**

![EKF on UDDS](plots/stage4_UDDS_03-21-17_00.29_25deg.png)

**Fixed-point C vs float Python**

![Fixed-point C vs float Python](plots/stage5_US06_3250.png)

**SoC RMSE vs temperature**

![SoC RMSE vs temperature](plots/stage6_temperature.png)

**SoC RMSE vs current-sensor bias**

![SoC RMSE vs current-sensor bias](plots/stage6_bias.png)
<!-- END:plots -->

## Limitations

- **Truth is coulomb counting** with lab-grade sensors from a known full start, the standard reference for this
  dataset, but not an independent SoC measurement.
- **Model error grows with current.** On aggressive cycles (US06, 15–20 A peaks) the 2-RC model's voltage error maps
  to several % of SoC in the flat middle of the OCV curve. Current-dependent measurement noise or a hysteresis state
  would be the next step.
- **No hysteresis state.** The model uses one OCV branch; a one-state hysteresis model (Plett) would handle mixed
  charge/discharge use better.
- **OCV is from 25 °C only**; other temperatures refit R and τ but keep the 25 °C OCV and capacity.
- **Aging is emulated**: the dataset has no drive cycles on the aged cell, so the test gives the estimators the
  fresh capacity on the 25 °C cycles, using the measured fade.
- **Cortex-M4 cycles are an estimate** from counted operations and per-operation cost ranges, not a measurement on
  hardware. The ns-per-update figures are measured on an Apple M4.

## Run it

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# put the dataset's temperature folders in data/raw/ (any nesting)
pytest -q                          # Python tests + C unit/UBSan tests
python scripts/inventory.py        # stage 1: data inventory
python scripts/plot_cycles.py
python scripts/fit_model.py        # stage 2: OCV, capacity, 2-RC fit, validation
python scripts/baselines.py        # stage 3
python scripts/run_ekf.py          # stage 4: tune + test the EKF
sh c/build.sh                      # stage 5: build the C port (cc only; or use c/CMakeLists.txt)
python scripts/compare_c.py
python scripts/robustness.py       # stage 6
python scripts/update_readme.py    # stage 7: refresh this README from results/
```

## Layout

```
soc_ekf/      data.py (loaders) · model.py (OCV, 2-RC, fit, gaps) · fitting.py · baselines.py · ekf.py · cycles.py
scripts/      one script per stage + inspect_file.py + update_readme.py
c/            include/ src/ (ekf_q.c fixed point, ekf_f.c double reference) · tests/ · tools/ekf_trace.c · build.sh
tests/        pytest suite (synthetic cells with known parameters; real-data checks skip without the dataset)
results/      *.json written by the scripts      plots/   *.png
```

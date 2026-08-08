# Synthetic Example

Synthetic data benchmark for the AquaPINN tracker. Simulated acoustic tag detections are generated for a Cabot Station-like hydrophone array, and the NN / NN+ solver is evaluated against the known ground-truth trajectory. Includes sensitivity analysis and CRLB (Cramér–Rao Lower Bound) comparison.

---

## Data Generation

**Scripts in [`dataGenerator/`](dataGenerator/):**

| Script | Description |
|--------|-------------|
| [`caseDesign_CabotSynthetic.py`](dataGenerator/caseDesign_CabotSynthetic.py) | Define hydrophone array geometry and simulation scenarios |
| [`TOASimulator.py`](dataGenerator/TOASimulator.py) | Simulate TOA detections with noise, missed detections, and false alarms |
| [`TrajectorySimulator.py`](dataGenerator/TrajectorySimulator.py) | Generate synthetic fish trajectories (straight-line, curved, etc.) |
| [`getCRLB.py`](dataGenerator/getCRLB.py) | Compute the CRLB for position estimation given the array geometry |

---

## AquaPINN Pipeline

**Entry point:** [`run/main.py`](run/main.py)

The synthetic pipeline iterates over multiple dataset scenarios and runs the NN / NN+ tracker on each, comparing results against the known ground truth.

**Run:**
```bash
python Examples/Synthetic/run/main.py
```

**Key settings in [`params.toml`](run/params.toml):**

| Parameter | Value | Notes |
|-----------|-------|-------|
| `trackMethod` | `"NN"` | Deep PINN solver |
| `nSamples` | `>1` | NN+ mode for UQ evaluation |
| `dam` | `"Cabot"` | Cabot Station array geometry |
| `ignoreZ` | `"False"` | Full 3-D tracking |

---

## Analysis Scripts

| Script | Description |
|--------|-------------|
| [`sensitivity_baseline.py`](run/sensitivity_baseline.py) | Run baseline configuration for sensitivity study |
| [`report_sensitivity.py`](run/report_sensitivity.py) | Aggregate results and generate sensitivity report |
| [`CompareVarCRLB.py`](run/CompareVarCRLB.py) | Compare NN+ position variance against the theoretical CRLB |
| [`ComparisonForShadTest.py`](run/ComparisonForShadTest.py) | Legacy script — compare NN vs. AML vs. YAPS on shad test scenarios (uses hardcoded PNNL network paths; superseded by `run_comparison.py` pattern) |
| [`run_comparison.py`](run/run_comparison.py) | Run the shared `AquaPINN_Tracker3D` `compareGPS.py` post-processing pipeline for all Critical scenarios |

---

## NN vs. NNplus Comparison

The comparison workflow uses the shared post-processing implementation in
`AquaPINN_Tracker3D/PostProcess/compareGPS.py`, following the same launcher
pattern as the FishHeart and Sequim examples.

The comparison configuration is [`compare.toml`](run/compare.toml). It defines
the methods, result folders, output folder, and plot types. The Synthetic
specific deployment values—tag code, PRI, hydrophone configuration, GPS
locations, and time window—are defined directly in
[`run_comparison.py`](run/run_comparison.py). No additional params or tag
metadata files are required.

**Run:**

```bash
python Examples/Synthetic/run/run_comparison.py
```

The launcher processes these Critical scenarios:

- `baseline`
- `sudden_burst`
- `sudden_turn`
- `tmp_Exit`

For each scenario it compares the `NN` and `NNplus` track CSV files with the
corresponding ground-truth GPS file in `data/Critical/`. The shared pipeline
generates comparison and XYZ plots and saves statistics in:

```text
run/results/Critical/comparisonGPS/<scenario>/
```

Each scenario directory contains the generated plots under `plots/` and a
`Stat.pickle` metrics file.

To use a different comparison configuration:

```bash
python Examples/Synthetic/run/run_comparison.py path/to/compare.toml
```

---

## Outputs

- Per-scenario track CSV files and error statistics in `run/results/`
- Variance vs. CRLB comparison plots
- Sensitivity analysis figures and summary tables (see [`sensitivity_analysis.md`](run/sensitivity_analysis.md))

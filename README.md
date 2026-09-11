# AquaPINN Tracker 3D

A physics-informed neural network (PINN) framework for 3-D acoustic tag tracking in aquatic environments. The system processes raw hydrophone detections, synchronizes receiver clocks, and estimates fish trajectories using either a deep-learning solver (NN / NN+) or a fast analytical solver (AML).

---

## Pipeline Overview

```
Raw receiver files (SR*.csv / RX*.csv / .mat)
        │
        ▼
  RawDataProcess.decoder          ← decode tag detections
        │
        ▼
  Sync.syncer                     ← synchronize hydrophone clocks
        │
        ▼
  Track.tracker                   ← estimate 3-D trajectories
        │
        ▼
  PostProcess                     ← plots, GPS error, reports
```

---

## Tracking Methods

| Method | Description | Key parameter |
|--------|-------------|---------------|
| `AML`  | Approximate Maximum Likelihood — fast, pointwise solver | `trackMethod = ["AML"]` |
| `NN`   | Deep PINN solver — batch-wise, GPU-accelerated | `trackMethod = ["NN"]`, `nSamples = [1]` |
| `NN+`  | NN with uncertainty quantification via ensemble sampling | `trackMethod = ["NN"]`, `nSamples > 1` |

---

## Configuration

Each deployment has a `params.toml` (or `params_*.json`) file that controls all pipeline settings. Key parameters:

| Parameter | Description |
|-----------|-------------|
| `phoneFile` | Hydrophone location CSV |
| `tagFile` | Tag code / PRI CSV |
| `dataFolder` | Raw receiver data directory |
| `testFile` | EA test description CSV (named drift periods) |
| `GPSFile` | Reference GPS track CSV |
| `time_start` / `time_end` | Processing time window (local time) |
| `timeZone` | UTC offset in hours |
| `temperature_ref` | Water temperature (°C) for sound speed |
| `trackMethod` | `"AML"` or `"NN"` |
| `nSamples` | 1 = NN solver; >1 = NN+ (UQ) |
| `output_folder` | Results directory |

---

## Post-Processing & Comparison

Each example includes a two-step post-processing workflow:

### Step 1 — Generate comparison plots
```bash
python Examples/fishHeart/run/run_comparison.py
python Examples/Sequim/run/run_comparison.py
```

Reads `compare.toml` and calls [`PostProcess/compareGPS.py`](AquaPINN_Tracker3D/AquaPINN_Tracker3D/PostProcess/compareGPS.py) to:
- Compare AML, NN, NN+, YAPS trajectories on one figure
- Overlay trajectories on a georeferenced background image
- Plot X/Y/Z time-series (optional, `plots.xyz = true`)
- Compute GPS accuracy metrics → `Stat.pickle`

### Step 2 — Generate summary table
```bash
python AquaPINN_Tracker3D/AquaPINN_Tracker3D/PostProcess/report.py Examples/fishHeart/run/compare.toml
```

### `compare.toml` key settings

| Key | Description |
|-----|-------------|
| `[methods]` | Method label → result folder mapping |
| `img_path` | Background image for georeferenced overlay |
| `[pixel2Phone1/2]` | Two pixel↔phone anchor points for image registration |
| `gps_file_pattern` | Per-tag GPS file (supports `{tagID}`, `{tagCode}`) |
| `[plots]` | `map`, `compare`, `xyz` flags to enable/disable plot types |
| `[report]` | Segments and methods for the summary table |

---

## Requirements

- Python ≥ 3.11 (or Python ≥ 3.8 with `pip install tomli`)
- PyTorch (CUDA recommended)
- NumPy, SciPy, pandas, matplotlib, utm, tomli, networkx, scikit-learn, tqdm, six

Install dependencies:
```bash
pip install torch numpy scipy pandas matplotlib utm tomli networkx scikit-learn tqdm six
```

---

## Quick Start

```bash
# FishHeart example — run pipeline
python Examples/fishHeart/run/main_FishHeart.py

# FishHeart example — run AML vs NN comparison
python Examples/fishHeart/run/run_comparison.py

# Sequim example — run pipeline
python Examples/Sequim/run/main_Sequim.py

# Sequim example — run AML vs NN comparison
python Examples/Sequim/run/run_comparison.py

# Synthetic benchmark
python Examples/Synthetic/run/main.py
```

---

## Examples

| Example | Solver | Data format | Description |
|---------|--------|-------------|-------------|
| [`fishHeart`](Examples/fishHeart/) | AML / NN | SR*.csv | Flume experiment with 4 drifting tags and 7 hydrophones |
| [`Sequim`](Examples/Sequim/) | AML / NN+ | `.mat` | Open-water drifting-tag deployment in Sequim Bay |
| [`Synthetic`](Examples/Synthetic/) | NN / NN+ | Simulated | Cabot Station synthetic benchmark for sensitivity analysis |
| YAPS | YAPS (R) | SR*.csv / CSV | R-based YAPS comparison for FishHeart and Sequim |


## LICENSE
This project is licensed under the GNU General Public License v3.0 (GPL v3.0).
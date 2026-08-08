# AquaPINN_Tracker3D — Package Reference

The `AquaPINN_Tracker3D` Python package implements the full acoustic-tag localization pipeline: raw data decoding, hydrophone clock synchronization, 3-D trajectory estimation via physics-informed neural networks (PINN) or an analytical AML solver, and post-processing comparison tools.

---

## Public API

Import via the package `__init__.py`:

```python
import importlib, sys
sys.path.insert(0, "path/to/AquaPINN_Tracker3D")

PINN = importlib.import_module("AquaPINN_Tracker3D")

getLogger          = PINN.getLogger           # logging setup
localizationSystem = PINN.localizationSystem  # deployment object
decoder            = PINN.decoder             # Stage 1
syncer             = PINN.syncer              # Stage 2
tracker            = PINN.tracker             # Stage 3
```

---

## Key Classes and Functions

### [`localizationSystem`](AquaPINN_Tracker3D/system.py)
Central object that holds all deployment geometry and metadata.

| Method | Description |
|--------|-------------|
| `deploy(param_file, doSync)` | Load phone/tag info from CSV; if `doSync=False`, load pre-synced phone CSV |
| `deployPhone(phoneFile, ...)` | Parse hydrophone locations, compute local XYZ frame |
| `deployTag(tagFile)` | Load tag codes and PRIs |
| `deployTest(testFile)` | Load named test periods (drift segments) from EA description CSV |
| `loadDecodes(decodesFile)` | Load detection pickle, map node SN → phone ID |
| `applySync2Decodes(...)` | Apply sync correction to detection timestamps |
| `applySync2Phone(...)` | Apply sync-corrected hydrophone positions |
| `getSyncedDecodesFile(...)` | Resolve expected synced-decodes pickle path |
| `loadGPS(GPS_file)` | Load GPS reference track (`.pos` or `.csv`) |

### [`config.oneKey_configure()`](AquaPINN_Tracker3D/config.py)
```python
config.oneKey_configure(use_cuda=True, precision=64, seed=42)
```
Sets PyTorch device, default float dtype, and all random seeds.

### [`load_params(param_file)`](AquaPINN_Tracker3D/SharedLibs/functions.py)
Reads a `.toml` or `.json` parameter file and returns a list of all parameter combinations (for grid-search runs).

---

## Pipeline Stages

### Stage 1 — Decode (`decoder.run`)
- Reads raw ATS Sonic Receiver SR*.csv or RX*.csv files from `dataFolder`
- Decodes tag and beacon detections within `[time_start, time_end]`
- Saves `*_TagDecodes.pickle` and `*_BeaconDecodes.pickle` to `output_folder`

### Stage 2 — Sync (`syncer.run`)
- Builds TDOA pairs from beacon detections
- Trains a piecewise-quadratic clock-offset model (Adam + BFGS)
- Applies corrections to tag detection timestamps
- Saves `*_TagDecodes_synced.pickle` and `*_synced.csv` phone locations

### Stage 3 — Track (`tracker.run`)
- Reads synced detection pickle
- Splits track into segments by time gap or test-file periods
- For each segment: trains NN (or runs AML) to estimate 3-D trajectory
- Saves track CSV, figures, and network weights to `output_folder`

---

## Post-Processing (`PostProcess/`)

| Script | Description |
|--------|-------------|
| [`compareGPS.py`](AquaPINN_Tracker3D/PostProcess/compareGPS.py) | Main comparison pipeline — loads `compare.toml`, generates plots, computes GPS accuracy metrics, saves `Stat.pickle` |
| [`plots.py`](AquaPINN_Tracker3D/PostProcess/plots.py) | All plot functions: `plot_track_OneFigureEqual`, `plot_track_with_Background`, `plot_track_XYZ`, `getMetrics` |
| [`utils.py`](AquaPINN_Tracker3D/PostProcess/utils.py) | Helper functions: `find_result_csv`, `get_gps_file`, `get_pri_from_csv`, `resolve_path` |
| [`report.py`](AquaPINN_Tracker3D/PostProcess/report.py) | Reads `Stat.pickle` → generates per-segment summary CSV table |

### `compare.toml` structure

```toml
params_file   = "params.toml"       # pipeline config
out_root_dir  = "results/comparison"
img_path      = "../config/bg.PNG"  # background image (optional)
gps_file_pattern = ""               # per-tag GPS path pattern
tag_ids       = []                  # [] = all tags

[methods]
AML = "results/Track/track"
NN  = "results/Track/track"

[pixel2Phone1]
px       = [col, row]
phone_id = N

[pixel2Phone2]
px       = [col, row]
phone_id = M

[plots]
map     = true   # background-image overlay
compare = true   # all-methods-in-one-figure
xyz     = false  # X/Y/Z time-series stacked plot

[report]
segments = ["All"]
methods  = ["AML", "NN"]
```

---

## Configuration Parameters

All parameters are read from `params.toml` (TOML) or `params_*.json` (JSON).
Each value is wrapped in a list to support grid-search sweeps.

| Parameter | Type | Description |
|-----------|------|-------------|
| `phoneFile` | str | Hydrophone location CSV |
| `tagFile` | str | Tag code / PRI CSV |
| `dataFolder` | str | Raw receiver data directory |
| `testFile` | str | EA test description CSV (named drift periods) |
| `GPSFile` | str | Reference GPS track |
| `output_folder` | str | Results output directory |
| `time_start` / `time_end` | str | Processing window (local time) |
| `timeZone` | int | UTC offset (hours) |
| `temperature_ref` | float | Water temperature °C (for sound speed) |
| `tol_signal` | float | TOA tolerance (s) for detection matching |
| `dis_detect` | float | Max detection range (m) |
| `trackMethod` | str | `"AML"` or `"NN"` |
| `nSamples` | int | 1 = NN; >1 = NN+ (UQ ensemble) |
| `nIterAdam` | int | Adam training iterations |
| `nIterBFGS` | int | BFGS fine-tuning iterations |
| `lr` | float | Adam learning rate |
| `weight_decay` | float | L2 regularization |
| `ignoreZ` | str | `"True"` = 2-D tracking only |
| `badPhones` | list | Phone IDs to exclude |

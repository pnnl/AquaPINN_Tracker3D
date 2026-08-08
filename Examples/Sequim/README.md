# Sequim Example

Acoustic-tag tracking for a drifting-tag deployment in **Sequim Bay, WA** on **2024-02-21**. Tags were attached to a drifting buoy and tracked by 6 hydrophones. A DGPS reference track is available for accuracy evaluation. Raw detections are stored in a pre-processed MATLAB `.mat` file (no separate sync step needed).

---

## AquaPINN Pipeline

**Entry point:** [`run/main_Sequim.py`](run/main_Sequim.py)

**Pipeline stages (Sequim-specific):**

Unlike the standard pipeline, Sequim uses a pre-processed `.mat` file instead of raw receiver CSVs. The `main_Sequim.py` script:
1. Locates `DriftingTags_timeCorrected.mat` in `../data/`
2. Calls [`convertDetections.convert_mat_to_synced_pickle()`](run/convertDetections.py) to convert it directly to the synced-decodes pickle format (no separate sync step)
3. Calls `tracker.run()` with the AML or NN+ solver

**Run:**
```bash
python Examples/Sequim/run/main_Sequim.py
```

**Key settings in [`params.toml`](run/params.toml):**

| Parameter | Value | Notes |
|-----------|-------|-------|
| `trackMethod` | `"AML"` or `"NN"` | Solver selection |
| `nSamples` | `1` | Set `>1` for NN+ (UQ ensemble) |
| `TagID` | `0` | Tag index to track (0-based) |
| `testFile` | `../config/EA_test_description.csv` | 5 named drift periods |
| `timeZone` | `0` | UTC |
| `time_start/end` | `2024-2-21 11:30:00` / `2024-2-21 13:30:00` | 2-hour window |
| `temperature_ref` | `8` °C | Sequim Bay water temperature |
| `output_folder` | `results/` | Under `run/` |

**Outputs** (written to `run/results/`):
- `Track/track/` — per-tag trajectory CSV files (`track_Tag0_<code>_AML.csv`, `track_Tag0_<code>_NN.csv`)
- `detections/` — detection visualization plots
- `history/` — training loss curves
- `net/` — saved network weights

---

## `convertDetections.py` Module

[`run/convertDetections.py`](run/convertDetections.py) provides two reusable functions:

| Function | Description |
|----------|-------------|
| `loadMsg_Mat(matFile, varName, badPhones, tagIndices)` | Load detections from a MATLAB `.mat` struct; filter bad receivers |
| `convert_mat_to_synced_pickle(mat_file, out_pickle, badPhones)` | Convert `.mat` → synced-decodes pickle (merges date + time-of-day columns into fractional MATLAB datenum) |

---

## Post-Processing Comparison

**Entry point:** [`run/run_comparison.py`](run/run_comparison.py)  
**Config:** [`run/compare.toml`](run/compare.toml)

```bash
python Examples/Sequim/run/run_comparison.py
```

Generates per-tag, per-drift-period comparison plots in `run/results/comparison/plots/`:

| Plot | Description |
|------|-------------|
| `Tag0_<code>_All.jpg` | All-methods-in-one-figure: XY map + X/Y time-series + error boxplot |
| `Tag0_<code>_All_map.jpg` | Trajectories overlaid on georeferenced satellite background image |
| `Tag0_<code>_drift_Y1/Y0/X2/X1/X0.jpg` | Per-drift-period comparison (5 segments) |
| `Tag0_<code>_drift_*_map.jpg` | Per-drift-period background overlay |

**`compare.toml` key settings:**
- `[methods]` — `AML` and `NN` both point to `results/Track/track/`
- `img_path` — `../config/Sequim_400_400.png` (Google satellite image)
- `[pixel2Phone1/2]` — pixel↔phone anchor points for image geo-registration
- `[plots]` — `map = true`, `compare = true`, `xyz = false`

---

## YAPS Comparison Pipeline

**Entry point:** [`run_YAPS/YAPS_for_Sequim_paper.R`](run_YAPS/YAPS_for_Sequim_paper.R)

Run with `Rscript` or Source in RStudio. Uses pre-synced detections from `config/YAPS/decodes4YAPS.csv`.

---

## Deployment Details

- **Date:** 2024-02-21
- **Tags:** 9 drifting tags (IDs 0–8); Tag 0 (`G72285C6F`) has AML and NN results
- **Hydrophones:** 6 ATS RX receivers (RX11072, RX11084, RX11106, RX11156, RX12042, RX12047)
- **Beacon tags:** 6 (one per hydrophone, for clock sync)
- **Sound speed:** ~1480 m/s (saltwater, ~8 °C)
- **GPS reference:** DGPS drifting buoy track, 11:30–13:30 UTC
- **Test periods:** 5 named drift segments (`drift_Y1`, `drift_Y0`, `drift_X2`, `drift_X1`, `drift_X0`) defined in `config/EA_test_description.csv`

# FishHeart Example

Acoustic-tag tracking for the FishHeart experiment conducted on **2024-03-21**. Four drifting tags were released and tracked by 7 hydrophones over a ~40-minute window. A DGPS reference track is available for accuracy evaluation.

---

## AquaPINN Pipeline

**Entry point:** [`run/main_FishHeart.py`](run/main_FishHeart.py)

**Pipeline stages:**
1. `decoder.run()` — decode raw SR*.csv files → detection pickles
2. `syncer.run()` — synchronize 7 hydrophone clocks using 7 beacon tags
3. `tracker.run()` — estimate 3-D trajectories (AML or NN solver)

**Run:**
```bash
python Examples/fishHeart/run/main_FishHeart.py
```

**Key settings in [`params.toml`](run/params.toml):**

| Parameter | Value | Notes |
|-----------|-------|-------|
| `trackMethod` | `"AML"` or `"NN"` | Solver selection |
| `nSamples` | `1` | Set `>1` for NN+ (UQ) |
| `TagID` | `3` | Tag index to track (0-based) |
| `testFile` | `../config/Fishheart_EA_test_description.csv` | 6 named drift periods |
| `timeZone` | `-4` | UTC-4 (EDT) |
| `time_start/end` | `2024-3-21 00:00:00` / `2024-4-2 00:00:00` | Full deployment window |
| `temperature_ref` | `10` °C | Sound speed reference |
| `output_folder` | `results/` | Under `run/` |

**Outputs** (written to `run/results/`):
- `Track/track/` — per-tag trajectory CSV files (`track_Tag3_<code>_AML.csv`, `track_Tag3_<code>_NN.csv`)
- `detections/` — detection visualization plots
- `history/` — training loss curves
- `net/` — saved network weights

---

## Post-Processing Comparison

**Entry point:** [`run/run_comparison.py`](run/run_comparison.py)  
**Config:** [`run/compare.toml`](run/compare.toml)

```bash
python Examples/fishHeart/run/run_comparison.py
```

Generates per-tag, per-drift-period comparison plots in `run/results/comparison/plots/`:

| Plot | Description |
|------|-------------|
| `Tag3_<code>_All.jpg` | All-methods-in-one-figure: XY map + X/Y time-series + error boxplot |
| `Tag3_<code>_All_map.jpg` | Trajectories overlaid on georeferenced flume background image |
| `Tag3_<code>_drift N.jpg` | Per-drift-period comparison (6 segments) |
| `Tag3_<code>_drift N_map.jpg` | Per-drift-period background overlay |

**`compare.toml` key settings:**
- `[methods]` — `AML` and `NN` both point to `results/Track/track/`
- `img_path` — `../config/background.PNG` (Google Earth flume image)
- `[pixel2Phone1/2]` — pixel↔phone anchor points for image geo-registration
- `[plots]` — `map = true`, `compare = true`, `xyz = false`

---

## YAPS Comparison Pipeline

**Entry point:** [`run_YAPS/YAPS_for_FishHeart_paper.R`](run_YAPS/YAPS_for_FishHeart_paper.R)

Run with `Rscript` or Source in RStudio. Uses pre-synced detections from `config/YAPS/decodes4YAPS.csv`.

---

## Deployment Details

- **Date:** 2024-03-21
- **Tags:** 4 focal drifting tags (`G72256C58`, `G7254813E`, `G72285C6F`, `G726777EF`)
- **Hydrophones:** 7 ATS Sonic Receivers (SR24009–SR24015)
- **Beacon tags:** 7 (one per hydrophone, for clock sync)
- **Sound speed:** ~1500 m/s (freshwater, ~10 °C)
- **GPS reference:** DGPS track, 12:06–12:46 local time
- **Test periods:** 6 named drift segments defined in `config/Fishheart_EA_test_description.csv`

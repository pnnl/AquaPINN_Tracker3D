# -*- coding: utf-8 -*-
"""
report.py — Config-driven summary statistics report.

Reads compare.toml (or a path passed as argv[1]) to locate the Stat.pickle
produced by compareGPS.py / run_comparison.py, then generates a per-segment
CSV table of tracking efficiency and position errors for each method and tag.

Usage:
    python report.py                      # uses compare.toml in CWD
    python report.py path/to/compare.toml
"""

from __future__ import annotations

import os
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import tomllib           # Python >= 3.11
except ImportError:
    import tomli as tomllib  # pip install tomli

# ---------------------------------------------------------------------------
# Locate and load compare.toml
# ---------------------------------------------------------------------------
if len(sys.argv) > 1:
    compare_file = Path(sys.argv[1]).resolve()
else:
    compare_file = Path("compare.toml").resolve()
    if not compare_file.is_file():
        # fall back: look in the same directory as this script
        compare_file = (Path(__file__).resolve().parent / "compare.toml").resolve()

assert compare_file.is_file(), f"compare.toml not found: {compare_file}"
COMPARE_DIR = compare_file.parent

with open(compare_file, "rb") as f:
    cfg = tomllib.load(f)

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
out_root_dir = (COMPARE_DIR / cfg["out_root_dir"]).resolve()
stat_file    = out_root_dir / "Stat.pickle"
assert stat_file.is_file(), (
    f"Stat.pickle not found: {stat_file}\n"
    "Run run_comparison.py first to generate it."
)

# Load tag codes from tagFile (via params.toml)
params_file = (COMPARE_DIR / cfg["params_file"]).resolve()
with open(params_file, "rb") as f:
    raw_params = tomllib.load(f)
params = {k: v[0] if isinstance(v, list) and len(v) == 1 else v
          for k, v in raw_params.items()}

tag_codes_file = (params_file.parent / params["tagFile"]).resolve()
tag_df         = pd.read_csv(tag_codes_file)
tag_codes_all  = list(tag_df["Code"])

# ---------------------------------------------------------------------------
# Report settings from compare.toml [report] section
# ---------------------------------------------------------------------------
report_cfg = cfg.get("report", {})
segments   = report_cfg.get("segments", ["All"])
methods    = report_cfg.get("methods",  list(cfg.get("methods", {}).keys()))

# Which tag IDs to report (empty = all)
tag_ids_cfg = cfg.get("tag_ids", [])
if tag_ids_cfg:
    tag_ids   = list(tag_ids_cfg)
    tag_codes = [tag_codes_all[t] for t in tag_ids]
else:
    tag_ids   = list(range(len(tag_codes_all)))
    tag_codes = tag_codes_all

# ---------------------------------------------------------------------------
# Load stats
# ---------------------------------------------------------------------------
with open(stat_file, "rb") as f:
    stat = pickle.load(f)

# ---------------------------------------------------------------------------
# Build summary table per segment × method
# ---------------------------------------------------------------------------
columns1 = ["Efficiency"] + sum(
    [[f"{d} Error (m)"] * 2 for d in ["Easting", "Northing", "Depth"]], []
)
columns2 = ["Efficiency"] + sum([["Median", "RMS"] for _ in range(3)], [])
multi_columns = pd.MultiIndex.from_arrays(
    [columns1, columns2], names=("", "")
)

for seg in segments:
    for method in methods:
        rows = []
        for tag_id in tag_ids:
            try:
                data = stat[tag_id][seg][method]
            except KeyError:
                print(f"  Warning: no data for tag {tag_id}, segment '{seg}', method '{method}' — skipping.")
                rows.append(np.full(7, np.nan))
                continue

            err = data["XYZ_error"]
            row = np.zeros(7)
            row[0]    = data["npoint"] / (data["npoint_ideal"] + 1e-10) * 100
            row[1::2] = np.nanmedian(np.abs(err), axis=0)
            row[2::2] = np.sqrt(np.nanmean(np.abs(err) ** 2, axis=0))
            rows.append(row)

        entry = np.stack(rows, axis=0)
        result_df = pd.DataFrame(entry, columns=multi_columns, index=tag_codes)

        # Format columns
        fmt_funcs = [lambda x: f"{x:.1f}%"] + [lambda x: f"{x:.2f}"] * 6
        for col, fmt in zip(result_df.columns, fmt_funcs):
            result_df[col] = result_df[col].map(fmt)

        out_csv = out_root_dir / f"report_{seg}_{method}.csv"
        result_df.to_csv(out_csv)
        print(f"Saved: {out_csv}")
        print(result_df.to_string())
        print()

print("Report complete.")

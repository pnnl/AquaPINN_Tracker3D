# -*- coding: utf-8 -*-
"""
run_comparison.py
=================
Entry point for the FishHeart post-processing comparison pipeline.

Loads compare.toml from the same directory, then calls compareGPS.py
(from the AquaPINN_Tracker3D PostProcess package) to generate plots and
compute accuracy metrics for all configured solvers.

Usage:
    python run_comparison.py              # uses compare.toml in this folder
    python run_comparison.py my.toml      # uses a custom compare.toml

After this script completes, run report.py to generate the summary CSV table.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent          # Examples/fishHeart/run/
PROJECT_DIR  = SCRIPT_DIR.parent                        # Examples/fishHeart/
REPO_ROOT      = PROJECT_DIR.parents[1]
TRACKER_ROOT   = REPO_ROOT / "AquaPINN_Tracker3D"
POSTPROCESS_DIR = TRACKER_ROOT / "PostProcess"

os.chdir(SCRIPT_DIR)

# Add the package root, inner package dir, and PostProcess directory to
# sys.path so that compareGPS.py can import its siblings (plots, utils,
# system) directly by name.
for p in [str(REPO_ROOT), str(POSTPROCESS_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------------------------------------------------------------------------
# Determine compare.toml path
# ---------------------------------------------------------------------------
compare_toml = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 \
               else SCRIPT_DIR / "compare.toml"

assert compare_toml.is_file(), f"compare.toml not found: {compare_toml}"

# ---------------------------------------------------------------------------
# Run compareGPS.py with the compare.toml as argument
# ---------------------------------------------------------------------------
# We exec compareGPS.py directly so it picks up sys.argv[1] as the config path.
sys.argv = ["compareGPS.py", str(compare_toml)]

compare_script = POSTPROCESS_DIR / "compareGPS.py"
with open(compare_script, "r") as f:
    code = f.read()

exec(compile(code, str(compare_script), "exec"), {"__name__": "__main__"})

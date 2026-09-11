# -*- coding: utf-8 -*-
"""
main_FishHeart.py
=================
Entry point for the FishHeart acoustic-tag tracking pipeline.

Raw detections are stored as ATS Sonic Receiver SR*.csv files in ../data/.
This script runs the standard AquaPINN pipeline:
  1. decoder.run()  — decode raw SR*.csv files into detection pickles
  2. syncer.run()   — time-sync hydrophone clocks
  3. tracker.run()  — NN / AML tracking
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve paths and change CWD to run/ so relative paths in params.toml work
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent          # Examples/fishHeart/run/
PROJECT_DIR  = SCRIPT_DIR.parent                        # Examples/fishHeart/
REPO_ROOT    = PROJECT_DIR.parents[1]
TRACKER_ROOT = REPO_ROOT / "AquaPINN_Tracker3D"

os.chdir(SCRIPT_DIR)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Import package components (mirrors Synthetic/run/main.py pattern)
# ---------------------------------------------------------------------------
PINN_PACKAGE       = importlib.import_module("AquaPINN_Tracker3D")
config             = importlib.import_module("AquaPINN_Tracker3D.config")
_functions         = importlib.import_module("AquaPINN_Tracker3D.SharedLibs.functions")

getLogger          = PINN_PACKAGE.getLogger
localizationSystem = PINN_PACKAGE.localizationSystem
decoder            = PINN_PACKAGE.decoder
syncer             = PINN_PACKAGE.syncer
tracker            = PINN_PACKAGE.tracker
load_params        = _functions.load_params

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
param_file = str(SCRIPT_DIR / "params.toml")

config.oneKey_configure(use_cuda=True, precision=64, seed=42)

train_params = load_params(param_file)[0]

TagStr = "Tag" + str(train_params["TagID"])
logger = getLogger(
    logFile=os.path.join(train_params["output_folder"], f"history_{TagStr}.log")
)

# ---------------------------------------------------------------------------
# Pipeline flags
# ---------------------------------------------------------------------------
doSync       = True   # a clean clone must create the synced phone CSV
trackBeacon  = False  # set True to also track beacon tags

# ---------------------------------------------------------------------------
# Deploy the localization system
# ---------------------------------------------------------------------------
loc_sys = localizationSystem().deploy(param_file, doSync=doSync)

# ---------------------------------------------------------------------------
# Decode raw SR*.csv files → detection pickles
# ---------------------------------------------------------------------------
decoder.run(param_file, loc_sys)

# ---------------------------------------------------------------------------
# Time-sync hydrophone clocks
# ---------------------------------------------------------------------------
if doSync:
    syncer.run(param_file, loc_sys)

# ---------------------------------------------------------------------------
# Run the tracker
# ---------------------------------------------------------------------------
tracker.run(param_file, loc_sys)

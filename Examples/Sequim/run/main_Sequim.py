# -*- coding: utf-8 -*-
"""
main_Sequim.py
==============
Entry point for the Sequim acoustic-tag tracking pipeline.

The raw detections are stored in a MATLAB .mat file (under ../data/).
This script:
  1. Locates the .mat file in the data folder defined in params.toml.
  2. Converts it to the synced-decodes pickle via convertDetections.py.
  3. Runs the NN tracker directly (no separate decode / sync steps needed).
"""

from __future__ import annotations

import glob
import importlib
import os
import sys
from pathlib import Path

from convertDetections import convert_mat_to_synced_pickle

# ---------------------------------------------------------------------------
# Resolve paths and change CWD to the run/ folder so that all relative paths
# in params.toml are resolved correctly (mirrors how pipeline.py is used).
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent          # Examples/Sequim/run/
PROJECT_DIR  = SCRIPT_DIR.parent                        # Examples/Sequim/
REPO_ROOT    = PROJECT_DIR.parents[1]
TRACKER_ROOT = REPO_ROOT / "AquaPINN_Tracker3D"

os.chdir(SCRIPT_DIR)   # all relative paths in params.toml now resolve from here

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Import package components (mirrors Synthetic/run/main.py)
# ---------------------------------------------------------------------------
PINN_PACKAGE       = importlib.import_module("AquaPINN_Tracker3D")
config             = importlib.import_module("AquaPINN_Tracker3D.config")
_functions         = importlib.import_module("AquaPINN_Tracker3D.SharedLibs.functions")

getLogger          = PINN_PACKAGE.getLogger
localizationSystem = PINN_PACKAGE.localizationSystem
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
# The TOA data under the data folder has already been synced and saved in the .mat file under ../data/.mat
# Pre-create the synced phone CSV that doSync=False requires.
# For Sequim there is no clock-sync step, so the original phone file IS the
# synced phone file.  We copy it into the output folder with the _synced suffix
# that system.deploy() expects.
# ---------------------------------------------------------------------------
import shutil

_phone_file    = train_params["phoneFile"]
_output_folder = train_params["output_folder"]
os.makedirs(_output_folder, exist_ok=True)

_phone_synced_name = os.path.splitext(os.path.basename(_phone_file))[0] + "_synced.csv"
_phone_synced_path = os.path.join(_output_folder, _phone_synced_name)
if not os.path.isfile(_phone_synced_path):
    shutil.copy(_phone_file, _phone_synced_path)

# ---------------------------------------------------------------------------
# Deploy the localization system
# doSync=False: the .mat conversion below produces the synced pickle directly
# ---------------------------------------------------------------------------
loc_sys = localizationSystem().deploy(param_file, doSync=False)

# ---------------------------------------------------------------------------
# Locate the .mat file in the data folder
# ---------------------------------------------------------------------------
data_folder = train_params["dataFolder"]
if not data_folder:
    data_folder = str(PROJECT_DIR / "data")
data_folder = os.path.abspath(data_folder)

mat_files = glob.glob(os.path.join(data_folder, "*.mat"))
assert len(mat_files) > 0, (
    f"No .mat file found in data folder: {data_folder}\n"
    "Please place the raw detection .mat file there."
)
if len(mat_files) > 1:
    logger.warning(f"Multiple .mat files found; using the first one: {mat_files[0]}")
mat_file = mat_files[0]

# ---------------------------------------------------------------------------
# Determine the synced-decodes pickle path that tracker.run() will look for
# ---------------------------------------------------------------------------
synced_pickle = loc_sys.getSyncedDecodesFile(
    train_params["tagFile"], "Tag", newFolder=train_params["output_folder"]
)

# ---------------------------------------------------------------------------
# Convert .mat → synced pickle, then run the tracker
# ---------------------------------------------------------------------------
convert_mat_to_synced_pickle(
    mat_file   = mat_file,
    out_pickle = synced_pickle,
    badPhones  = train_params.get("badPhones", []),
)

tracker.run(param_file, loc_sys)

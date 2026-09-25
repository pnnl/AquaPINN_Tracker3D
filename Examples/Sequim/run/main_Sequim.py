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

from convertDetections import convert_mat_to_synced_pickle, find_mat_tag_indices

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

# TagID = -1 means "track every tag in the tag file" (handled inside
# tracker.run, which reads the tag count from the synced-decodes pickle).
_track_all = int(train_params["TagID"]) == -1
TagStr = "AllTags" if _track_all else "Tag" + str(train_params["TagID"])
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
# Restrict processing to the tags listed in tagFile
# ---------------------------------------------------------------------------
# The raw .mat file holds every tag detected during the deployment (ten for
# this dataset), but config/DriftingTags.csv lists only the five analysed in
# the manuscript (Table 1, Salish Sea sea trial):
#
#   code        type    PRI (s)
#   ----------  ------  -------
#   G7270F55D   SS400        3
#   G726D7656   SS400        3
#   G726777EF   SSREF1       2
#   G72679FC4   SSREF2       3
#   G720EA94F   MSF          1
#
# The tagFile is the single source of truth: we look each code up in the .mat
# 'code' field rather than hard-coding positional indices, so editing the CSV
# is enough to change the tag set.  convert_mat_to_synced_pickle() keeps the
# tags in ascending .mat order, which is why the codes below are sorted by
# their resolved index before conversion -- that keeps pickle entry i aligned
# with tagFile row i, and hence with TagID i.
tag_codes = list(loc_sys.tagInfo["code"])

_mat_indices = find_mat_tag_indices(mat_file, tag_codes)
if _mat_indices != sorted(_mat_indices):
    raise ValueError(
        "tagFile rows are not in the same order as the .mat struct.\n"
        f"  tagFile order : {tag_codes}\n"
        f"  .mat indices  : {_mat_indices}\n"
        "Reorder config/DriftingTags.csv by ascending .mat index so that "
        "TagID stays aligned with the converted pickle."
    )

if _track_all:
    logger.info(
        "Sequim: TagID = -1 -> tracking all %d tag(s) from %s: %s",
        len(tag_codes), train_params["tagFile"], tag_codes,
    )
else:
    _tid = int(train_params["TagID"])
    assert 0 <= _tid < len(tag_codes), (
        f"TagID = {_tid} is out of range for a {len(tag_codes)}-tag tagFile "
        f"({train_params['tagFile']}). Valid values are 0..{len(tag_codes)-1}, "
        "or -1 to track all tags."
    )
    logger.info(
        "Sequim: TagID = %d -> tracking %s (1 of %d tags in %s)",
        _tid, tag_codes[_tid], len(tag_codes), train_params["tagFile"],
    )

# ---------------------------------------------------------------------------
# Convert .mat -> synced pickle, then run the tracker
# ---------------------------------------------------------------------------
convert_mat_to_synced_pickle(
    mat_file    = mat_file,
    out_pickle  = synced_pickle,
    badPhones   = train_params.get("badPhones", []),
    tagIndices  = _mat_indices,
)

tracker.run(param_file, loc_sys)

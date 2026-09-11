
# -*- coding: utf-8 -*-
"""
compareGPS.py — Standalone post-processing: compare solver tracks against GPS.

Usage:
    python compareGPS.py path/to/compare.toml

If no argument is given, looks for compare.toml in the current directory.

Configuration is split into two files:
    compare.toml  — PostProcess-specific settings (result folders, background
                    image, output directory). One per deployment comparison.
    params.toml   — Pipeline configuration (phoneFile, tagFile, GPSFile, etc.).
                    Referenced via the params_file key in compare.toml.

PRI values are read automatically from the tagFile CSV (no manual entry needed).
"""

import sys
import os
import pickle
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt

try:
    import tomllib           # Python >= 3.11 (stdlib)
except ImportError:
    import tomli as tomllib  # pip install tomli

from plots import plot_track_OneFigureEqual, plot_track_with_Background, plot_track_XYZ, getMetrics
from AquaPINN_Tracker3D import system as loc_system
from utils import resolve_path, find_result_csv, get_gps_file, get_pri_from_csv

# ── Load compare.toml ────────────────────────────────────────────────────────
compare_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("compare.toml")
compare_file = compare_file.resolve()
COMPARE_DIR  = compare_file.parent

with open(compare_file, 'rb') as f:
    cfg = tomllib.load(f)

# ── Load params.toml (pipeline config) ───────────────────────────────────────
params_file = (COMPARE_DIR / cfg['params_file']).resolve()
PARAMS_DIR  = params_file.parent

with open(params_file, 'rb') as f:
    raw_params = tomllib.load(f)

# Unwrap single-element lists (load_params() Cartesian-product convention)
params = {k: v[0] if isinstance(v, list) and len(v) == 1 else v
          for k, v in raw_params.items()}

def resolve_param(p):
    """Resolve a path from params.toml relative to its directory."""
    return resolve_path(PARAMS_DIR, p)

# ── Deploy receiver / tag / test info ────────────────────────────────────────
loc_sys   = loc_system.localizationSystem()
phoneInfo = loc_sys.deployPhone(resolve_param(params['phoneFile']))
tagInfo   = loc_sys.deployTag(resolve_param(params['tagFile']))
tagCodes  = tagInfo['code']
PRI_list  = tagInfo['PRI']   # read directly from tagFile CSV — no manual entry
print(f"Loaded {len(tagCodes)} tags  (PRI: {PRI_list})")

testFile = resolve_param(params.get('testFile', ''))
testInfo = loc_sys.deployTest(testFile) if testFile and os.path.isfile(testFile) else None

# ── GPS file ─────────────────────────────────────────────────────────────────
# Use gps_file_pattern from compare.toml if set; otherwise fall back to
# GPSFile from params.toml. Pattern may contain {tagID} or {tagCode}.
gps_pattern = cfg.get('gps_file_pattern', '') or resolve_param(params.get('GPSFile', ''))

# ── Output directory ──────────────────────────────────────────────────────────
out_root_dir = (COMPARE_DIR / cfg['out_root_dir']).resolve()
out_dir      = out_root_dir / 'plots'
os.makedirs(out_dir, exist_ok=True)

# ── Method → result folder mapping ───────────────────────────────────────────
methods_cfg = cfg.get('methods', {})

# ── Plot-type flags ───────────────────────────────────────────────────────────
plots_cfg    = cfg.get('plots', {})
do_map       = plots_cfg.get('map',     True)   # background-image overlay
do_compare   = plots_cfg.get('compare', True)   # all-methods-in-one-figure
do_xyz       = plots_cfg.get('xyz',     False)  # X/Y/Z time-series

# ── Background image config ───────────────────────────────────────────────────
_img_raw     = cfg.get('img_path', '') or ''
img_path     = str((COMPARE_DIR / _img_raw).resolve()) if _img_raw else None
if img_path and not os.path.isfile(img_path):
    print(f"Warning: background image not found: {img_path}")
    img_path = None
pixel2Phone1 = cfg.get('pixel2Phone1', None)
pixel2Phone2 = cfg.get('pixel2Phone2', None)
if pixel2Phone1:
    pixel2Phone1 = [tuple(pixel2Phone1['px']), pixel2Phone1['phone_id']]
if pixel2Phone2:
    pixel2Phone2 = [tuple(pixel2Phone2['px']), pixel2Phone2['phone_id']]

# ── Timing (from params.toml) ─────────────────────────────────────────────────
time_start = params.get('time_start', None)
time_end   = params.get('time_end',   None)

# ── Main comparison loop ──────────────────────────────────────────────────────
res_all = defaultdict(dict)
tag_ids = set(cfg.get('tag_ids', []))

for tagID, (tagCode, pri_fallback) in enumerate(zip(tagCodes, PRI_list)):
    if tag_ids and tagID not in tag_ids:
        continue
    print(f"\nTag {tagID}: {tagCode}  PRI={pri_fallback}s (from tagFile, may be overridden)")
    plt.close('all')

    # Collect result CSV per method (skip methods with no output).
    # Pass method= so files named *<tagCode>*<method>*.csv are matched correctly
    # when multiple methods share the same result folder.
    solutions_all_csv = {method: find_result_csv(folder, tagCode,
                                                  base_dir=COMPARE_DIR,
                                                  method=method)
                         for method, folder in methods_cfg.items()}
    found = [m for m, p in solutions_all_csv.items() if p is not None]
    if not found:
        print(f"  No results found for any method — skipping.")
        continue

    # Resolve PRI from the result CSV; fall back to tagFile value if not found
    PRI = get_pri_from_csv(solutions_all_csv, fallback_pri=pri_fallback)
    print(f"  Methods with results: {found}  (PRI resolved={PRI}s)")

    # GPS reference track
    track_GPS = None
    gps_file = get_gps_file(gps_pattern, tagID, tagCode)
    if gps_file:
        try:
            track_GPS = loc_sys.loadGPS(gps_file)
        except Exception as e:
            print(f"  Warning: could not load GPS ({gps_file}): {e}")

    # Build segment list from testInfo, always append an "All" segment
    if testInfo is None:
        segments = [('All', time_start, time_end)]
    else:
        segments = [(n, p[0], p[1]) for n, p in zip(testInfo['names'], testInfo['periods'])]
        if 'All' not in testInfo['names']:
            segments.append(('All', time_start, time_end))

    fout_tmpl = str(out_dir / 'hhhh.jpg')

    for seg_name, start, end in segments:
        fname = f"Tag{tagID}_{tagCode}_{seg_name}"

        # Background overlay plot (only when img_path and both registration
        # points are configured, and the 'map' plot flag is enabled)
        if do_map and img_path and pixel2Phone1 and pixel2Phone2:
            plot_track_with_Background(
                solutions_all_csv, phoneInfo,
                start_time=start, end_time=end,
                track_GPS=track_GPS, figTitle=None,
                fout=fout_tmpl.replace('hhhh', fname + '_map'), close=True,
                img_path=img_path,
                pixel2Phone1=pixel2Phone1, pixel2Phone2=pixel2Phone2,
                alpha=0.5,
            )

        # All-methods-in-one-figure comparison plot
        if do_compare:
            plot_track_OneFigureEqual(
                solutions_all_csv, phoneInfo,
                start_time=start, end_time=end,
                track_GPS=track_GPS, figTitle=None,
                fout=fout_tmpl.replace('hhhh', fname), close=True, PRI=PRI,
            )

        # X / Y / Z time-series plot (stacked, all solvers overlaid)
        if do_xyz:
            plot_track_XYZ(
                solutions_all_csv, phoneInfo,
                start_time=start, end_time=end,
                track_GPS=track_GPS, figTitle=None,
                fout=fout_tmpl.replace('hhhh', fname + '_XYZ'), close=True, PRI=PRI,
            )

        res_all[tagID][seg_name] = getMetrics(
            solutions_all_csv, phoneInfo,
            start_time=start, end_time=end, PRI=PRI,
        )

# ── Save stats ────────────────────────────────────────────────────────────────
stat_file = str(out_root_dir / 'Stat.pickle')
with open(stat_file, 'wb') as f:
    pickle.dump(res_all, f)
print(f"\nStats saved → {stat_file}")

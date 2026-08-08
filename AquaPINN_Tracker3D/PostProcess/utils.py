# -*- coding: utf-8 -*-
"""
utils.py — PostProcess helper functions shared across compareGPS.py and report.py.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd


def resolve_path(base_dir, p):
    """Resolve path *p* relative to *base_dir*; return '' if p is empty."""
    return str((Path(base_dir) / p).resolve()) if p else ""


def find_result_csv(folder, tagCode, base_dir=None, method=None):
    """Return the first CSV in *folder* whose name contains *tagCode*, or None.

    If *method* is given, the filename must also contain the method label
    (e.g. ``*G726777EF*AML*.csv``), allowing multiple methods to share the
    same result folder when files are named ``track_Tag3_<code>_<method>.csv``.

    *folder* may be absolute or relative to *base_dir*.
    """
    root = Path(folder)
    if not root.is_absolute() and base_dir is not None:
        root = (Path(base_dir) / root).resolve()
    if not root.exists():
        return None
    files = list(root.rglob(f"*{tagCode}*.csv"))
    if method:
        files = [f for f in files if method in f.name]
    return str(files[0]) if files else None


def get_gps_file(gps_pattern, tagID, tagCode):
    """Substitute {tagID}/{tagCode} in *gps_pattern* and return path if it exists.

    Returns None when gps_pattern is empty or the resolved path does not exist.
    """
    if not gps_pattern:
        return None
    path = gps_pattern.replace('{tagID}', str(tagID)).replace('{tagCode}', tagCode)
    return path if os.path.isfile(path) else None


def get_pri_from_csv(solutions_csv, fallback_pri):
    """Estimate PRI from the result CSV with the most tracked points.

    For a fair comparison across methods, all share the same PRI.  We pick
    the file with the most non-null 'datetime' rows (most tracked detections)
    and return the 20th percentile of positive inter-ping time differences —
    the same estimator used by get_PRI() in ML_NN.py.

    Falls back to *fallback_pri* (e.g. from tagFile) when no result CSV is
    available or contains too few points.
    """
    best_df   = None
    best_nrow = 0
    for csv_path in solutions_csv.values():
        if csv_path is None or not os.path.isfile(csv_path):
            continue
        try:
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.lower()
            if 'datetime' not in df.columns:
                continue
            nrow = df['datetime'].notna().sum()
            if nrow > best_nrow:
                best_nrow = nrow
                best_df   = df
        except Exception:
            continue
    if best_df is None or best_nrow < 2:
        return fallback_pri
    t  = pd.to_datetime(best_df['datetime'].dropna())
    dt = t.diff().dt.total_seconds().dropna().values
    dt = dt[dt > 0]
    if len(dt) == 0:
        return fallback_pri
    return float(np.percentile(dt, 20))

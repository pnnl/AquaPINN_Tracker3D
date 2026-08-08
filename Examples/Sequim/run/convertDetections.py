# -*- coding: utf-8 -*-
"""
convertDetections.py
====================
Utilities for converting raw MATLAB acoustic-tag detection files into the
synced-decodes pickle format expected by the AquaPINN tracker pipeline.
"""

from __future__ import annotations

import logging
import os
import pickle

import numpy as np
from scipy.io import loadmat

logger = logging.getLogger(__name__)


def loadMsg_Mat(msgMatFile: str, varName: str = "Tag",
                badPhones: list = [], tagIndices=None) -> list:
    """Load acoustic-tag detections from a MATLAB .mat file.

    Parameters
    ----------
    msgMatFile : str
        Path to the .mat file.
    varName : str
        Name of the MATLAB struct variable (default ``'Tag'``).
    badPhones : list
        Receiver IDs to exclude.
    tagIndices : list or None
        If given, only load these tag indices (0-based).

    Returns
    -------
    list of dict
        One dict per tag, each containing a ``'decodes'`` ndarray.
    """
    msgOri = loadmat(msgMatFile)
    logger.info(f"[loadMsg_Mat] .mat keys: {list(msgOri.keys())}")
    assert varName in msgOri.keys(), \
        f"Variable '{varName}' not found in {msgMatFile}"

    nTag = msgOri[varName].shape[1]
    msg  = [None] * nTag
    for TagID in range(nTag):
        if tagIndices is not None and TagID not in tagIndices:
            continue
        keys = msgOri[varName][0, 0].dtype.names
        msg[TagID] = {key: msgOri[varName][0][TagID][i]
                      for i, key in enumerate(keys)}
        decodes = msg[TagID]["decodes"]
        if decodes.ndim < 2 or decodes.shape[0] == 0 or decodes.shape[1] <= 3:
            msg[TagID]["decodes"] = np.empty((0, max(decodes.shape[1] if decodes.ndim == 2 else 0, 4)))
            continue
        pIDs = decodes[:, 3].astype(int)
        msg[TagID]["decodes"][:, 3] = pIDs
        ind_bad = np.zeros(len(pIDs), dtype=bool)
        for bp in badPhones:
            ind_bad |= (pIDs == bp)
        msg[TagID]["decodes"] = msg[TagID]["decodes"][~ind_bad]
    msg = [d for d in msg if d is not None]
    return msg


def convert_mat_to_synced_pickle(mat_file: str, out_pickle: str,
                                  badPhones: list = []) -> None:
    """Convert a .mat detection file to the synced-decodes pickle format.

    Time columns are merged into a single fractional-day timestamp
    (MATLAB datenum style) so the format matches what ``tracker.run()``
    expects from a synced decodes file.

    Parameters
    ----------
    mat_file : str
        Path to the source .mat file.
    out_pickle : str
        Destination pickle path (will be created including any missing dirs).
    badPhones : list
        Receiver IDs to exclude.
    """
    logger.info(f"Loading .mat file: {mat_file}")
    msg = loadMsg_Mat(mat_file, badPhones=badPhones)

    # col-0 = date (MATLAB datenum integer part)
    # col-1 = time-of-day in seconds  →  unified fractional-day number
    for i in range(len(msg)):
        if msg[i]["decodes"].shape[0] == 0:
            continue
        date     = msg[i]["decodes"][:, 0]
        time_sec = msg[i]["decodes"][:, 1]
        datetime_num = np.floor(date) + time_sec / 24 / 3600
        msg[i]["decodes"][:, 0] = datetime_num
        msg[i]["decodes"][:, 1] = datetime_num

    out_dir = os.path.dirname(out_pickle)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    with open(out_pickle, "wb") as f:
        pickle.dump(msg, f)
    logger.info(f"Synced decodes pickle saved to: {out_pickle}")

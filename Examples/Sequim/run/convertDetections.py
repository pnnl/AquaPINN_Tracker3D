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


def find_mat_tag_indices(msgMatFile: str, codes, varName: str = "Tag") -> list:
    """Resolve tag *codes* to their 0-based indices in a MATLAB .mat struct.

    The .mat struct carries a ``code`` field per tag, so the caller never has to
    hard-code positional indices.  Returned indices are in the same order as
    *codes*.

    Parameters
    ----------
    msgMatFile : str
        Path to the .mat file.
    codes : sequence of str
        Tag codes to locate (e.g. from the tagFile CSV).
    varName : str
        Name of the MATLAB struct variable (default ``'Tag'``).

    Returns
    -------
    list of int
        Index in the .mat struct for each requested code.

    Raises
    ------
    KeyError
        If any requested code is not present in the file.
    """
    msgOri = loadmat(msgMatFile)
    assert varName in msgOri.keys(), \
        f"Variable '{varName}' not found in {msgMatFile}"
    arr = msgOri[varName]
    keys = arr[0, 0].dtype.names
    if "code" not in keys:
        raise KeyError(
            f"'{varName}' struct in {msgMatFile} has no 'code' field "
            f"(fields: {keys}); cannot resolve tags by code."
        )
    icode = keys.index("code")

    code_to_idx = {}
    for i in range(arr.shape[1]):
        raw = arr[0][i][icode]
        flat = raw.flatten() if hasattr(raw, "flatten") else [raw]
        if len(flat) == 0:
            continue
        code_to_idx[str(flat[0]).strip()] = i

    missing = [c for c in codes if c not in code_to_idx]
    if missing:
        raise KeyError(
            f"Tag code(s) {missing} not found in {msgMatFile}. "
            f"Available: {sorted(code_to_idx)}"
        )
    resolved = [code_to_idx[c] for c in codes]
    logger.info(f"Resolved tag codes to .mat indices: {dict(zip(codes, resolved))}")
    return resolved


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
                                  badPhones: list = [],
                                  tagIndices=None) -> None:
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
    tagIndices : list or None
        Original 0-based tag indices in the .mat struct to keep.  When given,
        the output pickle contains only those tags, **in ascending index
        order**, so that entry *i* of the pickle corresponds to row *i* of the
        (correspondingly subset) tagFile.  Use this to restrict processing to a
        published subset of tags without renumbering the source .mat file.
    """
    logger.info(f"Loading .mat file: {mat_file}")
    msg = loadMsg_Mat(mat_file, badPhones=badPhones, tagIndices=tagIndices)

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

# Stage 1: Raw Data Processing (`RawDataProcess/`)

## Overview

This stage reads raw ATS Sonic Receiver SR\*.csv decode files, performs multipath
filtering, converts timestamps to the internal MATLAB-compatible datenum format, maps
receiver serial numbers to Phone IDs, and assembles the unified **Tag detection
dictionary** consumed by all downstream stages.

---

## Module Map

| Module | Role |
|--------|------|
| `decoder.py` | Entry-point orchestrator; dispatches ingest for fish tags and beacons |
| `read_recevier_file_csv.py` | Low-level fast CSV reader with binary-search time indexing |
| `receiver_dataprep.py` | Per-tag/receiver data preparation, multipath removal, assembly |

---

## Algorithm Details

### 1. `decoder.py` — Pipeline Orchestrator

`run(param_file, loc_sys)` performs the following steps:

1. Loads JSON parameter file via `load_params()`.
2. Converts `time_start` / `time_end` strings from local time to UTC using the
   deployment timezone offset stored in `loc_sys.timeZone_local`.
3. Calls `compile_tagdata2dict()` once for **fish tags** and once for **beacons**
   (beacon codes come from `phoneInfo['AttachedBeacons']`).
4. Each output is cached as a pickle file; if the file already exists it is skipped
   (incremental processing).
5. Calls `showDetections()` to generate detection-count visualisation plots.
6. Calls `cleanCache()` to release file-level in-memory caches.

**Key design decision:** beacons and fish tags are treated identically by ingest; their
distinction is carried by the caller (`loc_sys`) and preserved through to later stages.

---

### 2. `read_recevier_file_csv.py` — Fast CSV Reader

#### 2a. Binary Search Phase — `lower_bound_offset(filepath, target_key)`

Seeks the byte offset where the data time `>= target_key` without reading the whole file:

1. Get file size `fsize`.
2. Binary-search the byte range `[lo, hi]`:
   - Read a probe line near the midpoint.
   - Parse its datetime key ("YYYYMMDD HH:MM:SS.ffffff" lex-sortable string) via
     `probe_next_data_key()`.
   - If `key < target`: set `lo = mid + 1`, else `hi = mid`.
3. Return the byte offset of the first line with time `>= target`.

This reduces I/O from O(N) to O(log N) for large files.

#### 2b. Streaming Phase — `read_data_and_extract_SN_binary(filepath, start_time, end_time)`

Starting from the byte offset returned above:

1. Read lines sequentially using Python's `csv.reader` (handles quoted commas).
2. Strip NUL bytes (`\x00`) to handle occasional file corruption.
3. Parse fixed columns: `DateTime`, `TagCode`, `SigStr`, `Serial Number`.
4. Detect timezone from file header metadata (e.g., `"-8z"` → UTC offset −8 h).
5. Stop when the parsed datetime exceeds `end_time`.
6. Return a `pandas.DataFrame` with columns: `TagCode`, `DateTime`, `SigStr`,
   `NodeCode`, `timeZone`.

---

### 3. `receiver_dataprep.py` — Detection Preparation

#### 3a. `dtime2dnum(dt_str, timeZone)` — DateTime → MATLAB Datenum

Converts a datetime string of format `"MM/DD/YYYY HH:MM:SS[.ffffff]"` to a MATLAB
datenum (days since year 0, day 1):

```
unix_sec   = datetime.strptime(dt_str).timestamp()
datenum    = unix_sec / 86400 + 719529 - timeZone / 24
```

- `719529` is the number of days between MATLAB's epoch (Jan 0, year 0) and the UNIX
  epoch (Jan 1, 1970).
- The timezone offset `timeZone / 24` converts local time back to UTC.

**Why datenum?** Downstream stages (and the companion MATLAB pipeline) use MATLAB
datenums for absolute time arithmetic. Maintaining the same format ensures numerical
compatibility.

#### 3b. `filter_by_tdiff(df, tol=0.3)` — Multipath / Duplicate Removal

Removes acoustic multipath echoes, which appear as duplicate detections of the same tag
within a short time window at the same receiver:

1. Sort detections by `DateTime` within each receiver.
2. Compute inter-detection time differences `dt`.
3. Flag rows where `dt < tol` (default 0.3 s) as duplicates of the preceding detection.
4. Return boolean index of rows to **keep** (first occurrence is kept).

This mirrors `rmDupDecodes_parallel.m` in the MATLAB pipeline. The 0.3 s tolerance is
chosen to be shorter than the minimum inter-node acoustic propagation time but longer
than typical multipath delays (< 10 ms at < 3 m reflector).

#### 3c. `process_all_csv_files(datafolder, time_start_UTC, time_end_UTC)` — Batch Reader

1. Walks `datafolder` recursively for `SR*.csv` files.
2. For each file, calls `read_data_and_extract_SN_binary()` to extract detections within
   the time window.
3. Groups detections by receiver serial number (`NodeCode`).
4. Caches results in an in-memory `__cache` dict keyed by file path.

#### 3d. `compile_tagdata2dict(loc_sys, tagInfo, datafolder, ...)` — Main Assembly

Produces the final detection dictionary consumed by downstream stages:

```
output list:  [ {decodes: N×4 ndarray}, ... ]
              one element per tag (beacons first, then fish tags)
```

Each `decodes` array has four columns (fixed layout, never changes):

| Column | Content |
|--------|---------|
| 0 | MATLAB datenum (`floor` value = day number) |
| 1 | Seconds since midnight |
| 2 | Signal strength (dB) |
| 3 | Receiver Phone ID (after SN → ID mapping; see below) |

Steps:

1. Call `process_all_csv_files()` to get all detections in the time window.
2. For each tag code:
   - Collect detections from every receiver.
   - Convert datetimes to datenums via `dtime2dnum()`.
   - Apply `filter_by_tdiff()` to remove multipaths.
   - Stack into `N×4` array.
3. Map `NodeCode` (serial number) → `Phone ID` using `loc_sys.SN2ID` (built from the
   receiver locations CSV by `localizationSystem.deployPhone()`).
4. Save as pickle file.

---

## Data Flow

```
SR*.csv files (per receiver)
        │
        ▼   lower_bound_offset()   [binary search to start time]
read_recevier_file_csv
        │
        ▼   read_data_and_extract_SN_binary()
DataFrame [TagCode, DateTime, SigStr, NodeCode, timeZone]
        │
        ▼   process_all_csv_files()  [group by receiver SN, cache]
        │
        ▼   compile_tagdata2dict()
              ├─ dtime2dnum()          → MATLAB datenum
              ├─ filter_by_tdiff()     → multipath removed
              └─ SN → Phone ID map     → decodes[:,3] = Phone ID
        │
        ▼
pickle: [{decodes: N×4}, ...]   (one per tag; beacons first)
```

---

## Key Constants and Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tol` in `filter_by_tdiff` | 0.3 s | Multipath rejection window |
| MATLAB epoch offset | 719529 days | Days from MATLAB year-0 to UNIX epoch |
| DateTime column index | auto-detected from header | Varies per SR file version |

---

## Output Format

Each tag's detection array (`decodes`) is a NumPy `ndarray` of shape `(N, 4)`:

```
decodes[i] = [datenum_float, sec_since_midnight, signal_strength_dB, phone_id]
```

Absolute time in seconds: `floor(datenum) * 86400 + sec_since_midnight`

This layout is **fixed** and shared with the MATLAB pipeline; downstream stages in both
languages depend on it.

---

## Comparison with MATLAB Equivalent

| Python (`RawDataProcess`) | MATLAB (`read_ATS_SR_data.m`) |
|---------------------------|-------------------------------|
| Binary-search CSV seek | Sequential CSV read |
| `filter_by_tdiff` (0.3 s) | `rmDupDecodes_parallel` (0.3 s) |
| `dtime2dnum` | Built-in `datenum()` |
| Pickle output | `Tag` struct array in workspace |
| Timezone from file header | External parameter |

import os
import csv
import pandas as pd
from ..SharedLibs.decorators import timing
import logging
logger = logging.getLogger(__name__)


# default header 
# Internal, SiteName,,, DateTime, TagCode, Tilt, VBatt, Temp, Pressure, SigStr, Bit Period, Threshold    
# data rows are assumed to be sorted globally by DateTime

default_datetime_ind = 4
default_tagcode_ind = 5
default_sigstr_ind = 10
default_columns_min = 11

# ---------- Helpers ----------
def dt_key(dt: str) -> str:
    """
    Convert "MM/DD/YYYY HH:MM:SS[.ffffff]" -> "YYYYMMDD HH:MM:SS[.ffffff]" (lex-sortable).
    Works for your globally-sorted-by-DateTime file without calling strptime.
    """
    dt = dt.strip()
    # Expect at least "MM/DD/YYYY HH:MM:SS"
    #            0123456789 12345678
    # Example:   03/25/2025 17:10:56.399069
    if len(dt) < 19:
        return ""
    # YYYY + MM + DD + rest (time + optional fraction)
    return dt[6:10] + dt[0:2] + dt[3:5] + dt[11:]


def try_parse_row(line: bytes, dt_i=default_datetime_ind):
    """Parse a single CSV line safely (handles quoted commas). Skips blanks."""
    if not line or not line.strip():
        return None
    # remove NULs on the fly
    if b"\x00" in line:
        line = line.replace(b"\x00", b"")
    text = line.decode("utf-8", errors="ignore").strip()
    if not text:
        return None
    try:
        elements= next(csv.reader([text]))
        assert len(elements) >= default_columns_min
        assert len(elements[dt_i])>=19  # valid DateTime, example: "03/25/2025 17:10:56.399069"
        return elements
    except Exception:
        return None


def find_serial_and_tz(ifile, max_lines=5000):
    """Scan a small prefix to extract Serial Number and timeZone (fast, one-time)."""
    serial_number = None
    timeZone = None
    with open(ifile, "rb") as f:
        for _ in range(max_lines):
            line = f.readline()
            if not line:
                break
            if b"\x00" in line:
                line = line.replace(b"\x00", b"")
            txt = line.decode("utf-8", errors="ignore").strip()
            if not txt:
                continue

            if serial_number is None and "Serial Number" in txt and ":" in txt:
                try:
                    serial_str = txt.split(":", 1)[1].strip()
                    serial_str = serial_str.split(",", 1)[0].strip()
                    serial_number = int(serial_str)
                except Exception:
                    pass

            if timeZone is None and ("File Start" in txt or "Log Resume" in txt):
                for token in txt.split():
                    if token.endswith("z") and token[:-1].lstrip("-").isdigit():
                        timeZone = int(token[:-1])
                        break

            if serial_number is not None and timeZone is not None:
                break
            
    logger.debug(f"Serial Number: {serial_number}, TimeZone: {timeZone}")
    if serial_number is None:
        logger.warning(f"Serial Number not found in the first {max_lines} lines of {ifile}. Defaulting to -1.")
    if timeZone is None:
        logger.warning(f"TimeZone not found in the first {max_lines} lines of {ifile}. Defaulting to 0.")
    return (serial_number if serial_number is not None else -1,
            timeZone if timeZone is not None else 0)


# ---------- Binary search core ----------
def probe_next_data_key(f, dt_i, max_scan_lines=500):
    """
        From current file position, scan forward up to max_scan_lines to find
        a data row with a valid DateTime -> returns (key, line_start_pos).
    Returns None if not found.
    """
    for _ in range(max_scan_lines):
        line_start = f.tell()
        line = f.readline()
        if not line:
            return None

        row = try_parse_row(line, dt_i=dt_i)
        if row is None:
            continue

        dt_str = row[dt_i].strip()
        k = dt_key(dt_str)
        if not k:
            continue
        return (k, line_start)
    return None


def lower_bound_offset(ifile, start_key, dt_i):
    """
    Find byte offset of the first data row with DateTime_key >= start_key.
    Handles non-data rows + blank lines + NUL bytes.
    """
    size = os.path.getsize(ifile)

    with open(ifile, "rb") as f:
        lo, hi = 0, size
        iteration = 0
        max_iterations = 100  # Safety limit to prevent infinite loops
        
        while lo < hi and iteration < max_iterations:
            iteration += 1
            
            mid = (lo + hi) // 2
            
            # If the range is too small, just use lo
            if hi - lo < 100:
                break
            
            f.seek(mid)
            f.readline()  # discard partial line to align
            probe_start_pos = f.tell()

            res = probe_next_data_key(f, dt_i)
            if res is None:
                # No valid data found in this region, search in the lower half
                hi = mid
                continue

            k, line_start = res

            if k < start_key:
                # Move lo forward, but ensure progress
                new_lo = max(f.tell(), probe_start_pos, lo + 1)
                lo = new_lo
            else:
                # Found a key >= start_key, narrow to this position
                hi = line_start     
        return lo


# ---------- Public function: binary search + stream window ----------
@timing
def read_data_and_extract_SN_binary(ifile, time_start_UTC=None, time_end_UTC=None,
                                    dt_i=None, tag_i=None, sig_i=None):
    """
    Fastest for huge globally-sorted CSV:
      1) binary search to jump near start_time
      2) stream forward until end_time
    3) skips non-data rows by validity checks (fixed column indices)
      4) uses csv.reader per line (safe with quoted commas, assumes no multiline quoted fields)
    Returns a DataFrame with TagCode, DateTime, SigStr, NodeCode, timeZone
    """
    dt_i = default_datetime_ind if dt_i is None else dt_i
    tag_i = default_tagcode_ind if tag_i is None else tag_i
    sig_i = default_sigstr_ind if sig_i is None else sig_i
    if dt_i is None or tag_i is None or sig_i is None:
        raise ValueError("Provide dt_i/tag_i/sig_i (or set DEFAULT_*_COL_IDX constants)")

    
    time_start_UTC = time_start_UTC or pd.to_datetime("01/01/1970 00:00:00")
    time_end_UTC = time_end_UTC or pd.to_datetime("12/31/2222 00:00:00")

    serial_number, timeZone = find_serial_and_tz(ifile)
    
    time_start_local = time_start_UTC + pd.Timedelta(hours=timeZone)
    time_end_local = time_end_UTC + pd.Timedelta(hours=timeZone)
    
    time_start_local_str = time_start_local.strftime('%m/%d/%Y %H:%M:%S')
    time_end_local_str = time_end_local.strftime('%m/%d/%Y %H:%M:%S')
    
    start_k = dt_key(time_start_local_str) 
    end_k   = dt_key(time_end_local_str) 

    # Find approximate start position
    off = lower_bound_offset(ifile, start_k, dt_i)
    TagCode_list, DateTime_list, SigStr_list = [], [], []

    with open(ifile, "rb") as f:
        f.seek(off)
        # Stream forward collecting rows in [start_k, end_k]
        append_tag = TagCode_list.append
        append_dt = DateTime_list.append
        append_sig = SigStr_list.append

        while True:
            line = f.readline()
            if not line:
                break

            row = try_parse_row(line, dt_i=dt_i)
            if row is None:
                continue


            dt_str = row[dt_i].strip()

            k = dt_str[6:10] + dt_str[0:2] + dt_str[3:5] + dt_str[11:]
            if k > end_k:
                break
            if k < start_k:
                continue
            append_tag(row[tag_i].strip())
            append_dt(dt_str)
            append_sig(row[sig_i].strip())
    df = pd.DataFrame({"TagCode": TagCode_list, "DateTime": DateTime_list, "SigStr": SigStr_list})
    df["NodeCode"] = serial_number
    df["timeZone"] = timeZone 
    return df

if __name__ == "__main__":
    # test time cost for files of different sizes
    root = r'\\pnl\projects\JSATS_solver_dev\NN_testing\fishHeart'
    csv_files = [os.path.join(root, "data", "SR24009_240321_093846.csv"),
                 r'\\pnl\projects\JSATS_solver_dev\datasets\LGS_2025\data\S2503201.CSV',
                 os.path.join(root, "data", "SR24014_240321_102313.csv")][2:3]
    
    fun_names = ["binary", 'binary_start_end']
    fun_descriptions = ["Binary method", "Binary method with reading start and end date"]
                 
    import time             
    for csv_file in csv_files:
        fileSize = os.path.getsize(csv_file)
        print(f"Processing file: {csv_file} (Size: {fileSize/1024/1024:.2f} MB)")
        for i_fun in range(1,2):
            print(f"Processing {csv_file} using {fun_descriptions[i_fun]}")
            import tempfile
            import shutil
            temp_dir = tempfile.mkdtemp(prefix='bench_')
            csv_file_tmp = os.path.join(temp_dir, os.path.basename(csv_file))
            shutil.copy(csv_file, csv_file_tmp)
            print(f"Temporary file created at: {csv_file_tmp}")
            start_time = time.time()
            if i_fun == 0:
                df = read_data_and_extract_SN_binary(csv_file_tmp)
            else:                
                df = read_data_and_extract_SN_binary(csv_file_tmp,
                                                     time_start_UTC=pd.to_datetime('03/25/2024 06:00:00'),
                                                     time_end_UTC=pd.to_datetime('03/25/2024 07:00:00'))
            end_time = time.time()
            print(df.shape)
            print(f"Time taken for {fun_descriptions[i_fun]}: {end_time - start_time} seconds")
            if temp_dir is not None:
                shutil.rmtree(temp_dir)
    
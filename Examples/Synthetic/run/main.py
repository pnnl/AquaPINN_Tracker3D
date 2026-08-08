# -*- coding: utf-8 -*-
"""Run PINN tracking on generated Synthetic data."""

from __future__ import annotations

import csv
import importlib
import json
import shutil
import sys
from pathlib import Path

# tomllib is stdlib in Python ≥ 3.11; fall back to the tomli third-party package.
try:
    import tomllib  # type: ignore[import]
except ImportError:
    try:
        import tomli as tomllib  # type: ignore[import,no-redef]
    except ImportError as exc:
        raise ImportError(
            "A TOML library is required. "
            "Install tomli (`pip install tomli`) or use Python ≥ 3.11."
        ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATASET_DIRS = [PROJECT_DIR / "data" / "Critical", PROJECT_DIR / "data" / "CRLB"][1:2]
PHONE_FILE = PROJECT_DIR / "dataGenerator" / "Cabot_Station_hydrophone_configuration_update_3.csv"
PARAM_TEMPLATE_FILE = SCRIPT_DIR / "params.toml"


# Use the in-repo tracker package for synthetic runs.
REPO_ROOT = PROJECT_DIR.parents[1]
TRACKER_ROOT = REPO_ROOT / "AquaPINN_Tracker3D"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


PINN_PACKAGE = importlib.import_module("AquaPINN_Tracker3D")
config = importlib.import_module("AquaPINN_Tracker3D.config")
getLogger = PINN_PACKAGE.getLogger
localizationSystem = PINN_PACKAGE.localizationSystem
tracker = PINN_PACKAGE.tracker


config.oneKey_configure(use_cuda=True, precision=64, seed=42)


def load_modes(dataset_dir: Path) -> list[str]:
    scenario_file = dataset_dir / "Scenarios.pickle"
    if scenario_file.is_file():
        import pickle

        with scenario_file.open("rb") as handle:
            scenarios = pickle.load(handle)
        modes: list[str] = []
        for value in scenarios.values():
            if value is None:
                continue
            for mode in value:
                if (dataset_dir / f"{mode}.pickle").is_file():
                    modes.append(mode)
        if modes:
            return modes
    return sorted(path.stem for path in dataset_dir.glob("*.pickle") if path.name != "Scenarios.pickle")


def collect_jobs() -> list[dict[str, str]]:
    jobs: list[dict[str, str]] = []
    for dataset_dir in DATASET_DIRS:
        print(dataset_dir, dataset_dir.is_dir())
        if not dataset_dir.is_dir():
            continue
        for mode in load_modes(dataset_dir):
            decode_file = dataset_dir / f"{mode}.pickle"
            gps_file = dataset_dir / f"{mode}_GPS.csv"
            if not decode_file.is_file():
                continue
            jobs.append(
                {
                    "dataset": dataset_dir.name,
                    "mode": mode,
                    "decode_file": str(decode_file),
                    "gps_file": str(gps_file) if gps_file.is_file() else "",
                }
            )
        print(jobs)
    return jobs


def write_tag_file(tag_file: Path, tag_code: str, pri: float) -> None:
    with tag_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Code", "PRI"])
        writer.writerow([tag_code, pri])


def build_params(
    output_dir: Path,
    tag_file: Path,
    gps_file: str,
    n_samples: int = 1,
) -> Path:
    # Load defaults from params.toml (all values are lists per TOML convention).
    with PARAM_TEMPLATE_FILE.open("rb") as handle:
        params = tomllib.load(handle)

    # Override per-job fields.
    params["output_folder"] = [str(output_dir)]
    params["tagFile"] = [str(tag_file)]
    params["GPSFile"] = [gps_file if gps_file else ""]
    params["trackMethod"] = ["NN"]
    params["phoneFile"] = [str(PHONE_FILE)]
    params["nSamples"] = [n_samples]

    params_file = output_dir / "synthetic_params.json"
    with params_file.open("w", encoding="utf-8") as handle:
        json.dump(params, handle, indent=2)
    return params_file


def delete_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def stage_tracker_inputs(
    output_dir: Path,
    decode_file: Path,
    gps_file: str,
    n_samples: int = 1,
) -> tuple[Path, list[Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stage the files the tracker expects in each output folder.
    tag_file = output_dir / "synthetic_tags.csv"
    write_tag_file(tag_file, tag_code="SYNTHETIC_TAG", pri=2.0)

    synced_phone = output_dir / f"{PHONE_FILE.stem}_synced.csv"
    shutil.copy2(PHONE_FILE, synced_phone)

    synced_decodes = output_dir / f"{tag_file.stem}_TagDecodes_synced.pickle"
    shutil.copy2(decode_file, synced_decodes)

    params_file = build_params(output_dir, tag_file, gps_file, n_samples=n_samples)
    staged_files = [
        tag_file,
        synced_phone,
        synced_decodes,
        synced_decodes.with_name(synced_decodes.stem + "_TOA.pickle"),
        synced_decodes.with_name(synced_decodes.stem + "_TOA.pickle.csv"),
        params_file,
    ]
    return params_file, staged_files


# nSamples values to run for every job: first a quick single-sample run,
# then a full 10-sample UQ run.  Results go into separate subdirectories so
# neither run overwrites the other.
N_SAMPLES_SCHEDULE: list[int] = [1, 10]


def run_job(job: dict[str, str], output_root: Path) -> None:
    for n_samples in N_SAMPLES_SCHEDULE:
        # Each nSamples value gets its own subdirectory to avoid collisions.
        output_dir = output_root / job["dataset"] / job["mode"]
        params_file, staged_files = stage_tracker_inputs(
            output_dir=output_dir,
            decode_file=Path(job["decode_file"]),
            gps_file=job["gps_file"],
            n_samples=n_samples,
        )

        logger = getLogger(logFile=str(output_dir / "history.log"))
        logger.info("=" * 60)
        logger.info("Synthetic tracking run")
        logger.info(f"dataset      : {job['dataset']}")
        logger.info(f"mode         : {job['mode']}")
        logger.info(f"nSamples     : {n_samples}")
        logger.info(f"decode_file  : {job['decode_file']}")
        logger.info(f"gps_file     : {job['gps_file'] or 'None'}")
        logger.info(f"output_dir   : {output_dir}")
        logger.info("track_method : NN")
        logger.info("=" * 60)

        try:
            loc_sys = localizationSystem().deploy(str(params_file), doSync=False)
            tracker.run(str(params_file), loc_sys)
        finally:
            # Remove only staged helper files, not the tracking results.
            for staged_file in staged_files:
                delete_if_exists(staged_file)


def main(groupID, groupSize) -> int:
    output_root = (SCRIPT_DIR / "results").resolve()
    jobs = collect_jobs()

    if not jobs:
        print("No matching jobs found.", file=sys.stderr)
        return 1

    for id_job, job in enumerate(jobs):
        group_id = id_job % groupSize
        if group_id != groupID:
            continue
        run_job(job, output_root=output_root)
    return 0


if __name__ == "__main__":
    # read groupID and groupSize from command line arguments with parse_args
    import argparse
    parser = argparse.ArgumentParser(description="Run PINN tracking on generated Synthetic data.")
    parser.add_argument("--group-id", type=int, default=0, help="ID of the job group to run (0-based, default: 0).")
    parser.add_argument("--group-size", type=int, default=1, help="Total number of job groups (default: 1).")
    args = parser.parse_args()
    main(args.group_id, args.group_size)

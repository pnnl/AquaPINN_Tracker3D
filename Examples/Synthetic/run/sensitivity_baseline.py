# -*- coding: utf-8 -*-
"""Run baseline-only sensitivity sweeps for the AquaPINN synthetic tracker."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import shutil
import sys
from pathlib import Path
from typing import TypedDict

try:
    import tomllib  # type: ignore[import]
except ImportError:
    import tomli as tomllib  # type: ignore[import,no-redef]


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
REPO_ROOT = PROJECT_DIR.parents[1]
TRACKER_ROOT = REPO_ROOT / "AquaPINN_Tracker3D"
SENSITIVITY_DIR = PROJECT_DIR / "data" / "Sensitivity"
PHONE_FILE = PROJECT_DIR / "dataGenerator" / "Cabot_Station_hydrophone_configuration_update_3.csv"
PARAM_TEMPLATE_FILE = SCRIPT_DIR / "params.toml"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PINN_PACKAGE = importlib.import_module("AquaPINN_Tracker3D")
config = importlib.import_module("AquaPINN_Tracker3D.config")
getLogger = PINN_PACKAGE.getLogger
localizationSystem = PINN_PACKAGE.localizationSystem
tracker = PINN_PACKAGE.tracker


class SweepSpec(TypedDict):
    values: list[float | int]


class SolverSpec(TypedDict):
    name: str
    n_samples: int


class DatasetSpec(TypedDict):
    decode_file: str
    gps_file: str


SENSITIVITY_SWEEPS: dict[str, SweepSpec] = {
    "period_factor":        {"values": [75.0, 150.0, 300.0, 600.0, 1200.0, 2400.0, 4800.0, 9600.0, 19200.0]},
    "distance_threshold":   {"values": [0.75, 1.5, 3.0, 6.0, 12.0, 24.0, 48.0]},
    "std_deviation_factor": {"values": [1.25, 2.5, 5.0, 10.0, 20.0]},
    "time_shift_factor":    {"values": [0.01875, 0.0375, 0.075, 0.15, 0.3, 0.6, 1.2]},
    "uq_std":               {"values": [0.125, 0.25, 0.5, 1.0, 2.0, 4.0]},
}


SOLVER_SPECS: list[SolverSpec] = [
    {"name": "NN", "n_samples": 1},
    {"name": "NNplus", "n_samples": 10},
]

DATASET_CASES: dict[str, DatasetSpec] = {
    "baseline": {
        "decode_file": str(SENSITIVITY_DIR / "baseline.pickle"),
        "gps_file": str(SENSITIVITY_DIR / "baseline_GPS.csv"),
    },
    "baseline_outliers": {
        "decode_file": str(SENSITIVITY_DIR / "baseline_outliers.pickle"),
        "gps_file": str(SENSITIVITY_DIR / "baseline_outliers_GPS.csv"),
    },
}


def configure_runtime() -> None:
    config.oneKey_configure(use_cuda=True, precision=64, seed=42)


def write_tag_file(tag_file: Path, tag_code: str, pri: float) -> None:
    with tag_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Code", "PRI"])
        writer.writerow([tag_code, pri])


def delete_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def format_value_label(value: float | int) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def build_params(
    output_dir: Path,
    tag_file: Path,
    gps_file: Path,
    n_samples: int,
    overrides: dict[str, float | int],
) -> Path:
    with PARAM_TEMPLATE_FILE.open("rb") as handle:
        params = tomllib.load(handle)

    params["output_folder"] = [str(output_dir)]
    params["tagFile"] = [str(tag_file)]
    params["GPSFile"] = [str(gps_file)]
    params["trackMethod"] = ["NN"]
    params["phoneFile"] = [str(PHONE_FILE)]
    params["nSamples"] = [n_samples]
    params["sampleMethod"] = ["fixed distance"]
    params["uq_std"] = [0.5]

    for key, value in overrides.items():
        params[key] = [value]

    params_file = output_dir / "sensitivity_params.json"
    with params_file.open("w", encoding="utf-8") as handle:
        json.dump(params, handle, indent=2)
    return params_file


def stage_tracker_inputs(
    output_dir: Path,
    decode_file: Path,
    gps_file: Path,
    n_samples: int,
    overrides: dict[str, float | int],
) -> tuple[Path, list[Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)

    tag_file = output_dir / "synthetic_tags.csv"
    write_tag_file(tag_file, tag_code="SYNTHETIC_TAG", pri=2.0)

    synced_phone = output_dir / f"{PHONE_FILE.stem}_synced.csv"
    shutil.copy2(PHONE_FILE, synced_phone)

    synced_decodes = output_dir / f"{tag_file.stem}_TagDecodes_synced.pickle"
    shutil.copy2(decode_file, synced_decodes)

    params_file = build_params(output_dir, tag_file, gps_file, n_samples, overrides)
    staged_files = [
        tag_file,
        synced_phone,
        synced_decodes,
        synced_decodes.with_name(synced_decodes.stem + "_TOA.pickle"),
        synced_decodes.with_name(synced_decodes.stem + "_TOA.pickle.csv"),
        params_file,
    ]
    return params_file, staged_files


def locate_track_csv(output_dir: Path) -> Path:
    candidates = sorted(output_dir.glob("Track/track/*.csv"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected exactly one track CSV in {output_dir}, found {len(candidates)}")
    return candidates[0]


def run_single_case(
    dataset_name: str,
    decode_file: Path,
    gps_file: Path,
    parameter: str,
    value: float | int,
    solver_name: str,
    n_samples: int,
    output_root: Path,
) -> None:
    value_label = format_value_label(value)
    output_dir = output_root / dataset_name / solver_name / parameter / f"value_{value_label}"
    overrides = {parameter: value}

    params_file, staged_files = stage_tracker_inputs(
        output_dir=output_dir,
        decode_file=decode_file,
        gps_file=gps_file,
        n_samples=n_samples,
        overrides=overrides,
    )

    logger = getLogger(logFile=str(output_dir / "history.log"))
    logger.info("=" * 60)
    logger.info("Baseline sensitivity run")
    logger.info(f"dataset      : {dataset_name}")
    logger.info(f"solver       : {solver_name}")
    logger.info(f"parameter    : {parameter}")
    logger.info(f"value        : {value}")
    logger.info(f"nSamples     : {n_samples}")
    logger.info(f"output_dir   : {output_dir}")
    logger.info("=" * 60)

    try:
        configure_runtime()
        loc_sys = localizationSystem().deploy(str(params_file), doSync=False)
        tracker.run(str(params_file), loc_sys)
        track_csv_path = locate_track_csv(output_dir)
        logger.info(f"track_csv     : {track_csv_path}")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Sensitivity run failed")
        raise RuntimeError(
            f"Sensitivity run failed for {dataset_name}, {solver_name}, {parameter}={value}"
        ) from exc
    finally:
        for staged_file in staged_files:
            delete_if_exists(staged_file)


def get_requested_parameters(selected: list[str]) -> list[str]:
    if not selected or selected == ["all"]:
        return list(SENSITIVITY_SWEEPS)
    return selected


def get_requested_datasets(selected: list[str]) -> list[str]:
    if not selected or selected == ["all"]:
        return list(DATASET_CASES)
    return selected


def build_cases(parameters: list[str], datasets: list[str]) -> list[tuple[str, str, float | int, str, int]]:
    cases: list[tuple[str, str, float | int, str, int]] = []
    for dataset_name in datasets:
        for parameter in parameters:
            for solver in SOLVER_SPECS:
                for value in SENSITIVITY_SWEEPS[parameter]["values"]:
                    cases.append((dataset_name, parameter, value, solver["name"], solver["n_samples"]))
    return cases


def select_group_cases(
    cases: list[tuple[str, str, float | int, str, int]],
    group_id: int,
    group_size: int,
) -> list[tuple[str, str, float | int, str, int]]:
    return [case for index, case in enumerate(cases) if index % group_size == group_id]



def main() -> int:
    parser = argparse.ArgumentParser(description="Run baseline-only AquaPINN sensitivity sweeps.")
    parser.add_argument(
        "--parameter",
        nargs="+",
        choices=["all", *SENSITIVITY_SWEEPS.keys()],
        default=["all"],
        help="Parameters to sweep. Default: all.",
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        choices=["all", *DATASET_CASES.keys()],
        default=["all"],
        help="Datasets to sweep. Default: all.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=SCRIPT_DIR / "results" / "sensitivity_baseline",
        help="Output root for run folders and summary CSV.",
    )
    parser.add_argument(
        "--group-id",
        type=int,
        default=0,
        help="ID of the job group to run (0-based, default: 0).",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=1,
        help="Total number of job groups (default: 1).",
    )
    args = parser.parse_args()


    parameters = get_requested_parameters(args.parameter)
    datasets = get_requested_datasets(args.dataset)
    output_root = args.output_root.resolve()
    cases = build_cases(parameters, datasets)
    selected_cases = select_group_cases(cases, args.group_id, args.group_size)

    output_root.mkdir(parents=True, exist_ok=True)
    for dataset_name, parameter, value, solver_name, n_samples in selected_cases:
        dataset_spec = DATASET_CASES[dataset_name]
        run_single_case(
            dataset_name,
            Path(dataset_spec["decode_file"]),
            Path(dataset_spec["gps_file"]),
            parameter,
            value,
            solver_name,
            n_samples,
            output_root,
        )

    print(
        f"Completed {len(selected_cases)} sensitivity runs "
        f"for group {args.group_id}/{args.group_size}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

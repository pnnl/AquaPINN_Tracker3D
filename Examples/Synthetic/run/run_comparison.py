# -*- coding: utf-8 -*-
"""Run the Synthetic Critical comparison pipeline.

This is the Synthetic equivalent of the FishHeart/Sequim
``run_comparison.py`` entry point.  It loads a compare.toml template and
executes the shared AquaPINN_Tracker3D ``PostProcess/compareGPS.py`` script
once for each Critical scenario.

Usage
-----
    python run_comparison.py
    python run_comparison.py my_compare.toml
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
REPO_ROOT = PROJECT_DIR.parents[1]
TRACKER_ROOT = REPO_ROOT / "AquaPINN_Tracker3D"
POSTPROCESS_DIR = TRACKER_ROOT / "PostProcess"

SCENARIOS = ("baseline", "sudden_burst", "sudden_turn", "tmp_Exit")
TAG_CODE = "SYNTHETIC_TAG"
TAG_PRI = 2.0
PHONE_FILE = PROJECT_DIR / "dataGenerator" / "Cabot_Station_hydrophone_configuration_update_3.csv"
CRITICAL_DATA_DIR = PROJECT_DIR / "data" / "Critical"
TIME_START = "2023-10-16 15:30:00"
TIME_END = "2023-10-16 16:00:00"


def _add_import_paths() -> None:
    """Make compareGPS.py and its sibling imports available."""
    for path in (REPO_ROOT, POSTPROCESS_DIR):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)


def _scenario_config(template: str, scenario: str) -> str:
    """Create the compareGPS configuration for one scenario."""
    return template.replace("baseline", scenario)


def _run_compare(compare_text: str, scenario: str) -> None:
    """Execute compareGPS.py with temporary scenario configuration files."""
    compare_script = POSTPROCESS_DIR / "compareGPS.py"
    if not compare_script.is_file():
        raise FileNotFoundError(f"compareGPS.py not found: {compare_script}")

    with tempfile.TemporaryDirectory(prefix=f"synthetic_compare_{scenario}_") as temp_dir:
        temp_path = Path(temp_dir)
        compare_path = temp_path / "compare.toml"
        params_path = temp_path / "params_comparison.toml"
        tag_path = temp_path / "synthetic_tags.csv"

        # compareGPS requires a params_file and a tag CSV. Both are temporary
        # adapter files generated from the constants above.
        tag_path.write_text(
            f"Code,PRI\n{TAG_CODE},{TAG_PRI}\n",
            encoding="utf-8",
        )
        params_path.write_text(
            f'''phoneFile = ["{PHONE_FILE.as_posix()}"]
tagFile = ["{tag_path.as_posix()}"]
testFile = [""]
GPSFile = ["{(CRITICAL_DATA_DIR / f"{scenario}_GPS.csv").as_posix()}"]
time_start = ["{TIME_START}"]
time_end = ["{TIME_END}"]
timeZone = [0]
TagID = [-1]
badPhones = [[]]
''',
            encoding="utf-8",
        )
        compare_text = compare_text.replace(
            'params_file = "params_comparison.toml"',
            f'params_file = "{params_path.as_posix()}"',
        )
        compare_text = compare_text.replace(
            'out_root_dir = "results/Critical/',
            f'out_root_dir = "{(SCRIPT_DIR / "results" / "Critical").as_posix()}/',
        )
        compare_text = compare_text.replace(
            'NN = "results/Critical/',
            f'NN = "{(SCRIPT_DIR / "results" / "Critical").as_posix()}/',
        )
        compare_text = compare_text.replace(
            'NNplus = "results/Critical/',
            f'NNplus = "{(SCRIPT_DIR / "results" / "Critical").as_posix()}/',
        )

        compare_path.write_text(compare_text, encoding="utf-8")

        sys.argv = ["compareGPS.py", str(compare_path)]
        code = compare_script.read_text(encoding="utf-8")
        exec(compile(code, str(compare_script), "exec"), {"__name__": "__main__"})


def main() -> int:
    compare_toml = (
        Path(sys.argv[1]).resolve()
        if len(sys.argv) > 1
        else SCRIPT_DIR / "compare.toml"
    )
    if not compare_toml.is_file():
        raise FileNotFoundError(f"compare.toml not found: {compare_toml}")

    template = compare_toml.read_text(encoding="utf-8")
    _add_import_paths()

    original_directory = Path.cwd()
    os.chdir(SCRIPT_DIR)
    try:
        for scenario in SCENARIOS:
            print(f"\n=== Synthetic Critical scenario: {scenario} ===")
            compare_text = _scenario_config(template, scenario)
            _run_compare(compare_text, scenario)
    finally:
        os.chdir(original_directory)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# -*- coding: utf-8 -*-
"""Report baseline sensitivity results from saved AquaPINN runs."""

from __future__ import annotations

import argparse
import math
import runpy
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
RUNNER_FILE = SCRIPT_DIR / "sensitivity_baseline.py"

RUNNER_CONFIG = runpy.run_path(str(RUNNER_FILE), run_name="sensitivity_report_config")
SENSITIVITY_SWEEPS: dict[str, dict[str, list[float | int]]] = RUNNER_CONFIG["SENSITIVITY_SWEEPS"]
SOLVER_SPECS: list[dict[str, object]] = RUNNER_CONFIG["SOLVER_SPECS"]
DATASET_CASES: dict[str, dict[str, str]] = RUNNER_CONFIG["DATASET_CASES"]

# Human-readable axis / x-label for each sweep key.
PARAM_LABELS: dict[str, str] = {
    "period_factor":        "Period factor",
    "distance_threshold":   "Distance threshold",
    "std_deviation_factor": "Standard deviation factor",
    "time_shift_factor":    "Time shift factor",
    "uq_std":               "UQ standard deviation",
}


def format_value_label(value: float | int) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def locate_track_csv(output_dir: Path) -> Path | None:
    candidates = sorted(output_dir.glob("Track/track/*.csv"))
    if len(candidates) != 1:
        return None
    return candidates[0]


def compute_rms_3d_range_error(track_csv: Path, gps_file: Path) -> float:
    """Compute the RMS 3-D position error over all time-matched rows."""
    track_df = pd.read_csv(track_csv)
    gps_df = pd.read_csv(gps_file)

    track_df["datetime"] = pd.to_datetime(track_df["datetime"])
    gps_df["datetime"] = pd.to_datetime(gps_df["Datetime"])

    merged = pd.merge_asof(
        track_df.sort_values("datetime"),
        gps_df.sort_values("datetime"),
        on="datetime",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=0.3),
    )
    merged = merged.dropna(subset=["Easting", "Northing", "Elevation"])
    if merged.empty:
        return math.nan

    sq_errors = (
        (merged["X"] - merged["Easting"]) ** 2
        + (merged["Y"] - merged["Northing"]) ** 2
        + (merged["Z"] - merged["Elevation"]) ** 2
    )
    return math.sqrt(sq_errors.mean())


def compute_rms_3d_solution_std(track_csv: Path) -> float:
    """Compute the RMS combined solution std over all rows."""
    track_df = pd.read_csv(track_csv)
    if track_df.empty:
        return math.nan

    sq_stds = (
        track_df["X std"].astype(float) ** 2
        + track_df["Y std"].astype(float) ** 2
        + track_df["Z std"].astype(float) ** 2
    )
    return math.sqrt(sq_stds.mean())


def collect_results(results_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset_name, dataset_spec in DATASET_CASES.items():
        gps_file = Path(dataset_spec["gps_file"])
        for solver in SOLVER_SPECS:
            solver_name = str(solver["name"])
            for parameter, spec in SENSITIVITY_SWEEPS.items():
                for value in spec["values"]:
                    output_dir = results_root / dataset_name / solver_name / parameter / f"value_{format_value_label(value)}"
                    track_csv = locate_track_csv(output_dir)
                    rms_error = math.nan
                    rms_solution_std = math.nan
                    status = "missing"
                    if track_csv is not None:
                        try:
                            rms_error = compute_rms_3d_range_error(track_csv, gps_file)
                            rms_solution_std = compute_rms_3d_solution_std(track_csv)
                            status = "ok" if not math.isnan(rms_error) else "no_match"
                        except Exception as exc:  # noqa: BLE001
                            status = f"failed: {exc}"
                    rows.append(
                        {
                            "dataset": dataset_name,
                            "solver": solver_name,
                            "parameter": parameter,
                            "value": value,
                            "rms_3d_range_error_m": rms_error,
                            "rms_3d_solution_std_m": rms_solution_std,
                            "status": status,
                            "output_dir": str(output_dir),
                            "track_csv": "" if track_csv is None else str(track_csv),
                        }
                    )
    return pd.DataFrame(rows)


def plot_results(results_df: pd.DataFrame, figure_file: Path) -> None:
    import matplotlib
    from matplotlib.gridspec import GridSpec

    datasets = list(DATASET_CASES)
    parameters = list(SENSITIVITY_SWEEPS)  # expected: 5 parameters

    # ── Paper-quality style ──────────────────────────────────────────────────
    font_size = 10
    plt.rcParams.update({
        # Times New Roman throughout
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",          # matching math font
        # Sizes
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "legend.fontsize": font_size - 1,
        # Lines
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
        # Axes
        "axes.linewidth": 0.8,
        # Major ticks
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        # Minor ticks
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.minor.size": 2.5,
        "ytick.minor.size": 2.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        # Grid (major only — minor grid would be too busy)
        "axes.grid": True,
        "axes.grid.which": "major",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.4,
        "grid.linestyle": "--",
    })

    # Split 5 parameters: row 0 → first 3, row 1 → last 2 (centered).
    row0_params = parameters[:3]
    row1_params = parameters[3:]

    for dataset_name in datasets:
        # 6-virtual-column GridSpec:
        #   Row 0: cols 0-1, 2-3, 4-5  (3 equal panels)
        #   Row 1: cols 1-2, 3-4        (2 panels, centered)
        fig = plt.figure(figsize=(7.0, 5.2))
        gs = GridSpec(
            2, 6,
            figure=fig,
            left=0.09, right=0.97,
            top=0.95, bottom=0.10,
            hspace=0.55,   # generous vertical gap to avoid y-label overlap
            wspace=0.55,   # generous horizontal gap
        )

        row0_axes = [
            fig.add_subplot(gs[0, 0:2]),
            fig.add_subplot(gs[0, 2:4]),
            fig.add_subplot(gs[0, 4:6]),
        ]
        row1_axes = [
            fig.add_subplot(gs[1, 1:3]),
            fig.add_subplot(gs[1, 3:5]),
        ]

        all_axes = row0_axes + row1_axes
        all_params = row0_params + row1_params
        panel_labels = [f"({chr(ord('a') + i)})" for i in range(len(all_axes))]

        for ax, parameter, panel_label in zip(all_axes, all_params, panel_labels):
            param_df = results_df[
                (results_df["dataset"] == dataset_name)
                & (results_df["parameter"] == parameter)
                & (results_df["solver"] == "NNplus")
            ].copy()
            param_df = param_df.iloc[param_df["value"].to_numpy().argsort()]

            xlabel = PARAM_LABELS.get(parameter, parameter)

            if parameter == "uq_std":
                ax.plot(
                    param_df["value"],
                    param_df["rms_3d_range_error_m"],
                    label="RMS error",
                    color="#d62728",
                    marker="s",
                    clip_on=False,
                )
                ax.plot(
                    param_df["value"],
                    param_df["rms_3d_solution_std_m"],
                    label="RMS sol. std",
                    color="#1f77b4",
                    marker="^",
                    linestyle="--",
                    clip_on=False,
                )
                ax.set_ylabel("3D Range error / std (m)")
                ax.legend(loc="upper left", framealpha=0.9, edgecolor="0.7")

            elif parameter == "period_factor":
                # Wide range (75–19200): log x-scale spreads all points evenly.
                ax.plot(
                    param_df["value"],
                    param_df["rms_3d_range_error_m"],
                    color="#d62728",
                    marker="s",
                    clip_on=False,
                )
                ax.set_xscale("log")
                ax.set_ylabel("3D Range error (m)")

            elif parameter == "time_shift_factor":
                # Values [0.05, 0.1, 0.3, 0.6, 1.0]: log scale spreads the
                # low end (0.05–0.1) clearly.
                ax.plot(
                    param_df["value"],
                    param_df["rms_3d_range_error_m"],
                    color="#d62728",
                    marker="s",
                    clip_on=False,
                )
                ax.set_xscale("log")
                ax.set_ylabel("3D Range error (m)")

            else:
                ax.plot(
                    param_df["value"],
                    param_df["rms_3d_range_error_m"],
                    color="#d62728",
                    marker="s",
                    clip_on=False,
                )
                if ax in row0_axes:
                    pass
                else:
                    ax.set_ylabel("3D Range error (m)")

            # x-axis label with panel index on the line below
            ax.set_xlabel(f"{xlabel}\n{panel_label}", labelpad=4)
            ax.yaxis.set_tick_params(which="both", labelleft=True)

        # Save one figure per dataset
        stem = figure_file.stem
        suffix = figure_file.suffix
        out_file = figure_file.with_name(f"{stem}_{dataset_name}{suffix}")
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_file), dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Figure written to {out_file}")

    plt.rcParams.update(matplotlib.rcParamsDefault)


def main() -> int:
    parser = argparse.ArgumentParser(description="Report baseline sensitivity results from saved runs.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=SCRIPT_DIR / "results" / "sensitivity_baseline",
        help="Root folder containing sensitivity run outputs.",
    )
    parser.add_argument(
        "--figure-file",
        type=Path,
        default=SCRIPT_DIR / "results" / "sensitivity_baseline" / "sensitivity_report.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--table-file",
        type=Path,
        default=SCRIPT_DIR / "results" / "sensitivity_baseline" / "sensitivity_report.csv",
        help="Output CSV table with the computed final errors.",
    )
    args = parser.parse_args()

    results_root = args.results_root.resolve()
    results_df = collect_results(results_root)
    args.table_file.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.table_file, index=False)
    plot_results(results_df, args.figure_file.resolve())
    print(f"Figure written to {args.figure_file.resolve()}")
    print(f"Table written to {args.table_file.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

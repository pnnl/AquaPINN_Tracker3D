# AquaPINN Tracker 3D

AquaPINN Tracker 3D reconstructs three-dimensional trajectories of acoustically tagged animals from detections collected by hydrophone arrays. The tracker combines receiver locations, tag metadata, and time-of-arrival measurements to estimate a tag's position over time.

For raw receiver recordings, the workflow decodes tag and beacon detections, then uses beacon detections to estimate receiver-clock offsets when synchronization is needed. Deployments with already time-corrected detections, such as the bundled Sequim example, can proceed directly to tracking. The published neural-network (NN) solver represents the trajectory as a continuous function of ping transmission time and fits observed times of arrival (TOAs) using the acoustic travel-time relationship between the tag and each receiver. The project also includes approximate maximum likelihood (AML), a point-by-point comparison solver. NN+ extends NN with ensemble-based uncertainty quantification. Post-processing tools plot saved tracks and compare them with GPS reference data.

The method terminology follows Chen et al., [“A novel uncertainty-aware 3D tracking solver based on physics-informed neural networks”](https://doi.org/10.1139/cjfas-2025-0392) (2026).

This repository includes the Python research pipeline, example workflows for FishHeart, Sequim, and synthetic data, a Windows Auto desktop installer, and browser-based Jupyter notebooks for tutorials and exploring the examples.

## What the terms mean

| Term | Meaning in this project |
| --- | --- |
| Acoustic tag / transmitter | A device attached to an animal that emits identifiable acoustic pings. The tag code identifies the transmitter. |
| Ping | One acoustic signal emitted by a tag. The pulse repetition interval (PRI) is the time between consecutive pings. |
| Receiver / hydrophone | A sensor at a known location that detects and records a ping. Some parameter files call the receiver geometry `phoneFile`. |
| Detection | A receiver's record that it heard a particular tag ping, including its measured arrival time. A ping can be detected by several receivers or missed by some. |
| TOA | Time of arrival: the time a ping is recorded at a receiver. TOA measurements are affected by timing noise, multipath, and clock offsets. |
| TDOA | Time difference of arrival: the difference between TOA measurements at two receivers. |
| Beacon | A reference transmitter used to estimate clock offsets between receivers when detection times are not already synchronized. |
| AML | Approximate maximum likelihood. A point-by-point solver that estimates positions from individual transmissions. |
| NN (PINN) | The paper's physics-informed neural-network trajectory solver, called NN in the paper and configuration. It represents the trajectory as a continuous function of ping transmission time and fits TOA measurements using receiver distance and sound speed. |
| NN+ | An ensemble version of NN. It combines repeated NN fits and uses the ensemble mean and spread (such as standard deviation) to estimate trajectory uncertainty. |
| TagID | The zero-based row number in the tag catalogue. It selects a row; it is not the tag code or a receiver ID. |

## Windows desktop installer

The Auto Online Setup is in [Windows_installer](Windows_installer/AquaPINN-Tracker-Auto-0.3.1-auto-rc4-Online-Setup.exe).

The installer is a single EXE. Internet access is required during setup to download the locked dependencies. After installation, tracking runs locally. No separate Python, pip, CUDA Toolkit, or terminal installation is required.

Choose Automatic to check for a usable NVIDIA GPU; if one is unavailable, the app can use CPU processing. GPU use requires a compatible NVIDIA driver. AMD and Intel graphics use CPU processing. The interface runs in the CPU environment, with numerical fitting isolated in a worker process.

The included candidate installer is unsigned. Follow your organization’s software installation policies. For setup details, see [Installation](docs/installation.md). For the desktop workflows, see the [User Guide](docs/user-guide.md).

The Auto desktop application is delivered as an installer; its application source is not included in this checkout.

## Research pipeline

The stages can be run separately through the Python package and example launchers:

1. Decode raw receiver CSV recordings into tag and beacon detections.
2. Synchronize receiver clocks from beacon detections when required.
3. Prepare TOA observations and estimate tag trajectories.
4. Compare saved trajectories with GPS references and generate plots or reports.

FishHeart uses raw receiver CSV files and normally runs all three processing stages. Sequim starts from time-corrected MATLAB detections, converts them to the tracking format, and does not require a new synchronization run. Synthetic scenarios provide controlled data for method comparisons.

More detail is in [Research workflows](docs/research-workflows.md).

## Tracking methods

| Method | Description | Configuration |
| --- | --- | --- |
| AML | Approximate maximum likelihood; point-by-point position estimates | trackMethod = ["AML"] |
| NN | Physics-informed neural-network trajectory solver | trackMethod = ["NN"], nSamples = [1] |
| NN+ | NN ensemble; reports the ensemble mean and spread for uncertainty-aware interpretation | trackMethod = ["NN"], nSamples greater than 1 |

NN+ uses ensemble spread as a model-based uncertainty estimate and can support filtering less reliable positions. Check uncertainty calibration for the deployment before interpreting the spread as a formal confidence interval. In the configuration file, the method remains `trackMethod = ["NN"]`; set `nSamples` greater than 1 to run NN+, which is saved with the `NNplus` output label.

## Included examples

| Example | Inputs and workflow | Configuration |
| --- | --- | --- |
| FishHeart | Receiver CSV recordings, beacon synchronization, then tracking | Examples/fishHeart/run/params.toml |
| Sequim | Corrected MATLAB detections, conversion, then tracking | Examples/Sequim/run/params.toml |
| Synthetic | Generated trajectories and detections for benchmark scenarios | Examples/Synthetic/run/params.toml |

Check each configuration before running. Confirm the input paths, selected tag, time range, timezone, receiver geometry, and output folder. A TagID selects a row in the configured tag catalogue; it is not the tag code or receiver ID.

## Colab notebooks: an alternative to the desktop app

Use these browser-based notebooks to learn and try the example workflows without installing AquaPINN or Python on your computer. They are intended mainly for tutorials, teaching, and exploring the examples; the Windows installer remains available for local desktop use.

To start, you need internet access and a Google account with Google Drive. A free account is enough to open the notebooks and save results to Drive. Colab installs the notebook dependencies in its hosted runtime, so no local software installation is required.

| Workflow | Colab |
| --- | --- |
| Sequim Quickstart | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cwq2016/AquaPINN_tmp/blob/main/Examples/Colab/01_Sequim_Quickstart.ipynb) |
| FishHeart Full Pipeline | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cwq2016/AquaPINN_tmp/blob/main/Examples/Colab/02_FishHeart_Full_Pipeline.ipynb) |
| Synthetic Benchmark | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cwq2016/AquaPINN_tmp/blob/main/Examples/Colab/03_Synthetic_Benchmark.ipynb) |

Sequim and FishHeart load settings from their existing `params.toml` files; Synthetic has an editable setup block. Colab runtime limits and availability are controlled by Google. See the [Colab notebook guide](Examples/Colab/README.md) for workflow details.

## Research environment

Python 3.11 or later is recommended. The research pipeline uses PyTorch, NumPy, SciPy, pandas, Matplotlib, utm, NetworkX, scikit-learn, tqdm, and six. NumPy versions below 2.5 are recommended for the existing coordinate-conversion code.

There is no general package-install manifest or editable package setup. Install a PyTorch build appropriate for the machine, then install the remaining dependencies in an isolated environment. For example:

    python -m pip install torch "numpy<2.5" scipy pandas matplotlib utm networkx scikit-learn tqdm six

This is an unpinned starting point, not a fully reproduced research environment. CUDA is optional for the research scripts. Use the supplied example launchers rather than the legacy pipeline module.

## Run an example

Run commands from the repository root. Review the selected example’s parameters first.

    python Examples/fishHeart/run/main_FishHeart.py

The Sequim launcher follows its conversion-and-tracking workflow:

    python Examples/Sequim/run/main_Sequim.py

Synthetic data must be generated before tracking. The following commands perform generation, tracking, and comparison in sequence:

    python Examples/Synthetic/dataGenerator/caseDesign_CabotSynthetic.py
    python Examples/Synthetic/run/main.py
    python Examples/Synthetic/run/run_comparison.py

Training can take substantial time, especially on CPU. Repeated NN+ fits take longer than a single NN fit.

## Compare results

After generating the desired track files, run the example’s comparison launcher. For example:

    python Examples/fishHeart/run/run_comparison.py

Comparison scripts read existing results; they do not run missing solvers. The report script can summarize metrics from the comparison configuration:

    python AquaPINN_Tracker3D/PostProcess/report.py Examples/fishHeart/run/compare.toml

Use separate result directories for methods when comparing NN and NNplus. Confirm that the comparison selected the intended CSV files and that the GPS reference covers the interval being evaluated.

## Configuration notes

Pipeline parameter values are represented as lists to allow parameter combinations. For a single run, use one choice per setting. Vector settings are nested lists, for example region_lb = [[-200, -200, -20]].

Several boolean-like settings are strings, including ignoreZ and split_by_TestFile. Use ["True"] or ["False"] where the example configuration does so. Do not replace these with TOML booleans unless a parameter is documented as a real boolean.

Important settings include phoneFile, tagFile, dataFolder, GPSFile, time_start, time_end, timeZone, temperature_ref, trackMethod, nSamples, badPhones, and output_folder. Units, timezone conventions, receiver IDs, and tag-catalogue ordering must match the input data.

## Outputs

Depending on the stages and example, results can include:

- Decoded and synchronized detection pickle files.
- Corrected receiver-location CSV files.
- Prepared TOA data and detection summaries.
- Per-tag trajectory CSV files and solver sidecar files.
- Track plots, training histories, and parameter snapshots.
- GPS comparison plots, statistics, and report CSV files.

Outputs are written beneath the output folder configured for each example. Use a fresh output location when changing source data or preprocessing settings; some intermediate files are reused based on their existence.

## Interpretation limits

A completed run does not establish positional or clock accuracy. Coordinate units, vertical datum, time basis, sound-speed assumptions, GPS coverage, and receiver/tag offsets affect interpretation. Exported Z is elevation and should not automatically be interpreted as depth below water. Comparison reports may summarize error fields already stored in track CSVs rather than recomputing them from a newly selected GPS file.

For more, see [Limitations and validation](docs/limitations.md).

## License

The project is distributed under GNU GPL version 3 or later. Preserve the included license and disclaimer notices when redistributing. Confirm redistribution rights for example datasets and third-party components.


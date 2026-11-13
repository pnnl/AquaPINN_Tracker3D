# AquaPINN Colab notebooks

These notebooks are a free, browser-based alternative to the desktop app, intended mainly for tutorials, teaching, and exploring the example workflows. You need internet access and a Google account with Drive; a free account is sufficient. No local Python, AquaPINN, or dependency installation is required because Colab installs notebook dependencies in its hosted runtime.

Open a workflow directly in Google Colab:

| Workflow | Colab |
| --- | --- |
| Sequim Quickstart | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cwq2016/AquaPINN_tmp/blob/main/Examples/Colab/01_Sequim_Quickstart.ipynb) |
| FishHeart Full Pipeline | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cwq2016/AquaPINN_tmp/blob/main/Examples/Colab/02_FishHeart_Full_Pipeline.ipynb) |
| Synthetic Benchmark | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cwq2016/AquaPINN_tmp/blob/main/Examples/Colab/03_Synthetic_Benchmark.ipynb) |

The Sequim and FishHeart notebooks read settings from their supplied `params.toml` files. Synthetic has an editable setup block. In Colab, completed results are saved to Google Drive; local Jupyter uses the configured output folder.

Default input paths point to bundled examples. Custom paths can be entered in each notebook and may point to Drive. Only publish data approved for redistribution. Colab storage is temporary, so use the Drive-backed output folder for results you want to keep.

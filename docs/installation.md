AquaPINN Tracker Auto - Online / Offline candidate 0.3.1-auto-rc4

This edition includes Simple Example and Full Study. The CPU-only edition remains
a separate product and can be installed alongside it. Use the clearly named Auto
shortcuts for this edition. Setup does not change existing Python, drivers or CUDA.

OFFLINE: Keep Setup.exe and ALL matching Setup-*.bin files together in the same folder.
Run Setup.exe only. The companion files contain the offline dependencies and must
not be renamed or moved away. Windows' single-executable size limit requires these
files for the two-profile package; there is no manual dependency installation.

ONLINE: Run the single Online-Setup.exe. Private Python and the app are included;
internet is required DURING setup to download exact hash-locked binary CPU/Qt,
CUDA 12.6 and CUDA 13.2 dependencies from PyPI and official PyTorch hosts. Allow for
several gigabytes of downloads and substantial setup time. Restricted networks
may block downloads; contact IT or use Offline. No security bypass is needed.
CPU dependency failure stops setup; optional GPU dependency failure is logged and
keeps a usable CPU application. Logs are under setup-state. Uninstall normally
before retrying an incomplete setup or changing delivery modes. Both modes share
the same Auto installed identity and cannot be installed side by side with each
other; the separate CPU edition can coexist.
Online saves delivery-file size, not installed dependency size. Allow about 20 GB
free local storage for Auto setup plus results (provisional). After setup local
tracking needs no internet; optional map tiles are a separate opt-in feature.

Choose Automatic to try a usable NVIDIA GPU, otherwise use genuine CPU-only
PyTorch automatically. No preinstalled Python, pip, CUDA Toolkit or terminal is needed.
Offline setup needs no internet; Online setup requires it as described above.
GPU execution requires a compatible working NVIDIA driver. Missing/incompatible
drivers select CPU; updating drivers is optional and is not done by this installer.
AMD and Intel graphics use CPU in this candidate.

Two isolated GPU profiles are installed (CUDA 12.6 and CUDA 13.2), tested automatically
to cover both older and newer architectures. Unsupported devices fall back to CPU.
This larger offline package needs substantial disk space; no GPU model whitelist
or arbitrary memory floor is used, but the actual workload must fit GPU memory.

The interface always runs in the CPU environment. GPU checks run in isolated
processes. A detected GPU runtime failure restarts the whole request once in fresh
CPU outputs. That can take longer than the original attempt. The failed attempt's
logs are retained under technical evidence; partial GPU results are not combined
with CPU results. Invalid inputs/hash failures do not silently trigger CPU retry.
Choose CPU only inside the application to skip GPU checks. Cancel stops processing,
not starts a CPU retry. An indefinitely hung fit may require Cancel; probe checks
have timeouts but there is no guessed solver-duration cutoff.

Float64 precision and seed 42 are preserved. GPU is not guaranteed to be faster,
and CPU/GPU trajectories are not guaranteed bit-identical. This candidate is not
scientific accuracy certification, exhaustive hardware support or recipient
acceptance. Results show the chosen device/fallback details in run evidence.

Use a short empty installation folder; do not put results under the application.
Close this edition before its normal Uninstall action. Cleanup preserves external
results and new/modified generated files rather than deleting unknown files.
If setup is incomplete, use Uninstall before retrying. Do not manually delete an
unknown folder or disable security protections. The candidate is unsigned; signing,
third-party licensing/data rights and clean-recipient testing remain separate gates.

For scientific workflows, coordinate/depth/GPS/PRI limitations and setup guidance,
also read the included CPU recipient guide. CPU examples and scientific defaults
are unchanged; this Auto edition adds device selection and isolated fallback only.
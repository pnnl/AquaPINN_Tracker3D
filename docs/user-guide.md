# Desktop user guide

## Simple Example

Launch AquaPINN Tracker Auto from its installed shortcut. Choose a results location outside the application install folder, select the example and tracking method, then start a fresh run. The Simple Example uses bundled, already time-corrected Sequim detections; it does not run the raw-data decode and synchronization stages.

AML performs analytical multilateration. NN runs one neural fit. NN+ repeats the fit and reports the resulting ensemble spread. NN+ spread is not a calibrated confidence interval.

## Full Study

Use Full Study to configure a field-data workflow. Review the receiver and tag metadata, input recordings, time range, coordinate frame, and selected tags before starting. The workflow validates the setup, then decodes detections, synchronizes receiver clocks, and tracks selected tags. Results and run evidence are saved to a fresh run folder.

The exact options depend on the selected study setup. Inspect the generated run record and logs when reviewing a result. A cancelled or interrupted run is incomplete and should be restarted as a new run.

## Device selection

Automatic checks the installed NVIDIA GPU profiles in isolated processes. A compatible, usable GPU may be selected; otherwise processing uses CPU. A recoverable GPU execution failure can trigger one clean CPU retry. CPU-only skips GPU checks. The interface itself runs in the CPU environment.

## Results

Open completed runs from their saved run records. The viewer reads track CSVs and run metadata; it does not require loading solver pickle files. Preserve the complete run folder when sharing results.

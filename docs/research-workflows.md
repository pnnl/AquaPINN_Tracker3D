# Research workflows

## Included deployments

- FishHeart: receiver CSV detections, beacon-based clock synchronization, then tracking.
- Sequim: preprocessed MATLAB detections with corrected timing, then tracking.
- Synthetic: generated trajectories and detections for controlled method comparisons.

The desktop application provides supported guided workflows for the bundled Simple Example and Full Study. The Python package contains the lower-level decoder, synchronization, tracking, and post-processing stages for research use.

## Core stages

1. Decode receiver recordings into tag and beacon detections.
2. Synchronize receiver clocks using beacon detections when the source data are not already synchronized.
3. Build time-of-arrival observations and estimate trajectories using AML or NN.
4. Compare saved tracks with GPS references using the post-processing tools.

Use the deployment-specific parameter files and input metadata supplied with each example. Keep raw inputs, corrected detections, tracking results, and comparison outputs in separate locations. Confirm time zones, coordinate units, vertical datum, and tag catalogue order before interpreting results.

# Limitations and validation

AquaPINN is research software. A successful run demonstrates that the configured workflow completed; it does not certify positional accuracy, clock accuracy, or suitability for another deployment.

- Coordinate units, reference frame, vertical datum, time zone, sound-speed assumptions, and physical offsets affect trajectory interpretation.
- GPS is a reference track and may contain measurement error, gaps, or timing and coordinate differences.
- NN+ ensemble spread is based on repeated fits and is not automatically a calibrated confidence interval.
- GPU and CPU results are not guaranteed to be bit-identical. GPU execution is optional and depends on a compatible NVIDIA driver and workload memory.
- Tracking may return positions for only part of the configured time window. Low detection or tracking efficiency does not by itself establish why positions are missing.
- Included example data and third-party components may have separate redistribution terms; check permissions and notices before publishing or redistributing them.

Consult the deployment configuration and run evidence for the settings and inputs used in a particular result.

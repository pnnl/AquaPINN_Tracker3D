# Sensitivity Analysis of AquaPINN Tracker Hyperparameters

This document analyses how the five key hyperparameters of the AquaPINN neural-network tracker affect localisation accuracy, measured as the root-mean-square (RMS) 3-D position error over all time-matched track points.
All experiments use the synthetic baseline dataset with the NNplus solver (5 UQ samples, fixed-distance sampling).

---

## 1. Period Factor (`period_factor`)

**Role in the model.**  
`period_factor` (default 300) scales the reference velocity `Vel_ref` that controls the dominant spatial frequency of the Fourier-feature network:

```
Vel_ref = count(valid detections) / period_factor
```

A larger `period_factor` produces a smaller `Vel_ref`, which biases the network toward lower-frequency (smoother) trajectory representations; a smaller value allows higher-frequency components.

**Observed behaviour.**  
The RMS error is highest at very small values (≈75–150) and decreases rapidly as `period_factor` increases toward 300–600, after which it plateaus or rises only slightly out to 19 200.

**Physical explanation.**  
- *Too small (75–150):* `Vel_ref` is large, so the Fourier features encode very high spatial frequencies. The network can fit rapid oscillations that are not physically present in the trajectory, leading to over-fitting of TOA noise and inflated position error.  
- *Optimal range (300–1 200):* The network frequency matches the true trajectory dynamics. The smooth prior imposed by the Fourier features acts as an implicit regulariser, suppressing noise without over-smoothing genuine motion.  
- *Very large (4 800–19 200):* `Vel_ref` approaches zero, forcing the network into an almost constant-position solution. The trajectory is over-smoothed and cannot track real motion, so error rises again.

**Recommendation.** A value of 300–600 is robust for typical fish-tracking scenarios. The log-scale plateau suggests the tracker is not highly sensitive to this parameter once it is in the correct order of magnitude.

---

## 2. Distance Threshold (`distance_threshold`)

**Role in the model.**  
`distance_threshold` (default 3.0 m) defines the TOA residual threshold used in the adaptive residual-reweighting scheme:

```
dt = distance_threshold / sound_speed
noise_ratio = |residual_toa| // dt          # integer floor division
weight = 1 / noise_ratio   (for outliers)
```

Detections whose TOA residual exceeds `dt` are down-weighted in proportion to how many multiples of `dt` they deviate; detections within `dt` receive full weight.

**Observed behaviour.**  
The RMS error is relatively high at very small thresholds (0.75 m), decreases to a minimum around 3–6 m, and increases again at large values (24–48 m).

**Physical explanation.**  
- *Too small (0.75–1.5 m):* The threshold is tighter than the typical TOA noise level (converted to distance). Many legitimate detections are classified as outliers and down-weighted, starving the optimisation of useful signal. The effective number of observations drops, increasing uncertainty.  
- *Optimal range (3–6 m):* The threshold is commensurate with the expected TOA noise floor. Genuine outliers (multipath, false detections) are suppressed while clean detections retain full weight, giving the best bias–variance trade-off.  
- *Too large (24–48 m):* The threshold is so permissive that multipath arrivals and false detections are treated as valid observations. These corrupt the loss landscape and pull the estimated trajectory away from the true path.

**Recommendation.** Set `distance_threshold` to roughly 2–5× the expected one-sigma TOA noise converted to distance (noise_std × sound_speed). For typical freshwater deployments with σ_TOA ≈ 0.3–1 ms and c ≈ 1 450 m/s, this corresponds to 0.4–1.5 m, suggesting the default of 3 m already provides a comfortable margin.

---

## 3. Standard Deviation Factor (`std_deviation_factor`)

**Role in the model.**  
`std_deviation_factor` (default 5.0) controls the steepness of the sigmoid function that maps the learned per-detection quality variable τ (tau) to a positive weight:

```
tau_weight = 1 + exp(−std_deviation_factor × tau / n)
```

A larger value makes the sigmoid steeper, so τ transitions more sharply between "low quality" and "high quality" states; a smaller value produces a softer, more gradual transition.

**Observed behaviour.**  
The RMS error is relatively flat across the tested range (1–12), with a shallow minimum around 5–8 and a modest increase at the extremes.

**Physical explanation.**  
- *Too small (1–3):* The sigmoid is nearly linear. τ cannot effectively discriminate between good and bad detections; all observations receive similar weights regardless of their quality, reducing the benefit of the learned quality variable.  
- *Optimal range (5–8):* The sigmoid is steep enough to create a meaningful binary-like quality gate while remaining differentiable, allowing gradient-based optimisation to adjust τ effectively.  
- *Too large (≥12):* The sigmoid becomes a near-step function. Small numerical errors in τ cause abrupt weight changes, making the loss landscape rough and harder to optimise. Gradient flow through the quality variable is impeded.

**Recommendation.** The tracker is relatively insensitive to this parameter in the range 3–10. The default of 5 is a safe choice.

---

## 4. Time Shift Factor (`time_shift_factor`)

**Role in the model.**  
`time_shift_factor` (default 0.3 s) bounds the magnitude of the learned per-ping time correction `top`:

```
top = t − time_shift_factor × tanh(raw_output)
```

Because `tanh` saturates at ±1, the maximum absolute time correction is `time_shift_factor` seconds. This correction absorbs clock drift, synchronisation errors, and systematic ping-time offsets between the tag and the receiver network.

**Observed behaviour.**  
The RMS error is high at very small values (0.05 s), drops sharply to a minimum around 0.1–0.3 s, and rises again at large values (0.6–1.0 s).

**Physical explanation.**  
- *Too small (0.05 s):* The allowed correction is insufficient to absorb real synchronisation errors. Residual clock offsets appear as systematic position biases, inflating the RMS error.  
- *Optimal range (0.1–0.3 s):* The bound is large enough to correct realistic clock drift (typical acoustic tag systems have drift on the order of tens to hundreds of milliseconds) while remaining small enough to prevent the network from absorbing genuine travel-time variation as a spurious time shift.  
- *Too large (0.6–1.0 s):* The network can shift ping times by up to a full second. This degree of freedom is large enough to explain away real position changes as time offsets, effectively allowing the network to "cheat" by collapsing the trajectory toward a stationary solution. The result is a smooth but inaccurate track.

**Recommendation.** Set `time_shift_factor` to slightly exceed the expected worst-case clock drift. For well-synchronised systems (drift < 50 ms) a value of 0.1–0.2 s is appropriate; for loosely synchronised deployments 0.3–0.5 s provides more headroom.

---

## 5. UQ Standard Deviation (`uq_std`)

**Role in the model.**  
`uq_std` (default 1.0 m) is the perturbation radius used in the fixed-distance UQ sampling scheme. For each of the `nSamples − 1` additional training runs, the estimated position is perturbed by a Gaussian displacement of standard deviation `uq_std` metres, and the corresponding TOA noise is computed from the change in source–receiver distance:

```
toa_noise = (||x_perturbed − receiver|| − ||x_nominal − receiver||) / sound_speed
```

The spread of the resulting ensemble of tracks provides the reported position uncertainty (solution std).

**Observed behaviour.**  
The RMS error increases monotonically with `uq_std`, while the solution std also increases. At `uq_std = 0` both metrics are at their minimum; at `uq_std = 4 m` the error is substantially higher.

**Physical explanation.**  
- *`uq_std = 0` (no perturbation):* All samples start from the same nominal solution. The ensemble collapses to a single point, reporting zero uncertainty. The RMS error is lowest because the network is not perturbed away from its optimum.  
- *Small `uq_std` (0.5–1.0 m):* Perturbations are comparable to the TOA noise level. Each sample explores a slightly different region of the loss landscape, producing a realistic spread that reflects genuine localisation uncertainty without significantly degrading accuracy.  
- *Large `uq_std` (2–4 m):* Perturbations are much larger than the noise level. The injected TOA noise overwhelms the real signal, and each sample converges to a different local minimum far from the true position. The ensemble mean is biased and the reported uncertainty is inflated beyond what is physically meaningful.

**Recommendation.** Set `uq_std` to approximately 1–2× the expected one-sigma localisation error. For the synthetic baseline (sub-metre accuracy), a value of 0.5–1.0 m is appropriate. Larger values are only warranted when the prior uncertainty is genuinely large (e.g., at track initialisation or after a long detection gap).

---

## Summary

| Parameter | Optimal range | Primary failure mode (too small) | Primary failure mode (too large) |
|---|---|---|---|
| `period_factor` | 300–1 200 | Over-fitting (high-frequency artefacts) | Over-smoothing (trajectory flattened) |
| `distance_threshold` | 3–6 m | Signal starvation (valid obs. down-weighted) | Outlier contamination |
| `std_deviation_factor` | 3–10 | Weak quality discrimination | Rough loss landscape, poor gradient flow |
| `time_shift_factor` | 0.1–0.3 s | Residual clock bias | Trajectory collapsed to near-stationary |
| `uq_std` | 0.5–1.0 m | Zero / under-estimated uncertainty | Biased ensemble mean, inflated uncertainty |

Overall, the tracker is most sensitive to `period_factor`, `distance_threshold`, and `time_shift_factor`, which directly affect the physics of the localisation problem. The `std_deviation_factor` and `uq_std` parameters primarily influence the quality of the uncertainty quantification rather than the point-estimate accuracy.

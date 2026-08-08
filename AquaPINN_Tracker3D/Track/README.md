# Stage 3: Acoustic Tracking (`Track/`)

## Overview

Given time-synchronised detections from Stage 2, this stage localises each acoustic tag
in 3-D (or 2-D) space at each transmission time using observed Time-Differences-of-
Arrival (TDOA) between receiver pairs.

Two independent solvers are provided and can be run concurrently:

1. **AML** (`AML.py`) — closed-form algebraic multilateration; fast and interpretable.
2. **Neural Network** (`ML_NN.py`) — end-to-end learnable trajectory model; can capture
   smooth motion priors and produce uncertainty estimates.

---

## Module Map

| Module | Role |
|--------|------|
| `tracker.py` | Top-level orchestrator |
| `dataLoader.py` | Builds sparse TOA matrices for tracking (variant of Sync `dataLoader`) |
| `AML.py` | Algebraic Multilateration solver (closed-form, per-signal) |
| `ML_NN.py` | Neural-network trajectory estimator (PyTorch) |
| `postProcess.py` | Error analysis, GPS comparison, output CSV generation |
| `AMLlib/` | Low-level C-extension or NumPy TDOA linear algebra routines |

---

## Algorithm Details

### 1. `tracker.py` — Orchestrator

`run(param_file, loc_sys)` pipeline:

1. Load synced detections from pickle.
2. Call `dataLoader.TOAinfo_creator` → build TOA matrices.
3. Split signals into time intervals (by test periods or max-count windows).
4. For each interval:
   - Run `AML.AMLSolver()` (if enabled).
   - Run `ML_NN.Model` training (if enabled).
5. Call `postProcess.run()` → compute errors, compare to GPS, write CSV.

---

### 2. `dataLoader.py` — TOA Matrix Builder for Tracking

Closely parallels `Sync/dataLoader.py` but with important differences:

- **No preSync step**: positions are already corrected by the sync stage.
- **TOAD filter** is still applied to remove multipath echoes at the tracking stage.
- Signals are split into **segments** of at most `numTrack` events to keep memory
  manageable for the neural-network solver.

#### 2a. `TOAinfo_creator` (Tracking Variant)

1. Load synced detection pickle for the target tag.
2. For each signal transmission event $t$, collect TOAs from all receivers that detected
   it.
3. Build sparse CSR matrix: rows = events, columns = receivers, values = TOA (−999 if
   missing).
4. Apply `apply_TOAD_filter()` (same algorithm as in `Sync/dataLoader`; iterative
   median + 3σ criterion, up to 100 iterations) to flag multipath duplication at the
   tracking stage.
5. Clip idle intervals longer than 600 s to avoid accumulation of large time gaps in the
   neural network's input features.

#### 2b. Minimum Detection Thresholds

A signal is included only if it was detected at:
- $\geq 3$ receivers (3-D tracking, `ignoreZ = False`), or
- $\geq 4$ receivers (3-D tracking with $Z$ constrained, `ignoreZ = True`).

A receiver with fewer than 5 valid detections across all signals is flagged as "bad" and
excluded from that interval.

---

### 3. `AML.py` — Algebraic Multilateration Solver

#### 3a. Problem Formulation

For signal $t$ detected at receivers $\{p_0, p_1, \ldots, p_K\}$:

- Choose receiver $p_0$ as the reference.
- Define TDOAs:
  $$\tau_k = \frac{\text{TOA}_{t, p_k} - \text{TOA}_{t, p_0}}{c(T)}, \quad k=1\ldots K$$
  (units: metres equivalent travel time)

The source position $\mathbf{x}$ satisfies the hyperbolic constraint:
$$r_k - r_0 = c(T)\ \tau_k, \quad r_k = \|\mathbf{x} - \mathbf{x}_{p_k}\|$$

#### 3b. AML Linearisation

The Algebraic Multilateration approach eliminates the non-linear range term by
introducing an auxiliary variable $r_0 = \|\mathbf{x} - \mathbf{x}_{p_0}\|$:

$$r_k = r_0 + c(T)\ \tau_k$$

Squaring both sides and expanding:

$$\|\mathbf{x}\|^2 - 2\mathbf{x}_{p_k}^\top \mathbf{x} + \|\mathbf{x}_{p_k}\|^2
= (r_0 + c\tau_k)^2$$

After differencing against the reference equation and collecting terms, the linear
system is:

$$\mathbf{D}^\top Q_d^{-1} \mathbf{D}\ \mathbf{b} = \mathbf{D}^\top Q_d^{-1} \boldsymbol{\psi}$$

with:
- $\mathbf{D} = [\mathbf{x}_{p_1}-\mathbf{x}_{p_0},\ \ldots,\ \mathbf{x}_{p_K}-\mathbf{x}_{p_0}]^\top \in \mathbb{R}^{K\times 3}$
- $\psi_k = \frac{1}{2}\left(c\tau_k^2 + 2c\tau_k r_0 - \|\mathbf{x}_{p_k}\|^2 + \|\mathbf{x}_{p_0}\|^2\right)$ (depends on unknown $r_0$)
- $Q_d = \text{diag}(\text{range weights})$, initially uniform

Solving this linear system gives $\mathbf{b}$ (a proxy for position), from which $r_0$
is recovered via a **quadratic equation**:

$$(\|\mathbf{b}\|^2 - 1)\ r_0^2 + 2\mathbf{b}^\top(\mathbf{c} - \mathbf{x}_{p_0})\ r_0
  + \|\mathbf{c} - \mathbf{x}_{p_0}\|^2 = 0$$

Two candidate roots exist; the physically valid one (positive $r_0$, position inside
array bounds) is selected by checking TDOA residuals.

#### 3c. Iterative Range Weighting (AML Iterations)

After the first solve, the range weights are refined:

$$w_k = \frac{1}{r_0 + r_k + c\tau_k} \propto \frac{1}{\text{total path length}}$$

The system is re-solved with updated weights.  Two iterations are typically sufficient
for convergence.

#### 3d. Receiver Selection and Outlier Rejection

After each candidate solve:

1. **Boundary check**: $\mathbf{x}$ must lie within the pre-specified region bounds
   (`region_lb`, `region_ub`).
2. **Distance gate**: each receiver's implied range $r_k$ must be $<$ `dis_limit`.
3. **TDOA residual check**: $|c\tau_k^{\text{pred}} - c\tau_k^{\text{obs}}| <$ `dis_tol`.

If any receiver fails, it is removed and the solve is repeated without it.  This cycle
continues until all remaining receivers pass or too few remain for a valid solution.

Each reference receiver $p_0$ is tried in turn; the solve with the lowest total TDOA
residual is selected as the final position estimate.

#### 3e. Output

For each signal $t$ with a successful solve:

```
xyz[t]       = (X, Y, Z) in local coordinates (metres)
top[t]       = estimated time-of-transmission (datenum)
nValid[t]    = number of receivers used in the final solve
residual[t]  = RMSE of TDOA residuals (metres)
```

---

### 4. `ML_NN.py` — Neural Network Trajectory Estimator

#### 4a. Motivation

AML treats each transmission independently.  The NN solver instead models position as a
**continuous function of time**, imposing an implicit smoothness prior through the
network architecture.  It also produces **uncertainty estimates** via multi-sample training.

#### 4b. Input Features — Time Encoding

Raw timestamps are encoded into a rich feature vector using a learnable Fourier basis:

$$\phi(t) = \left[\cos(F t),\ \sin(F t)\right]$$

where $F \in \mathbb{R}^{d/2}$ is a learnable frequency matrix (not fixed as in
standard positional encodings).  This allows the network to adapt its temporal
resolution to the signal.

#### 4c. `Model` Class (PyTorch `nn.Module`)

**Learned parameters:**

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `xyz[t]` | $(N_t, 3)$ | Direct position at each signal time |
| `top[t]` | $(N_t,)$ | Time-of-transmission at each signal |
| `toa_tau` | B-spline coefficients | Noise std per receiver, varying smoothly over time |
| Network weights `W` | per layer | Maps time features to position trajectory |

**Trajectory model:**
The trajectory is parameterised by a `MFNN_feature` network (see `SharedLibs/networks.py`):
$$\mathbf{xyz}(t) = \text{MFNN}(\phi(t);\, W) + \text{xyz0}$$

where `xyz0` is an initial position estimate (e.g., centroid of array).

**Forward pass:**

For signal $t$ at receiver $p$:

$$\hat{\text{TOA}}(t, p) = \text{top}(t) + \frac{\|\mathbf{xyz}(t) - \mathbf{x}_p\|}{c(T)}$$

$$r(t, p) = \hat{\text{TOA}}(t, p) - \text{TOA}_{\text{sync}}(t, p)$$

#### 4d. Loss Function — Heteroscedastic NLL

$$\mathcal{L} = \sum_{t} \sum_{p \in \text{valid}(t)} \left[
  \frac{w(t,p)^2\ r(t,p)^2}{2\ \tau(t,p)^2} - \log \tau(t,p)
\right] + \lambda \cdot \mathcal{L}_{\text{bound}}$$

Components:
- **Data term**: Huber-weighted NLL identical to the sync stage.
- **Boundary penalty** $\mathcal{L}_{\text{bound}}$: penalises positions outside
  `region_lb`/`region_ub` via a soft hinge loss; coefficient $\lambda$ is annealed
  during training.
- **Noise model** $\tau(t,p)$: parameterised as B-spline coefficients (learnable),
  evaluated at each time $t$ for each receiver $p$.  Allows receiver noise to vary
  smoothly over time (e.g., due to temperature-induced phase shifts).

#### 4e. Adaptive Weights

Weights $w(t,p)$ are updated every 500 iterations:

$$w(t,p) = \begin{cases}
  1 & \text{if } |r(t,p)| \leq k \cdot \text{median}(|r|) \\
  \frac{k \cdot \text{median}(|r|)}{|r(t,p)|} & \text{otherwise (Huber-like downweighting)}
\end{cases}$$

with $k = 3$ by default.

#### 4f. Optimisation Schedule

| Phase | Optimiser | Iterations | Notes |
|-------|-----------|-----------|-------|
| Warm-up | Adam | 2 000 | High LR, position only |
| Main | Adam | 8 000 | All parameters, LR decay $0.98^{\lfloor \text{step}/250 \rfloor}$ |
| Refinement | L-BFGS | up to 500 | Fine-grained convergence |

#### 4g. Uncertainty Estimation — Multi-Sample Mode

When `nSamples > 1`:

1. Train the model on the original data → best-estimate trajectory.
2. For each of the $N$ samples:
   - Add noise perturbation to TOA observations: $\tilde{\text{TOA}} = \text{TOA} + \epsilon$, $\epsilon \sim \mathcal{N}(0, \sigma_p^2)$.
   - Re-optimise from the current parameter state.
   - Record resulting `xyz`.
3. Report:
   $$\bar{\mathbf{xyz}} = \frac{1}{N}\sum_n \mathbf{xyz}_n, \quad
     \sigma_{\mathbf{xyz}} = \text{std}_n(\mathbf{xyz}_n)$$

The per-axis standard deviation $\sigma_{\mathbf{xyz}}$ is saved alongside the position
estimate as a meaningful uncertainty metric.

---

### 5. `postProcess.py` — Error Analysis and Output

After both solvers run, `postProcess` generates:

1. **Per-signal tracking CSV**: `[datetime, X, Y, Z, X_std, Y_std, Z_std, nValid, residual]`
2. **GPS comparison** (if ground-truth GPS is provided):
   - Interpolate GPS track to acoustic detection times (nearest-neighbour, tolerance
     = 3 s by default).
   - Compute 2-D and 3-D tracking errors.
   - Generate error-vs-time and residual scatter plots.
3. **Tracking efficiency**: $\eta = N_{\text{solved}} / (T_{\text{span}} / \text{PRI})$
4. **Beacon QC**: solve beacon positions from tracking output, compare to known input
   positions, report RMSE.

---

## Data Flow

```
Synced decodes pickle  (from Sync stage)
        │
        ▼
dataLoader.TOAinfo_creator
  ├─ Group by transmission event
  ├─ Build sparse TOA matrix [N_signals × N_receivers]
  ├─ apply_TOAD_filter()   → outlier detections removed
  └─ Split into segments   (≤ numTrack events each)
        │
        ├─────────────────────────┐
        ▼                         ▼
AML.AMLSolver()           ML_NN.Model (PyTorch)
 per signal:               per segment:
  ├─ Compute TDOAs          ├─ Encode time → φ(t)
  ├─ Linearise (AML)        ├─ MFNN: φ(t) → xyz(t)
  ├─ Solve quadratic         ├─ Forward: predict TOA
  ├─ Iterative reweight      ├─ Huber NLL loss
  └─ Outlier rejection       ├─ Adam + L-BFGS
        │                    └─ Multi-sample uncertainty
        │                         │
        └───────────┬─────────────┘
                    ▼
            postProcess.run()
              ├─ Tracking CSV
              ├─ GPS comparison plots
              ├─ Beacon QC
              └─ Tracking efficiency report
```

---

## Comparison: AML vs Neural Network

| Aspect | AML | Neural Network |
|--------|-----|----------------|
| Solve type | Per-signal (independent) | Per-segment (trajectory) |
| Computation | Fast (closed-form algebra) | Slow (gradient-based optimisation) |
| Smoothness | None | Implicit via network architecture |
| Uncertainty | None | Per-axis std via multi-sample |
| Bad receiver handling | Explicit rejection loop | Soft (via adaptive weights / $\tau$) |
| Long gaps | Handled natively | Requires segment splitting |
| Robustness | Sensitive to poor geometry | More robust (trajectory prior) |
| Best for | Dense arrays, near-field | Sparse arrays, long trajectories |

---

## Key Tunable Parameters

| Parameter | Typical Value | Description |
|-----------|---------------|-------------|
| `dis_limit` | 500 m | Maximum acceptable receiver–tag range |
| `dis_tol` | 3 m | TDOA residual threshold for outlier rejection |
| `region_lb/ub` | deployment-specific | Spatial bounding box for valid positions |
| `numTrack` | 5 000 | Maximum signals per NN training segment |
| `nSamples` | 1 (or 10) | Number of noise-perturbation samples for uncertainty |
| `nn_layers` | [200, 50, 50, 50, 50, 50, 50, 3] | MFNN architecture |
| TOAD window | 21 points | Median filter window in `apply_TOAD_filter` |

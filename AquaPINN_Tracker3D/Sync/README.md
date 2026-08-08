# Stage 2: Time Synchronization (`Sync/`)

## Overview

Crystal oscillators in autonomous acoustic receivers drift at different rates, causing
each receiver's clock to diverge from true UTC over the deployment period.  This stage
estimates and removes that drift so that TDOA (Time-Difference-of-Arrival) calculations
in the tracking stage are valid.

The Python sync engine goes beyond the MATLAB pairwise Huber IRLS approach: it solves a
**joint physics-based optimization** over all receivers simultaneously using PyTorch,
allowing it to absorb non-linear (quadratic) drift, discrete clock step-jumps, and
uncertainty in beacon positions all in one pass.

---

## Module Map

| Module | Role |
|--------|------|
| `syncer.py` | Top-level orchestrator; calls the modules below in sequence |
| `dataLoader.py` | Pre-processing: builds TOA matrices, TOAD filtering, `preSync` |
| `jumpDetection.py` | CUSUM-based clock step-jump detection and piecewise linear fit |
| `graph.py` | Receiver connectivity graph; Floyd-Warshall; reference-node selection |
| `ML.py` | Joint physics ML model (PyTorch): clock-drift + beacon-location estimation |
| `ransac.py` | RANSAC helpers used inside ML training |

---

## Algorithm Details

### 1. `syncer.py` — Orchestrator

`run(param_file, loc_sys)` pipeline:

1. Load parameters and decode files.
2. Call `dataLoader.preSync()` → extract candidate time-of-transmission (top) values
   per beacon–receiver pair.
3. Build `TOAinfo_creator` → sparse TOA matrices for the ML model.
4. For each beacon:
   - Call `ML.analyzePairDetection()` → assess linearity of each pair, detect jumps.
   - Train `ML.Model` (PyTorch optimizer) → estimate clock-drift coefficients.
5. Call `loc_sys.applySync2Decodes()` → apply corrections to every detection.

---

### 2. `dataLoader.py` — Pre-processing and TOA Assembly

#### 2a. `preSync(beacon_decodes, phoneInfo, soundSpeed)` — Time-of-Flight Correction

For each beacon–receiver pair `(b, p)`:

$$\text{TOF}_{b,p} = \frac{d(x_b, x_p)}{c(T)}$$

$$\text{top}_{b,p,t} = \text{TOA}_{b,p,t} - \text{TOF}_{b,p}$$

where:
- $d(x_b, x_p)$ = Euclidean distance (metres) between beacon and receiver.
- $c(T)$ = freshwater sound speed from Marczak (1997) polynomial (see `SharedLibs`).
- $\text{TOA}_{b,p,t}$ = raw arrival time at receiver $p$ for beacon transmission $t$.

The resulting `top` estimates for a given receiver pair should be identical (within
noise) if the clocks are in sync.  Systematic offsets and drift are the signal we
estimate in later steps.

#### 2b. `modeFilter(values, int_bin)` — Outlier Rejection via Modal Consensus

Before building the TOA matrices for the ML model, isolated outlier detections (false
positives, interference) are removed:

1. Quantise `values` into bins of width `int_bin`.
2. Slide a window of length `window_size` across the sorted sequence.
3. Compute the **mode** (most-frequent bin value) in each window.
4. A detection is retained if its quantised value matches the local mode; otherwise it
   is marked as an outlier (`-999`).

This is analogous to RANSAC consensus: the majority of legitimate detections share a
consistent relative time offset, while outliers scatter randomly.

#### 2c. `TOAinfo_creator` Class — Sparse TOA Matrix Builder

Converts per-receiver detection lists into a compact sparse matrix:

- Rows = signal transmission events (indexed by nominal transmission number).
- Columns = receiver Phone IDs.
- Values = raw TOA (seconds since reference epoch), or `−999` for missing/outlier.

Steps:

1. Group detections by tag code and sort by time.
2. Assign transmission indices using the nominal PRI:
   $n = \text{round}((t - t_0) / \text{PRI})$
3. Resolve collisions (duplicate assignments) by keeping the detection with the highest
   SNR.
4. Store as `scipy.sparse.csr_matrix` for memory efficiency.

#### 2d. `apply_TOAD_filter(TOA_matrix, window=21)` — TDOA Consistency Check

Removes detections inconsistent with the smoothed TDOA time series:

1. For each receiver pair $(i, j)$:
   $$\text{TDOA}_{t} = \text{TOA}_{t,i} - \text{TOA}_{t,j}$$
2. Apply a median filter (window = 21 points) to get a smoothed reference TDOA.
3. Flag any point where $|\text{TDOA}_t - \text{TDOA}_{\text{smoothed}}| > 3\sigma$ as
   an outlier.
4. Set flagged entries to `−999` and iterate until no new outliers are found (up to
   100 iterations).

This iterative approach handles the case where a burst of bad detections distorts the
initial median estimate.

---

### 3. `jumpDetection.py` — Clock Step-Jump Detection

Crystal oscillators occasionally exhibit **discrete jump discontinuities** — step
changes of order microseconds to milliseconds — on top of the smooth linear drift.
These are identified per receiver before the ML model is trained.

#### 3a. `detect_jump(signal, threshold, max_median_ratio=10)` — CUSUM Algorithm

Applies the CUSUM (Cumulative Sum) algorithm on the detrended clock-offset time series:

```
g_pos[0] = 0
g_neg[0] = 0
for t = 1 .. N:
    diff = signal[t] - signal[t-1]
    g_pos[t] = max(0, g_pos[t-1] + diff - k)   # positive accumulator
    g_neg[t] = min(0, g_neg[t-1] + diff + k)   # negative accumulator
    if g_pos[t] > threshold: flag upward jump at t
    if g_neg[t] < -threshold: flag downward jump at t
```

A candidate jump is only accepted if:
$$|\text{jump height}| > \text{max\_median\_ratio} \times \text{median}(|\Delta \text{signal}|)$$

This rejects noise excursions and retains only genuine discontinuities that are
significantly larger than the background drift variation.

#### 3b. `signal_linear_fit(signal, times, threshold)` — Piecewise Polynomial Model

After jump times are detected, the full clock correction is modelled as a piecewise
polynomial with step functions:

$$\hat{f}(t) = a t^2 + b t + c + \sum_{i} h_i \cdot \mathbf{1}[t \geq t_{j,i}]$$

Parameters $(a, b, c, h_1, h_2, \ldots)$ are solved jointly via **ordinary least
squares** (NumPy `lstsq`) on the design matrix:

$$\mathbf{X} = \left[ t^2,\ t,\ \mathbf{1},\ \mathbf{1}[t \geq t_{j,1}],\ \ldots \right]$$

Spurious small jumps ($|h_i| < \epsilon$) detected in one iteration are removed and
the fit is repeated (`correctJump` loop).

---

### 4. `graph.py` — Connectivity Graph and Reference Selection

Before estimating clock corrections, the algorithm must determine which receivers are
mutually observable (i.e., share enough co-detected beacon signals) and choose one as
the global time reference.

#### 4a. `floyd_warshall(adj_matrix)` — All-Pairs Shortest Paths

Given a binary adjacency matrix $A$ (1 = connected pair, 0 = no link):

$$D[i][j] = \min_k \left( D[i][k] + D[k][j] \right)$$

Standard Floyd–Warshall initialised with $D[i][i]=0$, $D[i][j]=1$ if adjacent, else
$\infty$.  Returns the minimum hop count between every receiver pair.

This is used to:
- Identify isolated sub-graphs.
- Compute graph centrality for reference-node scoring.

#### 4b. `find_best_nodes(adj_matrix, quality_matrix=None)` — Reference Node Selection

Scores each receiver node and returns the one most suitable to act as the global
reference clock:

1. **Degree score**: number of directly connected neighbours.
2. **Centrality score**: sum of shortest-path distances to all other nodes (lower = more
   central).
3. If a `quality_matrix` is provided (e.g., pairwise sync RMSE), it is used as edge
   weights in the distance sum.
4. The node with the **highest degree**, broken by **lowest total distance**, is chosen
   as reference.

#### 4c. `check_solvability(adj_matrix, ref_node)` — Reachability Check

Performs a DFS from `ref_node` on the connectivity graph.  Any receiver not reachable
(i.e., in a disconnected sub-graph) is flagged as **unsolvable** and excluded from the
sync plan.  This prevents incorrect cross-component corrections.

---

### 5. `ML.py` — Joint Physics-Based ML Synchronizer

This is the core and most novel component.  Rather than fitting one receiver pair at a
time (as in the MATLAB Huber IRLS approach), it optimises all receiver clocks and beacon
positions **simultaneously** using automatic differentiation.

#### 5a. `analyzePairDetection(TOA_matrix, phoneInfo)` — Pre-training Quality Check

Before training, checks whether each beacon–receiver pair's TDOA series is consistent
with a smooth polynomial drift:

1. For each pair $(b, p)$, fit a quadratic $\text{TDOA}(t) = at^2 + bt + c + \text{jumps}$.
2. Compute the **inlier ratio**: fraction of detections within $3\sigma$ of the fit.
3. If inlier ratio $< 0.40$: flag the pair as problematic (`errFlag[b,p] = True`).
4. Also invoke `detect_jump()` to record jump times for model initialisation.

Flagged pairs are down-weighted or excluded during ML training.

#### 5b. `Model` Class (PyTorch `nn.Module`) — Learnable Parameters

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `top[t]` | $(N_{\text{sig}},)$ | Time-of-transmission for signal $t$ |
| `shift_polyVec[p, :, k]` | $(N_P, 3, N_{\text{seg}})$ | Piecewise quadratic drift coefficients $(a,b,c)$ per receiver $p$, time segment $k$ |
| `t_jump[p]` | $(N_P, N_J)$ | Normalised times of clock jumps per receiver |
| `value_jump[p]` | $(N_P, N_J)$ | Jump heights (seconds) per receiver |
| `phoneLocReal[p]` | $(N_P, 3)$ | Receiver location correction (small adjustment around nominal) |
| `beaconPhoneDis[p]` | $(N_P,)$ | Distance from beacon housing to hydrophone face |
| `toa_tau[p]` | $(N_P,)$ | TOA noise standard deviation per receiver |

#### 5c. Clock Correction Model — Forward Pass

Time is **normalised** to $[-1, 1]$ to avoid numerical cancellation at $\sim 10^{10}$ s
absolute time values:

$$\tau_{\text{norm}} = \frac{t - \frac{t_{\max}+t_{\min}}{2}}{\frac{t_{\max}-t_{\min}}{2}} \in [-1,\ 1]$$

The correction applied to the raw TOA at receiver $p$ for signal $t$ is:

$$\Delta(t, p) = a_p(\tau)\ \tau_{\text{norm}}^2 + b_p(\tau)\ \tau_{\text{norm}} + c_p(\tau) + \sum_{i} h_{p,i}\ \mathbf{1}\!\left[\tau_{\text{norm}} \geq \tau_{j,p,i}\right]$$

where $(a_p(\tau), b_p(\tau), c_p(\tau))$ are selected from the piecewise segments
based on which time segment $\tau$ falls in.

The **corrected TOA** is:

$$\text{TOA}_{\text{sync}}(t, p) = \text{TOA}_{\text{raw}}(t, p) + \Delta(t, p)$$

#### 5d. Physical Forward Model — Predicted TOA

Given corrected clock parameters and estimated locations, the predicted TOA at receiver
$p$ for signal $t$ emitted from beacon $b$ is:

$$\hat{\text{TOA}}(t, p) = \text{top}(t) + \frac{d(\hat{x}_b + \delta_b,\ \hat{x}_p + \delta_p)}{c(T)}$$

where:
- $\hat{x}_b$ = known nominal beacon position.
- $\delta_b$ = learned offset (`phoneLocReal` component).
- $\hat{x}_p$ = known nominal receiver position.
- $\delta_p$ = learned receiver position correction.
- $c(T)$ = Marczak (1997) freshwater sound speed.

The residual is:

$$r(t, p) = \hat{\text{TOA}}(t, p) - \text{TOA}_{\text{sync}}(t, p)$$

#### 5e. Loss Function — Huber-Like + Log-Variance

The loss is a robust heteroscedastic negative log-likelihood:

$$\mathcal{L} = \sum_{t} \sum_{p \in \text{valid}(t)} \left[
  \frac{w(t,p)^2\ r(t,p)^2}{2\ \tau_p^2} - \log \tau_p
\right]$$

where:
- $\tau_p$ = learnable noise standard deviation (per receiver).
- $w(t,p)$ = **adaptive weight** updated at fixed intervals:
  - If $|r(t,p)| > k \cdot \text{median}(|r|)$: weight is reduced (Huber-like downweighting).
  - If a pair has many valid detections: weight is boosted.

The $-\log \tau_p$ term prevents the model from trivially minimising loss by inflating
$\tau_p$ (Fisher regularisation).

#### 5f. Optimisation Schedule

1. **Adam** optimiser, 10 000 iterations, learning rate $= 10^{-3}$.
2. Learning rate decay: $\text{lr} \leftarrow \text{lr} \times 0.98^{\lfloor \text{step}/250 \rfloor}$.
3. **L-BFGS** refinement for final convergence (0–500 iterations depending on settings).
4. After training: identify "problematic phones" (receivers whose residuals remain
   consistently large) → fix their parameters and re-train the rest.

#### 5g. Output — Sync Coefficients

After training, the following are saved to `syncFile` (pickle):

| Key | Content |
|-----|---------|
| `coeff_sync[phoneID]` | `(a, b, c)` arrays per segment, `t_jump`, `v_jump` |
| `phoneLoc` | Corrected receiver XYZ coordinates |
| `phoneIDs` | List of synced receiver IDs |
| `tmin`, `tmax` | Absolute time bounds used for normalisation |

---

## Data Flow

```
Beacon decodes pickle  (from RawDataProcess stage)
        │
        ▼
preSync()                → top = TOA − TOF (per pair, nominal positions)
modeFilter()             → reject outlier detections (modal consensus)
        │
        ▼
TOAinfo_creator          → sparse TOA matrix [N_signals × N_receivers]
apply_TOAD_filter()      → iterative TDOA consistency pruning (≤100 iter)
        │
        ▼
analyzePairDetection()   → inlier ratio, errFlag matrix, jump init times
        │
        ▼
graph.floyd_warshall()   → all-pairs hop distances
graph.find_best_nodes()  → choose global reference receiver
graph.check_solvability() → flag isolated / unreachable receivers
        │
        ▼
ML.Model (PyTorch)
  ├─ Forward: predict TOA from top + position + polynomial drift
  ├─ Loss: Huber-weighted heteroscedastic NLL
  ├─ Optimise: Adam (10 000 iters) → L-BFGS refinement
  └─ Output: coeff_sync dict, corrected phoneLoc
        │
        ▼
applySync2Decodes()      → apply poly+jump corrections to all decodes
        │
        ▼
pickle: synced decodes  (same N×4 format, col 1 corrected)
```

---

## Comparison with MATLAB Equivalent

| Aspect | MATLAB (`receiver_connectivity_graph_fromDecodes_...`) | Python (`Sync/ML.py`) |
|--------|------------------------------------------------------|-----------------------|
| Scope | Pairwise (one receiver pair at a time) | Joint (all receivers simultaneously) |
| Drift model | Linear (affine: $at + c$) | Piecewise quadratic ($at^2 + bt + c$) |
| Jump handling | MAD-threshold iterative removal | CUSUM detection + LS piecewise fit |
| Robust fitting | Huber IRLS (iterative reweighting) | Adaptive-weight NLL with learnable $\tau$ |
| Reference selection | BFS from max-degree node | Floyd-Warshall + degree + centrality |
| TOA pre-filter | TOAD window (PRI-adaptive) | TOAD iterative TDOA + modeFilter |
| Position update | None (positions fixed) | Learns small corrections to beacon/receiver positions |
| Framework | MATLAB `lscov` / manual IRLS | PyTorch autograd + Adam + L-BFGS |

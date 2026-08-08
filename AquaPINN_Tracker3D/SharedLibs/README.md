# Shared Libraries (`SharedLibs/`)

## Overview

`SharedLibs` provides the common building blocks used by all three pipeline stages
(`RawDataProcess`, `Sync`, `Track`).  It contains physics models, neural network
architectures, training utilities, coordinate-transform helpers, and general-purpose
decorators.  Nothing in this folder runs independently — everything is imported by the
stage modules.

---

## Module Map

| Module | Role |
|--------|------|
| `functions.py` | Core utility functions: sound speed, parameter loading, tensor conversion |
| `networks.py` | Neural network architectures: MFNN, MFNN_feature, BSplineModel3D |
| `networks_numpy.py` | NumPy-only equivalents for inference without PyTorch |
| `batchSampler.py` | Custom PyTorch batch sampler for variable-length sequences |
| `BayesianNetworks.py` | Bayesian / variational inference wrappers (experimental) |
| `decorators.py` | `@timing`, `@treeFunction` decorator utilities |
| `visualization.py` | Detection and tracking visualisation helpers |

---

## Algorithm Details

### 1. `functions.py` — Core Physics and Utilities

#### 1a. `soundSpeed_freshWater(T)` — Marczak (1997) Polynomial

Computes the speed of sound in fresh water as a function of temperature $T$ (°C):

$$c(T) = 1.402385 \times 10^3
       + 5.038813\, T
       - 5.799136 \times 10^{-2}\, T^2
       + 3.287156 \times 10^{-4}\, T^3
       - 1.398845 \times 10^{-6}\, T^4
       + 2.78786  \times 10^{-9}\, T^5$$

Units: m/s.  Valid range: $0 \leq T \leq 40$ °C.

**Why this matters for timing accuracy:**  A 1 °C error in temperature at $T = 15$ °C
causes a sound-speed error of $\approx 3.5$ m/s, which translates to a TOF error of
$\approx 2.3\ \mu$s per 1 m of path length — comparable to the sync residuals achieved
by the ML model.

This function is identical to `sound_speed_fresh_water.m` in the MATLAB pipeline,
ensuring numerical consistency across both codebases.

#### 1b. `load_params(param_file)` — JSON Parameter Combinations

Reads a JSON configuration file where each key maps to a **list** of candidate values,
and returns the full **Cartesian product** of all combinations:

```json
{
  "learning_rate": [1e-3, 1e-4],
  "n_layers":      [4, 6]
}
```

Returns: `[{"learning_rate": 1e-3, "n_layers": 4}, {"learning_rate": 1e-3, "n_layers": 6}, ...]`

Used for hyperparameter sweeps across deployments.

#### 1c. `treeToNumpy` / `treeToTensor` — Recursive Structure Conversion

Implemented via the `@treeFunction` decorator (see `decorators.py`):

- **`treeToNumpy`**: Recursively traverses nested `list`/`dict`/`tuple` structures and
  calls `.detach().cpu().numpy()` on any `torch.Tensor` leaf node.
- **`treeToTensor`**: Converts any `np.ndarray` leaf to `torch.Tensor` with
  `torch.defaultReal` dtype and `torch.defaultDevice`.

These allow pipeline code to work with mixed NumPy/PyTorch data without manual
traversal boilerplate.

---

### 2. `networks.py` — Neural Network Architectures

#### 2a. `MFNN` — Modified Feed-Forward Neural Network

Standard fully-connected network with a **gated residual mechanism** at each hidden
layer that prevents gradient vanishing in deep architectures:

**Architecture (single hidden layer shown):**

```
Input x
   │
   ├──► U = ReLU(W_U x + b_U)   (gate branch 1)
   ├──► V = ReLU(W_V x + b_V)   (gate branch 2)
   │
   ▼ (for each middle layer l)
   Z_l = ReLU(W_l x + b_l)
   x   = Z_l ⊙ U + (1 − Z_l) ⊙ V   (gated blend)
   │
   ▼
Output = W_out x + b_out
```

The gating $Z \odot U + (1-Z) \odot V$ allows the network to **interpolate** between
two learned feature spaces.  This is particularly effective for smooth trajectory
modelling because the network can blend a "global trend" direction (U) with a "local
correction" direction (V).

Default layers for tracking: `[n_features, 200, 50, 50, 50, 50, 50, 50, 3]`
(output is 3D position).

#### 2b. `MFNN_feature` — MFNN with Learnable Distance Scaling

Extends `MFNN` with a trainable per-input scaling vector `filterRatio`:

$$\tilde{x} = x \cdot (1 + \alpha \cdot d)$$

where:
- $\alpha$ = `filterRatio` (learned $\in \mathbb{R}^{d_{\text{in}}}$).
- $d$ = `distance` (fixed scalar, default 0.2).

This allows the network to automatically up-weight or ignore specific input feature
dimensions, acting as a soft input gate.

**Initialisation:** Glorot (Xavier) uniform for all weight matrices; zero bias.  This
avoids the symmetric-weight problem and maintains activation variance across layers.

#### 2c. `BSplineModel3D` — Learnable B-Spline Basis Function

Models a 1-D smooth function over normalised time $t \in [0, 1]$ with learnable control
points.  Used to parameterise the **noise standard deviation** $\tau(t, p)$ in the
tracking NLL loss — allowing it to vary smoothly rather than being a fixed scalar.

**Basis function (order $k$, de Boor recursion):**

$$B_{i,1}(t) = \begin{cases} 1 & t_i \leq t < t_{i+1} \\ 0 & \text{otherwise} \end{cases}$$

$$B_{i,k}(t) = \frac{t - t_i}{t_{i+k-1} - t_i} B_{i,k-1}(t)
             + \frac{t_{i+k} - t}{t_{i+k} - t_{i+1}} B_{i+1,k-1}(t)$$

The full model output is:

$$f(t) = \sum_{i=0}^{N_c-1} c_i\, B_{i,k}(t)$$

where $c_i$ are the learnable control points (`ctrl_pts`).

**Knot vector** is constructed with clamped ends:
- $k$ repeated knots at 0 and 1 (for interpolation at boundaries).
- Uniform interior knots.

**Usage in tracking loss:**

$$\tau(t, p) = \text{BSpline}_p(t_{\text{norm}}) \quad \forall\ p$$

Each receiver has its own B-spline parameter set, allowing the noise model to adapt to
receiver-specific non-stationarity.

---

### 3. `decorators.py` — Utility Decorators

#### 3a. `@timing` — Function Execution Timing

Wraps a function to log its name and wall-clock execution time at `INFO` level:

```python
@timing
def run(param_file, loc_sys):
    ...
```

Output: `INFO: run completed in 12.34 s`

#### 3b. `@treeFunction` — Recursive Structure Applicator

Factory that creates a decorator applying a leaf-level transformation recursively
through nested Python containers (`list`, `tuple`, `dict`):

```python
treeToNumpy = treeFunction(lambda x: x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x)
```

This pattern allows functions returning complex nested structures of tensors to be
converted to NumPy arrays without any change to the function itself.

---

### 4. `batchSampler.py` — Training Batch Sampler

Provides a custom `torch.utils.data.Sampler` that handles **variable-length sequences**:

- Groups signals into batches such that each batch has at most `max_tokens` total
  detections (across all receivers), preventing GPU memory overflows on dense segments.
- Sequences can be padded or packed as required by the loss function.

Used by `ML_NN.py` when training on large data segments.

---

### 5. `visualization.py` — Detection and Tracking Plots

Helper functions called by both `decoder.py` (ingest) and `postProcess.py` (tracking):

- **`showDetections(decodesFile, tagType, outputdir, timeZone)`**: Plots detection count
  per receiver per time bin for all tags.  Useful for diagnosing gaps and multipath.
- Coverage heatmaps and receiver-pair co-detection matrices (used in sync QC).

---

## Dependency Map

```
RawDataProcess/
  └── functions.py     (dtime2dnum, load_params, treeToNumpy)
  └── decorators.py    (@timing, @treeFunction)
  └── visualization.py (showDetections)

Sync/
  └── functions.py     (soundSpeed_freshWater, treeToTensor/Numpy, load_params)
  └── networks.py      (MFNN — not used in sync ML directly)
  └── decorators.py    (@timing)

Track/
  └── functions.py     (soundSpeed_freshWater, treeToTensor/Numpy, load_params)
  └── networks.py      (MFNN_feature → position model, BSplineModel3D → τ(t,p))
  └── batchSampler.py  (ML training batching)
  └── visualization.py (GPS comparison plots)
```

---

## Design Principles

1. **No stage-specific logic** lives in `SharedLibs`.  Any code that makes assumptions
   about whether it is being called from sync or tracking belongs in those stage modules.

2. **Framework agnosticism where possible**: `networks_numpy.py` provides NumPy-only
   inference equivalents so that deployment scripts that lack a GPU or PyTorch
   installation can still run the position solver.

3. **Explicit tensor precision**: All tensors are created with `torch.defaultReal`
   (configured in `config.py`) to allow switching between `float32` (GPU, speed) and
   `float64` (CPU, precision) without changing algorithm code.

# Parameter drift evaluation

`parameter_drift/evaluate_parameter_drift.py` evaluates existing single AL-RNN checkpoints; it
does not train models. It uses the ODEs, observations, parameter conventions and
default endpoints in `generate_dynamic_regime_trajectories.py`.

## Quick Start On The Server

From `ALRNN-DSR/alrnn_python`, run:

```bash
bash parameter_drift/run_halvorsen_parameter_drift.sh 2
```

The wrapper activates `$ROOT_DIR/DynaMix-python-b-tipping/venv`, with
`ROOT_DIR=${ROOT_DIR:-$HOME}`. It uses the saved M=200, P=20 family model by
default. Extra arguments after the GPU ID select different or multiple runs:

```bash
bash parameter_drift/run_halvorsen_parameter_drift.sh 2 \
  runs/halvorsen_14to209_noisy_m200_p20_cphi \
  runs/halvorsen_9params_all_m100_p10_vanilla
```

There is no need to install the Python Plotly package. The HTML plots load Plotly
from its CDN when opened in a browser. For offline use, supply an existing
Plotly JavaScript bundle with `--plotly-js /path/to/plotly.min.js`; it will be
embedded in each HTML file. Standard evaluation dependencies are NumPy, SciPy,
PyTorch and the dependencies of the repository's `metrics.py`.

## Python CLI

With the venv activated:

```bash
python parameter_drift/evaluate_parameter_drift.py \
  --regime halvorsen \
  --steps 12000 \
  --burn-in-steps 4000 \
  --drift linear \
  --model-runs \
    runs/halvorsen_14to209_noisy_m200_p20_cphi \
    runs/halvorsen_9params_all_m100_p10_vanilla \
  --model-labels Cphi Vanilla \
  --ground-truth runs/parameter_drift/ground_truth/halvorsen_linear_12000.npz \
  --output-dir runs/parameter_drift/comparison_01 \
  --metric dh:8000:12000 \
  --metric dstsp:0:3000 \
  --metric mse:0:1000 \
  --device cuda --gpu-id 2 --cpu-threads 4
```

Each run must contain `model.pt` and `metrics.json`. Architecture and P are read
from these files; the model weights are not reinitialized. Vanilla models are
also accepted as controls and ignore phi. No context encoder or teacher forcing
is used. The observed initial state initializes `z = x0 @ B`, with its first
three dimensions overwritten by `x0`, exactly as in the tutorial.

## Ground Truth And Indexing

- The default is 4,000 **true-system** burn-in integration steps at the fixed
  start parameter, followed by 12,000 stored states with drifting parameter.
- `--steps` counts stored states including the initial state. Index 0 is the
  post-burn-in initial state. The model therefore makes 11,999 transitions for
  12,000 stored states. Its prediction at index 0 equals ground truth.
- `phi[t]` drives the model transition from index `t` to `t+1`. The true ODE uses
  the continuously varying parameter within the sampling interval. Sampling
  defaults to `dt=0.01`; the cached time array stores the actual times.
- The burn-in is not fed to the model. It is stored separately for inspection.
- Simulation uses DOP853 and the same derivative clipping as the generator.
  There is no observation noise in this evaluation.

The cache `.npz` contains standardized/model-coordinate observations (`x`),
physical observations (`observed`), full ODE states (`states`), `times`, normalized
`phi`, raw `parameter`, fixed `mean` and `std`, `burn_in_observed`, the sampled
`initial_before_burn`, and the JSON string `metadata_json`.

The cache is model-independent and automatically reused when present. Explicit
simulation options must agree with its stored settings; a mismatch raises an
error instead of silently changing the ground truth. To evaluate another model
against exactly the same data, use:

```bash
python parameter_drift/evaluate_parameter_drift.py \
  --reuse-ground-truth \
  --ground-truth runs/parameter_drift/ground_truth/halvorsen_linear_12000.npz \
  --model-runs runs/ANOTHER_RUN \
  --output-dir runs/parameter_drift/comparison_02 \
  --metric dh:8000:12000 --metric dstsp:0:3000 --metric mse:0:1000 \
  --device cuda --gpu-id 2
```

Reuse does not import the regime file or run the ODE solver. Omitted simulation
options use the cache settings. Pass `--regenerate-ground-truth` to explicitly
replace a cache, or choose another cache filename for a new experiment. Use
`--simulate-only` without `--model-runs` to generate just a reusable cache.

## Normalization

The default `--normalization trajectory` follows the existing generator: compute
one mean/std per observed dimension over the complete drift trajectory. The
same fixed transformation applies at every time and to every model. It includes
future ground-truth data in the choice of coordinates and is not an online
normalizer. Predictions never receive future state observations.

Training trajectories were standardized individually at constant parameters.
Standardizing an entire drifting trajectory can produce different coordinates;
reconstruction errors can therefore reflect both model dynamics and this change
of normalization. No time-local or parameter-dependent rescaling is performed.

For a controlled fixed coordinate system, use
`--normalization-stats stats.npz`, containing length-three `mean` and `std`
vectors. For example, an existing ground-truth cache already contains these
keys and can supply the scaling for a **new** simulation. Use
`--normalization none` only for models trained on physical observations. Scaling
is always stored with the ground truth and remains unchanged during cache reuse.

Phi uses the original regime's boundary and scale, also when custom drift
endpoints are supplied. Restricting a sweep never renormalizes it back to
[-1, 1]. Raw default endpoints are taken literally from the regime definition;
their normalized values need not be exactly -1 and +1. Use
`--parameter-start/--parameter-end` for raw endpoints, or
`--phi-start/--phi-end` for normalized endpoints. Do not mix the two forms.

## Metrics And Plots

`--metric NAME:START:STOP` is repeatable. Indices are zero-based and STOP is
exclusive, so `dh:8000:12000` evaluates exactly 4,000 samples. `end` means the
trajectory length. The supported names are `mse`, `dh` and `dstsp` (`dstps` is
accepted as an alias). With no metric options, all three use the full trajectory.

MSE averages squared errors over the selected states and all three observed
dimensions. DH and Dstsp call the repository's existing `metrics.py` functions,
with default spectral smoothing 20 and 30 histogram bins. All metrics are
computed in model/standardized coordinates, even with physical-coordinate plots.
Non-finite predictions or undefined metrics are recorded as `null` with an
explicit status, not as a misleading zero. Finite prefix intervals can still be
evaluated if the model diverges later.

DH and Dstsp summarize the selected interval. On drifting systems this is a
comparison of distributions/spectra over that interval, not proof of matching
instantaneous dynamics. Long-horizon pointwise MSE is also sensitive to chaotic
phase divergence. Interpret these metrics together with the time series.

Outputs:

- `rollout_3d.html`: rotatable 3D overlay of truth and all models, including
  step/phi hover information. Legend entries can hide or isolate models.
  Each trajectory has its own light-to-dark color scale encoding the current phi.
- `components.html`: x, y, z and phi versus step, with linked horizontal zoom.
  State samples use the same phi-dependent color gradients as the 3D plot.
- `model_XX_LABEL_rollout_3d.png`: one static 3D overlay per model, showing its
  prediction against the ground truth, with separate phi colorbars (200 dpi).
- `model_XX_LABEL.npz`: aligned model-coordinate and physical predictions,
  phi and time arrays, saved before metrics/plots.
- `metrics.json`: ground-truth SHA256, simulation settings, model weight hashes,
  model details and per-interval results. Updated after every model.
- `metrics.csv`: one row per model and requested metric interval.

All states are plotted without downsampling. Use `--plot-space physical` to
display observations and predictions in physical observation units.

Ground truth uses light-to-dark blue; models use orange, green, purple and other
distinct color families. Lighter colors always mean lower phi, darker colors
higher phi, using one shared numeric range across all trajectories. Colors track
the actual parameter, not elapsed time, so a decreasing drift becomes lighter
again. Static line segments use the mean phi of their two endpoints; HTML 3D
lines interpolate the colors between samples. Non-finite gaps are not connected.

## Other Regimes And Drift Shapes

On the server the default regime file is
`$HOME/diffrentbutconst_prams/generate_dynamic_regime_trajectories.py`. A sibling
local `generate_trainingdata` checkout is also detected. Override the location
with `--regime-file` and list the available systems with `--list-regimes`.

Examples of generation-only experiments:

```bash
python parameter_drift/evaluate_parameter_drift.py --simulate-only \
  --regime dadras_e_49_to_64 --drift sigmoid --sigmoid-steepness 10 \
  --ground-truth runs/parameter_drift/ground_truth/dadras_sigmoid.npz

python parameter_drift/evaluate_parameter_drift.py --simulate-only \
  --regime halvorsen --drift ramp-hold --ramp-fraction 0.5 \
  --ground-truth runs/parameter_drift/ground_truth/halvorsen_ramp_hold.npz
```

`ramp-hold` reaches the final parameter halfway through the recorded duration
when the fraction is 0.5, then holds it constant. Sigmoid is scaled to reach the
specified endpoints exactly. To add another shape, add a progress function to
the `DRIFTS` registry in the Python script. This one function controls both the
continuous ODE parameter and the sampled phi supplied to the models.

## Verification

```bash
python -m unittest discover -s alrnn_python/parameter_drift -p test_parameter_drift.py -v
```

Run this from the repository root. It checks the inference recurrence, temporal
alignment, an analytic time-varying ODE with burn-in, drift shapes, phi scaling,
cache reuse, checkpoint loading and interval/error handling.

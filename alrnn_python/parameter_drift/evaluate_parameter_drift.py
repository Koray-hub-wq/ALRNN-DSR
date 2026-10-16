"""Evaluate saved single AL-RNNs on a cached, time-varying dynamical system.

No training code is imported. Regime definitions and ODEs come from the existing
generate_dynamic_regime_trajectories.py. Plotly HTML needs no Python Plotly package.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import html
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys

os.environ.setdefault("MPLBACKEND", "Agg")

ALRNN_DIR = Path(__file__).resolve().parents[1]
if str(ALRNN_DIR) not in sys.path:
    sys.path.insert(0, str(ALRNN_DIR))

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import expit
import torch
from torch import nn


CACHE_VERSION = 1
DEFAULTS = {
    "steps": 12000,
    "burn_in_steps": 4000,
    "dt": 0.01,
    "seed": 8,
    "drift": "linear",
    "sigmoid_steepness": 10.0,
    "ramp_fraction": 0.5,
    "normalization": "trajectory",
    "rtol": 1e-6,
    "atol": 1e-8,
    "max_step_multiplier": 5.0,
}


def linear_progress(u, spec):
    return u


def sigmoid_progress(u, spec):
    steepness = spec["sigmoid_steepness"]
    lower, upper = expit(-steepness / 2), expit(steepness / 2)
    return (expit(steepness * (u - 0.5)) - lower) / (upper - lower)


def ramp_hold_progress(u, spec):
    return np.minimum(u / spec["ramp_fraction"], 1.0)


# New drift shapes only need a progress function mapping [0, 1] to [0, 1].
DRIFTS = {
    "linear": linear_progress,
    "sigmoid": sigmoid_progress,
    "ramp-hold": ramp_hold_progress,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-runs", nargs="+", type=Path, default=[])
    parser.add_argument("--model-labels", nargs="+")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/parameter_drift"))
    parser.add_argument("--ground-truth", type=Path, help="Reusable .npz file; defaults to output-dir/ground_truth.npz.")
    cache_mode = parser.add_mutually_exclusive_group()
    cache_mode.add_argument("--reuse-ground-truth", action="store_true", help="Require the cache; never simulate. Stored settings apply unless explicitly overridden with matching values.")
    cache_mode.add_argument("--regenerate-ground-truth", action="store_true", help="Explicitly replace the ground-truth cache.")
    parser.add_argument("--simulate-only", action="store_true")
    parser.add_argument("--regime-file", type=Path, help="Path to generate_dynamic_regime_trajectories.py.")
    parser.add_argument("--list-regimes", action="store_true")
    parser.add_argument("--regime", help="Exact name or unique substring; new simulations default to halvorsen.")
    parser.add_argument("--steps", type=int, help="Number of stored states, including the initial state (default: 12000).")
    parser.add_argument("--burn-in-steps", type=int, help="True-system burn-in at the fixed start parameter (default: 4000).")
    parser.add_argument("--dt", type=float, help="Sampling interval (default: 0.01).")
    parser.add_argument("--seed", type=int, help="Simulation seed (default: 8).")
    parser.add_argument("--initial-state", nargs=3, type=float, help="ODE state before burn-in; otherwise drawn as in the existing generator.")
    parser.add_argument("--parameter-start", type=float, help="Raw start parameter; defaults to regime.start.")
    parser.add_argument("--parameter-end", type=float, help="Raw end parameter; defaults to regime.end.")
    parser.add_argument("--phi-start", type=float, help="Normalized start parameter, instead of raw endpoints.")
    parser.add_argument("--phi-end", type=float, help="Normalized end parameter, instead of raw endpoints.")
    parser.add_argument("--drift", choices=tuple(DRIFTS), help="Default: linear.")
    parser.add_argument("--sigmoid-steepness", type=float, help="Default: 10.")
    parser.add_argument("--ramp-fraction", type=float, help="Fraction spent ramping before holding the end value (default: 0.5).")
    parser.add_argument("--normalization", choices=("trajectory", "none", "fixed"), help="One fixed scaling for the full trajectory, not time-local scaling (default: trajectory).")
    parser.add_argument("--normalization-stats", type=Path, help=".npz containing mean and std vectors; implies fixed normalization.")
    parser.add_argument("--rtol", type=float)
    parser.add_argument("--atol", type=float)
    parser.add_argument("--max-step-multiplier", type=float)
    parser.add_argument("--metric", action="append", default=[], metavar="NAME:START:STOP", help="Repeatable; names: dh, dstsp (alias dstps), mse. Indices are zero-based, stop-exclusive; 'end' is allowed. Default: all three on 0:end.")
    parser.add_argument("--dstsp-bins", type=int, default=30)
    parser.add_argument("--dh-smoothing", type=float, default=20)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--plot-space", choices=("standardized", "physical"), default="standardized", help="Metrics always use model/standardized coordinates.")
    parser.add_argument("--plotly-js", type=Path, help="Inline this plotly.min.js file for fully offline HTML; otherwise use the Plotly CDN.")
    return parser.parse_args(argv)


def load_regime_module(path=None):
    candidates = [path] if path is not None else [
        Path.home() / "diffrentbutconst_prams/generate_dynamic_regime_trajectories.py",
        Path(__file__).resolve().parents[3] / "generate_trainingdata/diffrentbutconst_prams/generate_dynamic_regime_trajectories.py",
    ]
    for candidate in candidates:
        if candidate.is_file():
            module_spec = importlib.util.spec_from_file_location("alrnn_drift_regimes", candidate)
            module = importlib.util.module_from_spec(module_spec)
            sys.modules[module_spec.name] = module
            module_spec.loader.exec_module(module)
            return module, candidate.resolve()
    raise FileNotFoundError("Regime definitions not found. Pass --regime-file /path/to/generate_dynamic_regime_trajectories.py")


def resolve_regime(module, name):
    regimes = module.default_regimes()
    exact = [regime for regime in regimes if regime.name == name]
    matches = exact or [regime for regime in regimes if name.lower() in regime.name.lower()]
    if len(matches) != 1:
        raise ValueError(f"Regime {name!r} is missing or ambiguous. Matches: {[r.name for r in matches]}. Use --list-regimes.")
    return matches[0]


def validate_options(args):
    if (args.parameter_start is not None or args.parameter_end is not None) and (args.phi_start is not None or args.phi_end is not None):
        raise ValueError("Use raw parameter endpoints OR normalized phi endpoints, not both.")
    if args.model_labels is not None and len(args.model_labels) != len(args.model_runs):
        raise ValueError("--model-labels must have one entry per --model-runs entry.")
    if args.normalization_stats is not None and args.normalization not in (None, "fixed"):
        raise ValueError("--normalization-stats requires fixed normalization.")
    if args.cpu_threads < 1 or args.dstsp_bins < 2 or args.dh_smoothing <= 0:
        raise ValueError("Threads and DH smoothing must be positive; Dstsp needs at least two bins.")


def make_simulation_spec(args, regime):
    spec = {key: getattr(args, key) if getattr(args, key) is not None else value for key, value in DEFAULTS.items()}
    scale = max(abs(regime.start - regime.boundary), abs(regime.end - regime.boundary), 1e-8)
    start = regime.start if args.parameter_start is None else args.parameter_start
    end = regime.end if args.parameter_end is None else args.parameter_end
    if args.phi_start is not None:
        start = regime.boundary + args.phi_start * scale
    if args.phi_end is not None:
        end = regime.boundary + args.phi_end * scale
    spec.update(regime=regime.name, parameter_start=float(start), parameter_end=float(end),
                phi_start=float((start - regime.boundary) / scale), phi_end=float((end - regime.boundary) / scale),
                phi_scale=float(scale), phi_boundary=float(regime.boundary), initial_state=args.initial_state)
    if args.normalization_stats is not None:
        spec["normalization"] = "fixed"
    if spec["steps"] < 3 or spec["burn_in_steps"] < 0:
        raise ValueError("Use at least three states and a non-negative burn-in.")
    for key in ("dt", "rtol", "atol", "max_step_multiplier", "sigmoid_steepness"):
        if not math.isfinite(spec[key]) or spec[key] <= 0:
            raise ValueError(f"{key} must be finite and positive.")
    if not 0 < spec["ramp_fraction"] <= 1:
        raise ValueError("--ramp-fraction must be in (0, 1].")
    if not np.isfinite([start, end]).all() or (args.initial_state is not None and not np.isfinite(args.initial_state).all()):
        raise ValueError("Endpoints and initial state must be finite.")
    if spec["normalization"] == "fixed" and args.normalization_stats is None:
        raise ValueError("New simulations with fixed normalization require --normalization-stats.")
    return spec


def parameter_at_time(t, spec):
    duration = (spec["steps"] - 1) * spec["dt"]
    u = np.clip(np.asarray(t) / duration, 0.0, 1.0)
    progress = DRIFTS[spec["drift"]](u, spec)
    return spec["parameter_start"] + (spec["parameter_end"] - spec["parameter_start"]) * progress


def load_scaling(path):
    with np.load(path, allow_pickle=False) as arrays:
        mean = np.asarray(arrays["mean"], dtype=np.float64).reshape(-1)
        std = np.asarray(arrays["std"], dtype=np.float64).reshape(-1)
    if mean.shape != (3,) or std.shape != (3,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Normalization statistics must be finite length-three mean/std vectors with std > 0.")
    return mean, std


def simulate_ground_truth(module, regime, spec, scaling_path=None):
    rng = np.random.default_rng(spec["seed"])
    initial = np.asarray(spec["initial_state"], dtype=float) if spec["initial_state"] is not None else rng.normal(regime.init_center, regime.init_scale)
    if spec["initial_state"] is None and regime.observation == "phase_sine":
        initial[2] = rng.uniform(0, 2 * np.pi)
    initial_before_burn = initial.copy()
    rhs = module.RHS[regime.rhs_key]
    solver_options = dict(method="DOP853", rtol=spec["rtol"], atol=spec["atol"], max_step=spec["dt"] * spec["max_step_multiplier"])

    # Match the derivative clipping used in the existing trajectory generator.
    def ode(t, state, parameter):
        return np.clip(rhs(t, state, float(parameter), **regime.params), -1e4, 1e4)

    burn_observed = np.empty((0, 3), dtype=np.float64)
    if spec["burn_in_steps"]:
        burn_times = np.arange(spec["burn_in_steps"] + 1) * spec["dt"]
        burn = solve_ivp(lambda t, state: ode(t, state, spec["parameter_start"]),
                         (0, burn_times[-1]), initial, t_eval=burn_times, **solver_options)
        if not burn.success or burn.y.shape[1] != len(burn_times):
            raise RuntimeError(f"Burn-in failed: {burn.message}")
        initial = burn.y[:, -1]
        burn_observed = module.observe_state(burn.y.T, regime.observation)

    times = np.arange(spec["steps"], dtype=np.float64) * spec["dt"]
    solution = solve_ivp(lambda t, state: ode(t, state, parameter_at_time(t, spec)),
                         (0, times[-1]), initial, t_eval=times, **solver_options)
    if not solution.success or solution.y.shape[1] != spec["steps"]:
        raise RuntimeError(f"Drift integration failed: {solution.message}")
    observed = module.observe_state(solution.y.T, regime.observation).astype(np.float64)
    if not np.isfinite(observed).all() or not np.isfinite(burn_observed).all():
        raise RuntimeError("The simulated trajectory contains non-finite states.")
    if spec["normalization"] == "trajectory":
        standardized, mean, std = module.standardize_trajectories(observed[None].astype(np.float32))
        x = standardized[0]
        mean, std = mean.reshape(3), std.reshape(3)
    else:
        mean, std = load_scaling(scaling_path) if spec["normalization"] == "fixed" else (np.zeros(3), np.ones(3))
        x = ((observed - mean) / std).astype(np.float32)
    raw_parameter = np.asarray(parameter_at_time(times, spec), dtype=np.float64)
    phi = ((raw_parameter - spec["phi_boundary"]) / spec["phi_scale"]).astype(np.float32)
    return dict(x=x, observed=observed, states=solution.y.T, times=times, phi=phi,
                parameter=raw_parameter, mean=mean, std=std, burn_in_observed=burn_observed,
                initial_before_burn=initial_before_burn)


def validate_cache_options(args, metadata, arrays):
    spec = metadata["simulation"]
    checks = {key: getattr(args, key) for key in DEFAULTS if getattr(args, key) is not None}
    for key in ("parameter_start", "parameter_end", "phi_start", "phi_end", "initial_state"):
        if getattr(args, key) is not None:
            checks[key] = getattr(args, key)
    for key, requested in checks.items():
        stored = spec.get(key)
        equal = requested == stored if isinstance(requested, str) else stored is not None and np.allclose(requested, stored, rtol=1e-7, atol=1e-7)
        if not equal:
            raise ValueError(f"Cached {key}={stored!r} does not match requested {requested!r}. Choose another --ground-truth path or explicitly --regenerate-ground-truth.")
    if args.regime is not None and args.regime.lower() not in spec["regime"].lower():
        raise ValueError(f"Cached regime is {spec['regime']}, not {args.regime}.")
    if args.normalization_stats is not None:
        mean, std = load_scaling(args.normalization_stats)
        if spec["normalization"] != "fixed" or not np.allclose(mean, arrays["mean"]) or not np.allclose(std, arrays["std"]):
            raise ValueError("Requested normalization statistics differ from the cached scaling.")


def get_ground_truth(args, path):
    if path.suffix != ".npz":
        raise ValueError("--ground-truth must use an .npz filename.")
    if args.reuse_ground_truth and not path.is_file():
        raise FileNotFoundError(f"Ground-truth cache not found: {path}")
    if path.exists() and not args.regenerate_ground_truth:
        with np.load(path, allow_pickle=False) as stored:
            metadata = json.loads(str(stored["metadata_json"].item()))
            arrays = {key: stored[key] for key in stored.files if key != "metadata_json"}
        if metadata.get("cache_version") != CACHE_VERSION:
            raise ValueError("Unsupported ground-truth cache version.")
        validate_cache_options(args, metadata, arrays)
        print(f"Reusing ground truth without simulation: {path}", flush=True)
        return arrays, metadata
    module, module_path = load_regime_module(args.regime_file)
    regime = resolve_regime(module, args.regime or "halvorsen")
    spec = make_simulation_spec(args, regime)
    print(f"Simulating {regime.name}: {spec['burn_in_steps']} fixed-parameter burn-in steps, {spec['steps']} stored drift states.", flush=True)
    arrays = simulate_ground_truth(module, regime, spec, args.normalization_stats)
    metadata = dict(cache_version=CACHE_VERSION, simulation=spec, regime_definition=asdict(regime),
                    regime_file=str(module_path), regime_file_sha256=file_hash(module_path),
                    sample_convention="x[0] is the post-burn-in initial state; phi[t] drives the model transition x[t] -> x[t+1]. steps includes x[0].",
                    normalization_note="One time-independent affine scaling. Trajectory mode uses statistics of the full observed drift trajectory, including its future.")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays, metadata_json=json.dumps(metadata))
    print(f"Saved reusable ground truth: {path}", flush=True)
    return arrays, metadata


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SavedALRNN(nn.Module):
    """Inference-only form of the tutorial recurrence, initialized from saved weights."""

    def __init__(self, state, pwl_units):
        super().__init__()
        required = {"A", "W", "h", "B"}
        if set(state) not in (required, required | {"C"}):
            raise ValueError(f"Unexpected single AL-RNN state_dict keys: {sorted(state)}")
        self.M = state["A"].numel()
        self.N = state["B"].shape[0]
        self.P = int(pwl_units)
        self.phi_dim = state["C"].shape[1] if "C" in state else 0
        if not 0 <= self.P <= self.M or self.N > self.M:
            raise ValueError("Invalid latent/nonlinear/observation dimensions.")
        shapes = {"A": (self.M,), "W": (self.M, self.M), "h": (self.M,), "B": (self.N, self.M)}
        if "C" in state:
            shapes["C"] = (self.M, self.phi_dim)
        for key, shape in shapes.items():
            if tuple(state[key].shape) != shape or not torch.isfinite(state[key]).all():
                raise ValueError(f"Invalid or non-finite saved tensor {key}.")
            self.register_buffer(key, state[key])
        if "C" not in state:
            self.register_buffer("C", None)

    def forward(self, z, phi=None):
        activated = z.clone()
        if self.P:
            activated[:, -self.P:] = torch.relu(activated[:, -self.P:])
        result = self.A * z + activated @ self.W.t() + self.h
        if self.C is not None:
            result = result + phi @ self.C.t()
        return result


def load_model(run_dir, device):
    with (run_dir / "metrics.json").open(encoding="utf-8-sig") as stream:
        metrics = json.load(stream)
    saved_args = metrics.get("args", {})
    pwl_units = saved_args.get("pwl_units", metrics.get("pwl_units"))
    if pwl_units is None:
        raise ValueError(f"Cannot infer pwl_units from {run_dir / 'metrics.json'}")
    model_path = run_dir / "model.pt"
    state = torch.load(model_path, map_location=device, weights_only=True)
    model = SavedALRNN(state, pwl_units).eval()
    if model.N != 3 or model.phi_dim not in (0, 1):
        raise ValueError("This evaluation supports three observations and zero or one external parameter.")
    if saved_args.get("latent_dim", model.M) != model.M or metrics.get("phi_dim", model.phi_dim) != model.phi_dim:
        raise ValueError(f"Checkpoint dimensions disagree with metadata in {run_dir}")
    return model, dict(run_dir=str(run_dir.resolve()), model_sha256=file_hash(model_path),
                       latent_dim=model.M, pwl_units=model.P, phi_dim=model.phi_dim,
                       training_phi_values=saved_args.get("train_phi_values"))


@torch.inference_mode()
def rollout(model, x0, phi, device):
    x0 = torch.as_tensor(np.asarray(x0).copy(), dtype=torch.float32, device=device).reshape(1, model.N)
    z = x0 @ model.B
    z[:, :model.N] = x0
    inputs = torch.as_tensor(np.asarray(phi).copy(), dtype=torch.float32, device=device).reshape(-1, 1)
    prediction = torch.full((len(inputs), model.N), float("nan"), device=device)
    prediction[0] = x0[0]
    failed_at = None
    for step in range(len(inputs) - 1):
        z = model(z, inputs[step:step + 1] if model.phi_dim else None)
        if not torch.isfinite(z).all():
            failed_at = step + 1
            break
        prediction[step + 1] = z[0, :model.N]
    return prediction.cpu().numpy(), failed_at


def parse_metric_requests(requests, steps):
    requests = requests or ["dh:0:end", "dstsp:0:end", "mse:0:end"]
    intervals = []
    for request in requests:
        parts = request.lower().split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid metric {request!r}; use NAME:START:STOP.")
        name, start, stop = parts
        name = "dstsp" if name == "dstps" else name
        if name not in ("dh", "dstsp", "mse"):
            raise ValueError(f"Unknown metric: {name}")
        start, stop = int(start), steps if stop == "end" else int(stop)
        minimum = 1 if name == "mse" else 3
        if not 0 <= start < stop <= steps or stop - start < minimum:
            raise ValueError(f"Invalid interval [{start}, {stop}) for {name}; trajectory has {steps} states.")
        intervals.append(dict(metric=name, start=start, stop=stop))
    return intervals


def calculate_metrics(prediction, truth, intervals, bins=30, smoothing=20):
    from metrics import power_spectrum_error, state_space_divergence_binning

    results = []
    for interval in intervals:
        start, stop, name = interval["start"], interval["stop"], interval["metric"]
        generated, target = prediction[start:stop], truth[start:stop]
        row = dict(interval, samples=stop - start, value=None, status="ok")
        if not np.isfinite(generated).all() or not np.isfinite(target).all():
            row["status"] = "non_finite_trajectory"
        elif name == "mse":
            value = float(np.mean((generated.astype(np.float64) - target.astype(np.float64)) ** 2))
            row["value"] = value if math.isfinite(value) else None
            if row["value"] is None:
                row["status"] = "non_finite_metric"
        elif np.any(np.ptp(target, axis=0) == 0) or (name == "dh" and np.any(np.std(generated, axis=0) == 0)):
            row["status"] = "undefined_for_constant_component"
        else:
            if name == "dstsp":
                value = state_space_divergence_binning(torch.from_numpy(generated.copy()), torch.from_numpy(target.copy()), n_bins=bins)
            else:
                value = float(power_spectrum_error(generated, target, smoothing=smoothing))
            if math.isfinite(value):
                row["value"] = float(value)
            else:
                row["status"] = "non_finite_metric"
        results.append(row)
    return results


def plot_values(values):
    array = np.asarray(values)
    return [float(value) if np.isfinite(value) else None for value in array]


def write_plot(path, title, traces, layout, plotly_js=None):
    script = ("<script>" + Path(plotly_js).read_text(encoding="utf-8") + "</script>") if plotly_js is not None else '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'
    figure = json.dumps(dict(data=traces, layout=layout), allow_nan=False).replace("<", "\\u003c")
    page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>{script}
<style>html,body,#figure{{width:100%;height:100%;margin:0;font-family:Arial,sans-serif}}</style></head>
<body><div id="figure"></div><script>const figure={figure};
Plotly.newPlot('figure',figure.data,figure.layout,{{responsive:true,displaylogo:false}});
</script></body></html>'''
    path.write_text(page, encoding="utf-8")


def save_plots(output_dir, arrays, models, metadata, space="standardized", plotly_js=None):
    truth = arrays["x"] if space == "standardized" else arrays["observed"]
    colors = ["#d95f02", "#1b9e77", "#984ea3", "#e7298a", "#666666", "#a6761d"]
    series = [("Ground truth", truth, "#1764a1")]
    for index, result in enumerate(models):
        prediction = result["prediction"]
        if space == "physical":
            prediction = prediction.astype(np.float64) * arrays["std"] + arrays["mean"]
        series.append((result["label"], prediction, colors[index % len(colors)]))
    spec = metadata["simulation"]
    title = f"{spec['regime']} | {spec['drift']} parameter drift | {space}"
    traces_3d = []
    for label, values, color in series:
        traces_3d.append(dict(type="scatter3d", mode="lines", name=label,
                              x=plot_values(values[:, 0]), y=plot_values(values[:, 1]), z=plot_values(values[:, 2]),
                              line=dict(color=color, width=3), connectgaps=False,
                              customdata=np.column_stack((np.arange(len(truth)), arrays["phi"])).tolist(),
                              hovertemplate="step=%{customdata[0]}<br>phi=%{customdata[1]:.4f}<br>x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra>%{fullData.name}</extra>"))
    write_plot(output_dir / "rollout_3d.html", title, traces_3d,
               dict(title=dict(text=title), scene=dict(xaxis=dict(title="x"), yaxis=dict(title="y"), zaxis=dict(title="z"), aspectmode="data"),
                    margin=dict(l=0, r=0, b=0, t=65), legend=dict(orientation="h")), plotly_js)
    traces = []
    steps = np.arange(len(truth)).tolist()
    for component in range(3):
        suffix = "" if component == 0 else str(component + 1)
        for label, values, color in series:
            traces.append(dict(type="scattergl", mode="lines", name=label, legendgroup=label,
                               showlegend=component == 0, x=steps, y=plot_values(values[:, component]),
                               xaxis=f"x{suffix}", yaxis=f"y{suffix}", line=dict(color=color, width=1.2), connectgaps=False))
    traces.append(dict(type="scattergl", mode="lines", name="phi", x=steps, y=plot_values(arrays["phi"]),
                       xaxis="x4", yaxis="y4", line=dict(color="#333333", width=2)))
    layout = dict(title=dict(text=title), grid=dict(rows=4, columns=1, pattern="independent"),
                  hovermode="x unified", legend=dict(orientation="h"), margin=dict(l=65, r=35, t=90, b=50))
    for row, label in enumerate(("x", "y", "z", "phi"), start=1):
        suffix = "" if row == 1 else str(row)
        layout[f"yaxis{suffix}"] = dict(title=label)
        layout[f"xaxis{suffix}"] = dict(title="step" if row == 4 else None, matches="x4" if row != 4 else None)
    write_plot(output_dir / "components.html", title, traces, layout, plotly_js)


def main(argv=None):
    args = parse_args(argv)
    validate_options(args)
    if args.list_regimes:
        module, _ = load_regime_module(args.regime_file)
        for regime in module.default_regimes():
            print(f"{regime.name}: {regime.parameter}={regime.start:g} -> {regime.end:g}")
        return
    if not args.simulate_only and not args.model_runs:
        raise ValueError("Pass --model-runs, or use --simulate-only to create just the ground truth.")
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    device = torch.device("cpu")
    if not args.simulate_only and args.device == "cuda":
        if not torch.cuda.is_available() or not 0 <= args.gpu_id < torch.cuda.device_count():
            raise ValueError("Requested CUDA device is unavailable. Select a valid --gpu-id or use --device cpu.")
        device = torch.device(f"cuda:{args.gpu_id}")
    if args.plotly_js is not None and not args.plotly_js.is_file():
        raise FileNotFoundError(args.plotly_js)
    for run_dir in args.model_runs:
        for filename in ("model.pt", "metrics.json"):
            if not (run_dir / filename).is_file():
                raise FileNotFoundError(run_dir / filename)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.ground_truth or args.output_dir / "ground_truth.npz"
    arrays, metadata = get_ground_truth(args, cache_path)
    print(f"Regime: {metadata['simulation']['regime']} | states: {len(arrays['x'])} | normalization: {metadata['simulation']['normalization']}", flush=True)
    print(f"Raw parameter: {arrays['parameter'][0]:g} -> {arrays['parameter'][-1]:g}; phi: {arrays['phi'][0]:g} -> {arrays['phi'][-1]:g}", flush=True)
    if args.simulate_only:
        return
    intervals = parse_metric_requests(args.metric, len(arrays["x"]))
    print(f"Using {device}. Metrics use zero-based [start, stop) intervals in model coordinates.", flush=True)
    labels = args.model_labels or [run_dir.name for run_dir in args.model_runs]
    if len(set(labels)) != len(labels):
        raise ValueError("Model labels must be unique.")
    report = dict(ground_truth=str(cache_path.resolve()), ground_truth_sha256=file_hash(cache_path),
                  metadata=metadata, metric_space="model/standardized", interval_convention="zero-based, stop-exclusive",
                  dstsp_bins=args.dstsp_bins, dh_smoothing=args.dh_smoothing, models=[])
    plot_models = []
    for index, (run_dir, label) in enumerate(zip(args.model_runs, labels), start=1):
        model, info = load_model(run_dir, device)
        print(f"[{index}/{len(labels)}] {label}: M={model.M}, P={model.P}, phi_dim={model.phi_dim}", flush=True)
        prediction, failed_at = rollout(model, arrays["x"][0], arrays["phi"], device)
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("._")[:100] or "model"
        filename = f"model_{index:02d}_{slug}.npz"
        np.savez_compressed(args.output_dir / filename, prediction=prediction,
                            prediction_physical=prediction.astype(np.float64) * arrays["std"] + arrays["mean"],
                            phi=arrays["phi"], times=arrays["times"])
        results = calculate_metrics(prediction, arrays["x"], intervals, args.dstsp_bins, args.dh_smoothing)
        for row in results:
            print(f"  {row['metric']} [{row['start']}:{row['stop']}): {row['value']} ({row['status']})", flush=True)
        if failed_at is not None:
            print(f"  Non-finite latent state at index {failed_at}; remaining predictions are NaN.", flush=True)
        report["models"].append(dict(info, label=label, prediction_file=filename, failed_at=failed_at, metrics=results))
        plot_models.append(dict(label=label, prediction=prediction))
        (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        del model
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["model", "metric", "start", "stop", "samples", "value", "status"])
        writer.writeheader()
        for model in report["models"]:
            for row in model["metrics"]:
                writer.writerow(dict(model=model["label"], **row))
    save_plots(args.output_dir, arrays, plot_models, metadata, args.plot_space, args.plotly_js)
    print(f"Saved predictions, metrics.json, metrics.csv, rollout_3d.html and components.html to {args.output_dir.resolve()}", flush=True)
    if args.plotly_js is None:
        print("HTML plots load Plotly from its CDN when opened. Use --plotly-js plotly.min.js for offline HTML.")


if __name__ == "__main__":
    main()

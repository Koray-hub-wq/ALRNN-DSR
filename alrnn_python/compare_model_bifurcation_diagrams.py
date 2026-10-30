"""Run the shared bifurcation backend and record model provenance beside plots."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


def install_provenance(backend, backend_path, normalization="per_simulated_trajectory"):
    recorded_sources = []
    cache_reused = False

    def json_value(value):
        if isinstance(value, Path):
            return str(value.expanduser().resolve())
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Cannot serialize {type(value).__name__}")

    def wrap_loader(original, model_type, checkpoint_index, info_index):
        def load(run_dir, *args, **kwargs):
            result = original(run_dir, *args, **kwargs)
            checkpoint = Path(result[checkpoint_index]).expanduser().resolve()
            digest = hashlib.sha256()
            with checkpoint.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            info = result[info_index]
            recorded_sources.append({
                "run_dir": str(Path(run_dir).expanduser().resolve()),
                "model_type": model_type,
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": digest.hexdigest(),
                "checkpoint_size_bytes": checkpoint.stat().st_size,
                "training_args": info.get("args") if model_type == "alrnn" else info,
                "model_config": info.get("model_config") if model_type == "alrnn" else None,
                "parameter_counts": info.get("parameter_counts") if model_type == "alrnn" else None,
                "provenance_status": "recorded_at_computation",
            })
            return result
        return load

    if hasattr(backend, "load_alrnn_model"):
        backend.load_alrnn_model = wrap_loader(backend.load_alrnn_model, "alrnn", 2, 1)
    if hasattr(backend, "load_model"):
        backend.load_model = wrap_loader(backend.load_model, "dynamix", 4, 2)

    original_save = backend.save_points_cache
    original_load = backend.load_points_cache
    original_write = backend.write_plots

    def save_cache(path, metadata, *args, **kwargs):
        # Provenance travels with the cache, not with today's checkpoint files.
        metadata = {**metadata, "model_sources": recorded_sources}
        return original_save(path, metadata, *args, **kwargs)

    def load_cache(*args, **kwargs):
        nonlocal cache_reused
        result = original_load(*args, **kwargs)
        cache_reused = True
        return result

    def cache_matches(path, expected_metadata):
        try:
            with np.load(path, allow_pickle=False) as data:
                metadata = json.loads(str(data["metadata"].item()))
            metadata.pop("model_sources", None)
            return metadata == expected_metadata
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            return False

    def write_plots(*args, **kwargs):
        written = original_write(*args, **kwargs)
        settings = backend.parse_args()
        cache = settings.cache if settings.cache is not None else settings.output.with_suffix(".npz")
        with np.load(cache, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
        sources = metadata.get("model_sources", [])
        models = []
        for index, run_dir in enumerate(settings.model_runs):
            resolved = str(Path(run_dir).expanduser().resolve())
            source = next((item for item in sources if item.get("run_dir") == resolved), None)
            models.append({
                **(source or {"run_dir": resolved, "checkpoint_path": None,
                              "checkpoint_sha256": None,
                              "provenance_status": "not_recorded_in_cache"}),
                "index": index, "label": kwargs["labels"][index],
            })
        for name in settings.pretrained_model_names:
            index = len(models)
            models.append({"index": index, "label": kwargs["labels"][index],
                           "model_type": "pretrained", "name": name,
                           "repo_root": settings.pretrained_repo_root,
                           "provenance_status": "pretrained_checkpoint_not_recorded"})
        payload = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "backend_script": str(Path(backend_path).resolve()),
            "command": list(sys.argv),
            "normalization": (
                "fixed_training" if metadata.get("training_coordinates")
                else "per_simulated_trajectory" if cache_reused else normalization
            ),
            "models": models,
            "settings": vars(settings),
            "cache": {"path": str(Path(cache).resolve()), "reused": cache_reused,
                      "metadata": metadata},
            "plots": [str(Path(path).resolve()) for path in written],
        }
        for path in written:
            path = Path(path)
            included = list(range(len(models)))
            component_name = None
            for component in kwargs["components"]:
                count = len(kwargs["components"])
                if path == backend.output_for_component(kwargs["output"], component, count):
                    component_name = component
                for index, label in enumerate(["Ground Truth", *kwargs["labels"]]):
                    if path == backend.individual_output_path(kwargs["output"], component, label, count):
                        included = [] if index == 0 else [index - 1]
                        component_name = component
            sidecar = path.with_suffix(".sources.json")
            document = {**payload, "plot": str(path.resolve()),
                        "component": component_name, "plot_model_indices": included}
            sidecar.write_text(json.dumps(document, indent=2, default=json_value) + "\n", encoding="utf-8")
            print(f"Saved model provenance: {sidecar.resolve()}", flush=True)
        return written

    backend.save_points_cache = save_cache
    backend.load_points_cache = load_cache
    backend.cache_matches = cache_matches
    backend.write_plots = write_plots


def find_backend(explicit=None):
    candidates = [explicit] if explicit is not None else [
        Path.home() / "diffrentbutconst_prams/compare_model_bifurcation_diagrams.py",
        Path(__file__).resolve().parents[2] / "generate_trainingdata/diffrentbutconst_prams/compare_model_bifurcation_diagrams.py",
    ]
    for candidate in candidates:
        if candidate.is_file() and candidate.resolve() != Path(__file__).resolve():
            return candidate.resolve()
    raise FileNotFoundError("Pass --backend-script /path/to/diffrentbutconst_prams/compare_model_bifurcation_diagrams.py")


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--backend-script", type=Path)
    args, remaining = parser.parse_known_args(argv)
    path = find_backend(args.backend_script)
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("alrnn_bifurcation_backend", path)
    backend = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = backend
    spec.loader.exec_module(backend)
    install_provenance(backend, path)
    original_argv = sys.argv
    try:
        sys.argv = [str(path), *remaining]
        backend.main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()

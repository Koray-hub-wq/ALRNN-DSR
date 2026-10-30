import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

import compare_model_bifurcation_diagrams as wrapper


class ProvenanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        alternative = os.environ.get("TEST_BIFURCATION_WRAPPER")
        cls.wrapper = wrapper
        if alternative:
            spec = importlib.util.spec_from_file_location("tested_provenance_wrapper", alternative)
            module = importlib.util.module_from_spec(spec)
            import sys
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            cls.wrapper = module

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run = self.root / "run"
        self.run.mkdir()
        self.checkpoint = self.run / "model.pt"
        self.checkpoint.write_bytes(b"original checkpoint")
        self.output = self.root / "compare.png"
        self.cache = self.root / "cache.npz"
        self.settings = SimpleNamespace(cache=self.cache, output=self.output,
                                        model_runs=[self.run], pretrained_model_names=[],
                                        pretrained_repo_root=None)
        self.metadata = {"model_runs": [str(self.run)], "regime": "halvorsen"}

    def backend(self):
        def save(path, metadata):
            np.savez(path, metadata=np.array(json.dumps(metadata)))

        def write(**kwargs):
            paths = [kwargs["output"], self.root / "truth.png", self.root / "model.png"]
            for path in paths:
                path.touch()
            return paths

        backend = SimpleNamespace(
            load_alrnn_model=lambda run, device: (
                object(), {"args": {"latent_dim": 300}, "model_config": {"pwl_units": 150}},
                self.checkpoint),
            save_points_cache=save,
            load_points_cache=lambda *args, **kwargs: None,
            write_plots=write,
            parse_args=lambda: self.settings,
            output_for_component=lambda output, component, count: output,
            individual_output_path=lambda output, component, label, count: (
                self.root / ("truth.png" if label == "Ground Truth" else "model.png")),
        )
        self.wrapper.install_provenance(backend, self.root / "backend.py")
        return backend

    def write(self, backend):
        backend.write_plots(output=self.output, labels=["Cphi"], components=["z"])
        return json.loads(self.output.with_suffix(".sources.json").read_text())

    def test_fresh_computation_records_paths_hash_labels_and_plot_membership(self):
        backend = self.backend()
        backend.load_alrnn_model(self.run, "cpu")
        backend.save_points_cache(self.cache, self.metadata)
        result = self.write(backend)
        source = result["models"][0]
        self.assertEqual(source["checkpoint_path"], str(self.checkpoint.resolve()))
        self.assertEqual(source["checkpoint_sha256"], hashlib.sha256(b"original checkpoint").hexdigest())
        self.assertEqual(source["label"], "Cphi")
        self.assertEqual(source["training_args"]["latent_dim"], 300)
        self.assertFalse(result["cache"]["reused"])
        self.assertTrue(backend.cache_matches(self.cache, self.metadata))
        self.assertEqual(result["plot_model_indices"], [0])
        truth = json.loads((self.root / "truth.sources.json").read_text())
        self.assertEqual(truth["plot_model_indices"], [])
        single = json.loads((self.root / "model.sources.json").read_text())
        self.assertEqual(single["plot_model_indices"], [0])

    def test_cached_plot_keeps_original_checkpoint_identity(self):
        backend = self.backend()
        backend.load_alrnn_model(self.run, "cpu")
        backend.save_points_cache(self.cache, self.metadata)
        self.checkpoint.write_bytes(b"replaced weights")
        cached_backend = self.backend()
        cached_backend.load_points_cache(self.cache)
        result = self.write(cached_backend)
        self.assertTrue(result["cache"]["reused"])
        self.assertEqual(result["models"][0]["checkpoint_sha256"], hashlib.sha256(b"original checkpoint").hexdigest())

    def test_old_cache_does_not_invent_checkpoint_provenance(self):
        np.savez(self.cache, metadata=np.array(json.dumps(self.metadata)))
        backend = self.backend()
        self.assertTrue(backend.cache_matches(self.cache, self.metadata))
        backend.load_points_cache(self.cache)
        result = self.write(backend)
        self.assertEqual(result["models"][0]["provenance_status"], "not_recorded_in_cache")
        self.assertIsNone(result["models"][0]["checkpoint_path"])

    def test_cache_settings_still_have_to_match(self):
        np.savez(self.cache, metadata=np.array(json.dumps(self.metadata)))
        backend = self.backend()
        self.assertFalse(backend.cache_matches(self.cache, {**self.metadata, "regime": "dadras"}))
        self.assertFalse(backend.cache_matches(self.root / "missing.npz", self.metadata))


if __name__ == "__main__":
    unittest.main()

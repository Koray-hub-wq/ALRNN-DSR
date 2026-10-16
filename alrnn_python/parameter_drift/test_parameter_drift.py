import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import evaluate_parameter_drift as drift


class ParameterDriftTests(unittest.TestCase):
    def regime(self):
        return SimpleNamespace(name="test_flow", start=0.0, end=2.0, boundary=1.0,
                               init_center=(0, 0, 0), init_scale=(1, 1, 1),
                               observation="identity", rhs_key="test", params={})

    def simulation_spec(self, *options):
        args = drift.parse_args(["--initial-state", "0", "0", "0", *options])
        return drift.make_simulation_spec(args, self.regime())

    def state_dict(self, phi=True):
        state = dict(A=torch.ones(5), W=torch.zeros(5, 5), h=torch.zeros(5), B=torch.zeros(3, 5))
        if phi:
            state["C"] = torch.tensor([[1.0], [2.0], [3.0], [0.0], [0.0]])
        return state

    def test_drifts_endpoints_hold_and_fixed_phi_scale(self):
        spec = self.simulation_spec("--steps", "11", "--dt", "0.1", "--parameter-start", "0.5", "--parameter-end", "1.5")
        self.assertEqual((spec["phi_start"], spec["phi_end"]), (-0.5, 0.5))
        for name in drift.DRIFTS:
            spec["drift"] = name
            values = drift.parameter_at_time(np.linspace(0, 1, 11), spec)
            np.testing.assert_allclose(values[[0, -1]], [0.5, 1.5])
            self.assertTrue(np.all(np.diff(values) >= 0))
        spec["drift"] = "ramp-hold"
        np.testing.assert_allclose(drift.parameter_at_time([0.5, 0.8, 1.0], spec), 1.5)

    def test_burn_in_and_continuous_drift_match_analytic_ode(self):
        module = SimpleNamespace(RHS={"test": lambda t, x, p: p * np.array([1, 2, 3])},
                                 observe_state=lambda x, observation: x)
        spec = self.simulation_spec("--steps", "11", "--dt", "0.1", "--burn-in-steps", "4",
                                    "--parameter-start", "0.5", "--parameter-end", "1.5", "--normalization", "none")
        arrays = drift.simulate_ground_truth(module, self.regime(), spec)
        t = arrays["times"]
        expected = (0.2 + 0.5 * t + 0.5 * t ** 2)[:, None] * [1, 2, 3]
        np.testing.assert_allclose(arrays["observed"], expected, atol=1e-7)
        np.testing.assert_allclose(arrays["x"], expected, atol=3e-7)
        np.testing.assert_allclose(arrays["phi"][[0, -1]], [-0.5, 0.5])

    def test_rollout_uses_left_endpoint_phi_and_keeps_initial_state(self):
        model = drift.SavedALRNN(self.state_dict(), pwl_units=0)
        inputs = np.array([-1, 0.5, 2, 9], dtype=np.float32)
        x0 = np.array([1, 4, -2], dtype=np.float32)
        result, failed = drift.rollout(model, x0, inputs, "cpu")
        expected = x0 + np.r_[0, np.cumsum(inputs[:-1])][:, None] * [1, 2, 3]
        np.testing.assert_allclose(result, expected)
        self.assertIsNone(failed)
        vanilla = drift.SavedALRNN(self.state_dict(phi=False), pwl_units=0)
        result, _ = drift.rollout(vanilla, x0, inputs, "cpu")
        np.testing.assert_allclose(result, np.tile(x0, (len(inputs), 1)))

    def test_inference_recurrence_matches_training_definition(self):
        rng = torch.Generator().manual_seed(42)
        state = self.state_dict()
        for tensor in state.values():
            tensor.copy_(torch.randn(tensor.shape, generator=rng))
        model = drift.SavedALRNN(state, pwl_units=2)
        z = torch.randn((3, 5), generator=rng)
        original = z.clone()
        phi = torch.randn((3, 1), generator=rng)
        activated = z.clone()
        activated[:, -2:] = torch.relu(activated[:, -2:])
        expected = state["A"] * z + activated @ state["W"].t() + state["h"] + phi @ state["C"].t()
        torch.testing.assert_close(model(z, phi), expected)
        torch.testing.assert_close(z, original)

    def test_cache_reuse_does_not_import_or_simulate_and_rejects_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truth.npz"
            spec = self.simulation_spec("--steps", "10")
            metadata = dict(cache_version=drift.CACHE_VERSION, simulation=spec)
            np.savez_compressed(path, x=np.zeros((10, 3)), metadata_json=json.dumps(metadata))
            args = drift.parse_args(["--reuse-ground-truth"])
            with patch.object(drift, "load_regime_module", side_effect=AssertionError("must not import")):
                arrays, stored = drift.get_ground_truth(args, path)
            self.assertEqual(arrays["x"].shape, (10, 3))
            self.assertEqual(stored, metadata)
            args.steps = 11
            with self.assertRaisesRegex(ValueError, "Cached steps"):
                drift.get_ground_truth(args, path)

    def test_metric_intervals_mse_and_divergent_rollout(self):
        intervals = drift.parse_metric_requests(["mse:2:5", "dstps:0:end", "dh:5:10"], 10)
        self.assertEqual(intervals[1]["metric"], "dstsp")
        truth = np.arange(30, dtype=np.float32).reshape(10, 3)
        result = drift.calculate_metrics(truth + 2, truth, intervals[:1])
        self.assertEqual(result[0]["value"], 4)
        self.assertEqual(result[0]["samples"], 3)
        with self.assertRaises(ValueError):
            drift.parse_metric_requests(["mse:0:11"], 10)
        prediction = truth.copy()
        prediction[4] = np.nan
        result = drift.calculate_metrics(prediction, truth, intervals[:1])
        self.assertIsNone(result[0]["value"])
        self.assertEqual(result[0]["status"], "non_finite_trajectory")

    def test_saved_checkpoint_load_and_nonfinite_latent_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            state = self.state_dict()
            torch.save(state, path / "model.pt")
            (path / "metrics.json").write_text(json.dumps(dict(args=dict(pwl_units=2, latent_dim=5), phi_dim=1)))
            model, info = drift.load_model(path, "cpu")
            self.assertEqual((info["latent_dim"], info["pwl_units"], info["phi_dim"]), (5, 2, 1))
            model.A.fill_(float("inf"))
            result, failed = drift.rollout(model, [1, 2, 3], [0, 0, 0], "cpu")
            self.assertEqual(failed, 1)
            self.assertTrue(np.isnan(result[1:]).all())


if __name__ == "__main__":
    unittest.main()

"""Numerical and temporal unit checks use invented observations only."""

import itertools
from pathlib import Path
import tempfile
import unittest

import numpy as np

from possession.learned import (FEATURE_NAMES, _decode_binary, fit_model, load_model,
                                predict_dead, save_model, validate_model)


class LearnedModelTests(unittest.TestCase):
    def example(self):
        rng = np.random.default_rng(9001)
        X = rng.normal(size=(400, len(FEATURE_NAMES)))
        labels = np.where(X[:, 0] + .5 * X[:, 1] > 0, "D", "H")
        return X, labels

    def test_fit_is_deterministic_and_learns_external_style_target(self):
        X, labels = self.example()
        one = fit_model(X, labels, l2=.01)
        two = fit_model(X, labels, l2=.01)
        self.assertEqual(one, two)
        prediction = predict_dead(X, one) >= .5
        self.assertGreater((prediction == (labels == "D")).mean(), .96)
        self.assertTrue(one["training"]["converged"])

    def test_model_json_roundtrip_reproduces_probability_bytes(self):
        X, labels = self.example()
        model = fit_model(X, labels)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.json"
            second = Path(folder) / "second.json"
            save_model(path, model)
            restored = load_model(path)
            save_model(second, restored)
            self.assertEqual(path.read_bytes(), second.read_bytes())
            np.testing.assert_array_equal(predict_dead(X, model), predict_dead(X, restored))

    def test_bad_artifact_and_unknown_training_labels_are_rejected(self):
        X, labels = self.example()
        model = fit_model(X, labels)
        for key, value in (("decoder_mode", "invalid"), ("feature_names", []),
                           ("scale", [0] * len(FEATURE_NAMES)), ("intercept", float("nan"))):
            with self.assertRaises(ValueError):
                validate_model(dict(model, **{key: value}))
        with self.assertRaises(ValueError):
            fit_model(X, np.full(len(X), "X"))
        with self.assertRaises(ValueError):
            fit_model(X, np.full(len(X), "D"))
        with self.assertRaises(ValueError):
            fit_model(X, labels, sample_weight=np.full(len(X), -1))

    def test_binary_duration_decoder_is_globally_optimal(self):
        rng = np.random.default_rng(8048)
        for durations in ([1, 1], [2, 3], [3, 1]):
            for _ in range(5):
                probabilities = rng.uniform(.05, 1, (8, 2))
                probabilities /= probabilities.sum(axis=1, keepdims=True)
                costs = np.array([[np.inf, 1.7], [2.1, np.inf]])
                path = _decode_binary(probabilities, durations, costs, .2)
                def score(candidate):
                    edges = [0] + [i for i in range(1, 8) if candidate[i] != candidate[i-1]] + [8]
                    for j in range(1, len(edges) - 2):
                        if edges[j+1] - edges[j] < durations[candidate[edges[j]]]:
                            return float("inf")
                    result = sum(-np.log(probabilities[i, candidate[i]]) * .2 for i in range(8))
                    result += sum(costs[candidate[i-1], candidate[i]] for i in range(1, 8)
                                  if candidate[i] != candidate[i-1])
                    return result
                optimum = min(score(p) for p in itertools.product((0, 1), repeat=8))
                self.assertAlmostEqual(score(path), optimum, places=10)


if __name__ == "__main__":
    unittest.main()

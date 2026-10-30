import copy
import importlib.util
import unittest
import numpy as np
from possession import joint_temporal as joint


def invented_model():
    rng = np.random.default_rng(427)
    model = {"schema_version": 1, "kind": "equivariant_joint_temporal_residual", "sample_hz": 5,
             "dilations": list(joint.DILATIONS), "residual_limit": joint.RESIDUAL_LIMIT,
             "control_names": list(joint.CONTROL_NAMES), "context_names": list(joint.CONTEXT_NAMES),
             "control_scale": np.ones(98).tolist(), "context_mean": np.zeros(156).tolist(),
             "context_scale": np.ones(156).tolist()}
    shapes = {"odd_input": (12, 98), "even_input": (12, 156), "even_input_bias": (12,),
              "home_output": (1, 12), "dead_output": (1, 12), "dead_output_bias": (1,)}
    for i in range(3):
        shapes.update({f"odd_conv_{i}": (12, 12, 3), f"even_conv_{i}": (12, 12, 3), f"even_conv_bias_{i}": (12,)})
    model.update({key: rng.normal(0, .04, shape).tolist() for key, shape in shapes.items()})
    return joint.validate_model(model)


class JointTemporalTests(unittest.TestCase):
    def test_joint_probabilities_preserve_team_exchange_symmetry(self):
        rng = np.random.default_rng(14); x = rng.normal(size=(200, 98)); z = rng.random((200, 58))
        h = rng.normal(size=200); d = rng.normal(size=200); blocks = [(1, 0, 199)]
        p = joint.predict_probabilities(h, d, x, z, blocks, 25, invented_model())
        q = joint.predict_probabilities(-h, d, -x, z, blocks, 25, invented_model())
        np.testing.assert_allclose(p[:, [1, 0, 2]], q, atol=2e-16)
        np.testing.assert_allclose(p.sum(axis=1), 1, atol=2e-16)
        r = joint.residual(invented_model(), x, z, blocks, 25)
        s = joint.residual(invented_model(), -x, z, blocks, 25)
        np.testing.assert_array_equal(r[:, 1], s[:, 1])
        np.testing.assert_array_equal(r[:, 0], -s[:, 0])

    def test_zero_residual_preserves_base_probabilities_and_unknown_gaps(self):
        model = invented_model()
        for name in ("home_output", "dead_output", "dead_output_bias"):
            model[name] = np.zeros_like(model[name]).tolist()
        h = np.linspace(-4, 4, 20); d = np.linspace(3, -3, 20)
        blocks = [(1, 0, 7), (2, 10, 19)]
        p = joint.predict_probabilities(h, d, np.ones((20, 98)), np.ones((20, 58)), blocks, 25, model)
        ph, pd = 1 / (1 + np.exp(-h)), 1 / (1 + np.exp(-d))
        expected = np.column_stack(((1 - pd) * ph, (1 - pd) * (1 - ph), pd)); expected[8:10] = 0
        np.testing.assert_array_equal(p, expected)

    def test_half_isolation_and_no_cross_gap_interpolation(self):
        rng = np.random.default_rng(45); x = rng.normal(size=(180, 98)); z = rng.random((180, 58))
        blocks = [(1, 0, 79), (2, 90, 179)]
        before = joint.residual(invented_model(), x, z, blocks, 25)
        x[90:] *= 200; z[90:] *= 200
        after = joint.residual(invented_model(), x, z, blocks, 25)
        np.testing.assert_array_equal(before[:90], after[:90])
        np.testing.assert_array_equal(after[80:90], 0)

    def test_schema_shapes_and_blocks_fail_closed(self):
        model = invented_model(); x = np.zeros((10, 98)); z = np.zeros((10, 58))
        for key, value in (("sample_hz", 10), ("residual_limit", 100), ("control_scale", [0] * 98),
                           ("context_mean", [float("nan")] * 156), ("odd_input", [[0]])):
            bad = copy.deepcopy(model); bad[key] = value
            with self.assertRaises(ValueError): joint.validate_model(bad)
        for blocks in ([(1, 0, 8), (2, 7, 9)], [(1, -1, 5)], [(1, 0, 10)], [(1, 0., 5)]):
            with self.assertRaises(ValueError): joint.residual(model, x, z, blocks, 25)
        with self.assertRaises(ValueError): joint.residual(model, x, z, [(1, 0, 9)], 24)
        with self.assertRaises(ValueError): joint.residual(model, x, z[:, :57], [(1, 0, 9)], 25)
        with self.assertRaises(ValueError): joint.predict_probabilities([0], np.zeros(10), x, z, [(1, 0, 9)], 25, model)

    def test_short_halves_and_bounded_residuals(self):
        model = invented_model()
        model["home_output"] = (np.ones((1, 12)) * 1e4).tolist()
        model["dead_output"] = (np.ones((1, 12)) * 1e4).tolist()
        for n in (1, 2, 4, 5, 11):
            value = joint.residual(model, np.ones((n, 98)), np.ones((n, 58)), [(1, 0, n - 1)], 25)
            self.assertEqual(value.shape, (n, 2))
            self.assertTrue(np.isfinite(value).all())
            self.assertLessEqual(np.abs(value).max(), joint.RESIDUAL_LIMIT)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Optional Torch training environment unavailable")
    def test_numpy_export_matches_masked_torch_chunks_and_half_edges(self):
        import torch
        from tools.train_joint import make_network
        torch.set_num_threads(1)
        model = invented_model(); net = make_network().double()
        state = {"odd_input.weight": model["odd_input"], "even_input.weight": model["even_input"],
                 "even_input.bias": model["even_input_bias"], "home_output.weight": model["home_output"],
                 "dead_output.weight": model["dead_output"], "dead_output.bias": model["dead_output_bias"]}
        for i in range(3):
            state.update({f"odd_convs.{i}.weight": model[f"odd_conv_{i}"],
                          f"even_convs.{i}.weight": model[f"even_conv_{i}"], f"even_convs.{i}.bias": model[f"even_conv_bias_{i}"]})
        net.load_state_dict({k: torch.tensor(v, dtype=torch.float64) for k, v in state.items()})
        rng = np.random.default_rng(105); halo = sum(joint.DILATIONS); width = 128
        for n in (1, 4, 35, 300):
            x, z = rng.normal(size=(n, 98)), rng.normal(size=(n, 58))
            ox, ez = joint.normalized_inputs(model, x, z); expected = joint.low_rate_residual(model, x, z)
            actual = np.zeros_like(expected)
            for start in range(0, n, width):
                left, right, stop = max(0, start - halo), min(n, start + width + halo), min(n, start + width)
                bx = np.zeros((1, width + 2 * halo, 98)); bz = np.zeros((1, width + 2 * halo, 156)); bm = np.zeros((1, width + 2 * halo))
                region = slice(left - start + halo, right - start + halo)
                bx[0, region] = ox[left:right]; bz[0, region] = ez[left:right]; bm[0, region] = 1
                with torch.no_grad():
                    result = net(torch.tensor(bx), torch.tensor(bz), torch.tensor(bm)).numpy()
                actual[start:stop] = result[0, halo:halo + stop - start]
            np.testing.assert_allclose(actual, expected, atol=3e-16, rtol=1e-13)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Optional Torch training environment unavailable")
    def test_joint_training_is_deterministic_and_excludes_unknown_scaling(self):
        from tools.train_joint import fit
        rng = np.random.default_rng(91); x = rng.normal(size=(160, 98)); z = rng.random((160, 58))
        y = np.array(list("HADHADXX") * 20); w = np.ones(160); w[3] = 0
        x[y == "X"] = 1e5; z[y == "X"] = 1e5
        sequence = {"X": x, "Z": z, "y": y, "w": w, "base_home": np.zeros(160), "base_dead": np.zeros(160)}
        a = fit([sequence], epochs=2); b = fit([sequence], epochs=2)
        self.assertEqual(a, b)
        use = (y != "X") & (w > 0)
        np.testing.assert_allclose(a["control_scale"], np.sqrt(np.mean(x[use] ** 2, axis=0)))
        np.testing.assert_allclose(a["context_mean"], joint.context_features(x[use], z[use]).mean(axis=0))
        self.assertEqual(a["training"]["positive_weight_rows"], int(use.sum()))
        self.assertGreater(a["training"]["dead_rows"], 0)
        self.assertEqual(set(a["training"]["class_rows"]), set("HAD"))
        with self.assertRaises(ValueError): fit([dict(sequence, w=(y == "H").astype(float))], epochs=1)
        with self.assertRaises(ValueError): fit([dict(sequence, w=(y != "A").astype(float))], epochs=1)
        with self.assertRaises(ValueError): fit([dict(sequence, y=np.repeat("X", 160))], epochs=1)


if __name__ == "__main__":
    unittest.main()

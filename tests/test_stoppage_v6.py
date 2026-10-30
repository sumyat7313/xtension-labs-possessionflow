import copy
import importlib.util
import unittest
import numpy as np
from tools.train_stoppage_v6 import stoppage_weights, WEIGHTING


class StoppageWeightTests(unittest.TestCase):
    def test_preserves_weighted_class_masses_and_bounds(self):
        y = np.array(list("H"*40 + "D"*8 + "A"*30 + "D"*100 + "X"*4))
        w = np.linspace(.1, 3, len(y)); w[9] = 0
        original = w.copy(); actual = stoppage_weights(y, w)
        active = (y != "X") & (w > 0)
        for cls in (y == "D", y != "D"):
            use = active & cls
            self.assertAlmostEqual(actual[use].sum(), w[use].sum(), places=12)
        np.testing.assert_array_equal(actual[~active], 0)
        self.assertTrue(np.all(actual[active]/w[active] >= .5-1e-14))
        self.assertTrue(np.all(actual[active]/w[active] <= 3+1e-14))
        np.testing.assert_array_equal(w, original)

    def test_short_stops_receive_more_mass_without_prior_shift(self):
        y = np.array(list("D"*5 + "X" + "D"*100))
        w = np.ones(len(y)); actual = stoppage_weights(y, w)
        self.assertGreater(actual[:5].mean(), actual[6:].mean())
        self.assertAlmostEqual(actual.sum(), 105)

    def test_boundary_focus_is_time_symmetric_and_team_invariant(self):
        y = np.array(list("H"*40 + "D"*50 + "A"*40))
        w = np.ones(len(y)); actual = stoppage_weights(y, w)
        np.testing.assert_allclose(actual, stoppage_weights(y[::-1], w[::-1])[::-1], atol=1e-14)
        swapped = np.where(y == "H", "A", np.where(y == "A", "H", y))
        np.testing.assert_array_equal(actual, stoppage_weights(swapped, w))
        self.assertGreater(actual[39], actual[10])
        self.assertGreater(actual[40], actual[65])

    def test_unknown_and_zero_weight_gaps_never_create_transitions(self):
        y = np.array(list("H"*20 + "X" + "D"*20))
        w = np.ones(len(y)); actual = stoppage_weights(y, w)
        np.testing.assert_allclose(actual[y != "X"], 1)
        y[20] = "H"; w[20] = 0
        np.testing.assert_allclose(stoppage_weights(y, w), actual)

    def test_neighborhoods_do_not_compound(self):
        y = np.array(list("H"*40 + "D"*4 + "A"*40))
        w = np.ones(len(y)); actual = stoppage_weights(y, w)
        np.testing.assert_allclose(actual[40:44], 1)
        self.assertTrue(np.isfinite(actual).all())

    def test_no_active_rows_and_single_class_are_supported(self):
        np.testing.assert_array_equal(stoppage_weights(np.array([]), np.array([])), [])
        np.testing.assert_array_equal(stoppage_weights(list("XXX"), [1, 2, 3]), [0, 0, 0])
        np.testing.assert_allclose(stoppage_weights(list("DDD"), [1, 2, 3]), [1, 2, 3])

    def test_bad_inputs_fail_closed(self):
        for y, w, fps in [(list("HD"), [1], 5), (list("HQ"), [1, 1], 5),
                          (list("HD"), [-1, 1], 5), (list("HD"), [1, np.nan], 5),
                          (list("HD"), [1, 1], True), (list("HD"), [1, 1], 0),
                          (list("HD"), [1, 1], np.inf)]:
            with self.assertRaises(ValueError): stoppage_weights(y, w, fps)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Optional Torch training environment unavailable")
    def test_deterministic_training_preserves_v5_scaling_and_schema(self):
        from tools.train_stoppage_v6 import fit
        from possession.joint_temporal import validate_model, context_features
        rng = np.random.default_rng(246)
        y = np.array(list("H"*20 + "D"*8 + "A"*20 + "D"*40 + "X"*8))
        x = rng.normal(size=(len(y), 98)); z = rng.normal(size=(len(y), 58)); w = np.ones(len(y))
        x[y == "X"] = 1e5; z[y == "X"] = 1e5; w[5] = 0
        seq = dict(X=x,Z=z,y=y,w=w,base_home=np.zeros(len(y)),base_dead=np.zeros(len(y)))
        untouched = copy.deepcopy(seq)
        a = fit([seq], epochs=1); b = fit([seq], epochs=1)
        self.assertEqual(a, b); validate_model(a)
        good = (y != "X") & (w > 0)
        np.testing.assert_allclose(a["control_scale"], np.sqrt(np.mean(x[good]**2, axis=0)))
        np.testing.assert_allclose(a["context_mean"], context_features(x[good],z[good]).mean(axis=0))
        self.assertEqual(a["training"]["stoppage_weighting"], WEIGHTING)
        for name in seq: np.testing.assert_array_equal(seq[name], untouched[name])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Optional Torch training environment unavailable")
    def test_unweighted_limit_reproduces_v5_training(self):
        from tools.train_stoppage_v6 import fit
        from tools.train_joint import fit as old_fit
        rng = np.random.default_rng(364); sequences=[]
        for label in "HAD":
            sequences.append(dict(X=rng.normal(size=(16,98)),Z=rng.normal(size=(16,58)),
                y=np.repeat(label,16),w=np.ones(16),base_home=np.zeros(16),base_dead=np.zeros(16)))
        old=old_fit(sequences,epochs=1); new=fit(sequences,epochs=1)
        for name in old:
            if name != "training": self.assertEqual(old[name],new[name],name)


if __name__ == "__main__":
    unittest.main()

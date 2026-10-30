import copy
import unittest
import numpy as np
from possession import boundary
from tools.train_boundary import targets


def invented_model():
    coefficients = np.zeros(len(boundary.FEATURE_NAMES))
    coefficients[boundary.FEATURE_NAMES.index("home_margin_reversal_0.2")] = 20
    return {"schema_version":1,"kind":"local_possession_boundary","feature_names":list(boundary.FEATURE_NAMES),
            "mean":np.zeros(len(coefficients)).tolist(),"scale":np.ones(len(coefficients)).tolist(),
            "coefficients":coefficients.tolist(),"intercept":float(np.log(.05/.95)),"prior":.05,
            "target_radius_seconds":.16,"maximum_cost_reduction":.75}


def observations(n=250):
    x = np.zeros((n, 98)); p = np.tile([.95, .04, .01], (n, 1))
    return x, p


class BoundaryTests(unittest.TestCase):
    def test_team_exchange_invariance(self):
        rng = np.random.default_rng(810); x = rng.normal(size=(200, 98))
        p = rng.uniform(.1,1,(200,3));p /= p.sum(axis=1)[:,None];blocks=[(1,0,89),(2,100,199)]
        a=boundary.feature_matrix(x,p,blocks,25)
        b=boundary.feature_matrix(-x,p[:,[1,0,2]],blocks,25)
        np.testing.assert_allclose(a,b,atol=2e-15)
        np.testing.assert_allclose(boundary.switch_costs(invented_model(),x,p,blocks,25,1),
                                   boundary.switch_costs(invented_model(),-x,p[:,[1,0,2]],blocks,25,1),atol=2e-15)

    def test_half_isolation_and_unknown_gap(self):
        x,p=observations(250);blocks=[(1,0,99),(2,150,249)]
        a=boundary.feature_matrix(x,p,blocks,25)
        x[150:]=20;p[150:]=[.04,.95,.01]
        b=boundary.feature_matrix(x,p,blocks,25)
        np.testing.assert_array_equal(a[:150],b[:150]);np.testing.assert_array_equal(a[100:150],0)
        prob=boundary.predict_probability(invented_model(),x,p,blocks,25)
        np.testing.assert_array_equal(prob[100:150],0);self.assertEqual(prob[0],0);self.assertEqual(prob[150],0)

    def test_supported_turnover_lowers_only_local_cost(self):
        x,p=observations();p[100:150]=[.04,.95,.01];blocks=[(1,0,249)]
        cost=boundary.switch_costs(invented_model(),x,p,blocks,25,1)
        self.assertLess(cost[100],.4);self.assertLess(cost[150],.4)
        np.testing.assert_allclose(cost[30:70],1);np.testing.assert_allclose(cost[115:135],1)
        self.assertTrue((cost>=.25).all());self.assertTrue((cost<=1).all())

    def test_weak_noisy_changes_do_not_get_maximum_discount(self):
        x,p=observations();p[:]=[.499,.491,.01];p[100:]=[.491,.499,.01]
        cost=boundary.switch_costs(invented_model(),x,p,[(1,0,249)],25,1)
        self.assertGreater(cost[100],.99)

    def test_no_artificial_half_boundary_target_or_gap_bridge(self):
        labels=np.array(list("HHHAAAXAAADHHHAAA"))
        y,use=targets(labels,[(1,0,5),(2,6,16)],25)
        self.assertEqual(y[6],0);self.assertEqual(y[10],0)
        self.assertFalse(use[0]);self.assertFalse(use[6]);self.assertFalse(use[10])
        y,_=targets(np.array(list("HHHAAA")),[(1,0,2),(2,3,5)],25)
        np.testing.assert_array_equal(y,0)
        y,_=targets(np.array(list("HHHXAAA")),[(1,0,6)],25)
        np.testing.assert_array_equal(y,0)

    def test_targets_remain_local_and_unknown_labels_never_train(self):
        labels=np.array(list("H"*20+"A"*20+"X"*4+"H"*20))
        target,use=targets(labels,[(1,0,len(labels)-1)],25)
        self.assertEqual(np.flatnonzero(target).tolist(),list(range(16,25)))
        self.assertFalse(use[40:44].any())

    def test_invalid_schema_features_probabilities_and_blocks(self):
        x,p=observations();model=invented_model()
        for key,value in [("prior",1),("scale",[0]*53),("maximum_cost_reduction",1),
                          ("intercept",float("nan")),("target_radius_seconds",.5)]:
            bad=copy.deepcopy(model);bad[key]=value
            with self.assertRaises(ValueError):boundary.validate_model(bad)
        with self.assertRaises(ValueError):boundary.feature_matrix(x,p,[(1,0,100),(2,100,249)],25)
        with self.assertRaises(ValueError):boundary.feature_matrix(x,p*2,[(1,0,249)],25)
        with self.assertRaises(ValueError):boundary.feature_matrix(x,p,[(1,0,249)],True)
        with self.assertRaises(ValueError):boundary.switch_costs(model,x,p,[(1,0,249)],25,-1)

    def test_single_frame_empty_blocks_and_zero_base_cost(self):
        x,p=observations(1);model=invented_model()
        np.testing.assert_array_equal(boundary.predict_probability(model,x,p,[(1,0,0)],25),0)
        np.testing.assert_array_equal(boundary.feature_matrix(x,p,[],25),0)
        np.testing.assert_array_equal(boundary.switch_costs(model,x,p,[(1,0,0)],25,0),0)


if __name__ == "__main__": unittest.main()

import copy
import unittest
import numpy as np
from possession import boundary, transition
from tools.train_transition_v6 import diagnostic


def classifier(probability=.2, prior=.02):
    n=len(boundary.FEATURE_NAMES)
    return {"schema_version":1,"kind":"local_possession_boundary","feature_names":list(boundary.FEATURE_NAMES),
            "mean":[0.]*n,"scale":[1.]*n,"coefficients":[0.]*n,
            "intercept":float(np.log(probability/(1-probability))),"prior":prior,
            "target_radius_seconds":.16,"maximum_cost_reduction":.75}


def observation(n=100):
    return np.zeros((n,98)),np.tile([.95,.04,.01],(n,1)),[(1,0,n-1)]


class TransitionV6Tests(unittest.TestCase):
    def test_prior_evidence_is_neutral_and_subprior_never_increases_cost(self):
        x,p,blocks=observation();p[30:35]=[.04,.95,.01]
        for probability in (.02,.001):
            out=transition.evidence(transition.from_boundary(classifier(probability)),x,p,blocks,25,4)
            np.testing.assert_allclose(out['team_switch_costs'],4,atol=1e-14)
            self.assertFalse(out['early_exit_allowed'].any())

    def test_rare_boundary_evidence_reduces_cost_by_relative_odds(self):
        x,p,blocks=observation();p[30:70]=[.04,.95,.01]
        # A .075 probability at prior .02 has odds ratio about four, well below
        # .5 absolute probability but substantial boundary evidence.
        c=classifier(.075)
        model=transition.from_boundary(c)
        new=transition.switch_costs(model,x,p,blocks,25,4)
        old=boundary.switch_costs(c,x,p,blocks,25,4)
        self.assertLess(new[30],old[30]);self.assertGreaterEqual(new.min(),1)
        self.assertLessEqual(new.max(),4);self.assertEqual(new[0],4)
        np.testing.assert_array_equal(new[40:60],4)

    def test_early_exit_requires_confident_live_reversal(self):
        x,p,blocks=observation();model=transition.from_boundary(classifier())
        self.assertFalse(transition.short_exit_allowed(model,x,p,blocks,25).any())
        np.testing.assert_array_equal(transition.switch_costs(model,x,p,blocks,25,4),4)
        p[30:35]=[.04,.95,.01]
        allow=transition.short_exit_allowed(model,x,p,blocks,25)
        self.assertTrue(allow[30]);self.assertTrue(allow[35])
        p[:]=[.51,.48,.01];p[30:35]=[.48,.51,.01]
        self.assertFalse(transition.short_exit_allowed(model,x,p,blocks,25).any())
        p[:]=[.57,.03,.4];p[30:35]=[.03,.57,.4]
        self.assertFalse(transition.short_exit_allowed(model,x,p,blocks,25).any())

    def test_team_exchange_invariant_costs_and_gates(self):
        rng=np.random.default_rng(911);x,p,blocks=observation()
        x[:]=rng.normal(size=x.shape);p[30:35]=[.04,.95,.01]
        c=classifier();c['coefficients']=rng.normal(size=len(boundary.FEATURE_NAMES)).tolist()
        model=transition.from_boundary(c)
        a=transition.evidence(model,x,p,blocks,25,4)
        b=transition.evidence(model,-x,p[:,[1,0,2]],blocks,25,4)
        np.testing.assert_allclose(a['team_switch_costs'],b['team_switch_costs'],atol=1e-14)
        np.testing.assert_array_equal(a['early_exit_allowed'],b['early_exit_allowed'])

    def test_half_isolation_gap_and_incomplete_context(self):
        x,p,_=observation();blocks=[(1,0,39),(2,50,99)]
        model=transition.from_boundary(classifier())
        original=transition.evidence(model,x,p,blocks,25,4)
        p[40:]=[.04,.95,.01];x[40:]=99
        changed=transition.evidence(model,x,p,blocks,25,4)
        for key in original:np.testing.assert_array_equal(original[key][:40],changed[key][:40])
        np.testing.assert_array_equal(changed['team_switch_costs'][40:51],4)
        self.assertFalse(changed['early_exit_allowed'][35:55].any())

    def test_schema_inputs_and_policy_fail_closed(self):
        x,p,blocks=observation();model=transition.from_boundary(classifier())
        for value in (.0,True,float('nan')):
            bad=copy.deepcopy(model);bad['policy']['minimum_short_team_seconds']=value
            with self.assertRaises(ValueError):transition.validate_model(bad)
        with self.assertRaises(ValueError):transition.evidence(model,x,p*2,blocks,25,4)
        with self.assertRaises(ValueError):transition.evidence(model,x,p,blocks,25,-1)
        with self.assertRaises(ValueError):transition.evidence(model,x,p,blocks,True,4)
        with self.assertRaises(ValueError):transition.evidence(model,x,p,[(1,0,50),(2,50,99)],25,4)

    def test_zero_cost_single_frame_and_empty_blocks(self):
        x,p,blocks=observation(1);model=transition.from_boundary(classifier())
        for current in (blocks,[]):
            out=transition.evidence(model,x,p,current,25,0)
            np.testing.assert_array_equal(out['team_switch_costs'],0)
            self.assertFalse(out['early_exit_allowed'].any())

    def test_diagnostic_counts_fragmentation_and_matches_globally(self):
        y=np.array(list('H'*50+'A'*5+'H'*50))
        exact=diagnostic(y,y)
        self.assertEqual(exact['bins']['<0.4s']['recall'],1)
        missed=diagnostic(y,np.repeat('H',len(y)))
        self.assertEqual(missed['bins']['<0.4s']['recall'],0)
        y=np.repeat('H',100);p=y.copy();p[50:55]='A'
        extra=diagnostic(y,p)
        self.assertEqual(extra['bins']['<0.4s']['n_pred'],1)
        self.assertEqual(extra['bins']['<0.4s']['precision'],0)

    def test_diagnostic_does_not_bridge_unknown_reference(self):
        y=np.array(list('HHHXXAAA'));p=np.repeat('H',len(y))
        out=diagnostic(y,p,25)
        self.assertEqual(out['bins']['<0.4s']['n_gt'],2)
        self.assertEqual(out['bins']['<0.4s']['n_matched_gt'],1)
        with self.assertRaises(ValueError):diagnostic(y,p[:-1])


if __name__=='__main__':unittest.main()

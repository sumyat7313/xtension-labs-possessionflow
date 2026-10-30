import copy
import unittest
import numpy as np
from possession import typed_boundary as tb
from possession.config import Config
from tools.train_typed_boundary_v7 import targets,fit


def invented_model(contact=False):
    names=tb.feature_names(contact);priors=np.array([.85,.08,.04,.03])
    return {'schema_version':1,'kind':'typed_boundary_softmax','classes':list(tb.CLASSES),
        'contact_features':contact,'feature_names':list(names),'policy':dict(tb.POLICY),
        'mean':np.zeros(len(names)).tolist(),'scale':np.ones(len(names)).tolist(),
        'coefficients':np.zeros((len(names),3)).tolist(),'intercepts':np.log(priors[1:]/priors[0]).tolist(),
        'priors':priors.tolist(),'training':{},'metadata':{}}


def observations(n=120):
    rng=np.random.default_rng(31);x=rng.normal(size=(n,98))*.1;z=rng.uniform(0,1,(n,58))
    p=np.tile([.8,.19,.01],(n,1));p[40:80]=[.19,.8,.01]
    return x,z,p


class TypedBoundaryTests(unittest.TestCase):
    def test_targets_distinguish_turnover_stop_restart(self):
        y=np.array(list('H'*20+'A'*20+'D'*20+'H'*20))
        t,use=targets(y,[(1,0,79)],25)
        self.assertEqual([t[i] for i in (20,40,60)],[1,2,3])
        for i in (20,40,60):self.assertTrue((t[i-3:i+4]==t[i]).all())
        self.assertFalse(use[0]);self.assertEqual(int((t>0).sum()),21)

    def test_targets_do_not_cross_unknown_or_half_edges_and_nearest_wins(self):
        y=np.array(list('HHHHAAAAXDDDDHHHH'))
        t,use=targets(y,[(1,0,3),(2,4,len(y)-1)],25)
        self.assertEqual(t[3],0);self.assertEqual(t[4],0);self.assertEqual(t[8],0);self.assertFalse(use[9])
        y=np.array(list('H'*10+'D'+'A'*10));t,_=targets(y,[(1,0,len(y)-1)],25)
        self.assertEqual(t[10],2);self.assertEqual(t[11],3)
        self.assertTrue(np.isin(t,[0,1,2,3]).all())

    def test_prior_probabilities_are_exactly_neutral_costs(self):
        x,z,p=observations();model=invented_model();cfg=Config()
        q=tb.predict_from_features(model,tb.feature_matrix(x,z,p,[(1,0,119)],25))
        np.testing.assert_allclose(q,np.tile(model['priors'],(120,1)),atol=1e-15)
        values=tb.evidence(model,x,z,p,[(1,0,119)],25,cfg)
        for key,value in [('team_switch_costs',cfg.team_switch_cost),('stop_costs',cfg.stop_cost),('restart_costs',cfg.restart_cost)]:
            np.testing.assert_array_equal(values[key],value)

    def test_discounts_are_typed_and_require_directional_context(self):
        x,z,p=observations();model=invented_model();cfg=Config()
        model['intercepts'][1]+=10
        p[:60]=[.49,.49,.02];p[60:]=[.01,.01,.98]
        costs=tb.evidence(model,x,z,p,[(1,0,119)],25,cfg)
        self.assertLess(costs['stop_costs'][60],cfg.stop_cost)
        np.testing.assert_array_equal(costs['restart_costs'],cfg.restart_cost)
        self.assertEqual(costs['stop_costs'][30],cfg.stop_cost)
        reversed_costs=tb.evidence(model,x,z,p[::-1].copy(),[(1,0,119)],25,cfg)
        self.assertEqual(reversed_costs['stop_costs'][60],cfg.stop_cost)

    def test_negative_evidence_penalizes_spurious_boundaries_with_bounds(self):
        x,z,p=observations();model=invented_model();model['intercepts']=(np.array(model['intercepts'])-20).tolist();cfg=Config()
        costs=tb.evidence(model,x,z,p,[(1,0,119)],25,cfg)
        for key,base in [('team_switch_costs',cfg.team_switch_cost),('stop_costs',cfg.stop_cost),('restart_costs',cfg.restart_cost)]:
            self.assertGreater(costs[key][60],base)
            self.assertTrue(np.all(costs[key]>=.75*base));self.assertTrue(np.all(costs[key]<=1.25*base))
            self.assertEqual(costs[key][0],base)

    def test_team_exchange_symmetry_and_half_isolation(self):
        x,z,p=observations();blocks=[(1,0,49),(2,60,119)]
        features=tb.feature_matrix(x,z,p,blocks,25)
        swapped=tb.feature_matrix(-x,z,p[:,[1,0,2]],blocks,25)
        np.testing.assert_allclose(features,swapped,atol=1e-14)
        for a,b in ((0,49),(60,119)):
            standalone=tb.feature_matrix(x[a:b+1],z[a:b+1],p[a:b+1],[(1,0,b-a)],25)
            np.testing.assert_allclose(features[a:b+1],standalone,atol=1e-14)
        np.testing.assert_array_equal(features[50:60],0)
        model=invented_model();model['intercepts'][0]+=4
        v=tb.evidence(model,x,z,p,blocks,25,Config());w=tb.evidence(model,-x,z,p[:,[1,0,2]],blocks,25,Config())
        for key in v:np.testing.assert_allclose(v[key],w[key],atol=1e-14)

    def test_contact_ablation_schema_and_features_are_explicit(self):
        x,z,p=observations();blocks=[(1,0,119)]
        basic=tb.feature_matrix(x,z,p,blocks,25);contact=tb.feature_matrix(x,z,p,blocks,25,contact=True)
        self.assertEqual(basic.shape[1],90);self.assertEqual(contact.shape[1],98)
        np.testing.assert_array_equal(contact[:,:90],basic)
        tb.validate_model(invented_model(True))
        with self.assertRaises(ValueError):tb.predict_from_features(invented_model(True),basic)

    def test_multiclass_fit_deterministic_and_prior_correction_valid(self):
        y=np.array(list(('H'*30+'A'*30+'D'*30+'H'*30)*2));n=len(y);t,use=targets(y,[(1,0,n-1)],25)
        rng=np.random.default_rng(4);x=rng.normal(size=(n,90))*.01
        x[:,:3]+=np.eye(4)[t,1:]
        items=[{'X':x,'y':y,'blocks':[(1,0,n-1)],'fps':25}]
        model=fit(items,stride=5);again=fit(items,stride=5)
        self.assertEqual(model,again)
        self.assertEqual(model['training']['class_rows'][1:],np.bincount(t[use],minlength=4)[1:].tolist())
        np.testing.assert_allclose(model['priors'],np.bincount(t[use],minlength=4)/use.sum(),atol=1e-14)
        q=tb.predict_from_features(model,x)
        np.testing.assert_allclose(q.sum(axis=1),1.,atol=1e-14)
        self.assertGreater(np.mean(np.argmax(q[use],axis=1)==t[use]),.98)

    def test_unknown_rows_are_excluded_from_fit_scaling(self):
        y=np.array(list('H'*30+'A'*30+'X'*10+'D'*30+'H'*30));n=len(y)
        rng=np.random.default_rng(6);x=rng.normal(size=(n,90));item={'X':x,'y':y,'blocks':[(1,0,n-1)],'fps':25}
        # Insert a valid stop so every class is present without bridging X.
        y[40:45]='D'
        model=fit([item]);altered=copy.deepcopy(item);altered['X'][60:71]=1e6
        same=fit([altered]);self.assertEqual(model,same)

    def test_constant_feature_optimizer_returns_empirical_priors(self):
        y=np.array(list('H'*101+'A'*31+'D'*47+'H'*22));n=len(y)
        x=np.zeros((n,90));model=fit([{'X':x,'y':y,'blocks':[(1,0,n-1)],'fps':25}],stride=5)
        t,use=targets(y,[(1,0,n-1)],25);expected=np.bincount(t[use],minlength=4)/use.sum()
        q=tb.predict_from_features(model,x)
        np.testing.assert_allclose(q,np.tile(expected,(n,1)),atol=1e-10)
        np.testing.assert_allclose(model['coefficients'],0,atol=1e-14)

    def test_invalid_inputs_and_missing_classes_fail_closed(self):
        x,z,p=observations();model=invented_model()
        for key,value in [('priors',[1,0,0,0]),('scale',[0]*90),('classes',['none']),('contact_features',1)]:
            bad=copy.deepcopy(model);bad[key]=value
            with self.assertRaises(ValueError):tb.validate_model(bad)
        with self.assertRaises(ValueError):tb.feature_matrix(x,z[:,:57],p,[(1,0,119)],25)
        with self.assertRaises(ValueError):targets(np.array(list('H?A')),[(1,0,2)],25)
        with self.assertRaises(ValueError):fit([{'X':np.zeros((10,90)),'y':np.array(list('H'*10)),'blocks':[(1,0,9)],'fps':25}])
        with self.assertRaises(ValueError):fit([{'X':np.zeros((10,90)),'y':np.array(list('H'*10)),'blocks':[(1,0,9)],'fps':25,'w':np.ones(10)}])


if __name__=='__main__':unittest.main()

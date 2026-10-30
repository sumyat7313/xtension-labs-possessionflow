import copy
import unittest
import numpy as np
from possession import temporal


def invented_model():
    rng=np.random.default_rng(912)
    return {'schema_version':1,'kind':'odd_temporal_residual','feature_names':list(temporal.FEATURE_NAMES),
        'sample_hz':5,'dilations':list(temporal.DILATIONS),'scale':np.ones(98).tolist(),
        'input_weight':rng.normal(0,.05,(16,98)).tolist(),'output_weight':rng.normal(0,.05,(1,16)).tolist(),
        **{f'conv_{i}':rng.normal(0,.03,(16,16,3)).tolist() for i in range(3)}}


class TemporalTests(unittest.TestCase):
    def test_team_symmetry_and_gate_symmetry(self):
        rng=np.random.default_rng(31); x=rng.normal(size=(300,98)); base=rng.normal(size=300)
        gate={'kind':'reliability_expert_gate','feature_names':list(temporal.GATE_NAMES),'coefficients':[0,1,-1,.3,.7,-.8]}
        for g in (None,gate):
            p=temporal.predict_home(base,x,[(1,0,299)],25,invented_model(),g)
            q=temporal.predict_home(-base,-x,[(1,0,299)],25,invented_model(),g)
            np.testing.assert_allclose(p,1-q,atol=2e-16)

    def test_half_isolation_and_unknown_gaps(self):
        rng=np.random.default_rng(9); x=rng.normal(size=(300,98)); changed=x.copy(); changed[150:]=100
        blocks=[(1,0,139),(2,150,299)]
        original=temporal.residual(invented_model(),x,blocks,25)
        np.testing.assert_array_equal(original[:150],temporal.residual(invented_model(),changed,blocks,25)[:150])
        np.testing.assert_array_equal(original[140:150],0)

    def test_zero_evidence_and_short_halves(self):
        for n in (1,2,4,5,12):
            result=temporal.residual(invented_model(),np.zeros((n,98)),[(1,0,n-1)],25)
            np.testing.assert_array_equal(result,0)

    def test_schema_and_bad_shapes(self):
        model=invented_model()
        for key,value in [('scale',[0]*98),('dilations',[1,2,3]),('sample_hz',10),('output_weight',[[float('nan')]*16])]:
            bad=copy.deepcopy(model);bad[key]=value
            with self.assertRaises(ValueError): temporal.validate_model(bad)
        with self.assertRaises(ValueError): temporal.residual(model,np.zeros((10,98)),[(1,0,9),(2,8,9)],25)
        with self.assertRaises(ValueError): temporal.residual(model,np.zeros((10,98)),[(1,0,9)],24)

    def test_reliability_gate_is_a_bounded_probability_mixture(self):
        rng=np.random.default_rng(42);x=rng.normal(size=(40,98));base=rng.normal(size=40)
        model=invented_model(); expert=temporal.predict_home(base,x,[(1,0,39)],25,model)
        p=1/(1+np.exp(-base)); gate={'kind':'reliability_expert_gate','feature_names':list(temporal.GATE_NAMES),'coefficients':[0]*6}
        mix=temporal.predict_home(base,x,[(1,0,39)],25,model,gate)
        np.testing.assert_array_equal(mix,.5*p+.5*expert)

    def test_nested_temporal_artifact_production_matches_shared_emissions(self):
        from pathlib import Path
        from possession import learned, robust_control
        from possession.config import Config
        from possession.features import extract
        from possession.tags import build
        from test_control import invented_tracking
        model=learned.load_model(Path(__file__).resolve().parents[1]/'models/baselines/v1.3.json')
        model['temporal_model']=invented_model()
        model['expert_gate']={'kind':'reliability_expert_gate','feature_names':list(temporal.GATE_NAMES),'coefficients':[0]*6}
        tracking=invented_tracking();blocks=[(1,0,89),(2,100,199)];pitch=(105,68);config=Config()
        actual=learned.infer_learned(tracking,[],blocks,pitch,25,config,model)
        f=extract(tracking,pitch,blocks,25,config);tags=build([],f,tracking.team_names,200,25,blocks,config)
        matrix=robust_control.feature_matrix(tracking,f,blocks,pitch,25)
        head=model['robust_control_model'];base=(matrix/np.asarray(head['scale']))@np.asarray(head['coefficients'])
        dead=learned.predict_dead(learned.feature_matrix(tracking,f,blocks,pitch,25),model)
        p=temporal.predict_home(base,matrix,blocks,25,model['temporal_model'],model['expert_gate'])
        np.testing.assert_array_equal(actual[1],learned.fuse_control_probabilities(dead,p,tags.team_log_odds))
        np.testing.assert_array_equal(actual[0][90:100],'X')

    def test_gate_without_temporal_expert_is_rejected(self):
        from pathlib import Path
        from possession import learned
        model=learned.load_model(Path(__file__).resolve().parents[1]/'models/baselines/v1.3.json')
        model['expert_gate']={'kind':'reliability_expert_gate','feature_names':list(temporal.GATE_NAMES),'coefficients':[0]*6}
        with self.assertRaises(ValueError): learned.validate_model(model)

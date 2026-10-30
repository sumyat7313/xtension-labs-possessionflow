import copy
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from possession import learned,robust_control,transition
from possession.config import Config
from possession.decoder import decode
from test_boundary import invented_model
from test_control import invented_tracking

ROOT=Path(__file__).resolve().parents[1]


class TransitionIntegrationTests(unittest.TestCase):
    def test_production_passes_both_costs_and_early_exit_flags_half_locally(self):
        model=learned.load_model(ROOT/'models/baselines/v1.5.json')
        model['transition_model']=transition.from_boundary(invented_model())
        tracking=invented_tracking();blocks=[(1,0,89),(2,100,199)];pitch=(105.3,68)
        config=learned.resolve_config(Config(),model)
        with patch('possession.learned.decode',wraps=decode) as dispatch:
            actual=learned.infer_learned(tracking,[],blocks,pitch,25,Config(),model)
        x=robust_control.feature_matrix(tracking,actual[2],blocks,pitch,25)
        signals=transition.evidence(model['transition_model'],x,actual[1],blocks,25,config.team_switch_cost)
        self.assertEqual(dispatch.call_count,2)
        expected=np.full(200,'X',dtype='<U1')
        for call,(_,a,b) in zip(dispatch.call_args_list,blocks):
            np.testing.assert_array_equal(call.kwargs['team_switch_costs'],signals['team_switch_costs'][a:b+1])
            np.testing.assert_array_equal(call.kwargs['early_exit_allowed'],signals['early_exit_allowed'][a:b+1])
            expected[a:b+1]=np.array(list('HAD'))[decode(actual[1][a:b+1],25,config,
                signals['team_switch_costs'][a:b+1],early_exit_allowed=signals['early_exit_allowed'][a:b+1])]
        np.testing.assert_array_equal(actual[0],expected)
        np.testing.assert_array_equal(actual[0][90:100],'X')

    def test_incompatible_optional_heads_rejected(self):
        base=learned.load_model(ROOT/'models/baselines/v1.5.json')
        base['transition_model']=transition.from_boundary(invented_model())
        for change in ('legacy','separate','missing'):
            bad=copy.deepcopy(base)
            if change=='legacy':bad['boundary_model']=invented_model()
            elif change=='separate':bad['decoder_mode']='separate'
            else:bad.pop('robust_control_model')
            with self.assertRaises(ValueError):learned.validate_model(bad)


if __name__=='__main__':unittest.main()

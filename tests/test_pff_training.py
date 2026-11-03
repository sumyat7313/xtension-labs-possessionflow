import copy
import unittest
from unittest.mock import patch

import numpy as np

from possession.io import Tracking
from possession.pipeline import infer
from possession.config import load_config
from tools import pff_training as pff


def item(name='train-a', n=250):
    return {'match': name, 'x': np.zeros((n, 98)), 'z': np.zeros((n, 58)),
            'y': np.resize(np.repeat(list('HAD'), 25), n), 'blocks': [(1, 0, n-1)],
            'fps': 25., 'base_home': np.full(n, .5), 'base_dead': np.full(n, .2),
            'usable': [True], 'reference_fields_used_for_inference': False}


class PFFTrainingTests(unittest.TestCase):
    def test_frozen_training_membership_rejects_final_and_missing(self):
        for values, train, held in [([item('final')], ['a'], ['final']),
                                    ([item('a')], ['a'], ['a']),
                                    ([item('a')], ['a','b'], ['final']),
                                    ([item('a'),item('a')], ['a'], ['final'])]:
            with self.assertRaises(ValueError):
                pff.build_sequences(values, training_matches=train, held_out_matches=held)

    def test_equal_fixture_weight_and_unknown_exclusion(self):
        a,b=item('a',250),item('b',500)
        b['y'][20:55]='X'
        sequences,support=pff.build_sequences([a,b],training_matches=['a','b'],held_out_matches=['final'])
        self.assertAlmostEqual(support['a']['sampled_weight'],10000)
        self.assertAlmostEqual(support['b']['sampled_weight'],10000)
        self.assertTrue(all(np.all(s['w'][s['y']=='X']==0) for s in sequences))

    def test_periods_do_not_join_training_context(self):
        a=item();a['blocks']=[(1,0,99),(2,150,249)];a['x'][150:]=20
        seq,_=pff.build_sequences([a],training_matches=['train-a'],held_out_matches=[])
        self.assertEqual([len(s['y']) for s in seq],[20,20])
        self.assertTrue((seq[0]['X']==0).all());self.assertTrue((seq[1]['X']==20).all())

    def test_reject_nonuniform_fps_and_bad_blocks(self):
        for key,value in [('fps',29.97),('blocks',[(1,0,150),(2,100,249)]),
                          ('blocks',[(1,0,251)]),('blocks',[])]:
            a=item();a[key]=value
            with self.assertRaises(ValueError):pff.build_sequences([a],training_matches=['train-a'],held_out_matches=[])

    def test_invalid_reference_and_upstream_values_rejected(self):
        for key,value in [('y',np.full(250,'Q')),('base_home',np.full(250,np.nan)),
                          ('base_dead',np.full(250,2)),('reference_fields_used_for_inference',True),
                          ('y',np.full(250,'X'))]:
            a=item();a[key]=value
            with self.assertRaises(ValueError):pff.build_sequences([a],training_matches=['train-a'],held_out_matches=[])

    def test_cache_matches_production_baseline_and_ignores_reference_payload(self):
        rng=np.random.default_rng(31);n=200
        players=rng.uniform(.1,.9,(n,22,2));ball=players[:,0].copy()
        tracking=Tracking(players,np.ones((n,22)),ball,np.ones(n),np.repeat([0,1],11),('home','away'))
        blocks=[(1,0,89),(2,100,199)];pitch=(105,68);model=pff.baseline_model()
        config=load_config(pff.ROOT/'configs/baselines/v1.5.json')
        cache,labels,p=pff.prepare_cache(tracking,blocks,pitch,25,model,config)
        expected,probabilities,*_=infer(tracking,[],blocks,pitch,25,config,model)
        np.testing.assert_array_equal(labels,expected);np.testing.assert_array_equal(p,probabilities)
        cache['y']=np.full(n,'D');cache['events']=[{'team':'away'}]
        changed,q=pff.predict_cache(cache,model,config)
        np.testing.assert_array_equal(changed,labels);np.testing.assert_array_equal(q,p)
        self.assertTrue(np.all(labels[90:100]=='X'))

    def test_fit_requires_frozen_protocol(self):
        for ph in ('', 'g'*64, '0'*63, None):
            with self.assertRaises(ValueError):pff.fit_candidate([],training_matches=[],held_out_matches=[],protocol_sha256=ph)

    def test_fit_changes_only_joint_head_and_research_metadata(self):
        baseline=pff.baseline_model();saved=copy.deepcopy(baseline)
        new_head=copy.deepcopy(baseline['joint_model']);new_head['metadata']={'synthetic-test':True}
        with patch('tools.train_joint.fit',return_value=new_head) as fit:
            candidate=pff.fit_candidate([item()],training_matches=['train-a'],held_out_matches=['final'],protocol_sha256='a'*64,baseline=baseline)
        self.assertEqual(baseline,saved)
        for key in baseline:
            if key not in ('joint_model','metadata'):self.assertEqual(candidate[key],baseline[key])
        self.assertEqual(fit.call_args.kwargs['epochs'],8)
        self.assertEqual(candidate['metadata']['pff_research']['held_out_matches'],['final'])
        self.assertFalse(candidate['metadata']['pff_research']['policy']['automatic_promotion'])


if __name__=='__main__':unittest.main()

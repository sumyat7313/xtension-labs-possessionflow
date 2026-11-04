"""Synthetic end-to-end inner campaign tests; no provider records or real fits."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools import clean_pff_adaptation_campaign as c
from tests.test_clean_pff_adaptation_fit import toy_parent,toy_record
from tests.test_joint_temporal import invented_model


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();(self.root/'runs').mkdir()
        self.out=self.root/'runs'/c.stages.OUT_NAME
        self.parent=toy_parent();self.y=np.tile(np.repeat(list('HDAD'),25),8)
        self.blocks=[[1,0,len(self.y)-1]]
        self.reference_calls=[];self.prediction_calls=[]
        self._patch(c,'ROOT',self.root)
        # Durability syscalls are covered by lifecycle tests; this integration
        # suite checks ordering and artifact semantics in disposable fixtures.
        self._patch(c.stages.os,'fsync',lambda fd:None)
        # Paths are exclusively generated beneath this synthetic temporary root;
        # link/junction defenses have dedicated input and lifecycle test suites.
        self._patch(c.stages,'_no_links',lambda path:Path(path))
        self._patch(c.stages,'descriptor',lambda root=c.ROOT:dict(descriptor_sha256='a'*64))
        self._patch(c.kernel,'validate_parent',lambda model:model)
        self._patch(c,'_context',self.context)
        self._patch(c,'_metadata',self.metadata)
        self._patch(c,'_observations',self.observations)
        self._patch(c,'_reference',self.reference)
        self._patch(c,'_predict',self.predict)
        self._patch(c.inputs,'make_inner_factory',self.factory)
        self._patch(c.kernel.train_joint,'fit',self.joint)
        metrics,short,support=c._measure(self.y,self.y,self.blocks)
        self.base=[dict(match_id=mid,condition='native',baseline_sha256=c.protocol.CLEAN_SHA256,
            baseline_prediction_sha256='b'*64,reference_sha256='c'*64,
            metrics=dict(metrics,spell=metrics['spell']-.01),reference_support=support) for mid in c.IDS]
        self._patch(c.protocol,'BASELINE_RECORDS_SHA256',c.protocol.original.digest(self.base))
        body=c.protocol._body({'tools/clean_pff_adaptation_protocol.py':'d'*64,
                              'tools/clean_pff_adaptation_fit.py':'e'*64},self.base)
        self.snapshot=dict(body,protocol_sha256=c.protocol.original.digest(body))
        c.stages.initialize(self.out,root=self.root)

    def _patch(self,obj,name,value):
        p=patch.object(obj,name,value);p.start();self.addCleanup(p.stop)

    def context(self):
        manifests={f'pff{mid}':dict(match_id=mid,rows=len(self.y),identity={'events':'c'*64},files={'alignment.json':'f'*64})
                   for mid in c.protocol.INNER_TRAIN_IDS}
        return dict(snapshot=copy.deepcopy(self.snapshot),parent=copy.deepcopy(self.parent),pff=manifests,
            inner={mid:dict(match_id=mid,rows=len(self.y),identity={'events':'c'*64},files={'alignment.json':'f'*64}) for mid in c.IDS},
            sportec_seals={mid:'1'*64 for mid in c.protocol.SPORTEC_IDS})

    def metadata(self,mid,ctx):
        c._mid(mid)
        return dict(blocks=self.blocks,fps=25,usable=[True],match=f'pff{mid}',reference_fields_used_for_inference=False,pitch=[105,68])

    def observations(self,mid,ctx):
        return dict(self.metadata(mid,ctx),x=np.zeros((len(self.y),98)),z=np.zeros((len(self.y),58)),synthetic_mid=mid)

    def predict(self,item,model):
        self.prediction_calls.append((model['candidate_id'],item['synthetic_mid']))
        return self.y.copy()

    def reference(self,mid,ctx):
        c._mid(mid)
        self.assertEqual(len(self.prediction_calls),16)
        self.assertTrue((self.out/'artifacts/freeze/seal.json').exists())
        self.reference_calls.append(mid)
        return self.y.copy()

    def factory(self,**kwargs):
        c.kernel.validate_fixture_ids(kwargs['fixture_ids'])
        parent=self.parent
        class Factory:
            def __call__(self,mid):return toy_record(mid)
            def parent_model(self):return copy.deepcopy(parent)
        return Factory()

    def joint(self,sequences,*,seed,epochs,metadata):
        model=invented_model();model['metadata']=copy.deepcopy(metadata)
        model['training']=dict(seed=seed,epochs=epochs,losses=[.1]*epochs)
        return model

    def advance(self,stop=None):
        for stage in c.stages.STAGES:
            c.execute_stage(stage,self.out)
            if stage==stop:break

    def test_complete_actual_artifact_chain_and_numerical_selection(self):
        self.advance()
        result=c.verify_completed(self.out)
        self.assertTrue(result['artifacts_verified']);self.assertTrue(result['lifecycle_complete'])
        self.assertEqual(result['selection']['selected_candidate_id'],'A')
        for key in c.FLAGS:self.assertIs(result[key],False)
        self.assertEqual(len(self.prediction_calls),16)
        self.assertEqual(len(self.reference_calls),24)
        self.assertEqual(c.verify_stage('fit-A',self.out)['execution_verified'],False)

    def test_no_reference_access_until_complete_prediction_freeze(self):
        self.advance('freeze')
        self.assertEqual(self.reference_calls,[])
        self.assertEqual(len(self.prediction_calls),16)
        c.execute_stage(f'score-{c.IDS[0]}',self.out)
        self.assertEqual(self.reference_calls,[c.IDS[0]])

    def test_early_score_and_unknown_stage_cannot_call_context(self):
        with patch.object(c,'_context') as context:
            for stage in (f'score-{c.IDS[0]}','predict-A-3814','refit-winner','development','final'):
                with self.assertRaises(ValueError):c.execute_stage(stage,self.out)
            context.assert_not_called()

    def test_execute_owns_attempt_and_refuses_stale_started_attempt(self):
        with self.assertRaises(RuntimeError):
            with c.stages.attempt(self.out,'initialize',root=self.root):raise RuntimeError('synthetic interruption')
        with patch.object(c,'_context') as context,self.assertRaises(ValueError):c.execute_stage('initialize',self.out)
        context.assert_not_called()

    def test_prior_artifact_tamper_blocks_next_stage_before_fitting(self):
        self.advance('initialize')
        (self.out/'artifacts/initialize/plan.json').write_text('{}')
        with patch.object(c.kernel,'fit_candidate') as fit,self.assertRaises(ValueError):c.execute_stage('fit-A',self.out)
        fit.assert_not_called()
        self.assertTrue(c.stages.status(self.out,root=self.root)['blocked'])

    def test_failed_fit_cannot_retry_or_choose_other_candidate(self):
        self.advance('initialize')
        with patch.object(c.kernel,'fit_candidate',side_effect=RuntimeError('synthetic fit failure')):
            with self.assertRaises(RuntimeError):c.execute_stage('fit-A',self.out)
        for stage in ('fit-A','fit-B'):
            with self.assertRaises(ValueError):c.execute_stage(stage,self.out)

    def test_prediction_tamper_after_freeze_blocks_reference_access(self):
        self.advance('freeze')
        with (self.out/f'artifacts/predict-A-{c.IDS[0]}/prediction.npy').open('ab') as f:f.write(b'changed')
        with self.assertRaises(ValueError):c.execute_stage(f'score-{c.IDS[0]}',self.out)
        self.assertEqual(self.reference_calls,[])

    def test_saved_selection_tamper_rejected_by_lifecycle_binding(self):
        self.advance()
        (self.out/'artifacts/select/selection.json').write_text('{}')
        with self.assertRaises(ValueError):c.verify_completed(self.out)

    def test_partial_family_cannot_verify_complete(self):
        self.advance('fit-A')
        with self.assertRaises(ValueError):c.verify_completed(self.out)

    def test_readonly_stage_verification_never_reads_references(self):
        self.advance('fit-A')
        with patch.object(c,'_reference',side_effect=AssertionError('reference opened')):
            self.assertEqual(set(c.verify_stage('fit-A',self.out)['artifact_sha256']),{'stage_seal'})

    def test_unexpected_output_file_rejected(self):
        self.advance('fit-A')
        (self.out/'artifacts/fit-A/extra.json').write_text('{}')
        with self.assertRaises(ValueError):c.verify_stage('fit-A',self.out)

    def test_model_fixed_control_change_rejected_even_before_sealing(self):
        self.advance('initialize')
        original=c.kernel.fit_candidate
        def changed(*args,**kwargs):
            model,report=original(*args,**kwargs);model['robust_control_model']['coefficients'][0]+=.1
            return model,report
        with patch.object(c.kernel,'fit_candidate',side_effect=changed),self.assertRaisesRegex(ValueError,'control'):
            c.execute_stage('fit-A',self.out)

    def test_reference_support_mismatch_consumes_scoring_attempt(self):
        self.advance('freeze')
        with patch.object(c,'_reference',return_value=np.repeat('H',len(self.y))),self.assertRaises(ValueError):
            c.execute_stage(f'score-{c.IDS[0]}',self.out)
        self.assertTrue(c.stages.status(self.out,root=self.root)['blocked'])

    def test_stage_verification_requires_lifecycle_completion(self):
        with self.assertRaises(ValueError):c.verify_stage('initialize',self.out)


if __name__=='__main__':unittest.main()

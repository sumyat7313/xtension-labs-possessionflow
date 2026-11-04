"""Synthetic-only winner-refit gates, exact recipe parity and consumed attempts."""
from contextlib import ExitStack, contextmanager
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from tools import clean_pff_adaptation_refit as refit
from tests.test_clean_pff_adaptation_fit import toy_parent, toy_record
from tests.test_joint_temporal import invented_model


def proof(candidate='A'):
    selection = dict(protocol_sha256=refit.stages.SNAPSHOT_PROTOCOL_SHA256,
                     selected_candidate_id=candidate, inner_numerical_passed=candidate is not None,
                     **refit.protocol.FLAGS)
    return dict(selection=selection, selection_sha256=refit.protocol.original.digest(selection),
                artifacts_verified=True, lifecycle_complete=True,
                protocol_sha256=refit.stages.SNAPSHOT_PROTOCOL_SHA256, inner_ledger_tip='a'*64,
                refit_authorized=False, final_access_authorized=False, promoted=False)


class WinnerRefitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve(); (self.root/'runs').mkdir()
        self.inner = self.root/'runs'/refit.stages.OUT_NAME
        self.root_patch = patch.object(refit, 'ROOT', self.root)
        self.root_patch.start(); self.addCleanup(self.root_patch.stop)
        self.env = patch.dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        self.env.start(); self.addCleanup(self.env.stop)

    @contextmanager
    def synthetic_run(self, candidate='A'):
        def joint(sequences, **kw):
            model=invented_model();model['metadata']=copy.deepcopy(kw['metadata'])
            model['training']={'seed':kw['seed'],'epochs':kw['epochs']}
            return model
        with ExitStack() as stack:
            stack.enter_context(patch.object(refit, '_verified_selection', return_value=proof(candidate)))
            stack.enter_context(patch.object(refit, '_sources', return_value={'synthetic-source.py':'b'*64}))
            stack.enter_context(patch.object(refit, '_refit_context', return_value={}))
            stack.enter_context(patch.object(refit.inputs, '_json', return_value=toy_parent()))
            stack.enter_context(patch.object(refit, '_record', side_effect=lambda match,ctx:toy_record(match)))
            stack.enter_context(patch.object(refit.kernel, 'validate_parent'))
            stack.enter_context(patch.object(refit.kernel.train_joint, 'fit', side_effect=joint))
            yield

    def test_sampler_exact_parity_with_frozen_inner_math(self):
        rng = np.random.default_rng(112)
        for n in (90, 137, 240):
            match = refit.protocol.fit_ids()[0]
            record = toy_record(match,n)
            record['y'] = rng.choice(list('HHHAAADDX'),n)
            record['blocks'] = [[1,3,n//2-3],[2,n//2+4,n-2]]
            a,sa = refit._sample(record,match)
            b,sb = refit.kernel.sample_record(record,match,refit.protocol.recipe())
            self.assertEqual(sa,sb)
            for x,y in zip(a,b):
                for key in x: np.testing.assert_array_equal(x[key],y[key])

    def test_sampler_allows_reserved_inner8_only_for_refit(self):
        for mid in refit.protocol.INNER_VALIDATION_IDS:
            match=f'pff{mid}'; record=toy_record(match)
            _,counts=refit._sample(record,match)
            self.assertAlmostEqual(counts['sampled_weight'],10000.)
            with self.assertRaises(ValueError):refit.kernel.sample_record(record,match,refit.protocol.recipe())

    def test_all_dev_final_and_unknown_fixtures_rejected_before_sources(self):
        for mid in (*refit.protocol.DEVELOPMENT_IDS,*refit.protocol.FINAL_IDS,99999):
            with patch.object(refit.inputs,'_sources') as src, self.assertRaises(ValueError):
                refit._record(f'pff{mid}',{})
            src.assert_not_called()

    def test_wrong_record_and_historical_probabilities_rejected_before_arrays(self):
        class Explosive:
            def __array__(self,*args,**kwargs):raise AssertionError('array accessed')
        match=refit.protocol.fit_ids('refit')[0]
        for change in ({'match':'pff3814'},{'base_dead':Explosive()},{'reference_fields_used_for_inference':True}):
            record=toy_record(match);record.update(change,x=Explosive())
            with self.assertRaises(ValueError):refit._sample(record,match)

    def test_zero_mass_and_period_overlap_fail(self):
        match=refit.protocol.fit_ids('refit')[0]
        for change in ({'y':np.repeat('X',90)},{'blocks':[[1,0,40],[2,40,89]]}):
            record=toy_record(match);record.update(change)
            with self.assertRaises(ValueError):refit._sample(record,match)

    def test_A_B_fit_same_math_parameters_and_upstream_recomputation(self):
        parent=toy_parent(); roster=refit.protocol.fit_ids('refit')
        def fitted_joint(sequences,**kw):
            value=invented_model();value['metadata']=copy.deepcopy(kw['metadata'])
            value['training']={'seed':kw['seed'],'epochs':kw['epochs']}
            return value
        for candidate in ('A','B'):
            replacement=copy.deepcopy(parent);replacement['intercept']=1.2
            replacement['training']={'l2':.01,'iterations':1}
            with patch.object(refit.kernel,'validate_parent'), \
                 patch.object(refit.kernel.learned,'fit_model',return_value=replacement) as dead, \
                 patch.object(refit.kernel.train_joint,'fit',side_effect=fitted_joint) as joint:
                model,report=refit._fit(candidate,toy_record,parent,'c'*64)
            self.assertEqual(len(joint.call_args.args[0]),47)
            self.assertEqual(joint.call_args.kwargs['seed'],20261101)
            self.assertEqual(joint.call_args.kwargs['epochs'],8)
            self.assertEqual(report['training_matches'],list(roster))
            self.assertEqual(model['robust_control_model'],parent['robust_control_model'])
            self.assertFalse(report['old_joint_used']);self.assertFalse(report['submission_eligible'])
            self.assertEqual(dead.call_count,int(candidate=='B'))
            if candidate=='B':
                self.assertEqual(dead.call_args.kwargs['l2'],.01)
                self.assertEqual(dead.call_args.kwargs['max_iter'],80)
                self.assertAlmostEqual(dead.call_args.kwargs['sample_weight'].sum(),470000.)
            for match,sequence in zip(roster,joint.call_args.args[0]):
                if match in refit.protocol.fit_ids():
                    expected,_=refit.kernel.sample_record(toy_record(match),match,refit.protocol.recipe())
                    for key in ('X','Z','y','w'):np.testing.assert_array_equal(sequence[key],expected[0][key])
                np.testing.assert_allclose(sequence['base_dead'],1.2 if candidate=='B' else -.3,atol=1e-14)

    def test_invalid_parent_precedes_all_data_access(self):
        factory=Mock()
        with self.assertRaises(ValueError):refit._fit('A',factory,toy_parent(),'a'*64)
        factory.assert_not_called()

    def test_actual_completed_proof_required_no_winner_override(self):
        from tools import clean_pff_adaptation_campaign as campaign
        for field,value in (('artifacts_verified',False),('lifecycle_complete',False),('selection_sha256','b'*64),
                            ('inner_ledger_tip','bad'),('refit_authorized',True)):
            p=proof();p[field]=value
            with patch.object(campaign,'verify_completed',return_value=p),self.assertRaises(ValueError):
                refit._verified_selection(self.inner)
        with patch.object(campaign,'verify_completed',return_value=proof(None)),self.assertRaises(ValueError):
            refit._verified_selection(self.inner)
        with patch.object(campaign,'verify_completed',return_value=proof('B')) as verify:
            self.assertEqual(refit._verified_selection(self.inner)['selection']['selected_candidate_id'],'B')
            verify.assert_called_once_with(self.inner)

    def test_invalid_path_and_threads_fail_before_proof(self):
        with patch.object(refit,'_verified_selection') as verify,self.assertRaises(ValueError):
            refit.run_refit(self.root/'runs'/'other')
        verify.assert_not_called()
        with patch.dict(os.environ,OMP_NUM_THREADS='2'),patch.object(refit,'_verified_selection') as verify,self.assertRaises(ValueError):
            refit.run_refit(self.inner)
        verify.assert_not_called()

    def test_no_winner_does_not_initialize_refit_directory(self):
        with patch.object(refit,'_verified_selection',side_effect=ValueError('no winner')):
            with self.assertRaises(ValueError):refit.run_refit(self.inner)
        self.assertFalse((self.root/'runs'/refit.OUT_NAME).exists())

    def test_success_authenticates_results_and_never_retries(self):
        with self.synthetic_run():
            result=refit.run_refit(self.inner)
            self.assertEqual(refit.verify_result(self.inner),result)
            self.assertTrue(result['refit_complete']);self.assertFalse(result['promoted'])
            self.assertEqual(result['fitting_fixtures'],47)
            with self.assertRaisesRegex(ValueError,'consumed'):refit.run_refit(self.inner)

    def test_failure_and_keyboard_interrupt_consume_before_labels(self):
        with self.synthetic_run(),patch.object(refit,'_refit_context',side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):refit.run_refit(self.inner)
            out=self.root/'runs'/refit.OUT_NAME
            self.assertTrue((out/'attempt.json').exists());self.assertTrue((out/'failed.json').exists())
            with self.assertRaisesRegex(ValueError,'consumed'):refit.run_refit(self.inner)

    def test_preexisting_partial_output_cannot_resume(self):
        out=self.root/'runs'/refit.OUT_NAME;out.mkdir();(out/'model.json').write_text('{}')
        with self.synthetic_run(),patch.object(refit,'_refit_context') as context,self.assertRaises(ValueError):
            refit.run_refit(self.inner)
        context.assert_not_called()

    def test_selection_mutation_after_fit_prevents_complete_seal(self):
        with self.synthetic_run(),patch.object(refit,'_verified_selection',side_effect=[proof('A'),proof('B')]):
            with self.assertRaisesRegex(ValueError,'changed'):refit.run_refit(self.inner)
        self.assertFalse((self.root/'runs'/refit.OUT_NAME/'complete.json').exists())

    def test_source_mutation_after_fit_prevents_complete_seal(self):
        with self.synthetic_run(),patch.object(refit,'_sources',side_effect=[{'s':'a'*64},{'s':'b'*64}]):
            with self.assertRaisesRegex(ValueError,'changed'):refit.run_refit(self.inner)
        self.assertFalse((self.root/'runs'/refit.OUT_NAME/'complete.json').exists())

    def test_completed_model_byte_tamper_is_rejected(self):
        with self.synthetic_run():
            refit.run_refit(self.inner)
            path=self.root/'runs'/refit.OUT_NAME/'model.json';path.write_bytes(path.read_bytes()+b' ')
            with self.assertRaisesRegex(ValueError,'bytes differ'):refit.verify_result(self.inner)

    def test_matching_resealed_report_cannot_hide_stale_joint_or_extra_head(self):
        with self.synthetic_run():
            refit.run_refit(self.inner)
            out=self.root/'runs'/refit.OUT_NAME
            model=json.loads((out/'model.json').read_text());report=json.loads((out/'report.json').read_text())
            complete=json.loads((out/'complete.json').read_text())
            for change in ('metadata','epochs','extra_head'):
                changed=copy.deepcopy(model);changed_report=copy.deepcopy(report)
                if change=='metadata':changed['joint_model']['metadata']={'stale':True}
                elif change=='epochs':
                    changed['joint_model']['training']['epochs']=9
                    changed_report['new_joint_training']=changed['joint_model']['training']
                else:changed['unexpected_old_head']={}
                (out/'model.json').write_text(json.dumps(changed));(out/'report.json').write_text(json.dumps(changed_report))
                complete['model_sha256']=refit.storage.file_hash(out/'model.json')
                complete['report_sha256']=refit.storage.file_hash(out/'report.json')
                (out/'complete.json').write_text(json.dumps(complete))
                with self.subTest(change=change),self.assertRaises(ValueError):refit.verify_result(self.inner)

    def test_B_resealed_report_cannot_relax_regularization_or_iteration_limit(self):
        with self.synthetic_run('B'):
            refit.run_refit(self.inner)
            out=self.root/'runs'/refit.OUT_NAME
            model=json.loads((out/'model.json').read_text());report=json.loads((out/'report.json').read_text())
            complete=json.loads((out/'complete.json').read_text())
            for key,value in (('l2',.1),('iterations',81),('iterations',True)):
                changed=copy.deepcopy(model);changed_report=copy.deepcopy(report)
                changed['training'][key]=value;changed_report['new_dead_training']=changed['training']
                (out/'model.json').write_text(json.dumps(changed));(out/'report.json').write_text(json.dumps(changed_report))
                complete['model_sha256']=refit.storage.file_hash(out/'model.json')
                complete['report_sha256']=refit.storage.file_hash(out/'report.json')
                (out/'complete.json').write_text(json.dumps(complete))
                with self.subTest(key=key,value=value),self.assertRaises(ValueError):refit.verify_result(self.inner)

    def test_refit_context_rejects_changed_or_incomplete_full_roster(self):
        ctx=dict(producer_sources={'source.py':'a'*64})
        manifests=[dict(match_id=mid,identity={'sources':ctx['producer_sources']}) for mid in refit.protocol.TRAIN_IDS]
        with patch.object(refit.inputs,'_evidence',return_value=ctx),patch.object(refit.inputs,'_json',return_value={'cache_manifests':manifests}):
            self.assertEqual(len(refit._refit_context()['pff']),40)
            manifests.pop()
            with self.assertRaises(ValueError):refit._refit_context()

    def test_new_refit_native_loader_opens_former_inner_validation_only_after_auth(self):
        from tests import test_clean_pff_adaptation_inputs as fixtures
        case=fixtures.AdaptationInputsTests();case.setUp();self.addCleanup(case.doCleanups)
        case.mid=refit.protocol.INNER_VALIDATION_IDS[0];case.match=f'pff{case.mid}'
        expected=case.pff_fixture();loaded=[];original=refit.inputs._array
        def load(name,identity):loaded.append(Path(name).name);return original(name,identity)
        with patch.object(refit.inputs,'_array',side_effect=load), \
             patch.object(refit.inputs.prior.pff_adapter,'references',return_value=(expected,{})) as refs:
            record=refit._record(case.match,case.context)
        self.assertEqual(loaded,['x.npy','z.npy']);refs.assert_called_once()
        np.testing.assert_array_equal(record['y'],expected)
        with self.assertRaises(ValueError):refit.inputs._pff_record(case.match,case.context)

    def test_changed_new_refit_feature_bytes_fail_before_reference_access(self):
        from tests import test_clean_pff_adaptation_inputs as fixtures
        case=fixtures.AdaptationInputsTests();case.setUp();self.addCleanup(case.doCleanups)
        case.mid=refit.protocol.INNER_VALIDATION_IDS[0];case.match=f'pff{case.mid}';case.pff_fixture()
        case.write(f'{refit.inputs.PFF_CACHE}/{case.mid}/x.npy',b'changed feature bytes')
        with patch.object(refit.inputs.prior.pff_adapter,'references') as refs,self.assertRaises(ValueError):
            refit._record(case.match,case.context)
        refs.assert_not_called()

    def test_refit_sources_cover_lazy_local_transitive_imports(self):
        (self.root/'tools').mkdir()
        (self.root/'tools/clean_pff_adaptation_refit.py').write_text('def load():\n    from tools import helper\n')
        (self.root/'tools/clean_pff_adaptation_campaign.py').write_text('from tools import helper\n')
        (self.root/'tools/helper.py').write_text('value=1\n')
        with patch.object(refit.stages,'source_hashes',return_value={}):
            sources=refit._sources()
            self.assertEqual(set(sources),{'tools/clean_pff_adaptation_refit.py','tools/clean_pff_adaptation_campaign.py','tools/helper.py'})
            (self.root/'tools/helper.py').write_text('value=2\n')
            self.assertNotEqual(sources['tools/helper.py'],refit._sources()['tools/helper.py'])


if __name__=='__main__':unittest.main()

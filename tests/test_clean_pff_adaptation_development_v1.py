"""Invented artifacts only: fixed-cohort development and owned execution guards."""
import copy
import json
import os
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import Mock, patch

from tools import clean_pff_adaptation_development_v1 as dev
from tools import clean_pff_adaptation_development_operator_v1 as operator


class DevelopmentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.repo = Path(self.temporary.name).resolve()
        self.study = dict(protocol_sha256='1'*64, candidate_sha256='2'*64,
                          source_hashes={'synthetic.py':'3'*64}, runtime_identity={'synthetic':True})
        for module in (dev, operator):
            context = patch.object(module,'ROOT',self.repo)
            context.start(); self.addCleanup(context.stop)
        dev.root().mkdir(parents=True)
        dev.write('protocol.json',self.study)

    def _source_fixture(self):
        parent = dev.policy.APPROVED_ROOT_MODEL
        dataset = 'd'*64
        clean = dev.adaptation.CLEAN_SHA256
        lineage = dict(schema_version=1,candidate_sha256=clean,
            comparator_sha256=dev.original.BASELINE_SHA256,fit_inputs=[parent,dataset],artifacts=[
                dict(sha256=parent,kind='model',scope='external',identifier='Approved external dead head',parents=[]),
                dict(sha256=dataset,kind='dataset',scope='external',identifier='sportec-idsse-28196177',parents=[]),
                dict(sha256=clean,kind='model',scope='external',identifier='Clean external parent',parents=[parent,dataset])])
        context = dict(lineage=lineage,model_paths={parent:'invented-parent.json',clean:'invented-clean.json'},
            cache_manifests=[{'match_id':mid,'invented':True} for mid in dev.adaptation.TRAIN_IDS])
        complete = dict(selected_candidate_id='A',refit_complete=True,model_sha256='f'*64,
                        inner_proof_sha256='a'*64)
        marker = dict(source_hashes={'synthetic-refit.py':'b'*64})
        return context, complete, marker

    def test_connected_lineage_includes_only_external_fit_ancestry(self):
        clean,complete,marker=self._source_fixture()
        graph,bindings=dev.connected_lineage(clean,complete,marker)
        audit=dev.policy.audit_lineage(graph)
        self.assertEqual(audit['external_dataset_ids'],['pff-world-cup-2022','sportec-idsse-28196177'])
        self.assertEqual(graph['candidate_sha256'],complete['model_sha256'])
        self.assertEqual(bindings[complete['model_sha256']],dev.REFIT+'/model.json')
        self.assertNotIn(dev.original.BASELINE_SHA256,audit['model_sha256s'])
        self.assertNotIn(dev.precision.PARENT_SHA256,audit['model_sha256s'])
        self.assertEqual(clean['lineage']['candidate_sha256'],dev.adaptation.CLEAN_SHA256)
        self.assertFalse(audit['provenance_semantically_certified'])

    def test_foreign_refit_and_missing_training_fixture_are_rejected(self):
        clean,complete,marker=self._source_fixture()
        for winner in ('B',None):
            with self.subTest(winner=winner):
                with self.assertRaises(ValueError):dev.connected_lineage(clean,dict(complete,selected_candidate_id=winner),marker)
        clean['cache_manifests'].pop()
        with self.assertRaises(ValueError):dev.connected_lineage(clean,complete,marker)

    def test_final_and_training_ids_rejected_before_any_io(self):
        for mid in (*dev.adaptation.FINAL_IDS,dev.adaptation.TRAIN_IDS[0],True,'3813'):
            with self.subTest(mid=mid), patch.object(dev.old,'observations') as observations, \
                 patch.object(dev,'verify_freeze') as freeze:
                with self.assertRaises(ValueError):dev._predict(mid,{}, {})
                with self.assertRaises(ValueError):dev._score(mid,{}, {})
                observations.assert_not_called(); freeze.assert_not_called()

    def test_unknown_stage_and_direct_kernel_admission_fail_before_authentication(self):
        with patch.object(dev,'authenticate') as authenticate:
            with self.assertRaises(ValueError):dev.run_stage('final',admission={})
            with self.assertRaises(ValueError):dev.run_stage('initialize',admission={})
            authenticate.assert_not_called()
        fake=dict(worker_pid=os.getpid(),runtime_verified=True)
        with patch.object(operator,'verify_admission',side_effect=ValueError('not owned')) as guard, \
             patch.object(dev,'authenticate') as authenticate:
            with self.assertRaises(ValueError):dev.run_stage('initialize',admission=fake)
            guard.assert_called_once(); authenticate.assert_not_called()

    def test_foreign_cohort_and_unexpected_root_artifacts_fail_closed(self):
        forbidden=dev.path('predictions')/str(dev.adaptation.FINAL_IDS[0])
        forbidden.mkdir(parents=True)
        with self.assertRaises(ValueError):dev.check_inventory()
        # No recursive delete: remove only the known empty synthetic directories.
        forbidden.rmdir(); forbidden.parent.rmdir()
        (dev.root()/'final-access.json').write_text('{}')
        with self.assertRaises(ValueError):dev.state()

    def test_score_requires_all_predictions_frozen_before_reference_io(self):
        with patch.object(dev,'verify_freeze',side_effect=ValueError('not frozen')), \
             patch.object(dev.old.prior,'reference') as reference, patch.object(dev,'write') as write:
            with self.assertRaises(ValueError):dev._score(dev.IDS[0],self.study,{})
            reference.assert_not_called(); write.assert_not_called()

    def test_prediction_cannot_regenerate_after_freeze(self):
        dev.write('predictions-before-references.json',{})
        with patch.object(dev.old,'observations') as observations:
            with self.assertRaises(ValueError):dev._predict(dev.IDS[0],self.study,{})
            observations.assert_not_called()

    def test_score_detail_inventory_rejects_extra_artifacts(self):
        mid=dev.IDS[0]
        for name in dev.NAMES:
            folder=dev.path(f'details/{mid}/{name}')
            folder.mkdir(parents=True)
            for filename in ('metrics.json','predictions.csv'):(folder/filename).write_text('synthetic')
        files=dev.stage_files(f'score-{mid}')
        self.assertEqual(len(files),8)
        (dev.path(f'details/{mid}/candidate')/'extra.csv').write_text('synthetic')
        with self.assertRaises(ValueError):dev.stage_files(f'score-{mid}')

    def test_access_marker_binds_exact_cohort_and_no_final_permission(self):
        value=dev.expected_access(self.study,dev.IDS[0],'4'*64)
        self.assertEqual(value['prediction_freeze_sha256'],'4'*64)
        for flag in ('final_accessed','final_access_authorized','promoted','submission_eligible'):
            self.assertIs(value[flag],False)
        self.assertTrue(value['development_previously_consumed'])

    def test_operator_partial_request_is_consumed_without_rerun(self):
        operator.directory().mkdir()
        request=dict(index=0,stage='initialize',nonce='a'*32,parent_pid=123,
                     protocol_sha256=self.study['protocol_sha256'],previous_result_sha256='0'*64)
        operator._write(operator.file(0,'request.json'),request)
        index,previous,blocked=operator.history(self.study)
        self.assertEqual(index,0);self.assertTrue(blocked)
        with patch.object(operator,'_manifest',return_value=(self.study,{})), \
             patch.object(operator.prior_operator,'_execute') as execute:
            with self.assertRaises(ValueError):operator.run()
            execute.assert_not_called()

    def test_operator_failed_launch_is_terminal_and_uses_owned_executor(self):
        operator.directory().mkdir()
        with patch.object(operator,'_manifest',return_value=(self.study,{})), \
             patch.object(operator.prior_operator,'_execute',side_effect=TimeoutError('invented timeout')) as execute:
            with self.assertRaises(TimeoutError):operator.run()
            self.assertEqual(execute.call_args.args[3],1800)
            self.assertIn(operator.MODULE,execute.call_args.args[0])
            self.assertEqual(operator._read(operator.file(0,'result.json'))['status'],'failed')
            with self.assertRaises(ValueError):operator.run()
            self.assertEqual(execute.call_count,1)

    def test_operator_cli_exposes_no_model_or_final_overrides(self):
        with self.assertRaises(SystemExit):operator.main(['plan','--model','invented.json'])
        with self.assertRaises(SystemExit):operator.main(['final'])
        with patch.object(dev,'plan') as plan:
            with self.assertRaises(ValueError):operator.main(['plan','--index','0'])
            plan.assert_not_called()

    def test_stage_completion_hashes_detect_changed_artifact(self):
        dev.write('lineage.json',{'synthetic':1});dev.write('model-bindings.json',{});dev.write('handoff.json',{})
        marker=dict(stage='initialize',protocol_sha256=self.study['protocol_sha256'],
                    candidate_sha256=self.study['candidate_sha256'],source_hashes=self.study['source_hashes'],pid=123)
        dev.write('initialize-started.json',marker)
        proof=dict(stage='initialize',protocol_sha256=self.study['protocol_sha256'],
            artifacts={name:dev.storage.file_hash(dev.path(name)) for name in dev.stage_files('initialize')})
        dev.write('stages/initialize.json',proof)
        self.assertEqual(dev.verify_stage('initialize'),proof)
        dev.path('lineage.json').write_text('{"synthetic":2}')
        with self.assertRaises(ValueError):dev.verify_stage('initialize')

    def test_full_freeze_requires_all_twelve_and_original_control_hashes(self):
        rows=[];controls=[]
        for mid in dev.IDS:
            folder=dev.path(f'predictions/{mid}');folder.mkdir(parents=True)
            for name in dev.NAMES:(folder/f'{name}.npy').write_bytes(f'invented-{mid}-{name}'.encode())
            identities={name:dev.storage.file_hash(folder/f'{name}.npy') for name in dev.NAMES}
            proof=dict(match_id=mid,protocol_sha256=self.study['protocol_sha256'],candidate_sha256=self.study['candidate_sha256'],
                       cache_manifest_sha256=dev.old.CACHE_HASHES[str(mid)],prediction_sha256=identities)
            dev.write(f'predictions/{mid}/complete.json',proof);rows.append(proof)
            controls.append(dict(match_id=mid,baseline_prediction_sha256=identities['baseline'],candidate_prediction_sha256=identities['parent']))
        context={'old_bundle':{'development':{'prediction_freeze':{'records':controls}}}}
        self.assertEqual(dev.prediction_records(self.study,context),rows)
        dev.write('predictions-before-references.json',dev.freeze_value(self.study,rows))
        dev.verify_freeze(self.study,context)
        with self.assertRaises(ValueError):dev.freeze_value(self.study,rows[:-1])
        dev.path(f'predictions/{dev.IDS[0]}/parent.npy').write_bytes(b'changed synthetic bytes')
        with self.assertRaises(ValueError):dev.verify_freeze(self.study,context)

    def _cached_context_fixture(self):
        context=dict(authenticated_tree_files={},input_hashes={})
        for relative in dev.AUTHENTICATED_TREES:
            folder=self.repo/relative;folder.mkdir(parents=True)
            target=folder/'invented.json';target.write_bytes(b'{"synthetic":true}\n')
            name=target.relative_to(self.repo).as_posix()
            context['authenticated_tree_files'][relative]=[name]
            context['input_hashes'][name]=dev.storage.file_hash(target)
        return context

    def test_process_local_context_rechecks_actual_bytes_and_attempt_inventory(self):
        context=self._cached_context_fixture()
        self.assertIs(dev.recheck_context(context),context)
        target=self.repo/next(iter(context['input_hashes']))
        target.write_bytes(b'changed synthetic source')
        with self.assertRaises(ValueError):dev.recheck_context(context)
        target.write_bytes(b'{"synthetic":true}\n')
        (target.parent/'failed.json').write_bytes(b'{}')
        with self.assertRaises(ValueError):dev.recheck_context(context)

    def test_source_and_runtime_changes_rejected_even_with_cached_context(self):
        context=self._cached_context_fixture()
        for field in ('source_hashes','runtime_identity'):
            changed=copy.deepcopy(self.study);changed[field]={'changed':True}
            with self.subTest(field=field),patch.object(dev,'build_protocol',return_value=changed), \
                 patch.object(dev,'context') as full_audit:
                with self.assertRaises(ValueError):dev.authenticate(context)
                full_audit.assert_not_called()

    def test_real_owned_executor_authenticates_new_worker_ancestry(self):
        # Reuse the exact helper used by the operator; no production bypass flag.
        # The child reads only code and invented launch receipts, never datasets.
        actual_repo=Path(dev.__file__).resolve().parents[1]
        launch=self.repo/'launch.json';result=self.repo/'owned-result.json';log=self.repo/'child.log'
        runtime=operator.owned.runtime_identity();request='a'*64
        script=("import json,os,sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);"
                "from tools import owned_python_launch as owned;"
                "value=owned.verify_worker(Path(sys.argv[2]),request_sha256=sys.argv[3],"
                "parent_pid=int(sys.argv[4]),expected_runtime=json.loads(sys.argv[5]));"
                "value['thread_env']={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')};"
                "Path(sys.argv[6]).write_text(json.dumps(value))")
        command=[sys.executable,'-B','-c',script,str(actual_repo),str(launch),request,
                 str(os.getpid()),json.dumps(runtime),str(result)]
        status=operator.prior_operator._execute(command,self.repo,log,20,launch,request)
        self.assertEqual(status,0)
        receipt=json.loads(result.read_bytes())
        self.assertTrue(receipt['runtime_verified'])
        self.assertEqual(receipt['request_sha256'],request)
        self.assertEqual(receipt['parent_pid'],os.getpid())
        self.assertEqual(set(receipt['thread_env'].values()),{'1'})
        self.assertIn(receipt['mode'],('direct','windows-venv'))


if __name__=='__main__':unittest.main()

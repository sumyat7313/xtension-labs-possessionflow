"""Synthetic lifecycle/source tests; no provider records or model fitting."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import clean_pff_adaptation_stages as s


RECEIPT = {'artifact_sha256': {'synthetic_receipt': 'b'*64}, 'execution_verified': False}


class AdaptationStageStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        (self.root/'runs').mkdir()
        self.out = self.root/'runs'/s.OUT_NAME
        self.sources = {'tools/synthetic.py': 'a'*64}
        source_patch = patch.object(s, 'source_hashes', side_effect=lambda root: dict(self.sources))
        source_patch.start(); self.addCleanup(source_patch.stop)
        version_patch = patch.object(s, 'runtime_versions', return_value={'python': 'synthetic'})
        version_patch.start(); self.addCleanup(version_patch.stop)
        s.initialize(self.out, self.root)

    def finish(self, stage):
        with s.attempt(self.out,stage,self.root) as handle:
            handle.complete(copy.deepcopy(RECEIPT))
        return handle

    def status(self):
        return s.status(self.out,self.root)

    def test_exact_stages_and_all_authorization_flags_false(self):
        self.assertEqual(len(s.STAGES),29)
        self.assertEqual(s.STAGES[:3],('initialize','fit-A','fit-B'))
        self.assertEqual(s.STAGES[19],'freeze')
        self.assertEqual(s.STAGES[-1],'select')
        self.assertEqual(self.status()['next_stage'],'initialize')
        for key in s.protocol.FLAGS:self.assertIs(self.status()[key],False)
        self.assertIs(self.status()['resource_limits_enforced'],False)
        self.assertIs(self.status()['provider_records_accessed_by_store'],False)
        self.assertIs(self.status()['caller_access_not_attested'],True)
        self.assertNotIn('final_accessed',self.status())
        self.assertEqual(s.LIMITS['stage_timeout_seconds'],1800)
        self.assertEqual(s.LIMITS['blas_threads_per_worker'],1)
        self.assertEqual(s.LIMITS['torch_threads_per_worker'],1)
        self.assertEqual(s.LIMITS['max_heavy_fits'],1)

    def test_complete_whole_synthetic_history_does_not_authorize_real_execution(self):
        for stage in s.STAGES:self.finish(stage)
        value=self.status()
        self.assertTrue(value['complete']);self.assertFalse(value['blocked'])
        self.assertEqual(value['recorded_completed_stages'],29)
        self.assertIsNone(value['next_stage'])
        for key in s.protocol.FLAGS:self.assertIs(value[key],False)
        with self.assertRaises(ValueError):self.finish('initialize')

    def test_completed_stage_resumes_only_next_stage(self):
        self.finish('initialize')
        self.assertEqual(self.status()['next_stage'],'fit-A')
        with self.assertRaises(ValueError):self.finish('initialize')
        with self.assertRaises(ValueError):self.finish('fit-B')
        self.finish('fit-A')
        self.assertEqual(self.status()['next_stage'],'fit-B')

    def test_unknown_or_final_stage_rejected_before_filesystem_access(self):
        other=self.root/'missing'
        for stage in ('refit','score-3814','final','predict-A-3813'):
            with self.assertRaises(ValueError):
                with s.attempt(other,stage,self.root):pass
        self.assertFalse(other.exists())

    def test_existing_study_cannot_be_reset(self):
        for _ in range(2):
            with self.assertRaises(ValueError):s.initialize(self.out,self.root)
        self.assertEqual(self.status()['next_stage'],'initialize')

    def test_exception_consumes_attempt_and_closed_handle_cannot_complete(self):
        with self.assertRaisesRegex(RuntimeError,'synthetic'):
            with s.attempt(self.out,'initialize',self.root) as handle:
                raise RuntimeError('synthetic')
        self.assertTrue(self.status()['blocked'])
        with self.assertRaises(ValueError):self.finish('initialize')
        with self.assertRaises(ValueError):handle.complete(RECEIPT)
        ledger=[json.loads(line) for line in (self.out/'ledger.jsonl').read_text().splitlines()]
        self.assertEqual(ledger[-1]['detail'],{'exception_type':'RuntimeError'})

    def test_keyboard_interrupt_also_consumes_attempt(self):
        with self.assertRaises(KeyboardInterrupt):
            with s.attempt(self.out,'initialize',self.root):raise KeyboardInterrupt()
        self.assertTrue(self.status()['blocked'])

    def test_missing_or_duplicate_completion_consumes_attempt(self):
        with self.assertRaises(ValueError):
            with s.attempt(self.out,'initialize',self.root):pass
        self.assertTrue(self.status()['blocked'])

    def test_repeated_completion_is_invalid_and_consumes_attempt(self):
        with self.assertRaises(ValueError):
            with s.attempt(self.out,'initialize',self.root) as handle:
                handle.complete(RECEIPT);handle.complete(RECEIPT)
        self.assertTrue(self.status()['blocked'])

    def test_successful_handle_is_closed_and_receipt_is_copied(self):
        receipt=copy.deepcopy(RECEIPT)
        with s.attempt(self.out,'initialize',self.root) as handle:
            handle.complete(receipt)
            receipt['artifact_sha256']['synthetic_receipt']='c'*64
        with self.assertRaises(ValueError):handle.complete(RECEIPT)
        last=json.loads((self.out/'ledger.jsonl').read_text().splitlines()[-1])
        self.assertEqual(last['detail'],RECEIPT)

    def test_lock_is_held_for_entire_attempt_body(self):
        with s.attempt(self.out,'initialize',self.root) as handle:
            with self.assertRaises(RuntimeError):self.status()
            with self.assertRaises(RuntimeError):
                with s.attempt(self.out,'initialize',self.root):pass
            handle.complete(RECEIPT)
        self.assertEqual(self.status()['next_stage'],'fit-A')

    def test_marker_created_before_start_is_permanently_consumed(self):
        manifest=json.loads((self.out/'manifest.json').read_text())
        s._exclusive(self.out/'attempts/000.json',s._marker_value(0,manifest,s.ZERO))
        self.assertTrue(self.status()['blocked'])
        with self.assertRaises(ValueError):self.finish('initialize')

    def test_started_event_without_terminal_is_permanently_consumed(self):
        manifest=json.loads((self.out/'manifest.json').read_text())
        s._exclusive(self.out/'attempts/000.json',s._marker_value(0,manifest,s.ZERO))
        s._append(self.out,[],b'',0,'started',{},manifest['descriptor_sha256'])
        self.assertTrue(self.status()['blocked'])
        with self.assertRaises(ValueError):self.finish('initialize')

    def test_source_drift_before_begin_prevents_attempt_marker(self):
        self.sources['tools/synthetic.py']='c'*64
        with self.assertRaises(ValueError):self.finish('initialize')
        self.assertEqual(list((self.out/'attempts').iterdir()),[])

    def test_source_drift_during_work_records_failure_not_completion(self):
        with self.assertRaises(ValueError):
            with s.attempt(self.out,'initialize',self.root) as handle:
                handle.complete(RECEIPT)
                self.sources['tools/synthetic.py']='c'*64
        self.sources['tools/synthetic.py']='a'*64
        self.assertTrue(self.status()['blocked'])
        last=json.loads((self.out/'ledger.jsonl').read_text().splitlines()[-1])
        self.assertEqual(last['state'],'failed')

    def test_manifest_tamper_failure_binds_original_descriptor(self):
        original=(self.out/'manifest.json').read_bytes()
        before=json.loads(original)
        with self.assertRaises(ValueError):
            with s.attempt(self.out,'initialize',self.root) as handle:
                handle.complete(RECEIPT)
                changed=copy.deepcopy(before);changed['descriptor_sha256']='c'*64
                (self.out/'manifest.json').write_text(json.dumps(changed))
        last=json.loads((self.out/'ledger.jsonl').read_text().splitlines()[-1])
        self.assertEqual(last['descriptor_sha256'],before['descriptor_sha256'])
        (self.out/'manifest.json').write_bytes(original)
        self.assertTrue(self.status()['blocked'])

    def test_ledger_tamper_and_reorder_rejected(self):
        self.finish('initialize')
        path=self.out/'ledger.jsonl';original=path.read_bytes()
        events=[json.loads(line) for line in original.splitlines()]
        events[0]['stage']='fit-A'
        path.write_text('\n'.join(json.dumps(e) for e in events)+'\n')
        with self.assertRaises(ValueError):self.status()
        path.write_bytes(b'\n'.join(reversed(original.splitlines()))+b'\n')
        with self.assertRaises(ValueError):self.status()

    def test_partial_ledger_and_deleted_marker_rejected(self):
        self.finish('initialize')
        path=self.out/'ledger.jsonl';original=path.read_bytes()
        path.write_bytes(original[:-1])
        with self.assertRaises(ValueError):self.status()
        path.write_bytes(original)
        (self.out/'attempts/000.json').unlink()
        with self.assertRaises((ValueError,FileNotFoundError)):self.status()

    def test_deleted_completion_does_not_allow_retry(self):
        self.finish('initialize')
        path=self.out/'ledger.jsonl'
        path.write_bytes(path.read_bytes().splitlines()[0]+b'\n')
        self.assertTrue(self.status()['blocked'])
        with self.assertRaises(ValueError):self.finish('initialize')

    def test_receipt_cannot_claim_execution_or_include_raw_content(self):
        invalid=[dict(RECEIPT,execution_verified=True),dict(RECEIPT,frame_records=[]),
                 {'artifact_sha256':{},'execution_verified':False},
                 {'artifact_sha256':{'../weights':'b'*64},'execution_verified':False},
                 {'artifact_sha256':{'model':'rawdata'},'execution_verified':False}]
        for receipt in invalid:
            with self.assertRaises(ValueError):s._receipt(receipt)

    def test_path_cannot_target_historical_or_nested_study(self):
        for out in (self.root/'runs/continuous-improvement',self.out/'nested',self.root/'models'):
            with self.assertRaises(ValueError):s.location(out,self.root)


class SourceManifestTests(unittest.TestCase):
    def test_actual_preserved_scientific_snapshot_matches(self):
        value=s.verify_snapshot()
        self.assertEqual(value['protocol_sha256'],s.SNAPSHOT_PROTOCOL_SHA256)
        self.assertEqual(len(value['source_hashes']),24)

    def test_static_closure_includes_lazy_relative_imports_and_both_bridges(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();(root/'tools').mkdir();(root/'models').mkdir()
            for name in s.SOURCE_ROOTS:(root/name).write_text('')
            (root/'tools/__init__.py').write_text('')
            (root/'tools/clean_pff_adaptation_inputs.py').write_text('def lazy():\n    from . import synthetic_dependency\n')
            (root/'tools/synthetic_dependency.py').write_text('VALUE=1\n')
            for name in ('requirements-solution.txt','requirements-training.txt',s.SNAPSHOT):(root/name).write_text('{}')
            with patch.object(s,'verify_snapshot',return_value={'source_hashes':{}}):
                hashes=s.source_hashes(root)
            self.assertIn('tools/synthetic_dependency.py',hashes)
            self.assertIn('tools/__init__.py',hashes)
            for name in s.SOURCE_ROOTS:self.assertIn(name,hashes)
            (root/'tools/clean_pff_adaptation_inputs.py').unlink()
            with patch.object(s,'verify_snapshot',return_value={'source_hashes':{}}),self.assertRaises(ValueError):
                s.source_hashes(root)


if __name__=='__main__':unittest.main()

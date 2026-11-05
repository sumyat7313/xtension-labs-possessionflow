"""Synthetic migration evidence and consumed-launch invariants; no provider data."""
from contextlib import contextmanager, ExitStack
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools import clean_pff_adaptation_operator_v2 as op


def write_json(path, value):
    path.write_bytes(op.storage.canonical(value)+b'\n')


def build_legacy(root):
    """Build an entirely synthetic, failed-before-claim v1 and empty inner."""
    root=Path(root).resolve();(root/'runs').mkdir(exist_ok=True)
    (root/'synthetic.py').write_text('# synthetic source\n',encoding='utf-8')
    descriptor={'schema_version':1,'synthetic':True}
    descriptor['descriptor_sha256']=op.storage.digest(descriptor)
    inner=root/op.INNER;inner.mkdir();(inner/'attempts').mkdir()
    (inner/'.lock').write_bytes(b'0');(inner/'ledger.jsonl').write_bytes(b'')
    write_json(inner/'manifest.json',descriptor)
    old=root/op.PRIOR;old.mkdir();(old/'attempts').mkdir();(old/'.lock').write_bytes(b'0')
    manifest=dict(schema_version=1,family='clean-pff-adaptation-operator-v1',
                  source_hashes={'synthetic.py':op.storage.file_hash(root/'synthetic.py')},
                  inner_descriptor=descriptor,stages=list(op.ORDER),limits=dict(op.stages.LIMITS),
                  python_executable=str(Path(sys.executable).resolve()))
    manifest['manifest_sha256']=op.storage.digest(manifest)
    write_json(old/'manifest.json',manifest)
    write_json(old/'initialized.json',{'manifest_sha256':manifest['manifest_sha256']})
    request=dict(index=0,stage='initialize',nonce='f'*32,parent_pid=os.getpid(),
                 manifest_sha256=manifest['manifest_sha256'],previous_result_sha256='0'*64)
    write_json(old/'attempts/000.request.json',request)
    write_json(old/'attempts/000.result.json',dict(request_sha256=op.storage.digest(request),
               manifest_sha256=manifest['manifest_sha256'],status='failed',detail={'exception_type':'ValueError'}))
    (old/'attempts/000.console.log').write_bytes(b'synthetic pre-stage parent mismatch\n')
    return {name:op.storage.file_hash(old/name) for name in op.PRIOR_FILE_HASHES},descriptor


@contextmanager
def fixture(root,pins,descriptor):
    """Test-only identity patches; all actual locks/files/lifecycle stay active."""
    with ExitStack() as stack:
        for replacement in (patch.object(op,'ROOT',root),
            patch.object(op,'PRIOR_FILE_HASHES',pins),
            patch.object(op,'INNER_MANIFEST_SHA256',op.storage.file_hash(root/op.INNER/'manifest.json')),
            patch.object(op,'sources',side_effect=lambda:{'synthetic.py':op.storage.file_hash(root/'synthetic.py')}),
            patch.object(op.stages,'descriptor',side_effect=lambda root:copy.deepcopy(descriptor))):
            stack.enter_context(replacement)
        yield


class RepairTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.pins,self.descriptor=build_legacy(self.root)
        self.context=fixture(self.root,self.pins,self.descriptor);self.context.__enter__()
        self.addCleanup(self.context.__exit__,None,None,None)
        self.out=self.root/op.OUT;self.inner=self.root/op.INNER

    def test_initialize_adopts_same_empty_inner_without_reset(self):
        original={name:op.storage.file_hash(self.root/op.PRIOR/name) for name in self.pins}
        inner=(self.inner/'manifest.json').read_bytes()
        op.initialize();status=op.status()
        self.assertEqual(status['next_stage'],'initialize');self.assertEqual(status['total_operator_attempts'],1)
        self.assertEqual(status['candidate_budget'],2);self.assertFalse(status['promoted'])
        self.assertEqual((self.inner/'manifest.json').read_bytes(),inner)
        self.assertEqual(original,{name:op.storage.file_hash(self.root/op.PRIOR/name) for name in self.pins})
        with self.assertRaisesRegex(ValueError,'consumed'):op.initialize()

    def test_all_three_locks_held_until_adoption_is_written(self):
        original=op._write
        def checked(path,value):
            for root in (self.root/op.PRIOR,self.out,self.inner):
                with self.assertRaisesRegex(RuntimeError,'holds the lock'):
                    with op.storage.locked(root):pass
            return original(path,value)
        with patch.object(op,'_write',side_effect=checked):op.initialize()

    def test_prior_byte_mutation_rejected(self):
        for name in self.pins:
            path=self.root/op.PRIOR/name;original=path.read_bytes();path.write_bytes(original+b' ')
            with self.subTest(name=name),self.assertRaises(ValueError):op.plan()
            path.write_bytes(original)

    def test_prior_claim_or_additional_attempt_rejected(self):
        for name in ('000.claim.json','000.worker.json','001.request.json'):
            path=self.root/op.PRIOR/'attempts'/name;path.write_bytes(b'{}')
            with self.subTest(name=name),self.assertRaises(ValueError):op.plan()
            path.unlink()

    def test_nonempty_inner_ledger_rejected_before_adoption(self):
        (self.inner/'ledger.jsonl').write_bytes(b'\n')
        with self.assertRaises(ValueError):op.initialize()
        self.assertFalse((self.out/'adoption.json').exists())

    def test_inner_attempt_marker_rejected_before_adoption(self):
        (self.inner/'attempts/000.json').write_bytes(b'{}')
        with self.assertRaises(ValueError):op.initialize()
        self.assertFalse((self.out/'manifest.json').exists())

    def test_even_empty_artifacts_directory_rejected(self):
        (self.inner/'artifacts').mkdir()
        with self.assertRaises(ValueError):op.initialize()

    def test_inner_manifest_or_frozen_source_drift_rejected(self):
        source=self.root/'synthetic.py';source.write_text('changed')
        with self.assertRaises(ValueError):op.plan()

    def test_modified_inner_descriptor_rejected(self):
        self.descriptor['changed']=True
        with self.assertRaises(ValueError):op.plan()

    def test_inner_manifest_bytes_changed_rejected(self):
        path=self.inner/'manifest.json';path.write_bytes(path.read_bytes()+b' ')
        with self.assertRaises(ValueError):op.plan()

    def test_prior_evidence_still_required_after_adoption(self):
        op.initialize();(self.root/op.PRIOR/'attempts/000.console.log').write_bytes(b'changed')
        with self.assertRaises(ValueError):op.status()

    def test_failed_repair_launch_is_consumed_and_chains_old_failure(self):
        op.initialize();manifest=op._read(self.out/'manifest.json')
        with patch.object(op,'_execute',side_effect=OSError('synthetic launch failure')):
            with self.assertRaises(OSError):op.run()
        request=op._read(op._file(self.out,0,'request.json'))
        self.assertEqual(request['previous_result_sha256'],manifest['predecessor']['failed_result_digest'])
        self.assertTrue(op.status()['blocked']);self.assertEqual(op.status()['total_operator_attempts'],2)
        with patch.object(op,'_execute') as execute,self.assertRaises(ValueError):op.run()
        execute.assert_not_called()

    def test_nonzero_or_zero_without_receipt_consumes_launch(self):
        op.initialize()
        with patch.object(op,'_execute',return_value=3),patch.object(op,'_verify_artifacts') as verify:
            with self.assertRaises(ValueError):op.run()
        verify.assert_not_called();self.assertTrue(op.status()['blocked'])

    def test_zero_exit_without_claim_cannot_complete(self):
        op.initialize()
        def fake(command,root,log,seconds,launch_path,request_sha256):
            write_json(launch_path,{'request_sha256':request_sha256,'parent_pid':os.getpid()})
            write_json(op._file(self.out,0,'worker.json'),dict(request_sha256=request_sha256,artifacts={},
                       launch_receipt_sha256=op.storage.digest(op._read(launch_path))))
            return 0
        with patch.object(op,'_execute',side_effect=fake),patch.object(op,'_verify_artifacts',return_value={}):
            with self.assertRaises(FileNotFoundError):op.run()
        self.assertTrue(op.status()['blocked'])

    def test_publication_failure_closes_job_and_reaps_owned_handle(self):
        from unittest.mock import Mock
        process=Mock(pid=123);process.poll.return_value=None
        job=Mock()
        with patch.object(op.subprocess,'Popen',return_value=process), \
             patch.object(op.owned,'contain_and_resume',return_value=job), \
             patch.object(op.owned,'publish_receipt',side_effect=ValueError('synthetic receipt rejection')), \
             patch.object(op.os,'killpg',create=True):
            with self.assertRaisesRegex(ValueError,'receipt rejection'):
                op._execute([sys.executable],self.root,self.root/'console.log',1800,self.root/'launch.json','a'*64)
        job.close.assert_called();process.kill.assert_called_once();process.wait.assert_called_once_with(timeout=15)

    def test_worker_rejects_invalid_launch_before_io(self):
        with patch.object(op,'_locations') as locations:
            for index,nonce in ((True,'a'*32),(-1,'a'*32),(0,'invalid')):
                with self.assertRaises(ValueError):op.worker(index,nonce)
        locations.assert_not_called()

    def test_worker_rejects_unbounded_threads_before_io(self):
        with patch.dict(os.environ,{'OPENBLAS_NUM_THREADS':'3'}),patch.object(op,'_locations') as locations:
            with self.assertRaises(ValueError):op.worker(0,'a'*32)
        locations.assert_not_called()

    def test_worker_requires_both_parent_locks(self):
        op.initialize()
        env={name:'1' for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}
        with patch.dict(os.environ,env),self.assertRaisesRegex(ValueError,'both parent'):
            op.worker(0,'a'*32)

    def test_actual_progress_does_not_require_empty_ledger_forever(self):
        from tools import clean_pff_adaptation_campaign as campaign
        op.initialize()
        with patch.object(campaign,'ROOT',self.root),patch.object(campaign,'_context',return_value={}), \
             patch.object(campaign,'_plan',return_value={'synthetic':True}):
            campaign.execute_stage('initialize',self.inner)
        self.assertTrue((self.inner/'ledger.jsonl').stat().st_size)
        # Prior proof remains valid, but unowned direct progress cannot be adopted.
        op._manifest(self.out)
        with self.assertRaisesRegex(ValueError,'histories disagree'):op.status()


if __name__=='__main__':unittest.main()

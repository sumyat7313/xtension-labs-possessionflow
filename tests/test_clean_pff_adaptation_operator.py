"""Synthetic launch consumption and real lightweight subprocess isolation tests."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from tools import clean_pff_adaptation_operator as op


class OperatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        (self.root/'runs').mkdir()
        self.completed = 0
        self.manifest = {'manifest_sha256':'a'*64,'synthetic':True}
        for replacement in (patch.object(op,'ROOT',self.root),
                            patch.object(op.stages,'_no_links',side_effect=lambda p:Path(p).absolute()),
                            patch.object(op.stages.os,'fsync'),
                            patch.object(op,'plan',side_effect=lambda:copy.deepcopy(self.manifest)),
                            patch.object(op.stages,'initialize',side_effect=self.init_inner),
                            patch.object(op.stages,'status',side_effect=self.inner_status)):
            replacement.start();self.addCleanup(replacement.stop)
        op.initialize()
        self.out,self.inner=op._locations()

    def init_inner(self,path,root):
        path.mkdir()

    def inner_status(self,*args):
        return dict(blocked=False,recorded_completed_stages=self.completed,
                    next_stage=op.ORDER[self.completed] if self.completed<len(op.stages.STAGES) else None,
                    complete=self.completed==len(op.stages.STAGES))

    def write_worker(self,command,root,log,seconds):
        self.assertEqual(command[:4],[sys.executable,'-m',op.MODULE,'worker'])
        self.assertEqual(seconds,1800)
        self.assertEqual(root,self.root)
        with Path(log).open('xb') as f:f.write(b'synthetic worker')
        index=int(command[5]);request=op._read(op._file(self.out,index,'request.json'))
        self.assertEqual(request['nonce'],command[7])
        op._write(op._file(self.out,index,'claim.json'),
                  dict(request_sha256=op.storage.digest(request),worker_pid=123))
        op._write(op._file(self.out,index,'worker.json'),
                  dict(request_sha256=op.storage.digest(request),artifacts={'verified':'b'*64}))
        self.completed+=1
        return 0

    def success(self):
        with patch.object(op,'_execute',side_effect=self.write_worker), \
             patch.object(op,'_verify_artifacts',return_value={'verified':'b'*64}):
            return op.run()

    def test_one_launch_per_advance_and_actual_receipt_verified(self):
        result=self.success()
        self.assertEqual(result['stage_completed'],'initialize')
        self.assertFalse(result['promoted'])
        self.assertEqual(op.status()['next_stage'],'fit-A')
        self.assertEqual(len(list((self.out/'attempts').glob('*.request.json'))),1)

    def test_zero_exit_without_artifact_receipt_is_consumed(self):
        with patch.object(op,'_execute',return_value=0), \
             patch.object(op,'_verify_artifacts',return_value={}):
            with self.assertRaises(FileNotFoundError):op.run()
        self.assertTrue(op.status()['blocked'])
        with patch.object(op,'_execute') as execute,self.assertRaises(ValueError):op.run()
        execute.assert_not_called()

    def test_nonzero_exit_consumes_launch_without_artifact_verification(self):
        with patch.object(op,'_execute',return_value=3),patch.object(op,'_verify_artifacts') as verify:
            with self.assertRaises(ValueError):op.run()
        verify.assert_not_called()
        self.assertTrue(op.status()['blocked'])

    def test_timeout_and_interrupt_are_durably_failed(self):
        for exception in (subprocess.TimeoutExpired(['fixed'],1800),KeyboardInterrupt()):
            with self.subTest(error=type(exception).__name__):
                # A separate fixture is necessary: attempts are never reset.
                with tempfile.TemporaryDirectory() as tmp:
                    root=Path(tmp).resolve();(root/'runs').mkdir()
                    with patch.object(op,'ROOT',root):
                        op.initialize();out,_=op._locations()
                        with patch.object(op,'_execute',side_effect=exception):
                            with self.assertRaises(type(exception)):op.run()
                        result=op._read(op._file(out,0,'result.json'))
                        self.assertEqual(result['status'],'failed')
                        self.assertEqual(result['detail']['exception_type'],type(exception).__name__)

    def test_sources_changed_after_worker_prevent_success(self):
        def changed(*args):
            result=self.write_worker(*args);self.manifest['drift']=True;return result
        with patch.object(op,'_execute',side_effect=changed):
            with self.assertRaisesRegex(ValueError,'manifest changed'):op.run()
        result=op._read(op._file(self.out,0,'result.json'))
        self.assertEqual(result['status'],'failed')
        self.assertEqual(result['manifest_sha256'],'a'*64)

    def test_artifact_mismatch_cannot_complete(self):
        with patch.object(op,'_execute',side_effect=self.write_worker), \
             patch.object(op,'_verify_artifacts',return_value={'verified':'c'*64}):
            with self.assertRaisesRegex(ValueError,'verification differs'):op.run()
        self.assertTrue(op.status()['blocked'])

    def test_parent_lock_is_held_during_subprocess(self):
        def locked(*args):
            with self.assertRaisesRegex(RuntimeError,'holds the lock'):
                with op.storage.locked(self.out):pass
            return self.write_worker(*args)
        with patch.object(op,'_execute',side_effect=locked), \
             patch.object(op,'_verify_artifacts',return_value={'verified':'b'*64}):op.run()

    def test_initialization_cannot_reset_existing_history(self):
        with self.assertRaises(ValueError):op.initialize()

    def test_launch_reserved_before_process_creation_failure(self):
        with patch.object(op,'_execute',side_effect=OSError('synthetic launch failure')):
            with self.assertRaises(OSError):op.run()
        self.assertTrue(op._file(self.out,0,'request.json').exists())
        self.assertTrue(op.status()['blocked'])

    def test_marker_only_crash_cannot_retry(self):
        request=dict(index=0,stage='initialize',nonce='f'*32,parent_pid=os.getpid(),
                     manifest_sha256='a'*64,previous_result_sha256='0'*64)
        op._write(op._file(self.out,0,'request.json'),request)
        self.assertTrue(op.status()['blocked'])
        with patch.object(op,'_execute') as execute,self.assertRaises(ValueError):op.run()
        execute.assert_not_called()

    def test_history_tamper_and_gap_are_rejected(self):
        self.success()
        request=op._read(op._file(self.out,0,'request.json'));request['stage']='fit-B'
        op._file(self.out,0,'request.json').write_text(json.dumps(request))
        with self.assertRaises(ValueError):op.status()

    def test_worker_rejects_invalid_launch_before_filesystem(self):
        with patch.object(op,'_locations') as locations:
            for index,nonce in ((True,'a'*32),(-1,'a'*32),(0,'not-a-nonce')):
                with self.assertRaises(ValueError):op.worker(index,nonce)
        locations.assert_not_called()

    def test_worker_rejects_unbounded_environment_before_io(self):
        with patch.dict(os.environ,{'OPENBLAS_NUM_THREADS':'4'}),patch.object(op,'_locations') as locations:
            with self.assertRaises(ValueError):op.worker(0,'a'*32)
        locations.assert_not_called()

    def test_worker_requires_parent_lock(self):
        env={name:'1' for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}
        with patch.dict(os.environ,env),self.assertRaisesRegex(ValueError,'parent operator lock'):
            op.worker(0,'a'*32)

    def test_worker_handshake_and_single_use_claim(self):
        from tools import clean_pff_adaptation_campaign as campaign
        def dispatched(stage,out):
            self.assertEqual(stage,'initialize');self.completed+=1
        def launch(command,root,log,seconds):
            with patch.object(op.os,'getppid',return_value=os.getpid()):
                op.worker(int(command[5]),command[7])
            return 0
        env={name:'1' for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}
        with patch.dict(os.environ,env),patch.object(op,'_execute',side_effect=launch), \
             patch.object(campaign,'execute_stage',side_effect=dispatched), \
             patch.object(op,'_verify_artifacts',return_value={'verified':'b'*64}):
            op.run()
        claim=op._read(op._file(self.out,0,'claim.json'))
        self.assertEqual(claim['worker_pid'],os.getpid())
        with patch.dict(os.environ,env),op.storage.locked(self.out),self.assertRaises(ValueError):
            op.worker(0,op._read(op._file(self.out,0,'request.json'))['nonce'])

    def test_no_qualified_winner_stops_without_refit_launch(self):
        for _ in op.stages.STAGES:self.success()
        with patch.object(op,'_selection',return_value={'selection':{'selected_candidate_id':None}}), \
             patch.object(op,'_execute') as execute:
            self.assertTrue(op.run()['complete'])
        execute.assert_not_called()

    def test_claim_deletion_invalidates_completed_launch(self):
        self.success();op._file(self.out,0,'claim.json').unlink()
        with self.assertRaises(FileNotFoundError):op.status()


class SubprocessBoundsTests(unittest.TestCase):
    def test_spawned_child_gets_single_thread_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            log=Path(tmp)/'console.log'
            code="import os; print(','.join(os.environ[k] for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')))"
            result=op._execute([sys.executable,'-c',code],Path(tmp),log,10)
            self.assertEqual(result,0)
            self.assertEqual(log.read_text().strip(),'1,1,1')

    def test_synthetic_sleeping_process_is_killed_on_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            log=Path(tmp)/'console.log'
            command=[sys.executable,'-c','import time; time.sleep(60)']
            real_popen=subprocess.Popen
            children=[]
            def capture(*args,**kwargs):
                child=real_popen(*args,**kwargs)
                if args and args[0]==command:children.append(child)
                return child
            with patch.object(op.subprocess,'Popen',side_effect=capture), \
                 self.assertRaises(subprocess.TimeoutExpired):
                op._execute(command,Path(tmp),log,.1)
            self.assertEqual(len(children),1)
            self.assertIsNotNone(children[0].poll(),'Owned worker was not terminated and reaped')
            # Windows can release inherited log handles shortly after process
            # termination. Require release within five seconds, rather than
            # letting immediate TemporaryDirectory cleanup race that release.
            # A surviving sleeper keeps its handle and must still fail here;
            # its natural exit is well beyond this bounded grace period.
            deadline=time.monotonic()+5
            while True:
                try:
                    log.unlink()
                    break
                except PermissionError as error:
                    if os.name!='nt' or getattr(error,'winerror',None)!=32:raise
                    if time.monotonic()>=deadline:
                        self.fail('Timed-out worker log remained held for more than five seconds')
                    time.sleep(.02)


if __name__=='__main__':unittest.main()

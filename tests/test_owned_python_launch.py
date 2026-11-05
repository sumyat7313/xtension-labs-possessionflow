"""Synthetic ownership tests; never import or access experiment data or models."""
from contextlib import ExitStack, contextmanager
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from tools import owned_python_launch as owned


REQUEST = 'a'*64
RUNTIME = dict(executable='venv-python', base_executable='base-python',
               executable_sha256='b'*64, base_executable_sha256='c'*64,
               prefix='venv', base_prefix='base', python_version='synthetic')


class OwnedLaunchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name).resolve()/'receipt.json'
        self.processes = {
            10: dict(pid=10, parent_pid=1, image='base-python', creation_token=100),
            20: dict(pid=20, parent_pid=10, image='venv-python', creation_token=200),
            30: dict(pid=30, parent_pid=20, image='base-python', creation_token=300),
        }

    def receipt(self, *, windows=True):
        return dict(schema_version=1, platform='windows' if windows else 'posix',
                    receipt_path=owned._normal(self.path), request_sha256=REQUEST,
                    parent_pid=10, owned_pid=20, runtime=copy.deepcopy(RUNTIME),
                    parent_identity=copy.deepcopy(self.processes[10]) if windows else None,
                    owned_identity=copy.deepcopy(self.processes[20]) if windows else None)

    def write(self, receipt):
        self.path.write_text(json.dumps(receipt)+'\n', encoding='utf-8')

    @contextmanager
    def fake_os(self, *, windows=True, pid=30, ppid=20):
        with ExitStack() as stack:
            stack.enter_context(patch.object(owned, 'IS_WINDOWS', windows))
            stack.enter_context(patch.object(owned, 'runtime_identity', return_value=copy.deepcopy(RUNTIME)))
            stack.enter_context(patch.object(owned.os, 'getpid', return_value=pid))
            stack.enter_context(patch.object(owned.os, 'getppid', return_value=ppid))
            stack.enter_context(patch.object(owned, '_windows_process', side_effect=lambda p: copy.deepcopy(self.processes[p])))
            stack.enter_context(patch.object(owned.os, 'kill'))
            yield

    def verify(self, **kwargs):
        return owned.verify_worker(self.path, request_sha256=REQUEST, parent_pid=10,
                                   expected_runtime=RUNTIME, **kwargs)

    def test_exact_windows_venv_hop(self):
        self.write(self.receipt())
        with self.fake_os():
            result = self.verify()
        self.assertEqual(result['mode'], 'windows-venv')
        self.assertTrue(result['runtime_verified'])

    def test_direct_windows_base_interpreter(self):
        self.processes[20]['image'] = 'base-python'
        self.write(self.receipt())
        with self.fake_os(pid=20, ppid=10):
            self.assertEqual(self.verify()['mode'], 'direct')

    def test_posix_direct_only(self):
        self.write(self.receipt(windows=False))
        with self.fake_os(windows=False, pid=20, ppid=10):
            self.assertEqual(self.verify()['mode'], 'direct')
            owned.os.kill.assert_called_once_with(10, 0)
        with self.fake_os(windows=False):
            with self.assertRaisesRegex(ValueError, 'direct owned child'):
                self.verify()

    def test_rejects_foreign_parent_or_image(self):
        for pid, field, value in [(20,'parent_pid',99),(20,'image','foreign'),
                                  (10,'image','foreign'),(30,'image','foreign')]:
            with self.subTest(pid=pid, field=field):
                before = copy.deepcopy(self.processes)
                self.write(self.receipt())
                self.processes[pid][field] = value
                with self.fake_os(), self.assertRaises(ValueError):
                    self.verify()
                self.processes = before

    def test_rejects_arbitrary_additional_hop(self):
        self.write(self.receipt())
        self.processes[30]['parent_pid'] = 40
        with self.fake_os(ppid=40), self.assertRaisesRegex(ValueError, 'exact owned Windows'):
            self.verify()

    def test_rejects_reused_parent_or_launcher_pid(self):
        for pid in (10,20):
            self.write(self.receipt())
            self.processes[pid]['creation_token'] += 1
            with self.fake_os(), self.assertRaisesRegex(ValueError, 'creation identity changed'):
                self.verify()

    def test_rejects_worker_predating_launcher(self):
        self.write(self.receipt())
        self.processes[30]['creation_token'] = 199
        with self.fake_os(), self.assertRaisesRegex(ValueError, 'exact owned Windows'):
            self.verify()

    def test_rechecks_ancestor_liveness(self):
        self.write(self.receipt())
        calls = []
        def process(pid):
            calls.append(pid)
            if len(calls) == 4:
                raise ValueError('Ancestor exited')
            return copy.deepcopy(self.processes[pid])
        with self.fake_os(), patch.object(owned, '_windows_process', side_effect=process):
            with self.assertRaisesRegex(ValueError, 'Ancestor exited'):
                self.verify()
        self.assertEqual(calls, [10,20,30,10])

    def test_wrong_request_parent_runtime_and_extra_schema_rejected(self):
        mutations = [('request_sha256','d'*64),('parent_pid',11),('platform','posix'),
                     ('runtime',{}),('extra',1),('schema_version',True),('owned_pid',True)]
        for field,value in mutations:
            with self.subTest(field=field):
                receipt = self.receipt(); receipt[field] = value; self.write(receipt)
                with self.fake_os(), self.assertRaises(ValueError):
                    self.verify()

    def test_expected_runtime_differs_before_os_queries(self):
        with self.fake_os(), patch.object(owned, 'runtime_identity', return_value={}), self.assertRaisesRegex(ValueError,'frozen runtime'):
            self.verify()

    def test_runtime_bytes_changed_during_verification(self):
        self.write(self.receipt())
        with self.fake_os(), patch.object(owned,'runtime_identity',side_effect=[RUNTIME,{}]):
            with self.assertRaisesRegex(ValueError,'bytes or environment changed'):
                self.verify()

    def test_copied_receipt_rejected(self):
        receipt = self.receipt();receipt['receipt_path'] = str(self.path.with_name('elsewhere.json'))
        self.write(receipt)
        with self.fake_os(), self.assertRaisesRegex(ValueError,'moved or replayed'):
            self.verify()

    def test_incomplete_missing_and_malformed_receipts(self):
        with self.assertRaises(TimeoutError): owned._wait_receipt(self.path,.03)
        self.path.write_bytes(b'{')
        with self.assertRaises(TimeoutError): owned._wait_receipt(self.path,.03)
        self.path.write_bytes(b'{\n')
        with self.assertRaises(json.JSONDecodeError): owned._wait_receipt(self.path,.03)
        for bound in (0,-1,11,float('nan'),float('inf'),True):
            with self.subTest(bound=bound), self.assertRaises(ValueError):
                owned._wait_receipt(self.path,bound)

    def test_linked_path_rejected(self):
        with patch.object(Path,'is_symlink',return_value=True), self.assertRaisesRegex(ValueError,'Linked'):
            owned._receipt_path(self.path)

    def test_publication_is_exclusive_and_actual_child_bound(self):
        process = Mock(pid=20);process.poll.return_value = None
        with self.fake_os(pid=10,ppid=1):
            receipt = owned.publish_receipt(self.path,process,request_sha256=REQUEST)
            self.assertEqual(receipt['owned_identity'],self.processes[20])
            with self.assertRaises(FileExistsError):
                owned.publish_receipt(self.path,process,request_sha256=REQUEST)
        self.assertEqual(json.loads(self.path.read_text()),receipt)

    def test_dead_or_unowned_process_cannot_publish(self):
        process=Mock(pid=20);process.poll.return_value=1
        with self.assertRaisesRegex(ValueError,'exited'):
            owned.publish_receipt(self.path,process,request_sha256=REQUEST)
        process.poll.return_value=None
        self.processes[20]['parent_pid']=99
        with self.fake_os(pid=10,ppid=1),self.assertRaisesRegex(ValueError,'direct Python child'):
            owned.publish_receipt(self.path,process,request_sha256=REQUEST)
        self.assertFalse(self.path.exists())

    def test_real_owned_interpreter_round_trip(self):
        """On Windows this exercises the real venv launcher plus base Python worker."""
        runtime = owned.runtime_identity()
        code = ('import json,sys; from tools.owned_python_launch import verify_worker; '
                'print(json.dumps(verify_worker(sys.argv[1],request_sha256=sys.argv[2],'
                'parent_pid=int(sys.argv[3]),expected_runtime=json.loads(sys.argv[4]))))')
        env = os.environ.copy()
        for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
            env[key]='1'
        options = dict(stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,
                       cwd=str(Path(__file__).resolve().parents[1]),env=env)
        if os.name=='nt': options['creationflags']=owned.windows_creationflags()
        process = subprocess.Popen([sys.executable,'-c',code,str(self.path),REQUEST,
                                    str(os.getpid()),json.dumps(runtime)],**options)
        job=None
        try:
            job=owned.contain_and_resume(process)
            receipt = owned.publish_receipt(self.path,process,request_sha256=REQUEST)
            stdout,stderr=process.communicate(timeout=15)
            self.assertEqual(process.returncode,0,stderr)
            result=json.loads(stdout)
            self.assertEqual(result['owned_pid'],process.pid)
            self.assertEqual(result['parent_pid'],os.getpid())
            self.assertTrue(result['runtime_verified'])
            expected='windows-venv' if os.name=='nt' and runtime['executable']!=runtime['base_executable'] else 'direct'
            self.assertEqual(result['mode'],expected)
            self.assertEqual(receipt['runtime'],runtime)
        finally:
            if job is not None: job.close()
            if process.poll() is None:
                if os.name=='nt':
                    subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],
                                   capture_output=True,timeout=10,creationflags=subprocess.CREATE_NO_WINDOW)
                if process.poll() is None: process.kill()
            process.wait(timeout=15)

    @unittest.skipUnless(os.name=='nt','Windows containment')
    def test_real_venv_job_kills_worker_without_taskkill(self):
        marker=self.path.with_name('synthetic-worker.json')
        code=('import json,os,sys,time; from pathlib import Path; '
              'Path(sys.argv[1]).write_text(json.dumps({"pid":os.getpid()}));time.sleep(30)')
        env=os.environ.copy()
        for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):env[key]='1'
        process=subprocess.Popen([sys.executable,'-c',code,str(marker)],env=env,
                                 creationflags=owned.windows_creationflags())
        job=None
        try:
            job=owned.contain_and_resume(process)
            deadline=time.monotonic()+5
            while not marker.exists() and time.monotonic()<deadline:time.sleep(.02)
            self.assertTrue(marker.exists(),'Contained synthetic worker never ran')
            worker_pid=json.loads(marker.read_text())['pid']
            worker_before=owned._windows_process(worker_pid)
            # The fallback tree killer is unavailable; kernel containment is sufficient.
            with patch.object(subprocess,'run',side_effect=OSError('taskkill unavailable')) as unavailable:
                job.close();process.wait(timeout=5)
                unavailable.assert_not_called()
            deadline=time.monotonic()+5
            while time.monotonic()<deadline:
                try:
                    current=owned._windows_process(worker_pid)
                    if current!=worker_before:break
                except (ValueError,OSError):break
                time.sleep(.02)
            else:self.fail('Actual Python worker survived closing its owned job')
            self.assertIsNotNone(process.poll())
        finally:
            if job is not None:job.close()
            if process.poll() is None:process.kill()
            process.wait(timeout=15)

    @unittest.skipUnless(os.name=='nt','Windows containment')
    def test_failed_containment_never_resumes_owned_stub(self):
        import ctypes
        for failure in ('identity','assignment'):
            with self.subTest(failure=failure):
                marker=self.path.with_name('should-not-run-'+failure)
                process=subprocess.Popen([sys.executable,'-c',
                    'from pathlib import Path;import sys,time;Path(sys.argv[1]).write_text("ran");time.sleep(30)',str(marker)],
                    creationflags=owned.windows_creationflags())
                try:
                    with ExitStack() as stack:
                        if failure=='identity':
                            stack.enter_context(patch.object(owned,'_windows_receipt_parent',side_effect=ValueError('bad identity')))
                        else:
                            kernel=ctypes.WinDLL('kernel32',use_last_error=True)
                            class DeniedAssignment:
                                AssignProcessToJobObject=Mock(return_value=0)
                                def __getattr__(self,name):return getattr(kernel,name)
                            stack.enter_context(patch.object(ctypes,'WinDLL',return_value=DeniedAssignment()))
                        with self.assertRaises((ValueError,OSError)):
                            owned.contain_and_resume(process)
                    self.assertIsNotNone(process.poll())
                    self.assertFalse(marker.exists(),'Failed containment executed uncontained worker')
                finally:
                    if process.poll() is None:process.kill()
                    process.wait(timeout=15)


if __name__ == '__main__':
    unittest.main()

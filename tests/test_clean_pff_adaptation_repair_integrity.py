"""Real subprocess admission with synthetic evidence and actual initialize seals.

No provider factory, observations, references, weights or fitting are used. The
test replaces only source/evidence boundaries in a temporary repository. Launch,
ancestry, locks, request/claim authentication and lifecycle/artifacts stay real.
"""
from contextlib import ExitStack, contextmanager
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from tools import clean_pff_adaptation_operator_v2 as op
from tools import clean_pff_adaptation_campaign as campaign


REPO = Path(__file__).resolve().parents[1]


def _write(path, value):
    path.write_bytes(op.storage.canonical(value) + b'\n')


def _build(root):
    (root/'runs').mkdir()
    old = root/op.PRIOR
    (old/'attempts').mkdir(parents=True)
    (old/'.lock').write_bytes(b'0')
    (root/'synthetic_source.py').write_text('# Synthetic source only\n')
    sources = {'synthetic_source.py':op.storage.file_hash(root/'synthetic_source.py')}
    descriptor = {'schema_version':1, 'synthetic':True, 'descriptor_sha256':'d'*64}
    with patch.object(op.stages,'descriptor',return_value=descriptor):
        op.stages.initialize(root/op.INNER,root)
    body = dict(family='clean-pff-adaptation-operator-v1', stages=list(op.ORDER),
                limits=dict(op.stages.LIMITS), python_executable=str(Path(sys.executable).resolve()),
                source_hashes=sources, inner_descriptor=descriptor)
    manifest = dict(body,manifest_sha256=op.storage.digest(body))
    request = dict(index=0,stage='initialize',nonce='1'*32,parent_pid=17,
                   manifest_sha256=manifest['manifest_sha256'],previous_result_sha256='0'*64)
    result = dict(request_sha256=op.storage.digest(request),manifest_sha256=manifest['manifest_sha256'],
                  status='failed',detail={'exception_type':'ValueError'})
    _write(old/'manifest.json',manifest)
    _write(old/'initialized.json',{'manifest_sha256':manifest['manifest_sha256']})
    _write(old/'attempts/000.request.json',request)
    _write(old/'attempts/000.result.json',result)
    (old/'attempts/000.console.log').write_text('Synthetic pre-stage launcher failure\n')
    config = dict(descriptor=descriptor, sources=sources,
                  pins={name:op.storage.file_hash(old/name) for name in op.PRIOR_FILE_HASHES},
                  inner_manifest_sha256=op.storage.file_hash(root/op.INNER/'manifest.json'))
    _write(root/'synthetic-config.json',config)
    return config


@contextmanager
def synthetic_runtime(root):
    """Same evidence-only replacements in parent and actual child interpreter."""
    root=Path(root).resolve()
    config=json.loads((root/'synthetic-config.json').read_text())
    with ExitStack() as stack:
        for replacement in (
            patch.object(op,'ROOT',root),
            patch.object(op,'PRIOR_FILE_HASHES',config['pins']),
            patch.object(op,'INNER_MANIFEST_SHA256',config['inner_manifest_sha256']),
            patch.object(op,'sources',return_value=config['sources']),
            patch.object(op.stages,'descriptor',side_effect=lambda *a,**k:copy.deepcopy(config['descriptor'])),
            patch.object(campaign,'ROOT',root),
            patch.object(campaign,'_context',return_value={}),
            patch.object(campaign,'_plan',return_value={'synthetic_initialization':True}),
        ):
            stack.enter_context(replacement)
        yield


class RepairIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve()
        self.config=_build(self.root)
        self.runtime=synthetic_runtime(self.root)
        self.runtime.__enter__()
        self.addCleanup(self.runtime.__exit__,None,None,None)
        self.child=self.root/'synthetic_worker.py'
        self.child.write_text(
            'import sys, json\n'
            f'sys.path.insert(0,{str(REPO)!r})\n'
            'from pathlib import Path\n'
            'import time\n'
            'from tests.test_clean_pff_adaptation_repair_integrity import synthetic_runtime, op\n'
            'from tools import clean_pff_adaptation_campaign as campaign\n'
            'root=Path(sys.argv[1]); index=int(sys.argv[2]); nonce=sys.argv[3]; mode=sys.argv[4]\n'
            'with synthetic_runtime(root):\n'
            '    if mode=="wrong-nonce": nonce="f"*32\n'
            '    if mode=="sleep-after-claim": campaign._context=lambda: time.sleep(30) or {}\n'
            '    result=op.worker(index,nonce)\n'
            '    if mode=="tamper-artifact":\n'
            '        (root/op.INNER/"artifacts/initialize/plan.json").write_text("{}\\n")\n'
            '    print(json.dumps(result))\n',encoding='utf-8')

    def _run(self,mode='ok'):
        with patch.object(op,'_command',side_effect=lambda i,n:[sys.executable,str(self.child),str(self.root),str(i),n,mode]):
            return op.run()

    def _unchanged_predecessor(self):
        self.assertEqual(self.config['pins'],{name:op.storage.file_hash(self.root/op.PRIOR/name)
                                             for name in self.config['pins']})

    def test_actual_worker_admits_owned_venv_and_seals_initialize(self):
        op.initialize()
        self.assertEqual(op.status()['next_stage'],'initialize')
        self.assertEqual(self._run()['stage_completed'],'initialize')
        state=op.status()
        self.assertEqual(state['recorded_completed_stages'],1)
        self.assertEqual(state['next_stage'],'fit-A')
        self.assertEqual(state['candidate_budget'],2)
        self.assertEqual(state['total_operator_attempts'],2)
        self.assertFalse(state['blocked'])
        out=self.root/op.OUT
        request=op._read(op._file(out,0,'request.json'))
        launch=op._read(op._file(out,0,'launch.json'))
        claim=op._read(op._file(out,0,'claim.json'))
        self.assertEqual(launch['parent_pid'],os.getpid())
        self.assertEqual(launch['request_sha256'],op.storage.digest(request))
        self.assertEqual(claim['launch_receipt_sha256'],op.storage.digest(launch))
        self.assertNotEqual(claim['worker_pid'],os.getpid())
        if os.name=='nt' and launch['runtime']['executable']!=launch['runtime']['base_executable']:
            self.assertNotEqual(claim['worker_pid'],launch['owned_pid'])
        self.assertEqual(campaign.verify_stage('initialize',self.root/op.INNER),
                         op._read(op._file(out,0,'worker.json'))['artifacts'])
        self.assertEqual(len((self.root/op.INNER/'ledger.jsonl').read_bytes().splitlines()),2)
        self._unchanged_predecessor()

    def test_actual_worker_wrong_nonce_consumes_launch_without_inner_attempt(self):
        op.initialize()
        with self.assertRaises(ValueError): self._run('wrong-nonce')
        out=self.root/op.OUT
        self.assertTrue(op.status()['blocked'])
        self.assertFalse(op._file(out,0,'claim.json').exists())
        self.assertEqual((self.root/op.INNER/'ledger.jsonl').read_bytes(),b'')
        self.assertEqual(list((self.root/op.INNER/'attempts').iterdir()),[])
        with patch.object(op,'_execute') as launch, self.assertRaises(ValueError):op.run()
        launch.assert_not_called()
        self._unchanged_predecessor()

    def test_parent_rejects_changed_actual_artifact_after_successful_worker(self):
        op.initialize()
        with self.assertRaises(ValueError): self._run('tamper-artifact')
        self.assertTrue(op.status()['blocked'])
        self.assertTrue(op._file(self.root/op.OUT,0,'claim.json').exists())
        self.assertEqual(len((self.root/op.INNER/'ledger.jsonl').read_bytes().splitlines()),2)
        with patch.object(op,'_execute') as launch, self.assertRaises(ValueError):op.run()
        launch.assert_not_called()
        self._unchanged_predecessor()

    def test_completed_worker_detects_coherent_launch_and_claim_tamper(self):
        op.initialize()
        self._run()
        out=self.root/op.OUT
        launch_path=op._file(out,0,'launch.json')
        claim_path=op._file(out,0,'claim.json')
        launch=op._read(launch_path)
        claim=op._read(claim_path)
        launch['owned_pid']+=1
        claim['launch_receipt_sha256']=op.storage.digest(launch)
        _write(launch_path,launch)
        _write(claim_path,claim)
        with self.assertRaisesRegex(ValueError,'launch binding'):op.status()
        self._unchanged_predecessor()

    def test_adoption_holds_inner_lock_through_proof_publication(self):
        original=op._write
        checked=[]
        def guarded(path,value):
            if Path(path).name=='adoption.json':
                with self.assertRaises(RuntimeError):
                    with op.storage.locked(self.root/op.INNER): pass
                checked.append(True)
            return original(path,value)
        with patch.object(op,'_write',side_effect=guarded):op.initialize()
        self.assertEqual(checked,[True])

    @unittest.skipUnless(os.name=='nt','Windows venv containment regression')
    def test_publication_failure_and_failed_taskkill_leave_no_native_worker(self):
        """Failure before claim must kill the native interpreter, not only its stub."""
        pid_path=self.root/'native-pid.json'
        sleeper=self.root/'native_sleeper.py'
        sleeper.write_text('import os,json,time\nfrom pathlib import Path\n'
                           f'Path({str(pid_path)!r}).write_text(json.dumps({{"pid":os.getpid()}}))\n'
                           'time.sleep(30)\n')
        native={}
        def fail_publication(*args,**kwargs):
            deadline=time.monotonic()+8
            while time.monotonic()<deadline:
                try:
                    native['pid']=json.loads(pid_path.read_text())['pid']
                    break
                except (FileNotFoundError,json.JSONDecodeError):time.sleep(.02)
            self.assertIn('pid',native,'Native child never started')
            native['identity']=op.owned._windows_process(native['pid'])
            raise RuntimeError('synthetic receipt publication failure')
        real_run=op.subprocess.run
        try:
            with patch.object(op.owned,'publish_receipt',side_effect=fail_publication), \
                 patch.object(op.subprocess,'run',side_effect=OSError('synthetic taskkill unavailable')):
                with self.assertRaisesRegex(RuntimeError,'synthetic receipt publication failure'):
                    op._execute([sys.executable,str(sleeper)],self.root,self.root/'cleanup.log',10,
                                self.root/'launch.json','a'*64)
            deadline=time.monotonic()+3
            alive=True
            while time.monotonic()<deadline:
                try:current=op.owned._windows_process(native['pid'])
                except (ValueError,OSError):alive=False;break
                self.assertEqual(current,native['identity'],'Synthetic worker PID was reused')
                time.sleep(.02)
            self.assertFalse(alive,'Native worker survived owned-launch cleanup')
        finally:
            # Only clean our identified synthetic child if a regression is found.
            if native:
                try:current=op.owned._windows_process(native['pid'])
                except (ValueError,OSError):current=None
                if current==native['identity']:
                    real_run(['taskkill','/PID',str(native['pid']),'/T','/F'],check=False,
                             capture_output=True,timeout=10,creationflags=op.subprocess.CREATE_NO_WINDOW)

    @unittest.skipUnless(os.name=='nt','Windows venv containment regression')
    def test_actual_worker_timeout_after_claim_consumes_both_lifecycles(self):
        op.initialize()
        execute=op._execute
        def bounded(command,root,log,seconds,launch_path,request_sha256):
            return execute(command,root,log,8,launch_path,request_sha256)
        with patch.object(op,'_execute',side_effect=bounded), \
             patch.object(op.subprocess,'run',side_effect=OSError('synthetic taskkill unavailable')):
            with self.assertRaises(op.subprocess.TimeoutExpired):self._run('sleep-after-claim')
        out=self.root/op.OUT
        claim=op._read(op._file(out,0,'claim.json'))
        self.assertTrue(op.status()['blocked'])
        self.assertTrue(op.stages.status(self.root/op.INNER,self.root)['blocked'])
        self.assertEqual(len((self.root/op.INNER/'ledger.jsonl').read_bytes().splitlines()),1)
        deadline=time.monotonic()+3
        while True:
            try:op.owned._windows_process(claim['worker_pid'])
            except (ValueError,OSError):break
            if time.monotonic()>=deadline:self.fail('Timed-out native worker survived containment')
            time.sleep(.02)
        with patch.object(op,'_execute') as launch, self.assertRaises(ValueError):op.run()
        launch.assert_not_called()
        self._unchanged_predecessor()


class RepairSourceClosureTests(unittest.TestCase):
    def test_new_closure_preserves_all_frozen_sources_and_adds_both_repair_modules(self):
        from tools import clean_pff_adaptation_operator as old
        previous=old.sources()
        actual=op.sources()
        self.assertEqual({name:actual[name] for name in previous},previous)
        self.assertEqual(set(actual)-set(previous),
                         {'tools/owned_python_launch.py','tools/clean_pff_adaptation_operator_v2.py'})


if __name__=='__main__':unittest.main()

"""Versioned repair for the exact failed pre-stage v1 launch; no final route.

The outer source manifest and consumed launch records complement the inner
lifecycle. Only fixed module entrypoints run. Dataset rights and promotion remain
unresolved; this operator cannot confer either. Logs and fitted artifacts stay local.
"""
from __future__ import annotations

from contextlib import contextmanager
import argparse
import ast
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid

from tools import clean_pff_adaptation_stages as stages
from tools import experiment_harness as storage
from tools import owned_python_launch as owned

ROOT = Path(__file__).resolve().parents[1]
OUT = 'runs/clean-pff-adaptation-operator-v2'
PRIOR = 'runs/clean-pff-adaptation-operator-v1'
INNER = 'runs/clean-pff-adaptation-inner-v1'
MODULE = 'tools.clean_pff_adaptation_operator_v2'
ROOTS = ('tools/clean_pff_adaptation_operator_v2.py', 'tools/owned_python_launch.py',
         'tools/clean_pff_adaptation_operator.py',
         'tools/clean_pff_adaptation_campaign.py', 'tools/clean_pff_adaptation_refit.py')
ORDER = (*stages.STAGES, 'refit-winner')
FLAGS = dict(final_access_authorized=False, development_access_authorized=False,
             submission_eligible=False, promoted=False, submission_hold=True)


def _need(value, message):
    if not value:
        raise ValueError(message)


def _read(path):
    return json.loads(stages._no_links(path).read_text(encoding='utf-8'))


def _write(path, value):
    stages._exclusive(path, value)


def sources():
    result = stages.source_hashes(ROOT)
    seen, pending = set(), [ROOT/name for name in ROOTS]
    def add(name):
        parts = name.split('.')
        if not all(p.isidentifier() for p in parts): return
        for i in range(1, len(parts)+1):
            p = ROOT.joinpath(*parts[:i])/'__init__.py'
            if p.is_file() and p not in seen: pending.append(p)
        p = ROOT.joinpath(*parts).with_suffix('.py')
        if p.is_file() and p not in seen: pending.append(p)
    while pending:
        path = pending.pop()
        if path in seen: continue
        stages._no_links(path)
        _need(path.is_file() and path.is_relative_to(ROOT), 'Missing or escaping operator source')
        seen.add(path)
        package = list(path.relative_to(ROOT).with_suffix('').parts[:-1])
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            if isinstance(node, ast.Import):
                for alias in node.names: add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                base = package[:len(package)-node.level+1] if node.level else []
                if node.module: base.extend(node.module.split('.'))
                if base:
                    add('.'.join(base))
                    for alias in node.names:
                        if alias.name != '*': add('.'.join([*base, alias.name]))
    result.update({p.relative_to(ROOT).as_posix(): storage.file_hash(p) for p in seen})
    return dict(sorted(result.items()))


def plan():
    # Aggregate provenance only; no provider data or fitted model is opened.
    value = dict(schema_version=1, family='clean-pff-adaptation-operator-v2',
                 predecessor=_prior_failure(), candidate_budget=2, prior_failed_launches=1,
                 runtime_identity=owned.runtime_identity(),
                 source_hashes=sources(), inner_descriptor=stages.descriptor(ROOT),
                 stages=list(ORDER), python_executable=str(Path(sys.executable).resolve()),
                 limits=dict(stages.LIMITS), artifact_verification_required=True,
                 resource_enforcement=dict(worker_timeout_seconds=1800, data_workers=1,
                     parent_verification_deadline_enforced=False, wakeup_seconds_is_scheduling_target=True),
                 real_fit_started=False, **FLAGS)
    return dict(value, manifest_sha256=storage.digest(value))


PRIOR_FILE_HASHES = {
    'manifest.json':'b3a8d49c7a926f204b17b4094be7952664cd21cf39c58542ba54d82723419f11',
    'initialized.json':'164a1e772fe8bb4cd04963193b940ddc315701d8cf7675b266df36b551b49026',
    'attempts/000.request.json':'49e78d069a33ab94d19dcedba2bce11583eaa53b5ead80b7a46a599c5ad61e89',
    'attempts/000.result.json':'3037325f68442d6f5b9f91db2769e3ee715dfba875c9eb64bdad0dd5d3d63fda',
    'attempts/000.console.log':'69e452d46181d999df4cb7b936ebdf30590f8a3da4bcd5a940e016382202b2b3',
}
INNER_MANIFEST_SHA256 = '53aefb6f1c2cf08cdaec78259bcb2bb45ae7cf08606baed63146ea429e4ba5a3'
EMPTY_SHA256 = 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'


def _checked(path, expected, parse=True):
    import hashlib
    raw=stages._no_links(path).read_bytes()
    _need(hashlib.sha256(raw).hexdigest()==expected,'Pinned predecessor bytes changed')
    return json.loads(raw) if parse else raw


def _prior_failure():
    """Authenticate only this exact terminal pre-stage failure; no data/model I/O."""
    old=stages._no_links(ROOT/PRIOR)
    _need({p.name for p in old.iterdir()}=={'.lock','manifest.json','initialized.json','attempts'},
          'Unexpected predecessor files')
    _need({p.name for p in (old/'attempts').iterdir()}==
          {'000.request.json','000.result.json','000.console.log'},'Predecessor claim, progress or extra attempt exists')
    values={name:_checked(old/name,identity,not name.endswith('.log')) for name,identity in PRIOR_FILE_HASHES.items()}
    manifest=values['manifest.json'];request=values['attempts/000.request.json'];result=values['attempts/000.result.json']
    body={k:v for k,v in manifest.items() if k!='manifest_sha256'}
    _need(manifest.get('manifest_sha256')==storage.digest(body) and manifest.get('family')=='clean-pff-adaptation-operator-v1',
          'Predecessor manifest identity differs')
    _need(values['initialized.json']=={'manifest_sha256':manifest['manifest_sha256']},'Predecessor initialization differs')
    _need(manifest['stages']==list(ORDER) and manifest['limits']==stages.LIMITS and
          manifest['python_executable']==str(Path(sys.executable).resolve()),'Predecessor scientific budget/runtime differs')
    for relative,identity in manifest['source_hashes'].items():
        path=stages._no_links(ROOT/relative)
        _need(path.is_relative_to(ROOT) and storage.file_hash(path)==identity,'Frozen predecessor source changed')
    descriptor=stages.descriptor(ROOT)
    _need(storage.canonical(manifest['inner_descriptor'])==storage.canonical(descriptor),'Original inner descriptor changed')
    inner_manifest=_checked(ROOT/INNER/'manifest.json',INNER_MANIFEST_SHA256)
    _need(storage.canonical(inner_manifest)==storage.canonical(descriptor),'Inner manifest changed')
    _need(set(request)=={'index','stage','nonce','parent_pid','manifest_sha256','previous_result_sha256'} and
          type(request['index']) is int and request['index']==0 and request['stage']=='initialize' and
          request['manifest_sha256']==manifest['manifest_sha256'] and request['previous_result_sha256']=='0'*64 and
          type(request['parent_pid']) is int and request['parent_pid']>0 and
          type(request['nonce']) is str and re.fullmatch('[0-9a-f]{32}',request['nonce']), 'Only failed initial v1 launch can migrate')
    _need(result==dict(request_sha256=storage.digest(request),manifest_sha256=manifest['manifest_sha256'],
                      status='failed',detail={'exception_type':'ValueError'}),'Predecessor is not the recorded terminal failure')
    return dict(operator='clean-pff-adaptation-operator-v1',file_sha256=dict(PRIOR_FILE_HASHES),
                original_inner_manifest_sha256=INNER_MANIFEST_SHA256,original_empty_ledger_sha256=EMPTY_SHA256,
                manifest_sha256=manifest['manifest_sha256'],failed_request_digest=storage.digest(request),
                failed_result_digest=storage.digest(result),failed_operator_launches=1,scientific_fit_attempts_consumed=0,
                original_source_count=len(manifest['source_hashes']),candidate_budget=2)


def _adoption_proof(manifest):
    return dict(predecessor_sha256=storage.digest(manifest['predecessor']),inner_descriptor_sha256=manifest['inner_descriptor']['descriptor_sha256'],
                original_empty_ledger_sha256=EMPTY_SHA256,inner_completed_stages=0,inner_attempt_count=0,
                no_inner_artifacts=True,prior_failed_launches_preserved=1,candidate_budget=2)


def _empty_inner(inner):
    _need({p.name for p in inner.iterdir()}=={'.lock','manifest.json','ledger.jsonl','attempts'},
          'Only the untouched empty inner lifecycle may be adopted')
    _need(not list(stages._no_links(inner/'attempts').iterdir()),'Inner attempt already consumed')
    _checked(inner/'ledger.jsonl',EMPTY_SHA256,False)
    _, events, _, index, blocked=stages._load(inner,ROOT)
    _need(not blocked and index==0 and not events,
          'Inner lifecycle is not untouched')


@contextmanager
def _locks(out):
    # Fixed order serializes legacy and repair operators without altering v1.
    with storage.locked(stages._no_links(ROOT/PRIOR)):
        with storage.locked(out):
            yield


def _locations():
    return stages._no_links(ROOT/OUT), stages.location(ROOT/INNER, ROOT)


def _manifest(out):
    value = _read(out/'manifest.json')
    _need(storage.canonical(value) == storage.canonical(plan()), 'Operator source/runtime manifest changed')
    _need(_read(out/'adoption.json')==_adoption_proof(value),'Migration proof changed')
    return value


def initialize():
    out, inner = _locations()
    _need(out.parent.is_dir(), 'Runs directory must already exist')
    out.mkdir(exist_ok=True)
    with _locks(out):
        _need(set(p.name for p in out.iterdir()) <= {'.lock'}, 'Operator initialization is consumed')
        with storage.locked(inner):
            manifest = plan()
            _empty_inner(inner)
            _write(out/'manifest.json', manifest)
            (out/'attempts').mkdir()
            _write(out/'adoption.json',_adoption_proof(manifest))
            _write(out/'initialized.json', {'manifest_sha256': manifest['manifest_sha256']})
    return dict(initialized=True, manifest_sha256=manifest['manifest_sha256'], **FLAGS)


def _file(out, index, suffix):
    return out/'attempts'/f'{index:03d}.{suffix}'


def _launch_binding(out, index, request):
    launch=_read(_file(out,index,'launch.json'))
    _need(launch.get('request_sha256')==storage.digest(request) and
          launch.get('parent_pid')==request['parent_pid'], 'Launch receipt request identity differs')
    identity=storage.digest(launch)
    claim=_read(_file(out,index,'claim.json'))
    _need(set(claim)=={'request_sha256','worker_pid','launch_receipt_sha256'} and
          claim['request_sha256']==storage.digest(request) and
          type(claim['worker_pid']) is int and claim['worker_pid']>0 and
          claim['launch_receipt_sha256']==identity, 'Worker claim identity differs')
    return identity


def _history(out, manifest):
    _need(_read(out/'initialized.json') == {'manifest_sha256': manifest['manifest_sha256']},
          'Incomplete operator initialization is consumed')
    index, previous, rows, consumed = 0, manifest['predecessor']['failed_result_digest'], [], False
    while _file(out, index, 'request.json').exists():
        request = _read(_file(out, index, 'request.json'))
        _need(index < len(ORDER) and set(request) == {'index', 'stage', 'nonce', 'parent_pid',
              'manifest_sha256', 'previous_result_sha256'}, 'Invalid launch request')
        _need(type(request['index']) is int and request['index'] == index and
              request['stage'] == ORDER[index] and request['manifest_sha256'] == manifest['manifest_sha256'] and
              request['previous_result_sha256'] == previous and type(request['parent_pid']) is int and request['parent_pid'] > 0 and
              type(request['nonce']) is str and re.fullmatch('[0-9a-f]{32}', request['nonce']), 'Launch identity/order differs')
        result_path = _file(out, index, 'result.json')
        if not result_path.exists():
            consumed = True; break
        result = _read(result_path)
        _need(set(result) == {'request_sha256', 'manifest_sha256', 'status', 'detail'} and
              result['request_sha256'] == storage.digest(request) and
              result['manifest_sha256'] == manifest['manifest_sha256'], 'Terminal launch identity differs')
        _need(result['status'] in ('complete', 'failed'), 'Invalid terminal launch state')
        if result['status'] == 'failed':
            consumed = True; break
        launch_sha = _launch_binding(out,index,request)
        receipt = _read(_file(out, index, 'worker.json'))
        _need(storage.digest(receipt) == result['detail']['worker_receipt_sha256'], 'Worker receipt bytes changed')
        _need(set(receipt)=={'request_sha256','artifacts','launch_receipt_sha256'} and
              receipt['request_sha256'] == storage.digest(request) and
              receipt['launch_receipt_sha256']==launch_sha, 'Worker receipt launch binding differs')
        previous = storage.digest(result); rows.append(request); index += 1
    allowed = {f'{i:03d}.{suffix}' for i in range(index + int(consumed))
               for suffix in ('request.json', 'result.json', 'worker.json', 'claim.json', 'launch.json', 'console.log')}
    _need(all(p.name in allowed for p in (out/'attempts').iterdir()), 'Unexpected attempt file or missing launch history')
    return index, previous, rows, consumed


def _selection(inner):
    from tools import clean_pff_adaptation_campaign as campaign
    return campaign.verify_completed(inner)


def _next(out, inner, manifest):
    index, previous, rows, consumed = _history(out, manifest)
    if consumed: return None, index, previous, True
    state = stages.status(inner, ROOT)
    if index < len(stages.STAGES):
        _need(not state['blocked'] and state['recorded_completed_stages'] == index and
              state['next_stage'] == ORDER[index], 'Inner and operator stage histories disagree')
        return ORDER[index], index, previous, False
    _need(state['complete'], 'Completed inner lifecycle required')
    selected = _selection(inner)
    if index == len(ORDER) or selected['selection']['selected_candidate_id'] is None:
        return None, index, previous, False
    return 'refit-winner', index, previous, False


def status():
    out, inner = _locations()
    with _locks(out):
        manifest = _manifest(out)
        stage, index, _, blocked = _next(out, inner, manifest)
        return dict(recorded_completed_stages=index, next_stage=stage, blocked=blocked,
                    complete=stage is None and not blocked,prior_failed_launches=1,
                    total_operator_attempts=1+index+int(blocked),candidate_budget=2, **FLAGS)


def _verify_artifacts(stage, inner):
    if stage == 'refit-winner':
        from tools import clean_pff_adaptation_refit as refit
        return refit.verify_result(inner)
    from tools import clean_pff_adaptation_campaign as campaign
    return campaign.verify_stage(stage, inner)


def _command(index, nonce):
    return [sys.executable, '-m', MODULE, 'worker', '--index', str(index), '--nonce', nonce]


def _execute(command, root, log, seconds, launch_path, request_sha256):
    """One bounded child; contain Windows descendants before the stub can run."""
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    options = dict(cwd=str(root), env=env, shell=False)
    if os.name == 'nt':
        options['creationflags'] = owned.windows_creationflags()
    else:
        options['start_new_session'] = True
    with Path(log).open('xb') as stream:
        deadline=time.monotonic()+seconds
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, **options)
        job=None
        try:
            job=owned.contain_and_resume(process)
            owned.publish_receipt(launch_path,process,request_sha256=request_sha256)
            return process.wait(timeout=max(0.,deadline-time.monotonic()))
        except BaseException:
            try:
                if os.name == 'nt':
                    if job is not None: job.close()
                else:
                    os.killpg(process.pid,signal.SIGKILL)
            except (OSError,subprocess.TimeoutExpired):
                pass
            finally:
                if process.poll() is None: process.kill()
                process.wait(timeout=15)
            raise
        finally:
            if job is not None: job.close()


def run():
    """Launch at most one fixed stage; failed launches can never be retried."""
    out, inner = _locations()
    with _locks(out):
        manifest = _manifest(out)
        stage, index, previous, blocked = _next(out, inner, manifest)
        _need(not blocked, 'A failed or interrupted launch consumes this campaign')
        if stage is None: return dict(complete=True, **FLAGS)
        request = dict(index=index, stage=stage, nonce=uuid.uuid4().hex, parent_pid=os.getpid(),
                       manifest_sha256=manifest['manifest_sha256'], previous_result_sha256=previous)
        _write(_file(out,index,'request.json'), request)
        terminal = dict(request_sha256=storage.digest(request), manifest_sha256=manifest['manifest_sha256'])
        try:
            code = _execute(_command(index,request['nonce']), ROOT,
                                   _file(out,index,'console.log'), stages.LIMITS['stage_timeout_seconds'],
                                   _file(out,index,'launch.json'),storage.digest(request))
            _need(code == 0, 'Worker failed; consult the local log')
            _manifest(out)
            actual = _verify_artifacts(stage, inner)
            receipt = _read(_file(out,index,'worker.json'))
            expected = dict(request_sha256=storage.digest(request), artifacts=actual,
                            launch_receipt_sha256=_launch_binding(out,index,request))
            _need(storage.canonical(receipt) == storage.canonical(expected), 'Actual artifact verification differs')
            _write(_file(out,index,'result.json'), dict(terminal,status='complete',
                   detail={'worker_receipt_sha256':storage.digest(receipt)}))
        except BaseException as error:
            _write(_file(out,index,'result.json'), dict(terminal,status='failed',detail={'exception_type':type(error).__name__}))
            raise
    return dict(stage_completed=stage, execution_completed=True, **FLAGS)


def worker(index, nonce):
    """Private subprocess entry; a parent-bound, single-use launch is mandatory."""
    _need(type(index) is int and 0 <= index < len(ORDER) and type(nonce) is str and
          re.fullmatch('[0-9a-f]{32}',nonce), 'Invalid worker launch')
    _need(all(os.environ.get(name) == '1' for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')),
          'One BLAS thread per worker is mandatory')
    out, inner = _locations()
    for lock_root in (ROOT/PRIOR,out):
        try:
            with storage.locked(lock_root):
                raise ValueError('Worker requires both parent operator locks to remain held')
        except RuntimeError as error:
            if 'holds the lock' not in str(error): raise
    manifest = _manifest(out)
    current, _, _, consumed = _history(out,manifest)
    _need(consumed and current == index, 'Worker must consume the exact current launch')
    request = _read(_file(out,index,'request.json'))
    _need(request.get('index') == index and request.get('stage') == ORDER[index] and
          request.get('nonce') == nonce and
          request.get('manifest_sha256') == manifest['manifest_sha256'], 'Worker request is not bound to this parent')
    _need(not _file(out,index,'result.json').exists(), 'Worker attempt already terminal')
    owned.verify_worker(_file(out,index,'launch.json'),request_sha256=storage.digest(request),
                        parent_pid=request['parent_pid'],expected_runtime=manifest['runtime_identity'])
    launch_sha=storage.digest(_read(_file(out,index,'launch.json')))
    _write(_file(out,index,'claim.json'), {'request_sha256':storage.digest(request),'worker_pid':os.getpid(),
                                        'launch_receipt_sha256':launch_sha})
    if ORDER[index].startswith('fit-') or ORDER[index] == 'refit-winner':
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    if ORDER[index] == 'refit-winner':
        from tools import clean_pff_adaptation_refit as refit
        refit.run_refit(inner)
    else:
        from tools import clean_pff_adaptation_campaign as campaign
        campaign.execute_stage(ORDER[index],inner)
    _manifest(out)
    artifacts = _verify_artifacts(ORDER[index],inner)
    _write(_file(out,index,'worker.json'), {'request_sha256':storage.digest(request),'artifacts':artifacts,'launch_receipt_sha256':launch_sha})
    return dict(stage_completed=ORDER[index], **FLAGS)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan','initialize','status','run','worker'))
    parser.add_argument('--index',type=int)
    parser.add_argument('--nonce')
    args = parser.parse_args(argv)
    if args.action == 'worker': result = worker(args.index,args.nonce)
    else:
        _need(args.index is None and args.nonce is None, 'Worker-only arguments are forbidden')
        result = {'plan':plan,'initialize':initialize,'status':status,'run':run}[args.action]()
    print(json.dumps(result,sort_keys=True))


if __name__ == '__main__':
    main()

"""One bounded local adaptation stage per invocation; no development/final route.

The outer source manifest and consumed launch records complement the inner
lifecycle. Only fixed module entrypoints run. Dataset rights and promotion remain
unresolved; this operator cannot confer either. Logs and fitted artifacts stay local.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import uuid

from tools import clean_pff_adaptation_stages as stages
from tools import experiment_harness as storage

ROOT = Path(__file__).resolve().parents[1]
OUT = 'runs/clean-pff-adaptation-operator-v1'
INNER = 'runs/clean-pff-adaptation-inner-v1'
MODULE = 'tools.clean_pff_adaptation_operator'
ROOTS = ('tools/clean_pff_adaptation_operator.py',
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
    value = dict(schema_version=1, family='clean-pff-adaptation-operator-v1',
                 source_hashes=sources(), inner_descriptor=stages.descriptor(ROOT),
                 stages=list(ORDER), python_executable=str(Path(sys.executable).resolve()),
                 limits=dict(stages.LIMITS), artifact_verification_required=True,
                 resource_enforcement=dict(worker_timeout_seconds=1800, data_workers=1,
                     parent_verification_deadline_enforced=False, wakeup_seconds_is_scheduling_target=True),
                 real_fit_started=False, **FLAGS)
    return dict(value, manifest_sha256=storage.digest(value))


def _locations():
    return stages._no_links(ROOT/OUT), stages.location(ROOT/INNER, ROOT)


def _manifest(out):
    value = _read(out/'manifest.json')
    _need(storage.canonical(value) == storage.canonical(plan()), 'Operator source/runtime manifest changed')
    return value


def initialize():
    out, inner = _locations()
    _need(out.parent.is_dir(), 'Runs directory must already exist')
    out.mkdir(exist_ok=True)
    with storage.locked(out):
        _need(set(p.name for p in out.iterdir()) <= {'.lock'}, 'Operator initialization is consumed')
        _need(not inner.exists(), 'Existing inner campaign cannot be adopted or reset')
        manifest = plan()
        _write(out/'manifest.json', manifest)
        (out/'attempts').mkdir()
        stages.initialize(inner, ROOT)
        _write(out/'initialized.json', {'manifest_sha256': manifest['manifest_sha256']})
    return dict(initialized=True, manifest_sha256=manifest['manifest_sha256'], **FLAGS)


def _file(out, index, suffix):
    return out/'attempts'/f'{index:03d}.{suffix}'


def _history(out, manifest):
    _need(_read(out/'initialized.json') == {'manifest_sha256': manifest['manifest_sha256']},
          'Incomplete operator initialization is consumed')
    index, previous, rows, consumed = 0, '0'*64, [], False
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
        claim = _read(_file(out,index,'claim.json'))
        _need(set(claim) == {'request_sha256','worker_pid'} and
              claim['request_sha256'] == storage.digest(request) and
              type(claim['worker_pid']) is int and claim['worker_pid'] > 0, 'Worker claim identity differs')
        receipt = _read(_file(out, index, 'worker.json'))
        _need(storage.digest(receipt) == result['detail']['worker_receipt_sha256'], 'Worker receipt bytes changed')
        _need(receipt['request_sha256'] == storage.digest(request), 'Worker receipt launch binding differs')
        previous = storage.digest(result); rows.append(request); index += 1
    allowed = {f'{i:03d}.{suffix}' for i in range(index + int(consumed))
               for suffix in ('request.json', 'result.json', 'worker.json', 'claim.json', 'console.log')}
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
    with storage.locked(out):
        manifest = _manifest(out)
        stage, index, _, blocked = _next(out, inner, manifest)
        return dict(recorded_completed_stages=index, next_stage=stage, blocked=blocked,
                    complete=stage is None and not blocked, **FLAGS)


def _verify_artifacts(stage, inner):
    if stage == 'refit-winner':
        from tools import clean_pff_adaptation_refit as refit
        return refit.verify_result(inner)
    from tools import clean_pff_adaptation_campaign as campaign
    return campaign.verify_stage(stage, inner)


def _command(index, nonce):
    return [sys.executable, '-m', MODULE, 'worker', '--index', str(index), '--nonce', nonce]


def _execute(command, root, log, seconds):
    """Hidden single-worker child; reap owned handle even if tree kill fails."""
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    options = dict(cwd=str(root), env=env, shell=False)
    if os.name == 'nt':
        options['creationflags'] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        options['start_new_session'] = True
    with Path(log).open('xb') as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, **options)
        try:
            return process.wait(timeout=seconds)
        except BaseException:
            try:
                if os.name == 'nt':
                    subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'], check=False,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10,
                        creationflags=subprocess.CREATE_NO_WINDOW)
                else:
                    os.killpg(process.pid,signal.SIGKILL)
            except (OSError,subprocess.TimeoutExpired):
                pass
            finally:
                if process.poll() is None: process.kill()
                process.wait(timeout=15)
            raise


def run():
    """Launch at most one fixed stage; failed launches can never be retried."""
    out, inner = _locations()
    with storage.locked(out):
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
                                   _file(out,index,'console.log'), stages.LIMITS['stage_timeout_seconds'])
            _need(code == 0, 'Worker failed; consult the local log')
            _manifest(out)
            actual = _verify_artifacts(stage, inner)
            receipt = _read(_file(out,index,'worker.json'))
            expected = dict(request_sha256=storage.digest(request), artifacts=actual)
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
    try:
        with storage.locked(out):
            raise ValueError('Worker requires the parent operator lock to remain held')
    except RuntimeError as error:
        if 'holds the lock' not in str(error): raise
    manifest = _manifest(out)
    current, _, _, consumed = _history(out,manifest)
    _need(consumed and current == index, 'Worker must consume the exact current launch')
    request = _read(_file(out,index,'request.json'))
    _need(request.get('index') == index and request.get('stage') == ORDER[index] and
          request.get('nonce') == nonce and request.get('parent_pid') == os.getppid() and
          request.get('manifest_sha256') == manifest['manifest_sha256'], 'Worker request is not bound to this parent')
    _need(not _file(out,index,'result.json').exists(), 'Worker attempt already terminal')
    _write(_file(out,index,'claim.json'), {'request_sha256':storage.digest(request),'worker_pid':os.getpid()})
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
    _write(_file(out,index,'worker.json'), {'request_sha256':storage.digest(request),'artifacts':artifacts})
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

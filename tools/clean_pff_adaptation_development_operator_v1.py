"""Owned, single-stage execution of the fixed selected-refit development study."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import uuid

for _thread_variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_thread_variable] = '1'

from tools import clean_pff_adaptation_development_v1 as campaign
from tools import clean_pff_adaptation_operator_v2 as prior_operator
from tools import experiment_harness as storage
from tools import owned_python_launch as owned

ROOT = Path(__file__).resolve().parents[1]
MODULE = 'tools.clean_pff_adaptation_development_operator_v1'
ORDER = campaign.STAGES
FLAGS = dict(campaign.FLAGS)
_CONTEXT = None  # Never serialized; each process begins with a complete actual audit.


def need(value, message):
    if not value:
        raise ValueError(message)


def directory():
    return campaign.path('operator')


def file(index, suffix):
    need(type(index) is int and 0 <= index < len(ORDER), 'Invalid stage index')
    need(suffix in ('request.json', 'result.json', 'worker.json', 'claim.json', 'launch.json', 'console.log'),
         'Unknown owned launch artifact')
    return campaign.path(f'operator/attempts/{index:03d}.{suffix}')


def _read(target):
    campaign.inner_store._no_links(target)
    return json.loads(Path(target).read_bytes())


def _write(target, value):
    campaign.inner_store._no_links(target)
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    with Path(target).open('xb') as stream:
        stream.write(storage.canonical(value)+b'\n'); stream.flush(); os.fsync(stream.fileno())


def initialize():
    need(not campaign.root().exists(), 'Development initialization is single-use')
    study = campaign.plan()  # completed winner and all original sources authenticated first
    campaign.root().mkdir()
    directory().mkdir()
    with storage.locked(directory()):
        campaign.write('protocol.json', study)
        campaign.write('operator/initialized.json', dict(protocol_sha256=study['protocol_sha256'],
            protocol_file_sha256=storage.file_hash(campaign.path('protocol.json'))))
    return dict(initialized=True, protocol_sha256=study['protocol_sha256'],
                candidate_sha256=study['candidate_sha256'], **FLAGS)


def _manifest():
    global _CONTEXT
    study, ctx = campaign.authenticate(_CONTEXT)
    _CONTEXT = ctx
    need(_read(directory()/'initialized.json') == dict(protocol_sha256=study['protocol_sha256'],
         protocol_file_sha256=storage.file_hash(campaign.path('protocol.json'))), 'Initialization bytes changed')
    return study, ctx


def verified_context():
    return _manifest()[1]


def verify_owned_receipts(index, request, launch, claim, study):
    launch_fields = {'schema_version','platform','request_sha256','receipt_path','parent_pid','owned_pid',
                     'runtime','parent_identity','owned_identity'}
    need(set(launch) == launch_fields and type(launch['schema_version']) is int and launch['schema_version'] == 1 and
         launch['platform'] == ('windows' if os.name == 'nt' else 'posix') and
         launch['request_sha256'] == storage.digest(request) and launch['parent_pid'] == request['parent_pid'] and
         launch['receipt_path'] == owned._normal(file(index,'launch.json')) and
         launch['runtime'] == study['runtime_identity'] and type(launch['owned_pid']) is int and launch['owned_pid'] > 0,
         'Historical owned launch schema or identity differs')
    need(set(claim) == {'request_sha256','worker_pid','launch_receipt_sha256','admission','resources'} and
         claim['request_sha256'] == storage.digest(request) and type(claim['worker_pid']) is int and claim['worker_pid'] > 0 and
         claim['launch_receipt_sha256'] == storage.digest(launch), 'Worker claim identity differs')
    admission = claim['admission']
    need(set(admission) == {'request_sha256','parent_pid','owned_pid','worker_pid','mode','runtime_verified','index','nonce','stage'} and
         all(type(admission[k]) is int for k in ('parent_pid','owned_pid','worker_pid','index')) and
         admission['runtime_verified'] is True and
         admission == dict(request_sha256=storage.digest(request),parent_pid=request['parent_pid'],
             owned_pid=launch['owned_pid'],worker_pid=claim['worker_pid'],mode=admission['mode'],runtime_verified=True,
             index=index,nonce=request['nonce'],stage=ORDER[index]), 'Worker admission binding differs')
    if launch['platform'] == 'posix':
        need(admission['mode'] == 'direct' and claim['worker_pid'] == launch['owned_pid'] and
             launch['parent_identity'] is None and launch['owned_identity'] is None, 'POSIX owned ancestry differs')
    else:
        parent, child = launch['parent_identity'], launch['owned_identity']
        fields = {'pid','parent_pid','image','creation_token'}
        need(isinstance(parent,dict) and isinstance(child,dict) and set(parent) == fields and set(child) == fields and
             parent['pid'] == request['parent_pid'] and child['pid'] == launch['owned_pid'] and
             child['parent_pid'] == parent['pid'] and parent['creation_token'] <= child['creation_token'] and
             parent['image'] == study['runtime_identity']['base_executable'] and
             child['image'] in (study['runtime_identity']['base_executable'],study['runtime_identity']['executable']),
             'Windows owned ancestry differs')
        need((admission['mode'] == 'direct' and claim['worker_pid'] == launch['owned_pid']) or
             (admission['mode'] == 'windows-venv' and claim['worker_pid'] != launch['owned_pid'] and
              child['image'] == study['runtime_identity']['executable'] != study['runtime_identity']['base_executable']),
             'Windows worker launch mode differs')
    need(type(claim['resources'].get('torch_threads')) is int and
         type(claim['resources'].get('torch_interop_threads')) is int and
         claim['resources'] == dict(blas_env={k:'1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
         torch_threads=1,torch_interop_threads=1), 'Observed worker thread counts differ')
    return claim


def history(study):
    """Append-only launch chain; an absent/failed terminal result consumes work."""
    previous = '0'*64
    index = 0
    blocked = False
    while index < len(ORDER) and file(index, 'request.json').exists():
        request = _read(file(index, 'request.json'))
        need(set(request) == {'index','stage','nonce','parent_pid','protocol_sha256','previous_result_sha256'} and
             type(request['index']) is int and request['index'] == index and request['stage'] == ORDER[index] and
             request['protocol_sha256'] == study['protocol_sha256'] and
             request['previous_result_sha256'] == previous and type(request['parent_pid']) is int and
             request['parent_pid'] > 0 and type(request['nonce']) is str and
             re.fullmatch('[0-9a-f]{32}', request['nonce']), 'Invalid owned stage request chain')
        if not file(index, 'result.json').exists():
            blocked = True; break
        result = _read(file(index, 'result.json'))
        need(set(result) == {'request_sha256','protocol_sha256','status','detail'} and
             result['request_sha256'] == storage.digest(request) and result['protocol_sha256'] == study['protocol_sha256'] and
             result['status'] in ('complete','failed'), 'Invalid terminal launch receipt')
        if result['status'] == 'failed':
            blocked = True; break
        worker = _read(file(index, 'worker.json'))
        launch = _read(file(index, 'launch.json'))
        claim = _read(file(index, 'claim.json'))
        verify_owned_receipts(index, request, launch, claim, study)
        launch_hash = storage.digest(launch)
        need(launch['request_sha256'] == storage.digest(request) and launch['parent_pid'] == request['parent_pid'] and
             launch['runtime'] == study['runtime_identity'] and
             worker == dict(request_sha256=storage.digest(request),
                            artifacts=campaign.verify_stage(ORDER[index]), launch_receipt_sha256=launch_hash) and
             result['detail'] == {'worker_receipt_sha256':storage.digest(worker)} and
             campaign.read(ORDER[index]+'-started.json')['pid'] == claim['worker_pid'], 'Completed owned stage proof differs')
        previous = storage.digest(result); index += 1
    attempts = directory()/'attempts'
    if attempts.exists():
        allowed = {f'{i:03d}.{suffix}' for i in range(index + int(blocked)) for suffix in
                   ('request.json','result.json','worker.json','claim.json','launch.json','console.log')}
        need(all(p.is_file() and p.name in allowed for p in attempts.iterdir()), 'Unexpected owned attempt artifact')
    completed = campaign.state()['completed']
    expected = len(ORDER) if index == len(ORDER) else index
    # A blocked worker may have completed its artifact proof before parent failure;
    # that does not authorize a retry or skip of the consumed request.
    need(len(completed) in ({expected, expected+1} if blocked else {expected}), 'Operator/stage history mismatch')
    return index, previous, blocked


def status():
    with storage.locked(directory()):
        study, _ = _manifest()
        index, _, blocked = history(study)
    return dict(recorded_completed_stages=index, next_stage=None if blocked or index == len(ORDER) else ORDER[index],
                blocked=blocked, complete=index == len(ORDER) and not blocked, **FLAGS)


def _parent_lock_required():
    try:
        with storage.locked(directory()):
            raise ValueError('Worker requires its live parent operator lock')
    except RuntimeError as error:
        if 'holds the lock' not in str(error):
            raise


def _admit(index, nonce):
    need(type(index) is int and 0 <= index < len(ORDER) and type(nonce) is str and
         re.fullmatch('[0-9a-f]{32}',nonce), 'Invalid owned worker request')
    need(all(os.environ.get(k) == '1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')),
         'One thread per worker is mandatory')
    _parent_lock_required()
    study, _ = _manifest()
    current, _, consumed = history(study)
    need(consumed and current == index and not file(index,'result.json').exists(), 'Exact current consumed launch required')
    request = _read(file(index,'request.json'))
    need(request['nonce'] == nonce and request['stage'] == ORDER[index], 'Worker nonce/stage mismatch')
    admission = owned.verify_worker(file(index,'launch.json'), request_sha256=storage.digest(request),
        parent_pid=request['parent_pid'], expected_runtime=study['runtime_identity'])
    return dict(admission, index=index, nonce=nonce, stage=ORDER[index])


def verify_admission(admission, stage):
    need(isinstance(admission,dict) and admission.get('stage') == stage,
         'Kernel requires an owned worker admission')
    actual = _admit(admission.get('index'),admission.get('nonce'))
    need(actual == admission, 'Worker admission changed')
    return actual


def worker(index, nonce):
    admission = _admit(index,nonce)
    request = _read(file(index,'request.json'))
    launch_hash = storage.digest(_read(file(index,'launch.json')))
    # Imported Torch is used for inference only, with explicit intra/inter-op limits.
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    resources = dict(blas_env={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
                     torch_threads=torch.get_num_threads(),torch_interop_threads=torch.get_num_interop_threads())
    _write(file(index,'claim.json'), dict(request_sha256=storage.digest(request), worker_pid=os.getpid(),
        launch_receipt_sha256=launch_hash,admission=admission,resources=resources))
    actual = campaign.run_stage(ORDER[index], admission=admission)
    _manifest()
    _write(file(index,'worker.json'), dict(request_sha256=storage.digest(request), artifacts=actual,
                                         launch_receipt_sha256=launch_hash))
    return dict(stage_completed=ORDER[index], **FLAGS)


def run():
    with storage.locked(directory()):
        study, _ = _manifest()
        index, previous, blocked = history(study)
        need(not blocked, 'Failed or interrupted work remains consumed; no retry')
        if index == len(ORDER):
            return dict(complete=True, **FLAGS)
        request = dict(index=index, stage=ORDER[index], nonce=uuid.uuid4().hex, parent_pid=os.getpid(),
                       protocol_sha256=study['protocol_sha256'], previous_result_sha256=previous)
        _write(file(index,'request.json'),request)
        terminal = dict(request_sha256=storage.digest(request),protocol_sha256=study['protocol_sha256'])
        try:
            command = [sys.executable,'-B','-m',MODULE,'worker','--index',str(index),'--nonce',request['nonce']]
            # The exact existing helper supplies CREATE_SUSPENDED, Windows Job
            # containment, launch ownership, deadline and descendant cleanup.
            code = prior_operator._execute(command, ROOT, file(index,'console.log'),1800,
                file(index,'launch.json'),storage.digest(request))
            need(code == 0, 'Development worker failed; local log retained')
            _manifest()
            actual = campaign.verify_stage(ORDER[index])
            worker_receipt = _read(file(index,'worker.json'))
            need(worker_receipt == dict(request_sha256=storage.digest(request),artifacts=actual,
                 launch_receipt_sha256=storage.digest(_read(file(index,'launch.json')))), 'Worker artifact receipt differs')
            _write(file(index,'result.json'),dict(terminal,status='complete',
                detail={'worker_receipt_sha256':storage.digest(worker_receipt)}))
        except BaseException as error:
            _write(file(index,'result.json'),dict(terminal,status='failed',detail={'exception_type':type(error).__name__}))
            raise
    return dict(stage_completed=ORDER[index],execution_completed=True,**FLAGS)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('plan','initialize','status','run','worker'))
    parser.add_argument('--index',type=int)
    parser.add_argument('--nonce')
    args = parser.parse_args(argv)
    if args.action == 'worker':
        value = worker(args.index,args.nonce)
    else:
        need(args.index is None and args.nonce is None, 'Worker parameters are not public overrides')
        if args.action == 'plan':
            full = campaign.plan()
            value = {key:full[key] for key in ('protocol_sha256','candidate_sha256','matches','stages','limits')}
            value.update(FLAGS)
        else:
            value = {'initialize':initialize,'status':status,'run':run}[args.action]()
    print(json.dumps(value,sort_keys=True))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps(dict(status='failed',exception_type=type(error).__name__)),file=sys.stderr)
        raise SystemExit(2)

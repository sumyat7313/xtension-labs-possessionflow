"""One fixed external candidate, development fixtures only, resumable evaluation.

Uses the existing harness's low-level ledger, lock and process helpers without
changing its constants or historical state. No fit, final, or promotion command
is exposed. A successful stage is authenticated before the next is queued.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import experiment_harness as storage

MODULE = 'tools.external_pff_development'
TRIAL = 'external-pff-development'
CANDIDATE_SHA256 = '7d79dc50cfc16a7f64c37b65908be90e588d3bfdec35f599071b25180089bf47'
REBUILD = 'runs/continuous-improvement/trials/external-rebuild-001/output'
DEVELOPMENT = (3813, 3816, 3818, 3823, 3831, 3844, 3850, 3854, 3858, 10502, 10514, 10517)
STAGES = ('initialize', *(f'predict-{mid}' for mid in DEVELOPMENT), 'freeze', 'score')
SOURCE_ENTRIES = ('tools/development_harness.py', 'tools/external_pff_development.py',
                  'tools/verify_external_pff_development.py')


def _plain_file(path, root):
    path, root = Path(path), Path(root).resolve()
    if path.resolve() != path or not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Required local artifact is missing or resolves outside its declared path')
    return path


def sources(root=ROOT):
    """Static local import closure, including package imports inside functions."""
    root = Path(root).resolve()
    pending = [_plain_file(root / name, root) for name in SOURCE_ENTRIES]
    found = set()

    def add(name):
        parts = name.split('.')
        if any(not p.isidentifier() for p in parts):
            return
        candidates = [root.joinpath(*parts).with_suffix('.py')]
        candidates.extend(root.joinpath(*parts[:n]) / '__init__.py' for n in range(1, len(parts) + 1))
        pending.extend(p for p in candidates if p.is_file() and p not in found)

    while pending:
        path = pending.pop()
        if path in found:
            continue
        _plain_file(path, root)
        found.add(path)
        package = list(path.relative_to(root).with_suffix('').parts[:-1])
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                base = package[:len(package) - node.level + 1] if node.level else []
                if node.module:
                    base += node.module.split('.')
                if base:
                    add('.'.join(base))
                    for alias in node.names:
                        if alias.name != '*':
                            add('.'.join([*base, alias.name]))
    for name in ('requirements-solution.txt', 'requirements-training.txt'):
        if (root / name).is_file():
            found.add(_plain_file(root / name, root))
    return {p.relative_to(root).as_posix(): storage.file_hash(p) for p in sorted(found)}


def inputs(root=ROOT):
    root = Path(root).resolve()
    names = [f'{REBUILD}/protocol.json', *[f'{REBUILD}/refit/{name}' for name in (
        'candidate.json', 'config.json', 'lineage.json', 'model-bindings.json', 'complete.json')]]
    result = {name: storage.file_hash(_plain_file(root / name, root)) for name in names}
    if result[f'{REBUILD}/refit/candidate.json'] != CANDIDATE_SHA256:
        raise ValueError('Fixed candidate model identity differs')
    return result


def metadata(root):
    return dict(schema_version=1, root=str(Path(root).resolve()), module=MODULE,
                trial_id=TRIAL, candidate_sha256=CANDIDATE_SHA256,
                final_access_enabled=False, promotion_enabled=False, fitting_enabled=False)


def initialize(out, root=ROOT):
    root = Path(root).resolve()
    out = storage.location(out, root)
    # Authenticate all bindings before creating an otherwise empty harness.
    frozen = dict(schema_version=1, candidate_sha256=CANDIDATE_SHA256, source_hashes=sources(root),
                  input_hashes=inputs(root), development_matches=list(DEVELOPMENT), stages=list(STAGES),
                  module=MODULE, final_access_enabled=False, promotion_enabled=False, fitting_enabled=False)
    out.mkdir(parents=True, exist_ok=False)
    with storage.locked(out):
        storage.atomic_json(out / 'harness.json', metadata(root))
        storage.atomic_json(out / 'recipe.json', frozen)
        storage.append(out, [], TRIAL, 'queued', storage.file_hash(out / 'recipe.json'),
                       {'action': 'run', 'next_stage': STAGES[0]})
    return out


def authenticate_location(out, root=ROOT):
    out = storage.location(out, root)
    if json.loads((out / 'harness.json').read_text(encoding='utf-8')) != metadata(root):
        raise ValueError('Development harness policy or fixed candidate changed')
    return out


def authenticate(out, identity, root=ROOT):
    out = authenticate_location(out, root)
    path = _plain_file(out / 'recipe.json', root)
    if storage.file_hash(path) != identity:
        raise ValueError('Frozen development recipe changed')
    frozen = json.loads(path.read_text(encoding='utf-8'))
    expected = dict(schema_version=1, candidate_sha256=CANDIDATE_SHA256, source_hashes=sources(root),
                    input_hashes=inputs(root), development_matches=list(DEVELOPMENT), stages=list(STAGES),
                    module=MODULE, final_access_enabled=False, promotion_enabled=False, fitting_enabled=False)
    if frozen != expected:
        raise ValueError('Frozen sources, model bindings or development protocol changed')
    output = out / 'output'
    if output.resolve() != output:
        raise ValueError('Output must not resolve through a symlink/junction')
    return [sys.executable, '-m', MODULE, 'run', '--out', str(output), '--stage', 'next']


def summary_proof(out, stage_index, previous=None):
    output = Path(out) / 'output'
    path = _plain_file(output / 'summary.json', out)
    value = json.loads(path.read_text(encoding='utf-8'))
    expected = dict(schema_version=1, candidate_sha256=CANDIDATE_SHA256,
                    status='complete' if stage_index == len(STAGES) - 1 else 'in_progress',
                    stage_completed=STAGES[stage_index],
                    next_stage=STAGES[stage_index + 1] if stage_index + 1 < len(STAGES) else None,
                    external_only=True, final_accessed=False, promoted=False,
                    submission_eligible=False, final_access_authorized=False)
    if any(value.get(key) != field or (isinstance(field, bool) and value.get(key) is not field)
           for key, field in expected.items()):
        raise ValueError('Runner summary violates the fixed stage/candidate/development-only contract')
    protocol = _plain_file(output / 'protocol.json', out)
    body = json.loads(protocol.read_text(encoding='utf-8'))
    protocol_sha = body.pop('protocol_sha256', None)
    protocol_file_sha = storage.file_hash(protocol)
    if (protocol_sha != storage.digest(body) or value.get('protocol_sha256') != protocol_sha or
            value.get('protocol_file_sha256') != protocol_file_sha):
        raise ValueError('Runner protocol identity differs from its summary')
    if previous and (previous.get('protocol_sha256') != protocol_sha or
                     previous.get('protocol_file_sha256') != protocol_file_sha):
        raise ValueError('Runner changed its frozen protocol between stages')
    return dict(summary_sha256=storage.file_hash(path), protocol_sha256=protocol_sha,
                protocol_file_sha256=protocol_file_sha,
                stage_completed=value['stage_completed'], next_stage=value['next_stage'],
                status=value['status'], development_gate_passed=value.get('development_gate_passed'))


def _snapshot(out, events, **extra):
    return dict(storage.checkpoint(out, events), **extra)


def run(out, *, max_seconds=900, root=ROOT):
    if type(max_seconds) is not int or not 1 <= max_seconds <= 2700:
        raise ValueError('Stage budget must be an integer in 1..2700 seconds')
    root = Path(root).resolve()
    out = authenticate_location(out, root)
    deadline = time.monotonic() + max_seconds
    with storage.locked(out):
        events = storage.read_ledger(out)
        trials = storage.project(events)
        if set(trials) != {TRIAL}:
            raise ValueError('Only the single fixed development trial is allowed')
        trial = trials[TRIAL]
        identity = trial['recipe_sha256']
        if trial['state'] == 'running':
            storage.append(out, events, TRIAL, 'blocked', identity,
                           {'reason': 'Interrupted attempt consumed; review orphan processes before a new harness',
                            'requires_process_review': True})
            return _snapshot(out, events, blocked_reason='Interrupted process requires explicit review')
        if trial['state'] != 'queued':
            return _snapshot(out, events)
        stage_index = trial['attempts']
        try:
            if not 0 <= stage_index < len(STAGES):
                raise ValueError('Fixed development stage budget exhausted')
            command = authenticate(out, identity, root)
            previous = trial['detail'] if stage_index else None
            if stage_index:
                path = out / 'output/summary.json'
                if not path.is_file() or storage.file_hash(path) != previous['summary_sha256']:
                    raise ValueError('Completed-stage summary changed')
                summary_proof(out, stage_index - 1, previous)
            elif (out / 'output').exists():
                raise ValueError('Unexplained output consumes this fresh trial')
            log = out / f'console-{stage_index + 1:04d}.log'
            if log.exists():
                raise ValueError('Existing stage log consumes this attempt')
        except (ValueError, OSError, KeyError, TypeError) as error:
            storage.append(out, events, TRIAL, 'blocked', identity, {'reason': str(error)})
            return _snapshot(out, events)
        storage.append(out, events, TRIAL, 'running', identity,
                       {'command': command, 'pid': os.getpid(), 'stage': STAGES[stage_index]})
        try:
            code = storage.execute(command, root, log, max(.001, deadline - time.monotonic()))
            if code != 0:
                raise RuntimeError(f'Development runner exited with code {code}')
            authenticate(out, identity, root)
            proof = summary_proof(out, stage_index, previous)
            progress = proof['status'] == 'in_progress'
            storage.append(out, events, TRIAL, 'queued' if progress else 'completed', identity,
                           dict(exit_code=code, console_sha256=storage.file_hash(log),
                                successful_stage=progress, **proof))
        except (Exception, KeyboardInterrupt) as error:
            detail = {'reason': f'{type(error).__name__}: {error}'}
            if log.is_file():
                detail['console_sha256'] = storage.file_hash(log)
            storage.append(out, events, TRIAL, 'failed', identity, detail)
        return _snapshot(out, events)


def status(out, root=ROOT):
    out = authenticate_location(out, root)
    try:
        with storage.locked(out):
            events = storage.read_ledger(out)
            if set(storage.project(events)) != {TRIAL}:
                raise ValueError('Unexpected development trial inventory')
            return _snapshot(out, events, busy=False)
    except RuntimeError as error:
        if 'holds the lock' not in str(error):
            raise
        return dict(json.loads((out / 'checkpoint.json').read_text(encoding='utf-8')),
                    busy=True, snapshot_only=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('init', 'run', 'status'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-seconds', type=int, default=900)
    args = parser.parse_args()
    result = ({'initialized': str(initialize(args.out))} if args.action == 'init' else
              run(args.out, max_seconds=args.max_seconds) if args.action == 'run' else status(args.out))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()

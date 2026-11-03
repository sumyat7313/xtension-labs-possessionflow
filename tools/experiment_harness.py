"""Bounded, resumable external-only experiments; never promotes or opens final data.

The append-only hash-chained ledger is authoritative. checkpoint.json is an atomic
derived view, so a crash between ledger append and checkpoint replacement loses no
attempt. Interrupted attempts become blocked rather than silently being retried.
This is orchestration and provenance, not a sandbox or a scientific quality claim.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 1
MODULE = 'tools.external_rebuild'
ACTIONS = frozenset(('plan', 'preflight', 'run'))
TERMINAL = frozenset(('completed', 'failed', 'blocked'))
ZERO = '0' * 64
DATA_DIRECTORIES = ('data/external/idsse', '../more-than-a-hack-2027/data/external/idsse')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as stream:
        stream.write(canonical(value) + b'\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _inside(path, parent):
    path, parent = Path(path).resolve(), Path(parent).resolve()
    if not path.is_relative_to(parent) or path == parent:
        raise ValueError('Harness paths must remain inside their declared local directory')
    return path


def location(out, root=ROOT):
    """No writes to source, production, submission, data or historical run roots."""
    root = Path(root).resolve()
    return _inside(out, root / 'runs')


@contextmanager
def locked(out):
    """Kernel lock: persists during work, automatically released if owner crashes."""
    path = Path(out) / '.lock'
    stream = path.open('a+b')
    stream.seek(0, os.SEEK_END)
    if stream.tell() == 0:
        stream.write(b'0')
        stream.flush()
    stream.seek(0)
    acquired = False
    try:
        if os.name == 'nt':
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError('Another harness process holds the lock') from error
        else:
            import fcntl
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise RuntimeError('Another harness process holds the lock') from error
        acquired = True
        yield
    finally:
        if acquired:
            stream.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_UN)
        stream.close()


def scientific_sources(root=ROOT):
    """Bind the static local import closure, not unrelated experimental tools.

    Traverse imports even inside functions. Runtime-computed imports cannot be
    inferred by AST; this runner uses static imports. Third-party dependencies
    are pinned by the two requirements manifests rather than copied here.
    """
    root = Path(root).resolve()
    if not (root / 'tools/external_rebuild.py').is_file():
        raise ValueError('External-only runner is not installed')
    pending = [root / name for name in ('tools/external_rebuild.py', 'tools/experiment_harness.py',
                                       'tools/improvement_policy.py') if (root / name).is_file()]
    paths = set()

    def add_module(name):
        parts = name.split('.')
        if any(not part.isidentifier() for part in parts):
            return
        for end in range(1, len(parts) + 1):
            package = root.joinpath(*parts[:end]) / '__init__.py'
            if package.is_file() and package not in paths:
                pending.append(package)
        module = root.joinpath(*parts).with_suffix('.py')
        if module.is_file() and module not in paths:
            pending.append(module)

    while pending:
        path = pending.pop()
        if path in paths:
            continue
        if path.resolve() != path or not path.is_relative_to(root):
            raise ValueError('Scientific source escapes repository or is a symlink')
        paths.add(path)
        relative = path.relative_to(root).with_suffix('')
        package = list(relative.parts[:-1])
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    add_module(alias.name)
            elif isinstance(node, ast.ImportFrom):
                base = (package[:len(package) - node.level + 1] if node.level else [])
                if node.module:
                    base.extend(node.module.split('.'))
                if base:
                    add_module('.'.join(base))
                    for alias in node.names:
                        if alias.name != '*':
                            add_module('.'.join([*base, alias.name]))
    for name in ('requirements-solution.txt', 'requirements-training.txt'):
        if (root / name).is_file():
            paths.add(root / name)
    return {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(paths)}


def data_directory(root, relative=None):
    """Only the two registered Sportec locations, never caller-supplied data."""
    root = Path(root).resolve()
    if relative is None:
        relative = next((name for name in DATA_DIRECTORIES if (root / name).is_dir()), DATA_DIRECTORIES[0])
    if relative not in DATA_DIRECTORIES:
        raise ValueError('Only registered external Sportec data locations are allowed')
    nominal = Path(os.path.abspath(root / relative))
    actual = nominal.resolve()
    # Reject junctions/symlinks even when they stay inside the containing workspace.
    if actual != nominal or not actual.is_relative_to(root.parent):
        raise ValueError('Registered data path resolves through a symlink/junction or escapes the workspace')
    return relative, actual


def validate_spec(spec):
    if not isinstance(spec, dict) or set(spec) != {'schema_version', 'trial_id', 'action', 'parameters'}:
        raise ValueError('Trial spec requires only schema_version, trial_id, action, parameters')
    if type(spec['schema_version']) is not int or spec['schema_version'] != SCHEMA:
        raise ValueError('Unknown trial schema')
    if not isinstance(spec['trial_id'], str) or not re.fullmatch(r'[a-z][a-z0-9-]{0,63}', spec['trial_id']):
        raise ValueError('Trial ID must be a safe lowercase slug')
    if spec['action'] not in ACTIONS:
        raise ValueError('Only external rebuild plan/preflight/run actions are allowed')
    params = spec['parameters']
    expected = {'schema_version', 'seed', 'epochs', 'control_l2', 'transition_weight'}
    if not isinstance(params, dict) or set(params) != expected:
        raise ValueError('Only external training parameters are allowed')
    if type(params['schema_version']) is not int or params['schema_version'] != 1:
        raise ValueError('Unknown parameter schema')
    if type(params['seed']) is not int or not 0 <= params['seed'] < 2 ** 31:
        raise ValueError('Seed must be a nonnegative 31-bit integer')
    if type(params['epochs']) is not int or params['epochs'] not in (4, 8, 12):
        raise ValueError('Epoch budget must be 4, 8 or 12')
    for name, allowed in (('control_l2', (.001, .01)), ('transition_weight', (1, 1.5, 2))):
        value = params[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value not in allowed:
            raise ValueError(f'Invalid bounded parameter: {name}')
    return spec


def read_ledger(out):
    path = Path(out) / 'ledger.jsonl'
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b'\n'):
        raise ValueError('Incomplete ledger write: preserve it for explicit recovery')
    events, previous = [], ZERO
    for number, line in enumerate(raw.splitlines(), 1):
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError('Invalid ledger entry')
        claimed = row.get('event_sha256')
        payload = {key: val for key, val in row.items() if key != 'event_sha256'}
        if payload.get('sequence') != number or payload.get('previous_sha256') != previous or digest(payload) != claimed:
            raise ValueError('Ledger integrity failure')
        events.append(row)
        previous = claimed
    return events


def project(events):
    trials = {}
    for event in events:
        tid, state = event['trial_id'], event['state']
        if state == 'queued':
            if tid in trials:
                before = trials[tid]
                if (before['state'] != 'running' or event['recipe_sha256'] != before['recipe_sha256'] or
                        event['detail'].get('successful_stage') is not True or
                        not re.fullmatch('[0-9a-f]{64}', event['detail'].get('summary_sha256', ''))):
                    raise ValueError('Only authenticated successful stages may requeue a trial')
                before['state'] = state
            else:
                if event['detail'].get('successful_stage'):
                    raise ValueError('Continuation cannot initialize a new trial')
                trials[tid] = dict(trial_id=tid, state=state, recipe_sha256=event['recipe_sha256'], attempts=0)
        else:
            if tid not in trials or event['recipe_sha256'] != trials[tid]['recipe_sha256']:
                raise ValueError('Unknown or changed trial identity')
            before = trials[tid]['state']
            if not ((before == 'queued' and state in ('running', 'blocked')) or
                    (before == 'running' and state in TERMINAL)):
                raise ValueError('Invalid or repeated trial transition')
            trials[tid].update(state=state)
            if state == 'running':
                trials[tid]['attempts'] += 1
        trials[tid].update(last_sequence=event['sequence'], updated_at=event['at'], detail=event['detail'])
    return trials


def append(out, events, tid, state, recipe_sha256, detail):
    row = dict(sequence=len(events) + 1, previous_sha256=events[-1]['event_sha256'] if events else ZERO,
               at=now(), trial_id=tid, state=state, recipe_sha256=recipe_sha256, detail=detail)
    row['event_sha256'] = digest(row)
    project([*events, row])
    with (Path(out) / 'ledger.jsonl').open('ab') as stream:
        stream.write(canonical(row) + b'\n')
        stream.flush()
        os.fsync(stream.fileno())
    events.append(row)
    checkpoint(out, events)


def checkpoint(out, events):
    result = dict(schema_version=SCHEMA, ledger_sha256=events[-1]['event_sha256'] if events else ZERO,
                  trials=project(events), final_accessed=False, promoted=False, submission_eligible=False,
                  meaning='Completed means command execution and artifact checks, not a scientific win.')
    atomic_json(Path(out) / 'checkpoint.json', result)
    return result


def initialize(out, root=ROOT):
    out = location(out, root)
    out.mkdir(parents=True, exist_ok=False)
    with locked(out):
        atomic_json(out / 'harness.json', dict(schema_version=SCHEMA, root=str(Path(root).resolve()),
                    allowed_module=MODULE, final_access_enabled=False, promotion_enabled=False, created_at=now()))
        (out / 'trials').mkdir()
        checkpoint(out, [])
    return out


def authenticate_harness(out, root=ROOT):
    out = location(out, root)
    meta = json.loads((out / 'harness.json').read_text(encoding='utf-8'))
    expected = dict(schema_version=SCHEMA, root=str(Path(root).resolve()), allowed_module=MODULE,
                    final_access_enabled=False, promotion_enabled=False)
    if any(meta.get(k) != v for k, v in expected.items()) or set(meta) != {*expected, 'created_at'}:
        raise ValueError('Harness policy or repository identity changed')
    return out


def enqueue(out, spec, root=ROOT):
    out = authenticate_harness(out, root)
    validate_spec(spec)
    with locked(out):
        events = read_ledger(out)
        if spec['trial_id'] in project(events):
            raise ValueError('Trial is already consumed or queued; IDs cannot be reused')
        folder = _inside(out / 'trials' / spec['trial_id'], out / 'trials')
        folder.mkdir(exist_ok=False)
        parameters_path = folder / 'parameters.json'
        atomic_json(parameters_path, spec['parameters'])
        recipe = dict(schema_version=SCHEMA, spec=spec, source_hashes=scientific_sources(root),
                      parameters_sha256=file_hash(parameters_path), module=MODULE,
                      data_directory=data_directory(root)[0], final_access_enabled=False, promotion_enabled=False)
        atomic_json(folder / 'recipe.json', recipe)
        identity = file_hash(folder / 'recipe.json')
        append(out, events, spec['trial_id'], 'queued', identity, {'action': spec['action']})
    return identity


def authenticate_trial(out, tid, identity, root=ROOT):
    folder = _inside(Path(out) / 'trials' / tid, Path(out) / 'trials')
    path = folder / 'recipe.json'
    if file_hash(path) != identity:
        raise ValueError('Frozen recipe changed')
    recipe = json.loads(path.read_text(encoding='utf-8'))
    spec = validate_spec(recipe['spec'])
    if (spec['trial_id'] != tid or recipe['module'] != MODULE or
            recipe['data_directory'] not in DATA_DIRECTORIES or
            recipe['final_access_enabled'] is not False or recipe['promotion_enabled'] is not False):
        raise ValueError('Forbidden recipe behavior')
    parameters = folder / 'parameters.json'
    if file_hash(parameters) != recipe['parameters_sha256']:
        raise ValueError('Frozen parameters changed')
    if json.loads(parameters.read_text(encoding='utf-8')) != spec['parameters']:
        raise ValueError('Parameter manifest mismatch')
    if scientific_sources(root) != recipe['source_hashes']:
        raise ValueError('Scientific source changed; create a new declared trial')
    # Resolve every existing parent to reject junction/symlink escape.
    _inside(parameters, out)
    _inside(folder / 'output', folder)
    _, data = data_directory(root, recipe['data_directory'])
    command = [sys.executable, '-m', MODULE, spec['action'], '--out', str(folder / 'output'),
               '--data-dir', str(data), '--recipe', str(parameters)]
    if spec['action'] == 'run':
        command.extend(('--stage', 'next'))
    return recipe, command, folder


def stop_process_tree(process):
    if os.name == 'nt':
        subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       creationflags=subprocess.CREATE_NO_WINDOW)
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait(timeout=15)


def execute(command, root, log, seconds):
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
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            stop_process_tree(process)
            raise


def verify_result(recipe, folder):
    if recipe['spec']['action'] != 'run':
        return {}
    path = folder / 'output/summary.json'
    result = json.loads(path.read_text(encoding='utf-8'))
    if result.get('external_only') is not True or result.get('final_accessed') is not False or result.get('promoted') is not False:
        raise ValueError('Runner did not preserve external-only/no-final/no-promotion contract')
    if result.get('submission_eligible') is not False:
        raise ValueError('Harness cannot clear submission eligibility')
    if result.get('status') not in ('in_progress', 'complete'):
        raise ValueError('Runner must explicitly distinguish stage completion from experiment completion')
    return dict(summary_sha256=file_hash(path), development_gate_passed=result.get('development_gate_passed'),
                status=result['status'], stage_completed=result.get('stage_completed'), next_stage=result.get('next_stage'))


def run(out, *, max_trials=1, max_seconds=2700, trial_seconds=1800, root=ROOT):
    if any(type(x) is not int or x < 1 for x in (max_trials, max_seconds, trial_seconds)):
        raise ValueError('Iteration and time budgets must be positive integers')
    if max_trials > 32 or max_seconds > 86400 or trial_seconds > 86400:
        raise ValueError('Bounded run exceeds maximum budget')
    out = authenticate_harness(out, root)
    deadline = time.monotonic() + max_seconds
    with locked(out):
        events = read_ledger(out)
        # A previous owner disappeared; no references/fit may be silently replayed.
        for tid, trial in list(project(events).items()):
            if trial['state'] == 'running':
                append(out, events, tid, 'blocked', trial['recipe_sha256'],
                       {'reason': 'Interrupted attempt consumed; orphan process review and a new harness required',
                        'requires_process_review': True})
        # A crashed parent can leave a training child alive. Do not launch ANY
        # further work in this harness until a reviewer checks processes and
        # deliberately creates a separate new harness, preserving this ledger.
        if any(trial['detail'].get('requires_process_review') for trial in project(events).values()):
            return dict(checkpoint(out, events), blocked_reason='Interrupted process requires explicit orphan-process review')
        attempted = 0
        for tid, trial in list(project(events).items()):
            if trial['state'] != 'queued' or attempted >= max_trials or time.monotonic() >= deadline:
                continue
            attempted += 1
            identity = trial['recipe_sha256']
            try:
                recipe, command, folder = authenticate_trial(out, tid, identity, root)
            except (ValueError, OSError, KeyError, TypeError) as error:
                append(out, events, tid, 'blocked', identity, {'reason': str(error)})
                continue
            attempt = trial['attempts'] + 1
            log = folder / f'console-{attempt:04d}.log'
            if trial['detail'].get('successful_stage'):
                summary = folder / 'output/summary.json'
                if not summary.is_file() or file_hash(summary) != trial['detail']['summary_sha256']:
                    append(out, events, tid, 'blocked', identity, {'reason': 'Completed-stage summary changed'})
                    continue
            elif (folder / 'output').exists():
                append(out, events, tid, 'blocked', identity, {'reason': 'Existing output consumes this trial'})
                continue
            if log.exists():
                append(out, events, tid, 'blocked', identity, {'reason': 'Existing attempt log consumes this trial'})
                continue
            append(out, events, tid, 'running', identity, {'command': command, 'pid': os.getpid()})
            try:
                seconds = min(trial_seconds, max(.001, deadline - time.monotonic()))
                code = execute(command, root, log, seconds)
                if code != 0:
                    raise RuntimeError(f'Runner exited with code {code}')
                # Detect source/recipe changes while a long-running child was active.
                authenticate_trial(out, tid, identity, root)
                proof = verify_result(recipe, folder)
                progress = proof.get('status') == 'in_progress'
                if recipe['spec']['action'] == 'run':
                    completed_stage = proof.get('stage_completed')
                    previous_stages = [event['detail'].get('stage_completed') for event in events
                                       if event['trial_id'] == tid and event['detail'].get('successful_stage')]
                    if not isinstance(completed_stage, str) or not completed_stage or completed_stage in previous_stages:
                        raise ValueError('Stage progress is missing or repeats an already consumed stage')
                    if progress and (not isinstance(proof.get('next_stage'), str) or
                                     not proof['next_stage'] or proof['next_stage'] == completed_stage):
                        raise ValueError('Incomplete study must identify a distinct next stage')
                    if trial['detail'].get('successful_stage') and completed_stage != trial['detail']['next_stage']:
                        raise ValueError('Runner did not execute the authenticated next stage')
                append(out, events, tid, 'queued' if progress else 'completed', identity,
                       dict(exit_code=code, console_sha256=file_hash(log), successful_stage=progress, **proof))
            except (Exception, KeyboardInterrupt) as error:
                detail = {'reason': f'{type(error).__name__}: {error}'}
                if log.is_file():
                    detail['console_sha256'] = file_hash(log)
                append(out, events, tid, 'failed', identity, detail)
                if isinstance(error, KeyboardInterrupt):
                    break
        return checkpoint(out, events)


def status(out, root=ROOT):
    out = authenticate_harness(out, root)
    try:
        with locked(out):
            return dict(checkpoint(out, read_ledger(out)), busy=False)
    except RuntimeError as error:
        if 'holds the lock' not in str(error):
            raise
        # Atomic checkpoint remains readable while a long-running stage owns the
        # lock. It is explicitly a snapshot, not a fresh ledger authentication.
        value = json.loads((out / 'checkpoint.json').read_text(encoding='utf-8'))
        return dict(value, busy=True, snapshot_only=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('init', 'enqueue', 'run', 'status'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--recipe', type=Path)
    parser.add_argument('--max-trials', type=int, default=1)
    parser.add_argument('--max-seconds', type=int, default=2700)
    parser.add_argument('--trial-seconds', type=int, default=1800)
    args = parser.parse_args()
    if args.action == 'init':
        result = {'initialized': str(initialize(args.out))}
    elif args.action == 'enqueue':
        if args.recipe is None:
            parser.error('enqueue requires --recipe')
        result = {'recipe_sha256': enqueue(args.out, json.loads(args.recipe.read_text(encoding='utf-8')))}
    elif args.action == 'run':
        result = run(args.out, max_trials=args.max_trials, max_seconds=args.max_seconds, trial_seconds=args.trial_seconds)
    else:
        result = status(args.out)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()

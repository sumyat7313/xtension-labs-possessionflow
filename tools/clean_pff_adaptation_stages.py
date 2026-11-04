"""Locked inner-stage bookkeeping; no fitting, data loader, CLI or execution authority.

This store authenticates its files and source snapshot, not provider rights,
artifact semantics or independent chronology. Resource limits are declarations
for the future dispatcher; no job is launched and no timeout is enforced here.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
import copy
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import re
import sys

from tools import clean_pff_adaptation_protocol as protocol
from tools import experiment_harness as storage

ROOT = Path(__file__).resolve().parents[1]
OUT_NAME = 'clean-pff-adaptation-inner-v1'
SNAPSHOT = 'models/clean-pff-adaptation-protocol.json'
SNAPSHOT_FILE_SHA256 = 'bbc6f5bc3c650a50112cfeeecde851643b028929700d9e3e66874acc0ce85c3c'
SNAPSHOT_PROTOCOL_SHA256 = '4df9683901ef0eafa08e5344a43870558b09173ac09fc1c3377514ec363cce53'
CORE_SHA256 = 'a1271ea8c6b95848d1379f5770fc4c1788c82d2e4472b7123edcbeb4fe66e02f'
SOURCE_ROOTS = ('tools/clean_pff_adaptation_stages.py', 'tools/clean_pff_adaptation_inputs.py',
                'tools/clean_pff_adaptation_fit.py', 'tools/clean_pff_adaptation_protocol.py')
STAGES = ('initialize', 'fit-A', 'fit-B',
          *(f'predict-{c}-{m}' for c in protocol.CANDIDATES for m in protocol.INNER_VALIDATION_IDS),
          'freeze', *(f'score-{m}' for m in protocol.INNER_VALIDATION_IDS), 'select')
LIMITS = dict(max_heavy_fits=1, max_data_workers=2, blas_threads_per_worker=1,
              torch_threads_per_worker=1, stage_timeout_seconds=1800,
              max_stages_per_advance=1, max_seconds_per_wakeup=2700)
FLAGS = dict(protocol.FLAGS, lifecycle_only=True, resource_limits_enforced=False,
             provider_records_accessed_by_store=False, caller_access_not_attested=True)
ZERO = '0'*64


def _need(value, message):
    if not value:
        raise ValueError(message)


def _hash(value):
    _need(type(value) is str and re.fullmatch('[0-9a-f]{64}', value), 'Invalid SHA-256 identity')
    return value


def _no_links(path):
    path = Path(os.path.abspath(path))
    _need(path.resolve() == path, 'Symlink or junction paths are forbidden')
    for item in (path, *path.parents):
        _need(not item.is_symlink() and not (hasattr(item, 'is_junction') and item.is_junction()),
              'Symlink or junction paths are forbidden')
    return path


def location(out, root=ROOT):
    root = _no_links(root)
    path = _no_links(out)
    _need(path == root/'runs'/OUT_NAME, 'Only the reserved adaptation inner-study directory is allowed')
    return path


def _read(path):
    return json.loads(_no_links(path).read_text(encoding='utf-8'))


def _exclusive(path, value):
    with _no_links(path).open('xb') as stream:
        stream.write(storage.canonical(value)+b'\n')
        stream.flush(); os.fsync(stream.fileno())


def verify_snapshot(root=ROOT):
    root = _no_links(root)
    path = _no_links(root/SNAPSHOT)
    _need(storage.file_hash(path) == SNAPSHOT_FILE_SHA256, 'Frozen protocol snapshot bytes changed')
    value = protocol.verify_protocol(_read(path))
    _need(value['protocol_sha256'] == SNAPSHOT_PROTOCOL_SHA256 and
          value['core_protocol_sha256'] == CORE_SHA256 and
          protocol.fixed_protocol_sha256() == CORE_SHA256, 'Scientific protocol identity changed')
    for name, expected in value['source_hashes'].items():
        _need(storage.file_hash(_no_links(root/name)) == expected, 'Frozen scientific source changed')
    return value


def source_hashes(root=ROOT):
    """Static import closure, including imports inside functions and both bridges."""
    root = _no_links(root)
    snapshot = verify_snapshot(root)
    pending = [root/name for name in SOURCE_ROOTS]
    seen = set()
    def add_module(name):
        parts = name.split('.')
        if any(not part.isidentifier() for part in parts): return
        for end in range(1, len(parts)+1):
            path = root.joinpath(*parts[:end])/'__init__.py'
            if path.is_file() and path not in seen: pending.append(path)
        path = root.joinpath(*parts).with_suffix('.py')
        if path.is_file() and path not in seen: pending.append(path)
    while pending:
        path = pending.pop()
        if path in seen: continue
        _no_links(path)
        _need(path.is_relative_to(root) and path.is_file(), 'Execution source missing or outside repository')
        seen.add(path)
        package = list(path.relative_to(root).with_suffix('').parts[:-1])
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            if isinstance(node, ast.Import):
                for alias in node.names: add_module(alias.name)
            elif isinstance(node, ast.ImportFrom):
                base = package[:len(package)-node.level+1] if node.level else []
                if node.module: base.extend(node.module.split('.'))
                if base:
                    add_module('.'.join(base))
                    for alias in node.names:
                        if alias.name != '*': add_module('.'.join([*base, alias.name]))
    seen.update(root/name for name in snapshot['source_hashes'])
    seen.update(root/name for name in ('requirements-solution.txt', 'requirements-training.txt', SNAPSHOT))
    return {path.relative_to(root).as_posix(): storage.file_hash(_no_links(path)) for path in sorted(seen)}


def runtime_versions():
    result = {'python': sys.version.split()[0]}
    for name in ('numpy', 'scipy', 'torch', 'pandas'):
        try: result[name] = metadata.version(name)
        except metadata.PackageNotFoundError: result[name] = None
    return result


def descriptor(root=ROOT):
    value = dict(schema_version=1, family='clean-pff-adaptation-inner-v1',
                 protocol_sha256=SNAPSHOT_PROTOCOL_SHA256, core_protocol_sha256=CORE_SHA256,
                 source_hashes=source_hashes(root), runtime_versions=runtime_versions(),
                 stages=list(STAGES), limits=dict(LIMITS), **FLAGS)
    return dict(value, descriptor_sha256=storage.digest(value))


def verify_descriptor(value, root=ROOT):
    expected = descriptor(root)
    _need(storage.canonical(value) == storage.canonical(expected), 'Execution descriptor/source/runtime changed')
    return value


def _marker(out, index):
    return out/'attempts'/f'{index:03d}.json'


def _marker_value(index, manifest, tip):
    return dict(schema_version=1, stage_index=index, stage=STAGES[index],
                descriptor_sha256=manifest['descriptor_sha256'], previous_sha256=tip)


def _load(out, root):
    manifest = verify_descriptor(_read(out/'manifest.json'), root)
    ledger = _no_links(out/'ledger.jsonl').read_bytes()
    _need(not ledger or ledger.endswith(b'\n'), 'Partial ledger write consumes the study')
    events, tip, index, active, blocked = [], ZERO, 0, False, False
    expected_markers = {}
    for line in ledger.splitlines():
        event = json.loads(line)
        fields = {'sequence', 'stage_index', 'stage', 'state', 'descriptor_sha256',
                  'previous_sha256', 'detail', 'event_sha256'}
        _need(isinstance(event, dict) and set(event) == fields, 'Ledger event schema differs')
        _need(index < len(STAGES) and type(event['sequence']) is int and event['sequence'] == len(events) and
              type(event['stage_index']) is int and event['stage_index'] == index and
              event['stage'] == STAGES[index] and event['descriptor_sha256'] == manifest['descriptor_sha256'] and
              event['previous_sha256'] == tip, 'Ledger order or identity differs')
        body = {key: val for key,val in event.items() if key != 'event_sha256'}
        _need(event['event_sha256'] == storage.digest(body), 'Ledger hash chain differs')
        state, detail = event['state'], event['detail']
        _need(not blocked, 'No event may follow a consumed failure')
        if state == 'started':
            _need(not active and detail == {}, 'Duplicate or malformed start')
            expected_markers[index] = _marker_value(index, manifest, tip)
            active = True
        elif state == 'recorded_complete':
            _need(active, 'Completion without start')
            _receipt(detail)
            active = False; index += 1
        elif state == 'failed':
            _need(active and isinstance(detail, dict) and set(detail) == {'exception_type'} and
                  type(detail['exception_type']) is str and re.fullmatch('[A-Za-z][A-Za-z0-9_]{0,79}', detail['exception_type']),
                  'Invalid failed-attempt record')
            blocked = True
        else:
            raise ValueError('Unknown lifecycle state')
        events.append(event); tip = event['event_sha256']
    actual = {_no_links(p).name for p in _no_links(out/'attempts').iterdir()}
    expected = {f'{i:03d}.json' for i in expected_markers}
    for i, value in expected_markers.items():
        _need(storage.canonical(_read(_marker(out, i))) == storage.canonical(value), 'Attempt marker differs')
    # Marker is created before started append. A crash in that gap is consumed.
    if actual != expected:
        _need(index < len(STAGES) and not active and not blocked and actual == expected | {f'{index:03d}.json'},
              'Unexpected attempt marker or missing history')
        _need(storage.canonical(_read(_marker(out,index))) == storage.canonical(_marker_value(index,manifest,tip)), 'Orphan attempt marker differs')
        blocked = True
    return manifest, events, ledger, index, bool(active or blocked)


def _append(out, events, original_bytes, stage_index, state, detail, identity):
    path = _no_links(out/'ledger.jsonl')
    _need(path.read_bytes() == original_bytes, 'Ledger changed during held attempt')
    event = dict(sequence=len(events), stage_index=stage_index, stage=STAGES[stage_index], state=state,
                 descriptor_sha256=identity,
                 previous_sha256=events[-1]['event_sha256'] if events else ZERO, detail=detail)
    event['event_sha256'] = storage.digest(event)
    data = storage.canonical(event)+b'\n'
    with path.open('ab') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    events.append(event)
    return original_bytes+data


def _receipt(value):
    _need(isinstance(value, dict) and set(value) == {'artifact_sha256', 'execution_verified'},
          'Only opaque artifact-identity receipts are supported')
    _need(value['execution_verified'] is False and isinstance(value['artifact_sha256'], dict) and value['artifact_sha256'],
          'This lifecycle store cannot verify execution')
    for name, digest in value['artifact_sha256'].items():
        _need(type(name) is str and re.fullmatch('[a-z][a-z0-9_]{0,63}', name), 'Invalid artifact name')
        _hash(digest)
    return value


class _Attempt:
    def __init__(self, stage):
        self.stage, self._active, self._receipt = stage, True, None

    def complete(self, receipt):
        """Record caller-declared identities; never grants execution permission."""
        _need(self._active and self._receipt is None, 'Attempt handle is closed or already completed')
        self._receipt = copy.deepcopy(_receipt(receipt))


def initialize(out, root=ROOT):
    out = location(out, root)
    _need(out.parent.is_dir(), 'Local runs directory must already exist')
    out.mkdir(exist_ok=True)
    with storage.locked(out):
        _need(set(p.name for p in out.iterdir()) <= {'.lock'}, 'Existing study cannot be reset')
        manifest = descriptor(root)
        _exclusive(out/'manifest.json', manifest)
        (out/'attempts').mkdir()
        with (out/'ledger.jsonl').open('xb') as stream:
            stream.flush(); os.fsync(stream.fileno())
    return dict(descriptor_sha256=manifest['descriptor_sha256'], next_stage=STAGES[0], **FLAGS)


def status(out, root=ROOT):
    out = location(out, root)
    with storage.locked(out):
        manifest, events, _, index, blocked = _load(out, root)
        return dict(descriptor_sha256=manifest['descriptor_sha256'], recorded_completed_stages=index,
                    blocked=blocked, complete=index == len(STAGES),
                    next_stage=None if blocked or index == len(STAGES) else STAGES[index],
                    ledger_tip=events[-1]['event_sha256'] if events else ZERO, **FLAGS)


@contextmanager
def attempt(out, stage, root=ROOT):
    """Reserve one ordered attempt before yielding; hold the lock until terminal.

    The body belongs to a future separately authenticated dispatcher. This API
    cannot identify what the caller actually does or enforce CPU/time bounds.
    Exceptions, missing completion and interrupted markers consume the attempt.
    """
    _need(type(stage) is str and stage in STAGES, 'Unknown or forbidden stage')
    out = location(out, root)
    with storage.locked(out):
        manifest, events, ledger, index, blocked = _load(out, root)
        _need(not blocked and index < len(STAGES) and STAGES[index] == stage, 'Attempt consumed or stage out of order')
        marker = _marker_value(index,manifest,events[-1]['event_sha256'] if events else ZERO)
        _exclusive(_marker(out,index),marker)
        ledger = _append(out,events,ledger,index,'started',{},manifest['descriptor_sha256'])
        handle = _Attempt(stage)
        try:
            yield handle
            _need(handle._receipt is not None, 'No completion receipt; attempt consumed')
            verify_descriptor(manifest,root)
            _need(storage.canonical(_read(out/'manifest.json')) == storage.canonical(manifest), 'Manifest changed during attempt')
            _append(out,events,ledger,index,'recorded_complete',handle._receipt,manifest['descriptor_sha256'])
        except BaseException as error:
            _append(out,events,ledger,index,'failed',{'exception_type':type(error).__name__},manifest['descriptor_sha256'])
            raise
        finally:
            handle._active = False

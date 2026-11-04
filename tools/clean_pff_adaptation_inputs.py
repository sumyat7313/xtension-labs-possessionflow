"""Authenticated, lazy inner-fitting inputs; no fit, stage or scoring command.

Only the pinned inner32 PFF and seven clean Sportec caches can be requested.
This bridge authenticates data identity, not dataset rights or stage permission.
Keep every returned array, reference and model in local ignored storage.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np

from tools import clean_pff_adaptation_fit as kernel
from tools import clean_pff_adaptation_protocol as protocol
from tools import external_pff_development as dev
from tools import external_rebuild as rebuild
from tools import improvement_policy as policy
from tools import pff_campaign as prior

ROOT = Path(__file__).resolve().parents[1]
ADAPTATION_SNAPSHOT_SHA256 = 'bbc6f5bc3c650a50112cfeeecde851643b028929700d9e3e66874acc0ce85c3c'
PFF_CACHE = 'runs/pff-campaign-v3/cache'
REBUILD = 'runs/continuous-improvement/trials/external-rebuild-001/output'
PFF_ITEM_KEYS = {'match', 'blocks', 'fps', 'usable', 'pitch', 'reference_fields_used_for_inference'}
SPORTEC_FILES = {'metadata.json', 'reference.npy', *[
    f'{regime}-{suffix}' for regime in ('clean', 'degraded', 'camera')
    for suffix in ('x.npy', 'z.npy', 'support.json')]}


def _path(relative):
    if (type(relative) is not str or not relative or '\\' in relative or ':' in relative or
            PurePosixPath(relative).is_absolute() or '..' in PurePosixPath(relative).parts):
        raise ValueError('Only fixed repository-relative artifact paths are accepted')
    path = ROOT / relative
    rebuild.no_links(path)
    if not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('Artifact escapes repository')
    return path


def _bytes(relative, expected):
    if not protocol.original._sha(expected):
        raise ValueError('Expected pinned SHA-256 identity')
    value = _path(relative).read_bytes()
    if hashlib.sha256(value).hexdigest() != expected:
        raise ValueError('Artifact bytes differ: ' + PurePosixPath(relative).name)
    return value


def _json(relative, expected):
    # Parse only the exact bytes just authenticated, not a second path read.
    return json.loads(_bytes(relative, expected))


def _hash(relative, expected):
    path = _path(relative)
    if not protocol.original._sha(expected) or dev.sha256(path) != expected:
        raise ValueError('Artifact bytes differ: ' + path.name)


def _array(relative, expected):
    """Authenticate the opened descriptor before and after NumPy materialization."""
    if not protocol.original._sha(expected):
        raise ValueError('Expected pinned array identity')
    with _path(relative).open('rb') as stream:
        def identity():
            stream.seek(0)
            digest = hashlib.sha256()
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
            return digest.hexdigest()
        if identity() != expected:
            raise ValueError('Array bytes differ before materialization')
        stream.seek(0)
        value = np.load(stream, allow_pickle=False)
        if not isinstance(value, np.ndarray) or identity() != expected:
            raise ValueError('Array bytes changed during materialization')
    return value


def _sources(source_hashes):
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError('Source identities are required')
    for relative, expected in source_hashes.items():
        _hash(relative, expected)


def _evidence():
    """Read pinned aggregates and model ancestry; no feature or label arrays."""
    snapshot = _json('models/clean-pff-adaptation-protocol.json', ADAPTATION_SNAPSHOT_SHA256)
    protocol.verify_protocol(snapshot)
    _sources(snapshot['source_hashes'])
    old = _json('models/pff-research.json', dev.PARENT_BUNDLE_SHA)
    clean = _json('models/external-rebuild-001-research.json', dev.REBUILD_BUNDLE_SHA)
    # This helper uses only published aggregates, source identities and the exact
    # old external v1.2 ancestor; unlike dev.context it never visits dev caches.
    ctx = dev.portable_context(old, clean)
    if (ctx['reservation']['protocol_sha256'] != protocol.RESERVATION_SHA256 or
            tuple(ctx['reservation']['split']['train']) != protocol.TRAIN_IDS):
        raise ValueError('Original training reservation differs')
    _sources(clean['protocol']['source_hashes'])
    for name, expected in [('refit/candidate.json', protocol.CLEAN_SHA256),
                           ('refit/config.json', protocol.CONFIG_SHA256),
                           ('refit/lineage.json', dev.LINEAGE_FILE_SHA),
                           ('protocol.json', dev.REBUILD_PROTOCOL_SHA),
                           ('summary.json', dev.REBUILD_SUMMARY_SHA)]:
        _hash(f'{REBUILD}/{name}', expected)
    lineage = _json(f'{REBUILD}/refit/lineage.json', dev.LINEAGE_FILE_SHA)
    if lineage != ctx['lineage']:
        raise ValueError('Actual clean lineage differs from published evidence')
    audited = policy.audit_lineage(lineage)
    if (audited['candidate_sha256'] != protocol.CLEAN_SHA256 or
            set(ctx['model_paths']) != set(audited['model_sha256s'])):
        raise ValueError('Clean lineage bindings are incomplete')
    for expected, relative in ctx['model_paths'].items():
        _hash(relative, expected)
    parent = _json(f'{REBUILD}/refit/candidate.json', protocol.CLEAN_SHA256)
    kernel.validate_parent(parent)
    manifests = [m for m in ctx['cache_manifests'] if m['match_id'] in protocol.INNER_TRAIN_IDS]
    pff = {f'pff{m["match_id"]}': m for m in manifests}
    if len(manifests) != 32 or set(pff) != {f'pff{m}' for m in protocol.INNER_TRAIN_IDS}:
        raise ValueError('Pinned inner PFF cache manifest roster differs')
    producer_sources = prior.sources()
    for m in manifests:
        if m['identity']['sources'] != producer_sources:
            raise ValueError('PFF observation producer sources changed')
    seals = {mid: clean['completed_stage_seals'][f'prepare-{mid}'] for mid in protocol.SPORTEC_IDS}
    # Source maps are checked again on each lazy call to catch a changed producer.
    return dict(snapshot=snapshot, parent=parent, pff=pff, sportec_seals=seals,
                sportec_inputs=clean['protocol']['inputs'],
                producer_sources=producer_sources, clean_sources=clean['protocol']['source_hashes'])


def _require_fixture(match):
    if type(match) is not str or match not in protocol.fit_ids('inner'):
        raise ValueError('Only the exact inner-fitting fixtures may be opened')
    return match


def _period_metadata(blocks, usable, fps, n):
    if type(fps) not in (int, float) or fps != 25:
        raise ValueError('Fixed 25 Hz observation timeline required')
    checked = kernel._blocks(blocks, n)
    if (not isinstance(usable, list) or len(usable) != len(checked) or
            any(type(v) is not bool for v in usable)):
        raise ValueError('Per-period observation support differs')
    return checked


def _validate_record(record):
    x, z, y = record['x'], record['z'], record['y']
    if (not isinstance(x, np.ndarray) or not isinstance(z, np.ndarray) or not isinstance(y, np.ndarray) or
            x.ndim != 2 or x.shape[1] != 98 or z.shape != (len(x), 58) or y.shape != (len(x),) or
            not np.isfinite(x).all() or not np.isfinite(z).all() or not np.isin(y, list('HADX')).all()):
        raise ValueError('Authenticated fitting arrays have invalid schema or values')
    kernel._blocks(record['blocks'], len(x))
    return record


def _validate_features(x, z, n):
    if (not isinstance(x, np.ndarray) or not isinstance(z, np.ndarray) or
            x.shape != (n, 98) or z.shape != (n, 58) or
            not np.isfinite(x).all() or not np.isfinite(z).all()):
        raise ValueError('Fitting observations have invalid schema or values')


def _pff_record(match, ctx):
    _require_fixture(match)
    manifest = ctx['pff'][match]
    mid = int(match[3:]); folder = f'{PFF_CACHE}/{mid}'
    # Accepted serializations are generated from the pinned published manifest.
    raw = _path(f'{folder}/complete.json').read_bytes()
    if (hashlib.sha256(raw).hexdigest() not in dev.old_verifier._serialized_hashes(manifest) or
            json.loads(raw) != manifest):
        raise ValueError('PFF cache manifest differs from pinned aggregate evidence')
    item = _json(f'{folder}/item.json', manifest['files']['item.json'])
    if (set(item) != PFF_ITEM_KEYS or item['match'] != match or
            item['reference_fields_used_for_inference'] is not False):
        raise ValueError('Native observation-only PFF metadata required')
    _period_metadata(item['blocks'], item['usable'], item['fps'], manifest['rows'])
    alignment = _json(f'{folder}/alignment.json', manifest['files']['alignment.json'])
    if (str(alignment.get('match')) != str(mid) or alignment.get('reference_fields_used_for_inference') is not False or
            alignment.get('fps') != 25 or len(alignment.get('source_frames', [])) != manifest['rows']):
        raise ValueError('PFF alignment belongs to another observation timeline')
    # Authenticate both features before opening this fitting fixture's events.
    x = _array(f'{folder}/x.npy', manifest['files']['x.npy'])
    z = _array(f'{folder}/z.npy', manifest['files']['z.npy'])
    _validate_features(x, z, manifest['rows'])
    events = _json(f'data/pff/Event Data/{mid}.json', manifest['identity']['events'])
    y, _ = prior.pff_adapter.references(events, alignment)
    return _validate_record(dict(match=match, x=x, z=z, y=y, blocks=item['blocks'], fps=item['fps'],
                                 reference_fields_used_for_inference=False))


def _sportec_record(match, ctx):
    _require_fixture(match)
    folder = f'{REBUILD}/cache/{match}'
    seal = _json(f'{folder}/complete.json', ctx['sportec_seals'][match])
    if not isinstance(seal, dict) or set(seal) != SPORTEC_FILES:
        raise ValueError('Clean rebuild cache seal schema differs')
    meta = _json(f'{folder}/metadata.json', seal['metadata.json'])
    if (set(meta) != {'blocks', 'pitch', 'alignment', 'inputs', 'match'} or meta['match'] != match):
        raise ValueError('Sportec metadata does not identify the requested fixture')
    expected_inputs = {f'{match}_{kind}.xml': {k: ctx['sportec_inputs'][f'{match}_{kind}.xml'][k]
                      for k in ('sha256', 'bytes')} for kind in ('metadata', 'tracking')}
    if meta['inputs'] != expected_inputs:
        raise ValueError('Sportec source identities differ from pinned external inputs')
    support = _json(f'{folder}/clean-support.json', seal['clean-support.json'])
    if set(support) != {'usable'}:
        raise ValueError('Clean Sportec support schema differs')
    x = _array(f'{folder}/clean-x.npy', seal['clean-x.npy'])
    z = _array(f'{folder}/clean-z.npy', seal['clean-z.npy'])
    _validate_features(x, z, len(x))
    _period_metadata(meta['blocks'], support['usable'], 25, len(x))
    y = _array(f'{folder}/reference.npy', seal['reference.npy'])
    return _validate_record(dict(match=match, x=x, z=z, y=y, blocks=meta['blocks'], fps=25,
                                 reference_fields_used_for_inference=False))


class _InnerFactory:
    def __init__(self, ctx):
        self.__ctx = ctx

    def __call__(self, match):
        _require_fixture(match)  # before any source, metadata or array filesystem access
        ctx = self.__ctx
        _sources(ctx['snapshot']['source_hashes'])
        _sources(ctx['producer_sources'])
        _sources(ctx['clean_sources'])
        return _pff_record(match, ctx) if match.startswith('pff') else _sportec_record(match, ctx)

    def parent_model(self):
        # Re-authenticate actual bytes at handoff, not merely a saved declaration.
        parent = _json(f'{REBUILD}/refit/candidate.json', protocol.CLEAN_SHA256)
        kernel.validate_parent(parent)
        return parent

    def provenance(self):
        return dict(schema_version=1, core_protocol_sha256=protocol.fixed_protocol_sha256(),
                    fitting_fixture_ids=list(protocol.fit_ids()),
                    parent_sha256=protocol.CLEAN_SHA256,
                    pff_research_sha256=dev.PARENT_BUNDLE_SHA, rebuild_research_sha256=dev.REBUILD_BUNDLE_SHA,
                    sportec_regime='clean', historical_upstream_arrays_used=False,
                    reference_quality='provider-derived references; independent dense truth unresolved',
                    fitted_weights_must_remain_local=True, artifact_delivery_rights_verified=False,
                    stage_execution_authorized=False, refit_authorized=False,
                    development_accessed=False, final_accessed=False, promoted=False, submission_eligible=False)


def make_inner_factory(*, fixture_ids, protocol_sha256):
    """Construct only after a future runner consumes an authorized inner-fit stage.

    No caller-supplied manifest/root is accepted. Construction reads aggregate
    evidence and model bytes only; fixture arrays and references remain lazy.
    This API is an input bridge, not an authorization or experiment runner.
    """
    kernel.validate_fixture_ids(fixture_ids, phase='inner')
    if type(protocol_sha256) is not str or protocol_sha256 != protocol.fixed_protocol_sha256():
        raise ValueError('Exact frozen scientific core identity required')
    return _InnerFactory(_evidence())

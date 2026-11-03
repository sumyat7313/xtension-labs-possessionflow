"""Fixed training-only attribution of clean-model false stoppages; no fitting.

All forty three-model predictions are frozen before the first training reference
read. A command advances one stage. Failed/partial stages remain consumed.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from possession import robust_control
from possession.config import load_config
from possession.decoder import decode
from possession.io import sha256, write_json
from possession.learned import FEATURE_NAMES, load_model, predict_dead, fuse_control_probabilities, joint_probabilities
from tools import external_pff_development as dev
from tools import pff_campaign as prior, pff_validation as original
from tools import external_rebuild as rebuild, improvement_policy as policy
from tools.experiment_harness import locked
from evaluation import possession_metrics as pm

PARENT, REBUILD = dev.PARENT, dev.REBUILD
CANDIDATE_SHA, CONFIG_SHA = dev.CANDIDATE_SHA, dev.CONFIG_SHA
TRAIN_IDS = (3812, 3817, 3819, 3820, 3821, 3822, 3825, 3826, 3827, 3828, 3829,
             3832, 3833, 3834, 3835, 3836, 3838, 3839, 3842, 3843, 3845, 3846,
             3847, 3849, 3851, 3853, 3855, 3856, 3857, 3859, 10503, 10504, 10505,
             10507, 10508, 10509, 10512, 10513, 10515, 10516)
NAMES = ('base', 'ha_only', 'full')
STAGES = ('initialize', *[f'predict-{m}' for m in TRAIN_IDS], 'freeze',
          *[f'score-{m}' for m in TRAIN_IDS], 'finalize')
FLAGS = dict(external_only=True, training_partition_only=True, fitted=False,
             final_accessed=False, development_accessed=False, final_access_authorized=False,
             promoted=False, submission_eligible=False)
BIN_LABELS = {
    'base_dead_confidence': ['[0,.2)', '[.2,.4)', '[.4,.6)', '[.6,.8)', '[.8,1]'],
    'ball_availability': ['observed', 'interpolated', 'unavailable'],
    'valid_player_motion_count': ['0', '1-5', '6-11', '12-17', '18+ (capped at 22)'],
    'missing_run_seconds': ['observed', '(0,.6]', '(.6,2]', '(2,5]', '(5,8]', '>8'],
    'missing_run_edge_censor': ['observed', 'interior_missing_run', 'left_period_edge', 'right_period_edge', 'both_period_edges'],
}
METHOD = {
    'schema_version': 1,
    'ablations': {'base': 'clean control and frozen dead head, no joint residual',
                  'ha_only': 'full conditional H:A probabilities with exact base dead emission',
                  'full': 'unchanged clean joint inference'},
    'bin_labels': BIN_LABELS,
    'confidence_edges': [0., .2, .4, .6, .8, 1.],
    'confidence_anchor': 'base D emission shared by all ablations; last bin includes one',
    'ball_availability': 'cleaned accepted ball_observed; otherwise positive ball_reliability means interpolation',
    'player_count': 'round(22 * z.player_coverage); valid coordinates and speed; capped at 22',
    'missing_duration': 'full contiguous unobserved run length / fps, reset each period; edge runs marked potentially censored',
    'reference_mask': 'known H/A/D within observation period blocks; X excluded; predicted X counts as missed D or team error',
    'team_accuracy': 'posterior H:A argmax on all known live references (ties H); decoded accuracy separately reports D/X errors and live coverage',
    'short_spells': 'known-run H/A spells <2s; same period and unknown barriers; edge-touching spells marked potentially censored',
    'restart_distance': 'not used',
    'selection_or_promotion': 'none; component attribution only; decoder paths can differ despite identical D emissions',
}


def read(path):
    return dev.read(path)


def require_match(mid):
    if type(mid) is not int or mid not in TRAIN_IDS:
        raise ValueError('Only the original forty PFF training fixtures are permitted')
    return mid


def recipe(value=None):
    expected = dict(schema_version=1, candidate_sha256=CANDIDATE_SHA, ablations=list(NAMES))
    if value is not None and (not isinstance(value, dict) or value != expected or type(value.get('schema_version')) is not int):
        raise ValueError('Only the fixed three-ablation recipe is permitted')
    return expected


def sources():
    names = ('tools/clean_pff_training_diagnosis.py', 'docs/CLEAN_PFF_DIAGNOSIS_METHOD.md')
    return dict(dev.sources(), **{n: sha256(ROOT / n) for n in names})


def manifests(ctx):
    entries = [m for m in ctx['cache_manifests'] if m['match_id'] in TRAIN_IDS]
    result = {str(m['match_id']): m for m in entries}
    if len(entries) != 40 or len(result) != 40:
        raise ValueError('Training observation manifests incomplete or duplicated')
    return result


def context():
    # portable_context reads aggregate evidence and tracked code/model identities,
    # never local development caches. Do not replace this with dev.context().
    old_path = ROOT / 'models/pff-research.json'
    rebuild_path = ROOT / 'models/external-rebuild-001-research.json'
    dev.checked_hash(old_path, dev.PARENT_BUNDLE_SHA)
    dev.checked_hash(rebuild_path, dev.REBUILD_BUNDLE_SHA)
    ctx = dev.portable_context(read(old_path), read(rebuild_path))
    if ctx['reservation']['split']['train'] != list(TRAIN_IDS):
        raise ValueError('Training partition changed')
    for name, expected in [('refit/candidate.json', CANDIDATE_SHA), ('refit/config.json', CONFIG_SHA),
                           ('refit/lineage.json', dev.LINEAGE_FILE_SHA), ('protocol.json', dev.REBUILD_PROTOCOL_SHA),
                           ('summary.json', dev.REBUILD_SUMMARY_SHA)]:
        dev.checked_hash(REBUILD / name, expected)
    if read(REBUILD / 'refit/lineage.json') != ctx['lineage'] or read(REBUILD / 'refit/model-bindings.json') != ctx['model_paths']:
        raise ValueError('Candidate lineage differs')
    policy.authenticate_models(policy.audit_lineage(ctx['lineage']), ctx['model_paths'], ROOT)
    expected_sources = prior.sources()
    for mid, manifest in manifests(ctx).items():
        path = PARENT / 'cache' / mid / 'complete.json'
        rebuild.no_links(path)
        if sha256(path) not in dev.old_verifier._serialized_hashes(manifest) or read(path) != manifest:
            raise ValueError('Training cache manifest differs from published provenance')
        if manifest['identity']['sources'] != expected_sources:
            raise ValueError('Observation producer source changed')
    return ctx


def build_protocol(ctx, parameters=None):
    if ctx['reservation']['split']['train'] != list(TRAIN_IDS):
        raise ValueError('Training reservation differs')
    value = dict(schema_version=1, recipe=recipe(parameters), candidate_sha256=CANDIDATE_SHA,
                 config_sha256=CONFIG_SHA, lineage_digest=dev.LINEAGE_DIGEST,
                 reservation_sha256=ctx['reservation']['protocol_sha256'],
                 parent_research_sha256=dev.PARENT_BUNDLE_SHA, rebuild_research_sha256=dev.REBUILD_BUNDLE_SHA,
                 matches=list(TRAIN_IDS), stages=list(STAGES), method=METHOD,
                 cache_manifest_digests={m: original.digest(v) for m, v in manifests(ctx).items()},
                 source_hashes=sources(), reference_quality='provider event-derived silver; training diagnosis only',
                 **FLAGS)
    value['protocol_sha256'] = original.digest(value)
    return value


def preflight(out, parameters=None):
    dev.location(out)
    return build_protocol(context(), parameters)


def observation_meta(mid, ctx):
    require_match(mid)
    folder = PARENT / 'cache' / str(mid)
    manifest = manifests(ctx)[str(mid)]
    if read(folder / 'complete.json') != manifest:
        raise ValueError('Training cache declaration differs')
    dev.checked_hash(folder / 'item.json', manifest['files']['item.json'])
    value = read(folder / 'item.json')
    if value.get('match') != f'pff{mid}' or value.get('reference_fields_used_for_inference') is not False:
        raise ValueError('Observation-only training fixture declaration differs')
    return {k: value[k] for k in ('blocks', 'fps', 'usable', 'pitch', 'reference_fields_used_for_inference')}


def observations(mid, ctx):
    require_match(mid)
    item = observation_meta(mid, ctx)
    manifest = manifests(ctx)[str(mid)]
    for key in ('x', 'z'):
        path = PARENT / 'cache' / str(mid) / f'{key}.npy'
        dev.checked_hash(path, manifest['files'][f'{key}.npy'])
        item[key] = np.load(path, mmap_mode='r', allow_pickle=False)
    return item


def validate_observations(item, model):
    if set(item) != {'x', 'z', 'blocks', 'fps', 'usable', 'pitch', 'reference_fields_used_for_inference'}:
        raise ValueError('Only x/z and observation metadata are permitted')
    x, z = np.asarray(item['x']), np.asarray(item['z'])
    if (item['reference_fields_used_for_inference'] is not False or item['fps'] != 25 or
            x.ndim != 2 or x.shape[1] != len(robust_control.FEATURE_NAMES) or
            z.shape != (len(x), len(FEATURE_NAMES)) or not np.isfinite(x).all() or not np.isfinite(z).all()):
        raise ValueError('Invalid observation features')
    prior.pff_training._blocks(item['blocks'], len(x), item['fps'])
    if len(item['usable']) != len(item['blocks']) or any(type(v) is not bool for v in item['usable']):
        raise ValueError('Invalid support declarations')
    if any(k in model for k in ('temporal_model', 'expert_gate', 'boundary_model', 'transition_model', 'typed_boundary_model')):
        raise ValueError('Unexpected learned component')
    return x, z


def ha_only_probabilities(full, base_dead):
    full, base_dead = np.asarray(full), np.asarray(base_dead)
    if (full.ndim != 2 or full.shape[1] != 3 or base_dead.shape != (len(full),) or
            not np.isfinite(full).all() or not np.isfinite(base_dead).all() or
            np.any((full < 0) | (full > 1)) or np.any((base_dead < 0) | (base_dead >= 1)) or
            not np.allclose(full.sum(axis=1), 1., atol=1e-12, rtol=0)):
        raise ValueError('Invalid full probabilities or base dead emission')
    live = full[:, :2].sum(axis=1)
    if np.any(live <= 0):
        raise ValueError('Full conditional team probability is undefined')
    conditional = full[:, :2] / live[:, None]
    result = np.column_stack(((1-base_dead)[:, None] * conditional, base_dead))
    if not np.array_equal(result[:, 2], base_dead) or not np.allclose(result[:, :2] / result[:, :2].sum(axis=1, keepdims=True), conditional, atol=1e-12, rtol=0):
        raise ValueError('H:A-only emission identities failed')
    return result


def ablations(item, model, config):
    x, z = validate_observations(item, model)
    home = robust_control.predict_home(model['robust_control_model'], x)
    dead = predict_dead(z, model)
    base = fuse_control_probabilities(dead, home, np.zeros(len(x)))
    full = joint_probabilities(dead, home, x, z, item['blocks'], item['fps'], model['joint_model'], np.zeros(len(x)))
    probabilities = dict(base=base, ha_only=ha_only_probabilities(full, dead), full=full)
    labels = {}
    for name, p in probabilities.items():
        value = np.full(len(x), 'X', dtype='<U1')
        for (_, a, b), usable in zip(item['blocks'], item['usable']):
            if usable:
                value[a:b+1] = np.array(list('HAD'))[decode(p[a:b+1], item['fps'], config)]
        labels[name] = value
    return labels, probabilities


def observation_bins(item, base_dead):
    z, fps = np.asarray(item['z']), item['fps']
    observed = z[:, FEATURE_NAMES.index('ball_observed')]
    reliability = z[:, FEATURE_NAMES.index('ball_reliability')]
    coverage = z[:, FEATURE_NAMES.index('player_coverage')]
    base_dead = np.asarray(base_dead)
    if (base_dead.shape != (len(z),) or not np.isfinite(base_dead).all() or np.any((base_dead < 0) | (base_dead > 1)) or
            not np.isfinite(z).all() or not np.isin(observed, [0., 1.]).all() or np.any((reliability < 0) | (reliability > 1)) or
            np.any((coverage < 0) | (coverage > 1)) or not np.allclose(coverage*22, np.rint(coverage*22), atol=1e-8, rtol=0)):
        raise ValueError('Observation feature semantics differ from fixed bins')
    n = len(z)
    active, _ = prior.pff_training._blocks(item['blocks'], n, fps)
    missing_bins = np.zeros(n, dtype=np.uint8)
    edge_bins = np.zeros(n, dtype=np.uint8)
    for _, a, b in item['blocks']:
        mask = observed[a:b+1] == 0
        edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
        for lo, hi in zip(edges[::2], edges[1::2]):
            seconds = (hi-lo) / fps
            code = 1 + int(np.searchsorted([.6, 2., 5., 8.], seconds, side='left'))
            missing_bins[a+lo:a+hi] = code
            edge_bins[a+lo:a+hi] = 4 if lo == 0 and hi == len(mask) else 2 if lo == 0 else 3 if hi == len(mask) else 1
    player_count = np.rint(coverage*22).astype(int)
    result = dict(
        base_dead_confidence=np.searchsorted([.2, .4, .6, .8], base_dead, side='right').astype(np.uint8),
        ball_availability=np.where(observed == 1, 0, np.where(reliability > 0, 1, 2)).astype(np.uint8),
        valid_player_motion_count=np.searchsorted([0, 5, 11, 17], player_count, side='left').astype(np.uint8),
        missing_run_seconds=missing_bins, missing_run_edge_censor=edge_bins)
    for values in result.values():
        values[~active] = 255
    return result


def predict_one(out, mid, study, ctx):
    require_match(mid)
    if any((out / p).exists() for p in ('predictions-before-references.json', 'reference-access.json', 'diagnosis-summary.json')):
        raise ValueError('Prediction regeneration after freeze/reference access prohibited')
    if study != build_protocol(ctx, study['recipe']):
        raise ValueError('Frozen study differs')
    dev.checked_hash(REBUILD / 'refit/candidate.json', CANDIDATE_SHA)
    dev.checked_hash(REBUILD / 'refit/config.json', CONFIG_SHA)
    item = observations(mid, ctx)
    labels, probabilities = ablations(item, load_model(REBUILD / 'refit/candidate.json'), load_config(REBUILD / 'refit/config.json'))
    bins = observation_bins(item, probabilities['base'][:, 2])
    folder = out / 'predictions' / str(mid)
    folder.mkdir(parents=True, exist_ok=False)
    for name in NAMES:
        np.save(folder / f'{name}.npy', labels[name], allow_pickle=False)
        np.save(folder / f'{name}-probabilities.npy', probabilities[name], allow_pickle=False)
    np.savez(folder / 'bins.npz', **bins)
    proof = dict(match_id=mid, protocol_sha256=study['protocol_sha256'], candidate_sha256=CANDIDATE_SHA,
                 config_sha256=CONFIG_SHA,
                 cache_manifest_digest=original.digest(manifests(ctx)[str(mid)]),
                 files={p.name: sha256(p) for p in sorted(folder.iterdir())})
    write_json(folder / 'complete.json', proof)
    return proof


def prediction_records(out, study, ctx):
    if study != build_protocol(ctx, study['recipe']):
        raise ValueError('Frozen protocol/source differs')
    expected = {f'{n}.npy' for n in NAMES} | {f'{n}-probabilities.npy' for n in NAMES} | {'bins.npz'}
    records = []
    for mid in TRAIN_IDS:
        folder = out / 'predictions' / str(mid)
        proof = read(folder / 'complete.json')
        if (proof['match_id'] != mid or proof['protocol_sha256'] != study['protocol_sha256'] or
                proof['candidate_sha256'] != CANDIDATE_SHA or proof.get('config_sha256') != CONFIG_SHA or set(proof['files']) != expected or
                proof['cache_manifest_digest'] != original.digest(manifests(ctx)[str(mid)])):
            raise ValueError('Prediction completion differs')
        for name, identity in proof['files'].items():
            dev.checked_hash(folder / name, identity)
        records.append(proof)
    return records


def freeze_value(study, records):
    return dict(schema_version=1, protocol_sha256=study['protocol_sha256'], candidate_sha256=CANDIDATE_SHA,
                source_hashes=study['source_hashes'], records=records, references_accessed_in_this_study=False, **FLAGS)


def verify_freeze(out, study, ctx):
    value = freeze_value(study, prediction_records(out, study, ctx))
    if read(out / 'predictions-before-references.json') != value:
        raise ValueError('Complete forty-fixture prediction freeze differs')
    return value


def access_value(out, study):
    return dict(schema_version=1, protocol_sha256=study['protocol_sha256'], candidate_sha256=CANDIDATE_SHA,
                source_hashes=study['source_hashes'], references_consumed=True,
                prediction_freeze_sha256=sha256(out / 'predictions-before-references.json'), **FLAGS)


def read_reference(out, mid, study, ctx):
    require_match(mid)
    verify_freeze(out, study, ctx)
    if read(out / 'reference-access.json') != access_value(out, study):
        raise ValueError('Exclusive reference access differs')
    if read(out / f'reference-access-{mid}.json') != dict(match_id=mid, protocol_sha256=study['protocol_sha256'],
                                                        reference_access_sha256=sha256(out / 'reference-access.json')):
        raise ValueError('Fixture reference access differs')
    manifest = manifests(ctx)[str(mid)]
    dev.checked_hash(PARENT / 'cache' / str(mid) / 'alignment.json', manifest['files']['alignment.json'])
    dev.checked_hash(ROOT / 'data/pff/Event Data' / f'{mid}.json', manifest['identity']['events'])
    return prior.reference(PARENT, mid)


def with_rates(result):
    result = dict(result)
    def rate(num, den):
        return result[num] / result[den] if result[den] else None
    result.update(dead_precision=rate('true_dead', 'predicted_dead'), dead_recall=rate('true_dead', 'reference_dead'),
        conditional_team_accuracy=rate('conditional_team_correct', 'reference_live'),
        decoded_team_accuracy=rate('decoded_team_correct', 'reference_live'),
        decoded_live_team_accuracy=rate('decoded_team_correct', 'decoded_live_on_reference_live'),
        decoded_live_coverage=rate('decoded_live_on_reference_live', 'reference_live'))
    return result


def counts(reference, prediction, probabilities, mask):
    live = mask & np.isin(reference, ['H', 'A'])
    dead = mask & (reference == 'D')
    predicted_dead = prediction == 'D'
    decoded_live = np.isin(prediction, ['H', 'A'])
    team = np.where(probabilities[:, 0] >= probabilities[:, 1], 'H', 'A')
    result = dict(known_frames=int(mask.sum()), reference_live=int(live.sum()), reference_dead=int(dead.sum()),
        true_dead=int((dead & predicted_dead).sum()), false_dead=int((live & predicted_dead).sum()),
        missed_dead=int((dead & ~predicted_dead).sum()), predicted_dead=int((mask & predicted_dead).sum()),
        predicted_unknown=int((mask & (prediction == 'X')).sum()),
        conditional_team_correct=int((live & (team == reference)).sum()),
        decoded_team_correct=int((live & (prediction == reference)).sum()),
        decoded_live_on_reference_live=int((live & decoded_live).sum()))
    return with_rates(result)


def short_false_dead(reference, labels, blocks, fps):
    code = pm.label_codes(reference)
    runs = []
    for _, a, b in blocks:
        runs.extend((a+lo, a+hi) for lo, hi in pm.valid_runs(code[a:b+1]))
    spells = [(a, b, c) for a, b, c in pm.possession_spells(code, runs) if (b-a)/fps < 2.]
    boundaries = {point for a, b in runs for point in (a, b)}
    total_frames = sum(b-a for a, b, _ in spells)
    result = dict(n_gt=len(spells), reference_frames=total_frames,
                  touches_scored_run_edge=sum(a in boundaries or b in boundaries for a, b, _ in spells),
                  censored_reference_frames=sum(b-a for a, b, _ in spells if a in boundaries or b in boundaries), models={})
    for name in NAMES:
        prediction = labels[name]
        masses = [int((prediction[a:b] == 'D').sum()) for a, b, _ in spells]
        result['models'][name] = dict(false_dead_frames=sum(masses), spells_with_any_dead=sum(n > 0 for n in masses),
            spells_erased_by_dead=sum(n == b-a for n, (a, b, _) in zip(masses, spells)),
            interruption_rate=sum(n > 0 for n in masses)/len(spells) if spells else None,
            erasure_rate=sum(n == b-a for n, (a, b, _) in zip(masses, spells))/len(spells) if spells else None,
            false_dead_frame_fraction=sum(masses)/total_frames if total_frames else None)
    return result


def aggregate_fixture(reference, labels, probabilities, bins, blocks, fps):
    n = len(reference)
    active, _ = prior.pff_training._blocks(blocks, n, fps)
    if (not np.isin(reference, list('HADX')).all() or set(labels) != set(NAMES) or
            set(probabilities) != set(NAMES) or set(bins) != set(BIN_LABELS)):
        raise ValueError('Diagnostic schema differs')
    for name in NAMES:
        p = probabilities[name]
        if (labels[name].shape != (n,) or not np.isin(labels[name], list('HADX')).all() or
                p.shape != (n, 3) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)) or
                not np.allclose(p.sum(axis=1), 1., atol=1e-12, rtol=0)):
            raise ValueError('Invalid saved inference arrays')
    for name, values in bins.items():
        if values.shape != (n,) or not np.isin(values[active], range(len(BIN_LABELS[name]))).all() or np.any(values[~active] != 255):
            raise ValueError('Invalid saved diagnostic bins')
    known = active & np.isin(reference, list('HAD'))
    result = dict(models={}, bins={}, paired={}, short_false_dead=short_false_dead(reference, labels, blocks, fps))
    for name in NAMES:
        result['models'][name] = counts(reference, labels[name], probabilities[name], known)
    for key, names in BIN_LABELS.items():
        result['bins'][key] = {label: {name: counts(reference, labels[name], probabilities[name], known & (bins[key] == i))
                                         for name in NAMES} for i, label in enumerate(names)}
    for left, right in [('base', 'ha_only'), ('ha_only', 'full'), ('base', 'full')]:
        a, b = labels[left] == 'D', labels[right] == 'D'
        live, dead = known & np.isin(reference, ['H', 'A']), known & (reference == 'D')
        result['paired'][left + '->' + right] = dict(
            added_false_dead=int((live & ~a & b).sum()), removed_false_dead=int((live & a & ~b).sum()),
            recovered_true_dead=int((dead & ~a & b).sum()), lost_true_dead=int((dead & a & ~b).sum()))
    return result


def score_one(out, mid, study, ctx):
    require_match(mid)
    verify_freeze(out, study, ctx)
    for previous in TRAIN_IDS[:TRAIN_IDS.index(mid)]:
        if not (out / 'scores' / str(previous) / 'aggregate.json').exists():
            raise ValueError('Training references must be scored in frozen order')
    access_path = out / 'reference-access.json'
    if mid == TRAIN_IDS[0]:
        with access_path.open('x', encoding='utf-8') as stream:
            json.dump(access_value(out, study), stream, sort_keys=True, indent=2)
    elif not access_path.exists():
        raise ValueError('First training reference access has not been recorded')
    if (out / 'scores' / str(mid) / 'aggregate.json').exists():
        raise ValueError('No repeated training scoring')
    with (out / f'reference-access-{mid}.json').open('x', encoding='utf-8') as stream:
        json.dump(dict(match_id=mid, protocol_sha256=study['protocol_sha256'],
                       reference_access_sha256=sha256(access_path)), stream, sort_keys=True)
    y, support = read_reference(out, mid, study, ctx)
    item = observation_meta(mid, ctx)
    folder = out / 'predictions' / str(mid)
    labels = {name: np.load(folder / f'{name}.npy', allow_pickle=False) for name in NAMES}
    probabilities = {name: np.load(folder / f'{name}-probabilities.npy', allow_pickle=False) for name in NAMES}
    with np.load(folder / 'bins.npz', allow_pickle=False) as saved:
        bins = {key: saved[key] for key in saved.files}
    diagnostics = aggregate_fixture(y, labels, probabilities, bins, item['blocks'], item['fps'])
    metrics, details = {}, []
    for name in NAMES:
        detail, metrics[name] = prior.score_one(out, 'scores', mid, name, y, labels[name], item['blocks'])
        details.append(detail)
    result = dict(match_id=mid, candidate_sha256=CANDIDATE_SHA, protocol_sha256=study['protocol_sha256'],
                  reference_support=support, diagnostics=diagnostics, metrics=metrics, details=details, **FLAGS)
    write_json(out / 'scores' / str(mid) / 'aggregate.json', result)
    return result


def final_report(out, study):
    records = [read(out / 'scores' / str(mid) / 'aggregate.json') for mid in TRAIN_IDS]
    if any(row['match_id'] != mid or row['protocol_sha256'] != study['protocol_sha256'] or row['candidate_sha256'] != CANDIDATE_SHA for mid, row in zip(TRAIN_IDS, records)):
        raise ValueError('Training fixture score identity differs')
    means = {name: {key: float(np.mean([r['metrics'][name][key] for r in records]))
                    for key in records[0]['metrics'][name]} for name in NAMES}
    fields = ('known_frames', 'reference_live', 'reference_dead', 'true_dead', 'false_dead', 'missed_dead',
              'predicted_dead', 'predicted_unknown', 'conditional_team_correct', 'decoded_team_correct', 'decoded_live_on_reference_live')
    totals = {name: {key: sum(r['diagnostics']['models'][name][key] for r in records) for key in fields} for name in NAMES}
    totals = {name: with_rates(value) for name, value in totals.items()}
    rate_names = ('dead_precision', 'dead_recall', 'conditional_team_accuracy', 'decoded_team_accuracy', 'decoded_live_team_accuracy', 'decoded_live_coverage')
    mean_rates = {}
    for name in NAMES:
        mean_rates[name] = {}
        for key in rate_names:
            values = [r['diagnostics']['models'][name][key] for r in records if r['diagnostics']['models'][name][key] is not None]
            mean_rates[name][key] = dict(mean=float(np.mean(values)) if values else None, supported_fixtures=len(values))
    paired = {pair: {key: sum(r['diagnostics']['paired'][pair][key] for r in records)
                     for key in records[0]['diagnostics']['paired'][pair]} for pair in records[0]['diagnostics']['paired']}
    result = dict(schema_version=1, protocol_sha256=study['protocol_sha256'], candidate_sha256=CANDIDATE_SHA,
                  source_hashes=study['source_hashes'], records=records, mean_metrics=means, frame_count_totals=totals,
                  equal_fixture_diagnostic_rates=mean_rates,
                  paired_totals=paired, reference_access_sha256=sha256(out / 'reference-access.json'),
                  prediction_freeze_sha256=sha256(out / 'predictions-before-references.json'),
                  conclusion='Component diagnosis only; no selection, fit, fresh test or promotion', **FLAGS)
    write_json(out / 'diagnosis-summary.json', result)
    return result


def stage_files(out, stage):
    if stage == 'initialize': return ['protocol.json']
    if stage.startswith('predict-'):
        mid = require_match(int(stage[8:]))
        return [f'predictions/{mid}/{n}{suffix}' for n in NAMES for suffix in ('.npy', '-probabilities.npy')] + [f'predictions/{mid}/bins.npz', f'predictions/{mid}/complete.json']
    if stage == 'freeze': return ['predictions-before-references.json']
    if stage.startswith('score-'):
        mid = require_match(int(stage[6:]))
        return ['reference-access.json', f'reference-access-{mid}.json'] + [p.relative_to(out).as_posix() for p in sorted((out / 'scores' / str(mid)).rglob('*')) if p.is_file()]
    if stage == 'finalize': return ['diagnosis-summary.json']
    raise ValueError('Unknown diagnostic stage')


def authenticate(out, parameters=None, ctx=None):
    ctx = context() if ctx is None else ctx
    study, state = read(out / 'protocol.json'), read(out / 'state.json')
    if study != build_protocol(ctx, parameters): raise ValueError('Frozen source/protocol changed')
    done = state.get('completed')
    if not isinstance(done, dict) or set(done) != set(STAGES[:len(done)]) or state.get('protocol_sha256') != study['protocol_sha256']:
        raise ValueError('Stages are not a completed prefix')
    for stage, identity in done.items():
        path = out / 'stages' / f'{stage}.json'
        dev.checked_hash(path, identity)
        proof = read(path)
        if proof['stage'] != stage or proof['protocol_sha256'] != study['protocol_sha256'] or set(proof['artifacts']) != set(stage_files(out, stage)):
            raise ValueError('Stage evidence differs')
        for name, expected in proof['artifacts'].items(): dev.checked_hash(out / name, expected)
    return study, state, ctx


def status(out):
    out = dev.location(out)
    study, state, _ = authenticate(out)
    pending = [s for s in STAGES if s not in state['completed']]
    return dict(status='in_progress' if pending else 'complete', completed_stages=len(state['completed']),
                next_stage=pending[0] if pending else None,
                consumed_failure=bool(pending and (out / f'{pending[0]}-started.json').exists()),
                protocol_sha256=study['protocol_sha256'], candidate_sha256=CANDIDATE_SHA, **FLAGS)


def run(out, parameters=None):
    parameters = recipe(parameters)
    out = dev.location(out); out.mkdir(parents=True, exist_ok=True)
    with locked(out):
        ctx = context()
        if not (out / 'protocol.json').exists():
            if any(p.name != '.lock' for p in out.iterdir()): raise ValueError('Interrupted/nonempty study cannot be reset')
            study, stage = build_protocol(ctx, parameters), 'initialize'
            state = dict(protocol_sha256=study['protocol_sha256'], completed={})
        else:
            study, state, ctx = authenticate(out, parameters, ctx)
            pending = [s for s in STAGES if s not in state['completed']]
            if not pending: return read(out / 'summary.json')
            stage = pending[0]
        with (out / f'{stage}-started.json').open('x', encoding='utf-8') as stream:
            json.dump(dict(stage=stage, protocol_sha256=study['protocol_sha256'], source_hashes=study['source_hashes'], pid=os.getpid()), stream, sort_keys=True)
        if stage == 'initialize': write_json(out / 'protocol.json', study)
        elif stage.startswith('predict-'): predict_one(out, int(stage[8:]), study, ctx)
        elif stage == 'freeze': write_json(out / 'predictions-before-references.json', freeze_value(study, prediction_records(out, study, ctx)))
        elif stage.startswith('score-'): score_one(out, int(stage[6:]), study, ctx)
        elif stage == 'finalize': final_report(out, study)
        if study != build_protocol(context(), parameters): raise ValueError('Source/model/cache identity changed during stage')
        path = out / 'stages' / f'{stage}.json'
        write_json(path, dict(stage=stage, protocol_sha256=study['protocol_sha256'],
                             artifacts={name: sha256(out / name) for name in stage_files(out, stage)}))
        state['completed'][stage] = sha256(path)
        rebuild._atomic(out / 'state.json', state)
        pending = [s for s in STAGES if s not in state['completed']]
        summary = dict(schema_version=1, status='in_progress' if pending else 'complete', stage_completed=stage,
                       next_stage=pending[0] if pending else None, protocol_sha256=study['protocol_sha256'],
                       protocol_file_sha256=sha256(out / 'protocol.json'), candidate_sha256=CANDIDATE_SHA,
                       source_hashes=study['source_hashes'], **FLAGS)
        if not pending: summary['diagnosis_summary_sha256'] = sha256(out / 'diagnosis-summary.json')
        rebuild._atomic(out / 'summary.json', summary)
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'preflight', 'run', 'status'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stage', choices=('next',), default='next')
    args = parser.parse_args()
    result = run(args.out) if args.action == 'run' else status(args.out) if args.action == 'status' else preflight(args.out)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__': main()

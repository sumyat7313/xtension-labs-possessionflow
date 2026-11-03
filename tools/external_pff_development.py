"""One fixed external-only model, evaluated on previously used PFF development.

No fitting, threshold search, final-fixture entry point or automatic promotion.
Each invocation advances one authenticated stage; failed attempts stay consumed.
"""
from __future__ import annotations
import argparse
import copy
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
from possession.learned import load_model, predict_dead, joint_probabilities
from tools import pff_campaign as prior, pff_validation as original
from tools import pff_precision_validation as precision, improvement_policy as policy
from tools import external_rebuild as rebuild, verify_pff_research as old_verifier
from tools.experiment_harness import locked

PARENT = ROOT / 'runs/pff-campaign-v3'
REBUILD = ROOT / 'runs/continuous-improvement/trials/external-rebuild-001/output'
IDS = (3813, 3816, 3818, 3823, 3831, 3844, 3850, 3854, 3858, 10502, 10514, 10517)
CANDIDATE_SHA = '7d79dc50cfc16a7f64c37b65908be90e588d3bfdec35f599071b25180089bf47'
CONFIG_SHA = '2755c140130f4ef9821e7909f8da794d6e4c8d4a7691b75de77eada7c866612b'
LINEAGE_FILE_SHA = '3d31772a8821e869b8569563ab43fcca3b7b2afe11159072ebfd76c1620f0e8c'
LINEAGE_DIGEST = '9051b999f5c3b76fad879614d9157590a7e736ba95c7057278097a3834d16a56'
REBUILD_PROTOCOL_SHA = '0cd152ae7105f4561b5fa6f97f23fc71f1dd2afde49fe32301fb7d57dca6f31f'
REBUILD_SUMMARY_SHA = 'c605d16d3550b36db281c95ecee964ffe8e97538d096f9dbef6c6e4b8d5cdb34'
PARENT_BUNDLE_SHA = '46e49be1e4fcbcc75a6ae7193d73886738a76d0bdd5a622f2f9c8f1c5a980654'
REBUILD_BUNDLE_SHA = '9c4150f3ba9cdbb59df3270fa2ca4e667d6eca76898b40a0debf5be39d2867c0'
CACHE_MAP_DIGEST = 'c1f9e252ff447dfa8f2eb0621f10151a612bf3e41f07926caf57e977bae4587f'
CACHE_HASHES = {
    '3813': 'f49bf299873addc53e74d23f3df4c71ede5898882458a094e39cd1a3e346caec',
    '3816': 'd5628886adb6d85063ee58dc893183441c3c5d2cf414253cbeffb2b2d1149c14',
    '3818': 'fbfbf095c905e4836ed0b7efc96fe338d7d95abbcb0a142a680f5a15411d6be3',
    '3823': '3fd6c5d43e6598c5fff1d974e6873b62be3ab5a46b6ceb0bcbe75f8f3f2d28f7',
    '3831': 'd9d2f26456a596fe3f6ae8b673519496156874409ee0869fb912950519421b51',
    '3844': '34142bdb30561dc18978abab602518a843174ae8d91e5795dcbe57ec473e9505',
    '3850': '969229d5ca25692edb5b697ee85d2961998f6318b76c1240e5c7b445cbd29105',
    '3854': '39ee67cd5a170ddd51b9762ed96cca46486f0386e41e3cbadaf74911bd0c8086',
    '3858': 'fab4d9f99e638ae371abd934bf7244903404262e3f105547fbe9405c958a7157',
    '10502': '10f8379c90966c18518015c21535da53a4a8c2ecba9f50da447efe80074ab2cb',
    '10514': 'c594a14f25cd56a5fb26a18718d266d4f561ce8148e1f10c70acedf572c0dbfe',
    '10517': '7b14aef657d6c0e74f6af0cebab7adf8a4275a1d818ad5a9a1d20cf5adc60b87',
}
STAGES = ('initialize', *[f'predict-{m}' for m in IDS], 'freeze', 'score')
NAMES = ('baseline', 'parent', 'candidate')
FLAGS = dict(external_only=True, final_accessed=False, final_access_authorized=False,
             promoted=False, submission_eligible=False)


def read(path):
    rebuild.no_links(path)
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require_match(mid):
    if type(mid) is not int or mid not in IDS:
        raise ValueError('Only the twelve declared PFF development fixtures are permitted')
    return mid


def recipe(value=None):
    expected = {'schema_version': 1, 'candidate_sha256': CANDIDATE_SHA}
    if value is not None and (not isinstance(value, dict) or value != expected or type(value.get('schema_version')) is not int):
        raise ValueError('Only the fixed candidate recipe is permitted')
    return expected


def location(out):
    rebuild.no_links(out)
    path = Path(out).resolve()
    if not path.is_relative_to((ROOT / 'runs').resolve()) or path == (ROOT / 'runs').resolve():
        raise ValueError('Study output must remain in its own local runs directory')
    for protected in (PARENT, REBUILD):
        protected = protected.resolve()
        if path == protected or path.is_relative_to(protected) or protected.is_relative_to(path):
            raise ValueError('Historical study directories cannot be overwritten')
    return path


def checked_hash(path, expected):
    rebuild.no_links(path)
    if sha256(path) != expected:
        raise ValueError('Artifact bytes differ: ' + Path(path).name)


def sources():
    # Old scientific modules stay immutable. Include imported orchestration,
    # ancestry and bundle checks; the separate new verifier is not inference code.
    names = ('tools/external_pff_development.py', 'tools/external_rebuild.py',
             'tools/experiment_harness.py', 'tools/improvement_policy.py',
             'tools/pff_precision_validation.py', 'tools/verify_pff_research.py',
             'tools/pff_migrate_preparation.py', 'requirements-solution.txt',
             'requirements-training.txt')
    return dict(prior.sources(), **{n: sha256(ROOT / n) for n in names})


def portable_context(old_bundle, rebuild_bundle):
    old_verifier.verify_bundle(old_bundle)
    reservation = original.verify_protocol(old_bundle['protocol'])
    if reservation['split']['development'] != list(IDS):
        raise ValueError('Development reservation differs')
    previous = old_bundle['development']['summary']['records']
    if original.digest(previous) != policy.ORIGINAL_PARENT_RECORDS_SHA256:
        raise ValueError('Original PFF gain anchor differs')
    summary = rebuild_bundle['summary']
    if (summary['release_candidate_model_sha256'] != CANDIDATE_SHA or
            summary['status'] != 'complete' or summary['stage_completed'] != 'refit' or
            summary['next_stage'] is not None or summary['external_only'] is not True or
            summary['final_accessed'] is not False or summary['promoted'] is not False or
            summary['submission_eligible'] is not False):
        raise ValueError('Clean external rebuild is not complete')
    selection = rebuild.select(summary['records'])
    if any(summary[k] != v for k, v in selection.items()):
        raise ValueError('External rebuild development selection differs')
    if rebuild_bundle['protocol'] != rebuild.plan(rebuild_bundle['protocol']['recipe']) | {'inputs': rebuild_bundle['protocol']['inputs']}:
        raise ValueError('External rebuild source or plan changed')
    lineage = rebuild_bundle['lineage']
    audited = policy.audit_lineage(lineage)
    if audited['candidate_sha256'] != CANDIDATE_SHA or original.digest(lineage) != LINEAGE_DIGEST:
        raise ValueError('Clean candidate lineage differs')
    return dict(reservation=reservation, prior_records=previous, lineage=lineage,
                model_paths=summary['model_paths'], old_bundle=old_bundle,
                cache_manifests=old_bundle['cache_manifests'], rebuild_bundle=rebuild_bundle,
                parent_research_sha256=PARENT_BUNDLE_SHA, rebuild_research_sha256=REBUILD_BUNDLE_SHA)


def manifests(ctx):
    result = {str(m['match_id']): m for m in ctx['cache_manifests'] if m['match_id'] in IDS}
    if set(result) != {str(m) for m in IDS}:
        raise ValueError('Development observation manifests incomplete')
    return result


def context():
    parent_path = ROOT / 'models/pff-research.json'
    rebuild_path = ROOT / 'models/external-rebuild-001-research.json'
    checked_hash(parent_path, PARENT_BUNDLE_SHA)
    checked_hash(rebuild_path, REBUILD_BUNDLE_SHA)
    ctx = portable_context(read(parent_path), read(rebuild_path))
    for relative, expected in [('refit/candidate.json', CANDIDATE_SHA), ('refit/config.json', CONFIG_SHA),
                               ('refit/lineage.json', LINEAGE_FILE_SHA), ('protocol.json', REBUILD_PROTOCOL_SHA),
                               ('summary.json', REBUILD_SUMMARY_SHA)]:
        checked_hash(REBUILD / relative, expected)
    if read(REBUILD / 'refit/lineage.json') != ctx['lineage'] or read(REBUILD / 'refit/model-bindings.json') != ctx['model_paths']:
        raise ValueError('Actual model ancestry differs from published rebuild')
    policy.authenticate_models(policy.audit_lineage(ctx['lineage']), ctx['model_paths'], ROOT)
    checked_hash(ROOT / 'models/submission.json', original.BASELINE_SHA256)
    checked_hash(ROOT / 'configs/baselines/v1.5.json', CONFIG_SHA)
    for mid, manifest in manifests(ctx).items():
        path = PARENT / 'cache' / mid / 'complete.json'
        checked_hash(path, CACHE_HASHES[mid])
        if read(path) != manifest:
            raise ValueError('Local observation manifest differs from published evidence')
    return ctx


def build_protocol(ctx, parameters=None, source_hashes=None):
    config = recipe(parameters)
    if original.digest(CACHE_HASHES) != CACHE_MAP_DIGEST:
        raise ValueError('Fixed cache identities changed')
    known = manifests(ctx)
    for mid, manifest in known.items():
        if CACHE_HASHES[mid] not in old_verifier._serialized_hashes(manifest):
            raise ValueError('Cache manifest serialization differs')
        for name, value in ctx['rebuild_bundle']['protocol']['source_hashes'].items():
            if name in manifest['identity']['sources'] and manifest['identity']['sources'][name] != value:
                raise ValueError('Observation-producing scientific source differs')
    value = dict(schema_version=1, recipe=config, candidate_sha256=CANDIDATE_SHA,
        baseline_sha256=original.BASELINE_SHA256, parent_sha256=precision.PARENT_SHA256,
        config_sha256=CONFIG_SHA, lineage_file_sha256=LINEAGE_FILE_SHA, lineage_digest=LINEAGE_DIGEST,
        rebuild_protocol_sha256=REBUILD_PROTOCOL_SHA, rebuild_summary_sha256=REBUILD_SUMMARY_SHA,
        parent_research_sha256=ctx['parent_research_sha256'], rebuild_research_sha256=ctx['rebuild_research_sha256'],
        reservation_sha256=ctx['reservation']['protocol_sha256'],
        prior_records_digest=original.digest(ctx['prior_records']), matches=list(IDS), stages=list(STAGES),
        cache_manifest_sha256=dict(CACHE_HASHES), source_hashes=sources() if source_hashes is None else source_hashes,
        inference='x/z only; recompute clean upstream heads; fixed joint model and decoder',
        scope='Previously consumed development; single fixed candidate; no fitting or tuning',
        policy=dict(policy.POLICY), **FLAGS)
    value['protocol_sha256'] = original.digest(value)
    return value


def plan(parameters=None):
    return build_protocol(context(), parameters)


def preflight(out, parameters=None):
    location(out)
    return plan(parameters)


def _observation_meta(mid, ctx):
    require_match(mid)
    manifest = manifests(ctx)[str(mid)]
    folder = PARENT / 'cache' / str(mid)
    checked_hash(folder / 'complete.json', CACHE_HASHES[str(mid)])
    checked_hash(folder / 'item.json', manifest['files']['item.json'])
    item = read(folder / 'item.json')
    if item.get('reference_fields_used_for_inference') is not False or item.get('match') != f'pff{mid}':
        raise ValueError('Observation-only fixture declaration differs')
    return {k: item[k] for k in ('blocks', 'fps', 'usable', 'pitch', 'reference_fields_used_for_inference')}


def observations(mid, ctx):
    require_match(mid)  # before path lookup or any cache read
    item = _observation_meta(mid, ctx)
    manifest = manifests(ctx)[str(mid)]
    for key in ('x', 'z'):
        path = PARENT / 'cache' / str(mid) / f'{key}.npy'
        checked_hash(path, manifest['files'][f'{key}.npy'])
        item[key] = np.load(path, mmap_mode='r', allow_pickle=False)
    return item


def predict_observations(item, model, config):
    if set(item) != {'x', 'z', 'blocks', 'fps', 'usable', 'pitch', 'reference_fields_used_for_inference'}:
        raise ValueError('Only whitelisted observation inputs are permitted')
    if item['reference_fields_used_for_inference'] is not False:
        raise ValueError('Reference-derived inputs prohibited')
    x, z = np.asarray(item['x']), np.asarray(item['z'])
    if (x.ndim != 2 or x.shape[1] != len(robust_control.FEATURE_NAMES) or
            z.ndim != 2 or z.shape != (len(x), len(model['feature_names'])) or
            not np.isfinite(x).all() or not np.isfinite(z).all() or item['fps'] != 25):
        raise ValueError('Invalid observation feature shape, values or FPS')
    prior.pff_training._blocks(item['blocks'], len(x), item['fps'])
    if len(item['usable']) != len(item['blocks']) or any(type(v) is not bool for v in item['usable']):
        raise ValueError('Observation support declarations differ')
    if any(k in model for k in ('temporal_model', 'expert_gate', 'boundary_model', 'transition_model', 'typed_boundary_model')):
        raise ValueError('Unexpected learned component in fixed clean candidate')
    home = robust_control.predict_home(model['robust_control_model'], x)
    dead = predict_dead(z, model)
    probabilities = joint_probabilities(dead, home, x, z, item['blocks'], item['fps'], model['joint_model'], np.zeros(len(x)))
    labels = np.full(len(x), 'X', dtype='<U1')
    for (_, a, b), supported in zip(item['blocks'], item['usable']):
        if supported:
            labels[a:b+1] = np.array(list('HAD'))[decode(probabilities[a:b+1], item['fps'], config)]
    return labels


def predict_one(out, mid, study, ctx):
    require_match(mid)
    if any((out / name).exists() for name in ('predictions-before-references.json', 'reference-access.json', 'development-summary.json')):
        raise ValueError('Predictions cannot be regenerated after cohort freeze or reference access')
    if study != build_protocol(ctx, study['recipe']):
        raise ValueError('Prediction protocol differs')
    checked_hash(REBUILD / 'refit/candidate.json', CANDIDATE_SHA)
    checked_hash(REBUILD / 'refit/config.json', CONFIG_SHA)
    item = observations(mid, ctx)
    labels = predict_observations(item, load_model(REBUILD / 'refit/candidate.json'), load_config(REBUILD / 'refit/config.json'))
    folder = out / 'predictions' / str(mid)
    folder.mkdir(parents=True, exist_ok=False)
    np.save(folder / 'candidate.npy', labels, allow_pickle=False)
    previous = next(r for r in ctx['old_bundle']['development']['prediction_freeze']['records'] if r['match_id'] == mid)
    for name, path, expected in (
        ('baseline', PARENT / 'cache' / str(mid) / 'baseline.npy', previous['baseline_prediction_sha256']),
        ('parent', PARENT / 'development' / str(mid) / 'candidate.npy', previous['candidate_prediction_sha256'])):
        checked_hash(path, expected)
        control = np.load(path, allow_pickle=False)
        if control.shape != labels.shape or not np.isin(control, list('HADX')).all():
            raise ValueError('Historical control shape or labels differ')
        (folder / f'{name}.npy').write_bytes(path.read_bytes())
    proof = dict(match_id=mid, protocol_sha256=study['protocol_sha256'], candidate_sha256=CANDIDATE_SHA,
        cache_manifest_sha256=CACHE_HASHES[str(mid)],
        prediction_sha256={n: sha256(folder / f'{n}.npy') for n in NAMES})
    write_json(folder / 'complete.json', proof)
    return proof


def prediction_records(out, study, ctx):
    if study != build_protocol(ctx, study['recipe']):
        raise ValueError('Study source or protocol changed')
    records = []
    for mid in IDS:
        proof = read(out / 'predictions' / str(mid) / 'complete.json')
        if (set(proof) != {'match_id', 'protocol_sha256', 'candidate_sha256', 'cache_manifest_sha256', 'prediction_sha256'} or
                proof['match_id'] != mid or proof['protocol_sha256'] != study['protocol_sha256'] or
                proof['candidate_sha256'] != CANDIDATE_SHA or proof['cache_manifest_sha256'] != CACHE_HASHES[str(mid)] or
                set(proof['prediction_sha256']) != set(NAMES)):
            raise ValueError('Prediction completion declaration differs')
        old = next(r for r in ctx['old_bundle']['development']['prediction_freeze']['records'] if r['match_id'] == mid)
        if proof['prediction_sha256']['baseline'] != old['baseline_prediction_sha256'] or proof['prediction_sha256']['parent'] != old['candidate_prediction_sha256']:
            raise ValueError('Frozen historical control bytes differ')
        for name, identity in proof['prediction_sha256'].items():
            checked_hash(out / 'predictions' / str(mid) / f'{name}.npy', identity)
        records.append(proof)
    return records


def freeze_value(study, records):
    return dict(schema_version=1, part='development', protocol_sha256=study['protocol_sha256'],
                candidate_sha256=CANDIDATE_SHA, source_hashes=study['source_hashes'], records=records,
                references_accessed_in_this_study=False, development_previously_consumed=True, **FLAGS)


def verify_freeze(out, study, ctx):
    value = freeze_value(study, prediction_records(out, study, ctx))
    if read(out / 'predictions-before-references.json') != value:
        raise ValueError('Complete prediction cohort freeze differs')
    return value


def access_value(out, study):
    return dict(schema_version=1, part='development', protocol_sha256=study['protocol_sha256'],
                candidate_sha256=CANDIDATE_SHA, source_hashes=study['source_hashes'],
                prediction_freeze_sha256=sha256(out / 'predictions-before-references.json'),
                references_consumed=True, development_previously_consumed=True, **FLAGS)


def begin_access(out, study):
    marker = access_value(out, study)
    with (out / 'reference-access.json').open('x', encoding='utf-8') as stream:
        json.dump(marker, stream, indent=2, sort_keys=True, allow_nan=False)
    return marker


def read_reference(out, mid, study, ctx):
    require_match(mid)
    verify_freeze(out, study, ctx)
    if read(out / 'reference-access.json') != access_value(out, study):
        raise ValueError('Exclusive reference-access marker differs')
    manifest = manifests(ctx)[str(mid)]
    checked_hash(PARENT / 'cache' / str(mid) / 'alignment.json', manifest['files']['alignment.json'])
    checked_hash(ROOT / 'data/pff/Event Data' / f'{mid}.json', manifest['identity']['events'])
    return prior.reference(PARENT, mid)


def score(out, study, ctx):
    verify_freeze(out, study, ctx)
    begin_access(out, study)  # failure/partial scoring remains consumed
    records, parent_records, details, supports = [], [], [], {}
    for mid in IDS:
        item = _observation_meta(mid, ctx)
        y, supports[str(mid)] = read_reference(out, mid, study, ctx)
        values = {}
        for name in NAMES:
            labels = np.load(out / 'predictions' / str(mid) / f'{name}.npy', allow_pickle=False)
            detail, values[name] = prior.score_one(out, 'scores', mid, name, y, labels, item['blocks'])
            details.append(detail)
        d = detail['temporal_diagnostics']
        shared = dict(match_id=mid, condition='native', baseline_sha256=original.BASELINE_SHA256,
                      baseline=values['baseline'], reference_dead_segments=d['dead_segments']['n_gt'],
                      reference_short_spells=d['short_possessions']['n_gt'])
        parent_records.append(dict(shared, candidate=values['parent'], candidate_sha256=precision.PARENT_SHA256))
        records.append(dict(shared, candidate=values['candidate'], candidate_sha256=CANDIDATE_SHA))
    if original.digest(parent_records) != policy.ORIGINAL_PARENT_RECORDS_SHA256:
        raise ValueError('Historical control metrics/support differ; preserve consumed attempt')
    if supports != ctx['old_bundle']['development']['summary']['reference_support']:
        raise ValueError('Reference reconstruction support differs')
    gate = precision.development_gate(records, ctx['prior_records'], ctx['reservation'])
    # Missing receipts explicitly remain unresolved; numerical success alone is
    # not a provider-rights decision or an authorization to read final fixtures.
    screening = policy.screen_development(records, ctx['prior_records'], ctx['reservation'],
        ctx['lineage'], [], ROOT, model_paths=ctx['model_paths'], artifact_root=ROOT)
    result = dict(schema_version=1, part='development', protocol_sha256=study['protocol_sha256'],
        candidate_sha256=CANDIDATE_SHA, source_hashes=study['source_hashes'], records=records,
        parent_records=parent_records, details=details, reference_support=supports, gate=gate,
        evidence_screen=screening, development_gate_passed=gate['numeric_gate_passed'],
        prediction_freeze_sha256=sha256(out / 'predictions-before-references.json'),
        reference_access_sha256=sha256(out / 'reference-access.json'),
        reference_quality='PFF provider-event-derived silver; previously used development', **FLAGS)
    write_json(out / 'development-summary.json', result)
    return result


def _atomic(path, value):
    temporary = path.with_name(path.name + '.pending')
    write_json(temporary, value)
    temporary.replace(path)


def stage_files(out, stage):
    if stage == 'initialize':
        return ['protocol.json']
    if stage.startswith('predict-'):
        mid = require_match(int(stage[8:]))
        return [f'predictions/{mid}/{name}.npy' for name in NAMES] + [f'predictions/{mid}/complete.json']
    if stage == 'freeze':
        return ['predictions-before-references.json']
    if stage == 'score':
        return ['development-summary.json', 'reference-access.json'] + [p.relative_to(out).as_posix() for p in sorted((out / 'scores').rglob('*')) if p.is_file()]
    raise ValueError('Unknown development stage')


def _complete(out, stage, study, state):
    proof = dict(stage=stage, protocol_sha256=study['protocol_sha256'],
                 artifacts={name: sha256(out / name) for name in stage_files(out, stage)})
    path = out / 'stages' / f'{stage}.json'
    write_json(path, proof)
    state['completed'][stage] = sha256(path)
    _atomic(out / 'state.json', state)


def authenticate(out, parameters=None, ctx=None):
    ctx = context() if ctx is None else ctx
    study = read(out / 'protocol.json')
    if study != build_protocol(ctx, parameters):
        raise ValueError('Frozen scientific source or protocol changed')
    state = read(out / 'state.json')
    completed = state.get('completed')
    if (state.get('protocol_sha256') != study['protocol_sha256'] or not isinstance(completed, dict) or
            set(completed) != set(STAGES[:len(completed)])):
        raise ValueError('Stages must be completed in frozen order')
    for stage, identity in completed.items():
        path = out / 'stages' / f'{stage}.json'
        checked_hash(path, identity)
        proof = read(path)
        if proof['stage'] != stage or proof['protocol_sha256'] != study['protocol_sha256'] or set(proof['artifacts']) != set(stage_files(out, stage)):
            raise ValueError('Stage evidence differs')
        for name, expected in proof['artifacts'].items():
            checked_hash(out / name, expected)
    return study, state, ctx


def _started(out, stage, study):
    with (out / f'{stage}-started.json').open('x', encoding='utf-8') as stream:
        json.dump(dict(stage=stage, protocol_sha256=study['protocol_sha256'], candidate_sha256=CANDIDATE_SHA,
                       source_hashes=study['source_hashes'], pid=os.getpid()), stream, sort_keys=True)


def run(out, parameters=None):
    parameters = recipe(parameters)
    out = location(out)
    out.mkdir(parents=True, exist_ok=True)
    with locked(out):
        ctx = context()
        if not (out / 'protocol.json').exists():
            if any(p.name != '.lock' for p in out.iterdir()):
                raise ValueError('Use an empty study directory; interrupted initialization stays consumed')
            study = build_protocol(ctx, parameters)
            stage = 'initialize'
            _started(out, stage, study)
            write_json(out / 'protocol.json', study)
            state = dict(protocol_sha256=study['protocol_sha256'], completed={})
        else:
            study, state, ctx = authenticate(out, parameters, ctx)
            pending = [s for s in STAGES if s not in state['completed']]
            if not pending:
                return read(out / 'summary.json')
            stage = pending[0]
            _started(out, stage, study)
            if stage.startswith('predict-'):
                predict_one(out, int(stage[8:]), study, ctx)
            elif stage == 'freeze':
                write_json(out / 'predictions-before-references.json', freeze_value(study, prediction_records(out, study, ctx)))
            elif stage == 'score':
                score(out, study, ctx)
            else:
                raise ValueError('Unexpected pending stage')
        if study != build_protocol(context(), parameters):
            raise ValueError('Source, model or observation identity changed during stage')
        _complete(out, stage, study, state)
        remaining = [s for s in STAGES if s not in state['completed']]
        summary = dict(schema_version=1, status='in_progress' if remaining else 'complete',
            stage_completed=stage, next_stage=remaining[0] if remaining else None,
            protocol_sha256=study['protocol_sha256'], candidate_sha256=CANDIDATE_SHA,
            protocol_file_sha256=sha256(out / 'protocol.json'),
            source_hashes=study['source_hashes'], **FLAGS)
        if not remaining:
            result = read(out / 'development-summary.json')
            summary.update(development_gate_passed=result['development_gate_passed'],
                           development_summary_sha256=sha256(out / 'development-summary.json'))
        _atomic(out / 'summary.json', summary)
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'preflight', 'run'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--recipe', type=Path)
    parser.add_argument('--stage', choices=('next',), default='next')
    args = parser.parse_args()
    parameters = recipe(read(args.recipe) if args.recipe else None)
    result = run(args.out, parameters) if args.action == 'run' else preflight(args.out, parameters)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()

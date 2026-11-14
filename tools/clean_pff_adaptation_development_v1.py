"""One authenticated A refit on the original development twelve; never final.

Provider arrays and fitted weights stay local. No fit, model override, candidate
grid, submission or final-access interface exists. Use the owned operator.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np

from tools import clean_pff_adaptation_protocol as adaptation
from tools import clean_pff_adaptation_refit as refit
from tools import clean_pff_adaptation_operator_v2 as predecessor
from tools import clean_pff_adaptation_stages as inner_store
from tools import external_pff_development as old
from tools import experiment_harness as storage
from tools import improvement_policy as policy
from tools import owned_python_launch as owned
from tools import pff_precision_validation as precision
from tools import pff_validation as original

ROOT = Path(__file__).resolve().parents[1]
OUT = 'runs/clean-pff-adaptation-development-v1'
INNER = 'runs/clean-pff-adaptation-inner-v1'
REFIT = 'runs/clean-pff-adaptation-refit-v1'
IDS = tuple(old.IDS)
NAMES = tuple(old.NAMES)
STAGES = ('initialize', *(f'predict-{mid}' for mid in IDS), 'freeze',
          *(f'score-{mid}' for mid in IDS), 'summarize')
FLAGS = dict(external_only=True, final_accessed=False, final_access_authorized=False,
             promoted=False, submission_eligible=False)
NEW_SOURCES = (
    'tools/clean_pff_adaptation_development_v1.py',
    'tools/clean_pff_adaptation_development_operator_v1.py',
    'tools/verify_clean_pff_adaptation_development_v1.py',
    'tools/verify_external_pff_development.py',
    'tests/test_clean_pff_adaptation_development_v1.py',
    'tests/test_clean_pff_adaptation_development_safety_v1.py',
)
AUTHENTICATED_TREES = (INNER, REFIT, predecessor.OUT, predecessor.PRIOR)


def need(value, message):
    if not value:
        raise ValueError(message)


def root():
    return inner_store._no_links(ROOT / OUT)


def path(relative):
    need(isinstance(relative, str) and relative and '\\' not in relative and ':' not in relative
         and not Path(relative).is_absolute() and '..' not in Path(relative).parts,
         'Plain local artifact path required')
    result = inner_store._no_links(root() / relative)
    need(result.is_relative_to(root()), 'Artifact escapes fixed study')
    return result


def read(relative):
    return json.loads(path(relative).read_bytes())


def write(relative, value):
    target = path(relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as stream:
        stream.write(storage.canonical(value) + b'\n')
        stream.flush()
        os.fsync(stream.fileno())


def require_match(mid):
    need(type(mid) is int and mid in IDS, 'Only fixed development fixtures are allowed')
    return mid


def require_stage(stage):
    need(type(stage) is str and stage in STAGES, 'Unknown or forbidden development stage')
    return stage


def sources():
    # The original closure stays immutable; this version adds only named files.
    result = predecessor.sources()
    for name in NEW_SOURCES:
        target = inner_store._no_links(ROOT / name)
        need(target.is_file(), 'Complete reviewed development source closure is required')
        result[name] = storage.file_hash(target)
    return dict(sorted(result.items()))


def connected_lineage(clean, complete, marker):
    """Declare the actual clean-parent refit, with selection separate from weights."""
    need(complete['selected_candidate_id'] == 'A' and complete['refit_complete'] is True,
         'Only the verified selected A refit is admissible')
    candidate = complete['model_sha256']
    need(original._sha(candidate) and candidate != adaptation.CLEAN_SHA256,
         'New refit byte identity required')
    graph = copy.deepcopy(clean['lineage'])
    train = [m for m in clean['cache_manifests'] if m['match_id'] in adaptation.TRAIN_IDS]
    need(len(train) == 40 and {m['match_id'] for m in train} == set(adaptation.TRAIN_IDS),
         'Exactly original forty training manifests required')
    dataset = storage.digest(dict(train_ids=list(adaptation.TRAIN_IDS),
                                  manifests=sorted(train, key=lambda row: row['match_id'])))
    code = storage.digest(marker['source_hashes'])
    recipe = storage.digest(adaptation.recipe())
    selection = complete['inner_proof_sha256']
    nodes = [
        dict(sha256=dataset, kind='dataset', scope='external', identifier=policy.EVALUATION_DATASET, parents=[]),
        dict(sha256=code, kind='code', scope='synthetic', identifier='Fixed adaptation refit producer source closure', parents=[]),
        dict(sha256=recipe, kind='configuration', scope='synthetic', identifier='Fixed adaptation fitting recipe', parents=[code]),
        dict(sha256=selection, kind='configuration', scope='external',
             identifier='Verified fixed A selection on original training partition; not a weight warm start',
             parents=[dataset, code]),
        dict(sha256=candidate, kind='model', scope='external', identifier='Clean-parent A refit original forty plus Sportec seven',
             parents=[adaptation.CLEAN_SHA256, dataset, code, recipe, selection]),
    ]
    graph['artifacts'].extend(nodes)
    graph['candidate_sha256'] = candidate
    graph['fit_inputs'] = sorted(set(graph['fit_inputs']) | {
        adaptation.CLEAN_SHA256, dataset, code, recipe, selection})
    bindings = dict(clean['model_paths'], **{candidate: f'{REFIT}/model.json'})
    audit = policy.audit_lineage(graph)
    need(audit['external_dataset_ids'] == ['pff-world-cup-2022', 'sportec-idsse-28196177'],
         'Unexpected fitting dataset ancestry')
    return graph, bindings


def context():
    """Actual local byte authentication; call only in authorized local execution."""
    status = predecessor.status()
    need(status['complete'] is True and status['blocked'] is False,
         'Original bounded A/B and winner refit must be complete')
    complete = refit.verify_result(ROOT / INNER)
    need(complete['selected_candidate_id'] == 'A' and complete['fitting_fixtures'] == 47,
         'Exactly the verified A winner refit is required')
    marker_path = inner_store._no_links(ROOT / REFIT / 'attempt.json')
    marker = json.loads(marker_path.read_bytes())
    need(storage.digest(marker['inner_proof']) == complete['inner_proof_sha256'],
         'Refit selection binding differs')
    clean = old.context()
    graph, bindings = connected_lineage(clean, complete, marker)
    policy.authenticate_models(policy.audit_lineage(graph), bindings, ROOT)
    names = [f'{REFIT}/{name}' for name in ('attempt.json', 'model.json', 'report.json', 'complete.json')]
    names += [f'{INNER}/manifest.json', f'{predecessor.OUT}/manifest.json',
              'models/pff-research.json', 'models/external-rebuild-001-research.json',
              'configs/baselines/v1.2.json']
    names += list(bindings.values())
    trees = {}
    for relative in AUTHENTICATED_TREES:
        folder = inner_store._no_links(ROOT/relative)
        members = sorted(inner_store._no_links(p).relative_to(ROOT).as_posix()
                         for p in folder.rglob('*') if p.is_file() and p.name != '.lock')
        trees[relative] = members
        names.extend(members)
    identities = {name: storage.file_hash(inner_store._no_links(ROOT/name)) for name in names}
    need(identities[f'{REFIT}/model.json'] == complete['model_sha256'] and
         identities['configs/baselines/v1.2.json'] == adaptation.CONFIG_SHA256,
         'Fixed refit or decoder bytes differ')
    return dict(clean, lineage=graph, model_paths=bindings,
                candidate_sha256=complete['model_sha256'], refit_proof=complete,
                refit_marker=marker, inner_proof=marker['inner_proof'], input_hashes=identities,
                authenticated_tree_files=trees)


def recheck_context(ctx):
    """Within one process only: retain full audit, recheck all bound actual bytes."""
    need(set(ctx['authenticated_tree_files']) == set(AUTHENTICATED_TREES), 'Authenticated tree roster differs')
    for relative, expected in ctx['authenticated_tree_files'].items():
        folder = inner_store._no_links(ROOT/relative)
        current = sorted(inner_store._no_links(p).relative_to(ROOT).as_posix()
                         for p in folder.rglob('*') if p.is_file() and p.name != '.lock')
        need(current == expected, 'Authenticated completed attempt inventory changed')
    for name, identity in ctx['input_hashes'].items():
        need(storage.file_hash(inner_store._no_links(ROOT/name)) == identity,
             'Authenticated completed input bytes changed')
    return ctx


def build_protocol(ctx, source_hashes=None):
    need(ctx['reservation']['protocol_sha256'] == policy.RESERVATION_SHA256 and
         ctx['reservation']['split']['development'] == list(IDS) and
         storage.digest(ctx['prior_records']) == policy.ORIGINAL_PARENT_RECORDS_SHA256,
         'Original reservation or retention anchor differs')
    need(ctx['refit_proof']['selected_candidate_id'] == 'A' and
         ctx['candidate_sha256'] == ctx['refit_proof']['model_sha256'] == ctx['lineage']['candidate_sha256'],
         'Candidate/selection/ancestry mismatch')
    value = dict(schema_version=1, family='clean-pff-adaptation-development-v1',
        candidate_sha256=ctx['candidate_sha256'], selected_candidate_id='A',
        config_sha256=adaptation.CONFIG_SHA256, baseline_sha256=original.BASELINE_SHA256,
        parent_sha256=precision.PARENT_SHA256, reservation_sha256=policy.RESERVATION_SHA256,
        original_parent_records_sha256=policy.ORIGINAL_PARENT_RECORDS_SHA256,
        refit_proof_sha256=storage.digest(ctx['refit_proof']), inner_proof_sha256=storage.digest(ctx['inner_proof']),
        lineage_sha256=storage.digest(ctx['lineage']), model_bindings_sha256=storage.digest(ctx['model_paths']),
        input_hashes=ctx['input_hashes'], source_hashes=sources() if source_hashes is None else source_hashes,
        authenticated_tree_files=ctx['authenticated_tree_files'],
        cache_manifest_sha256=dict(old.CACHE_HASHES), matches=list(IDS), stages=list(STAGES),
        runtime_identity=owned.runtime_identity(), runtime_versions=inner_store.runtime_versions(),
        limits=dict(data_workers=1, blas_threads=1, torch_threads=1, stage_timeout_seconds=1800),
        policy=dict(policy.POLICY), candidate_budget=1, fitting_permitted=False,
        development_previously_consumed=True, reference_quality='PFF provider-event-derived silver',
        fitted_weights_must_remain_local=True, artifact_delivery_rights_verified=False, **FLAGS)
    return dict(value, protocol_sha256=storage.digest(value))


def plan():
    return build_protocol(context())


def authenticate(ctx=None):
    check_inventory()
    ctx = context() if ctx is None else recheck_context(ctx)
    study = read('protocol.json')
    need(storage.canonical(study) == storage.canonical(build_protocol(ctx)),
         'Frozen development sources, runtime, inputs or policy changed')
    return study, ctx


def state():
    check_inventory()
    study = read('protocol.json')
    completed = {}
    missing = False
    for stage in STAGES:
        target = path(f'stages/{stage}.json')
        if target.exists():
            need(not missing, 'Development stage completion is out of order')
            completed[stage] = storage.file_hash(target)
        else:
            missing = True
    folder = path('stages')
    if folder.exists():
        need({p.name for p in folder.iterdir()} == {s+'.json' for s in completed},
             'Unknown development stage proof')
    return dict(protocol_sha256=study['protocol_sha256'], completed=completed)


def check_inventory():
    """Reject unexpected/cohort artifacts without opening their contents."""
    allowed = {'protocol.json','operator','stages','predictions','scores','details',
               'lineage.json','model-bindings.json','handoff.json',
               'predictions-before-references.json','development-summary.json'}
    allowed |= {stage+'-started.json' for stage in STAGES}
    need(root().is_dir() and all(p.name in allowed for p in root().iterdir()),
         'Unexpected study artifact or final activity')
    for item in root().rglob('*'):
        inner_store._no_links(item)
    operator_folder = path('operator')
    if operator_folder.exists():
        need(operator_folder.is_dir() and all(p.name in {'.lock','initialized.json','attempts'}
             for p in operator_folder.iterdir()), 'Unexpected operator artifact')
    for category in ('predictions','scores','details'):
        folder = path(category)
        if not folder.exists():
            continue
        need(folder.is_dir(), 'Artifact category must be a directory')
        for match in folder.iterdir():
            need(match.is_dir() and match.name in {str(mid) for mid in IDS},
                 'Forbidden cohort artifact')
            if category == 'predictions':
                permitted = {'candidate.npy','baseline.npy','parent.npy','complete.json'}
                need(all(p.is_file() and p.name in permitted for p in match.iterdir()), 'Unexpected prediction artifact')
            elif category == 'scores':
                need(all(p.is_file() and p.name in {'access.json','aggregate.json'} for p in match.iterdir()),
                     'Unexpected score artifact')
            else:
                need(all(p.is_dir() and p.name in NAMES for p in match.iterdir()), 'Unexpected score-detail model')
                for model in match.iterdir():
                    need(all(p.is_file() and p.name in {'predictions.csv','metrics.json'} for p in model.iterdir()),
                         'Unexpected score-detail artifact')


def expected_access(study, mid, freeze_hash):
    require_match(mid)
    return dict(schema_version=1, part='development', match_id=mid,
        protocol_sha256=study['protocol_sha256'], candidate_sha256=study['candidate_sha256'],
        prediction_freeze_sha256=freeze_hash, references_consumed=True,
        development_previously_consumed=True, **FLAGS)


def prediction_records(study, ctx):
    rows = []
    for mid in IDS:
        proof = read(f'predictions/{mid}/complete.json')
        expected_keys = {'match_id', 'protocol_sha256', 'candidate_sha256',
                         'cache_manifest_sha256', 'prediction_sha256'}
        need(set(proof) == expected_keys and type(proof['match_id']) is int and proof['match_id'] == mid and
             proof['protocol_sha256'] == study['protocol_sha256'] and
             proof['candidate_sha256'] == study['candidate_sha256'] and
             proof['cache_manifest_sha256'] == old.CACHE_HASHES[str(mid)] and
             set(proof['prediction_sha256']) == set(NAMES), 'Invalid prediction completion')
        previous = next(r for r in ctx['old_bundle']['development']['prediction_freeze']['records'] if r['match_id'] == mid)
        need(proof['prediction_sha256']['baseline'] == previous['baseline_prediction_sha256'] and
             proof['prediction_sha256']['parent'] == previous['candidate_prediction_sha256'],
             'Historical evaluation controls changed')
        for name, identity in proof['prediction_sha256'].items():
            need(storage.file_hash(path(f'predictions/{mid}/{name}.npy')) == identity,
                 'Prediction bytes changed')
        rows.append(proof)
    return rows


def freeze_value(study, rows):
    need([r['match_id'] for r in rows] == list(IDS), 'Complete development prediction cohort required')
    return dict(schema_version=1, part='development', protocol_sha256=study['protocol_sha256'],
        candidate_sha256=study['candidate_sha256'], source_hashes=study['source_hashes'], records=rows,
        references_accessed_in_this_study=False, development_previously_consumed=True, **FLAGS)


def verify_freeze(study, ctx):
    frozen = freeze_value(study, prediction_records(study, ctx))
    need(read('predictions-before-references.json') == frozen, 'Cohort prediction freeze differs')
    return frozen


def _predict(mid, study, ctx):
    require_match(mid)  # cohort guard precedes all dataset I/O
    need(not path('predictions-before-references.json').exists() and not path('scores').exists(),
         'Prediction regeneration after freeze/reference access is forbidden')
    item = old.observations(mid, ctx)  # exact model-independent x/z, never cached probabilities
    model = old.load_model(ROOT / REFIT / 'model.json')
    config = old.load_config(ROOT / 'configs/baselines/v1.2.json')
    labels = old.predict_observations(item, model, config)
    folder = path(f'predictions/{mid}')
    folder.mkdir(parents=True, exist_ok=False)
    with (folder/'candidate.npy').open('xb') as stream:
        np.save(stream, labels, allow_pickle=False)
    previous = next(r for r in ctx['old_bundle']['development']['prediction_freeze']['records'] if r['match_id'] == mid)
    for name, source, identity in (
        ('baseline', old.PARENT/'cache'/str(mid)/'baseline.npy', previous['baseline_prediction_sha256']),
        ('parent', old.PARENT/'development'/str(mid)/'candidate.npy', previous['candidate_prediction_sha256'])):
        old.checked_hash(source, identity)
        control = np.load(source, allow_pickle=False)
        need(control.shape == labels.shape and np.isin(control, list('HADX')).all(), 'Historical control shape differs')
        with (folder/f'{name}.npy').open('xb') as stream:
            stream.write(source.read_bytes())
    write(f'predictions/{mid}/complete.json', dict(match_id=mid, protocol_sha256=study['protocol_sha256'],
        candidate_sha256=study['candidate_sha256'], cache_manifest_sha256=old.CACHE_HASHES[str(mid)],
        prediction_sha256={name: storage.file_hash(folder/f'{name}.npy') for name in NAMES}))


def _score(mid, study, ctx):
    require_match(mid)  # deny final IDs even before inspecting local reference paths
    verify_freeze(study, ctx)
    freeze_hash = storage.file_hash(path('predictions-before-references.json'))
    write(f'scores/{mid}/access.json', expected_access(study, mid, freeze_hash))
    manifest = old.manifests(ctx)[str(mid)]
    old.checked_hash(old.PARENT/'cache'/str(mid)/'alignment.json', manifest['files']['alignment.json'])
    old.checked_hash(ROOT/'data/pff/Event Data'/f'{mid}.json', manifest['identity']['events'])
    y, support = old.prior.reference(old.PARENT, mid)
    need(support == ctx['old_bundle']['development']['summary']['reference_support'][str(mid)],
         'Historical reference support differs')
    item = old._observation_meta(mid, ctx)
    values, details = {}, []
    for name in NAMES:
        labels = np.load(path(f'predictions/{mid}/{name}.npy'), allow_pickle=False)
        detail, values[name] = old.prior.score_one(root(), 'details', mid, name, y, labels, item['blocks'])
        details.append(detail)
    diagnostic = details[-1]['temporal_diagnostics']
    shared = dict(match_id=mid, condition='native', baseline_sha256=original.BASELINE_SHA256,
        baseline=values['baseline'], reference_dead_segments=diagnostic['dead_segments']['n_gt'],
        reference_short_spells=diagnostic['short_possessions']['n_gt'])
    prior_row = dict(shared, candidate=values['parent'], candidate_sha256=precision.PARENT_SHA256)
    need(prior_row == next(r for r in ctx['prior_records'] if r['match_id'] == mid),
         'Recomputed historical baseline/parent metrics differ')
    write(f'scores/{mid}/aggregate.json', dict(match_id=mid, protocol_sha256=study['protocol_sha256'],
        candidate_sha256=study['candidate_sha256'], candidate_record=dict(shared, candidate=values['candidate'],
        candidate_sha256=study['candidate_sha256']), parent_record=prior_row, details=details,
        reference_support=support, prediction_freeze_sha256=freeze_hash,
        reference_access_sha256=storage.file_hash(path(f'scores/{mid}/access.json'))))


def aggregate_scores(study, ctx):
    verify_freeze(study, ctx)
    freeze_hash = storage.file_hash(path('predictions-before-references.json'))
    records, parent_records, details, support, access_hashes = [], [], [], {}, {}
    for mid in IDS:
        value = read(f'scores/{mid}/aggregate.json')
        access = read(f'scores/{mid}/access.json')
        need(access == expected_access(study, mid, freeze_hash), 'Reference access proof differs')
        identity = storage.file_hash(path(f'scores/{mid}/access.json'))
        need(value['match_id'] == mid and value['protocol_sha256'] == study['protocol_sha256'] and
             value['candidate_sha256'] == study['candidate_sha256'] and
             value['prediction_freeze_sha256'] == freeze_hash and value['reference_access_sha256'] == identity,
             'Score identity differs')
        records.append(value['candidate_record']); parent_records.append(value['parent_record'])
        details.extend(value['details']); support[str(mid)] = value['reference_support']; access_hashes[str(mid)] = identity
    need(storage.digest(parent_records) == policy.ORIGINAL_PARENT_RECORDS_SHA256 and
         support == ctx['old_bundle']['development']['summary']['reference_support'],
         'Original comparator anchor or reference support differs')
    gate = precision.development_gate(records, ctx['prior_records'], ctx['reservation'])
    screen = policy.screen_development(records, ctx['prior_records'], ctx['reservation'],
        ctx['lineage'], [], ROOT, model_paths=ctx['model_paths'], artifact_root=ROOT)
    need(screen['ready_for_independent_final_review'] is False and screen['final_access_authorized'] is False,
         'Unresolved rights/reference quality must keep final review blocked')
    return dict(schema_version=1, part='development', protocol_sha256=study['protocol_sha256'],
        candidate_sha256=study['candidate_sha256'], source_hashes=study['source_hashes'], records=records,
        parent_records=parent_records, details=details, reference_support=support, gate=gate,
        evidence_screen=screen, development_gate_passed=gate['numeric_gate_passed'],
        prediction_freeze_sha256=freeze_hash, reference_access_sha256=access_hashes,
        reference_quality='PFF provider-event-derived silver; previously used development', **FLAGS)


def stage_files(stage):
    require_stage(stage)
    if stage == 'initialize':
        return ['lineage.json', 'model-bindings.json', 'handoff.json']
    if stage == 'freeze':
        return ['predictions-before-references.json']
    if stage == 'summarize':
        return ['development-summary.json']
    category, value = stage.split('-'); mid = require_match(int(value))
    if category == 'predict':
        return [f'predictions/{mid}/{name}' for name in ('candidate.npy', 'baseline.npy', 'parent.npy', 'complete.json')]
    result = [f'scores/{mid}/{name}' for name in ('access.json', 'aggregate.json')]
    folder = path(f'details/{mid}')
    need(folder.is_dir(), 'Score detail artifacts missing')
    expected = {f'details/{mid}/{name}/{filename}' for name in NAMES
                for filename in ('predictions.csv','metrics.json')}
    need({p.relative_to(root()).as_posix() for p in folder.rglob('*') if p.is_file()} == expected and
         {p.name for p in folder.iterdir()} == set(NAMES), 'Exact score-detail inventory required')
    result += sorted(expected)
    return result


def verify_stage(stage):
    require_stage(stage)
    study = read('protocol.json')
    proof = read(f'stages/{stage}.json')
    files = stage_files(stage)
    expected = dict(stage=stage, protocol_sha256=study['protocol_sha256'],
                    artifacts={name: storage.file_hash(path(name)) for name in files})
    need(proof == expected, 'Completed stage artifact bytes changed')
    marker = read(stage+'-started.json')
    need(set(marker) == {'stage', 'protocol_sha256', 'candidate_sha256', 'source_hashes', 'pid'} and
         marker['stage'] == stage and marker['protocol_sha256'] == study['protocol_sha256'] and
         marker['candidate_sha256'] == study['candidate_sha256'] and marker['source_hashes'] == study['source_hashes']
         and type(marker['pid']) is int and marker['pid'] > 0, 'Stage start identity differs')
    return expected


def run_stage(stage, *, admission):
    require_stage(stage)
    need(isinstance(admission, dict) and admission.get('worker_pid') == os.getpid() and
         admission.get('runtime_verified') is True, 'Owned worker admission is required')
    from tools import clean_pff_adaptation_development_operator_v1 as operator
    operator.verify_admission(admission, stage)
    need(all(os.environ.get(k) == '1' for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')),
         'One thread per worker required')
    study, ctx = authenticate(operator.verified_context())
    completed = state()['completed']
    need(len(completed) < len(STAGES) and STAGES[len(completed)] == stage, 'Fixed stage order required')
    for previous in completed:
        verify_stage(previous)
    write(stage+'-started.json', dict(stage=stage, protocol_sha256=study['protocol_sha256'],
        candidate_sha256=study['candidate_sha256'], source_hashes=study['source_hashes'], pid=os.getpid()))
    if stage == 'initialize':
        write('lineage.json', ctx['lineage']); write('model-bindings.json', ctx['model_paths'])
        write('handoff.json', dict(refit_proof=ctx['refit_proof'], inner_proof=ctx['inner_proof'],
                                  input_hashes=ctx['input_hashes'], **FLAGS))
    elif stage.startswith('predict-'):
        _predict(require_match(int(stage[8:])), study, ctx)
    elif stage == 'freeze':
        write('predictions-before-references.json', freeze_value(study, prediction_records(study, ctx)))
    elif stage.startswith('score-'):
        _score(require_match(int(stage[6:])), study, ctx)
    else:
        write('development-summary.json', aggregate_scores(study, ctx))
    need(authenticate(ctx)[0] == study, 'Sources or authenticated inputs changed during execution')
    write(f'stages/{stage}.json', dict(stage=stage, protocol_sha256=study['protocol_sha256'],
        artifacts={name: storage.file_hash(path(name)) for name in stage_files(stage)}))
    return verify_stage(stage)

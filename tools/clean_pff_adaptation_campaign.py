"""Fixed inner29 data dispatcher and artifact verification; no development/final routes.

All arrays and fitted weights stay under ignored local artifacts. Public execution
owns a consumed lifecycle attempt. A separate operator must bind these sources
and enforce process/resource limits. A verified numerical winner is not a release.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np

from possession import learned, robust_control
from possession.config import Config
from possession.decoder import decode
from evaluation import possession_metrics as pm
from tools import clean_pff_adaptation_protocol as protocol
from tools import clean_pff_adaptation_fit as kernel
from tools import clean_pff_adaptation_inputs as inputs
from tools import clean_pff_adaptation_stages as stages
from tools import clean_pff_training_diagnosis as diagnosis

ROOT = Path(__file__).resolve().parents[1]
IDS = protocol.INNER_VALIDATION_IDS
FLAGS = dict(refit_authorized=False, development_access_authorized=False,
             final_access_authorized=False, promoted=False, submission_eligible=False)


def _need(value, message):
    if not value: raise ValueError(message)


def _location(out):
    return stages.location(out, ROOT)


def _stage(stage):
    _need(type(stage) is str and stage in stages.STAGES, 'Only fixed inner stages are permitted')
    return stage


def _mid(mid):
    _need(type(mid) is int and mid in IDS, 'Only the declared inner-validation eight are permitted')
    return mid


def _path(out, stage, name):
    _stage(stage)
    _need(type(name) is str and re.fullmatch(r'[a-z][a-z_]*\.(json|npy)', name), 'Invalid artifact name')
    return stages._no_links(_location(out)/'artifacts'/stage/name)


def _json_write(path, value):
    stages._exclusive(path, value)


def _array_write(path, value):
    with stages._no_links(path).open('xb') as stream:
        np.save(stream, value, allow_pickle=False)
        stream.flush()


def _read(path):
    return json.loads(stages._no_links(path).read_bytes())


def _hash(path):
    return stages.storage.file_hash(stages._no_links(path))


def _array(path, expected):
    with stages._no_links(path).open('rb') as stream:
        raw = stream.read()
        _need(hashlib.sha256(raw).hexdigest() == expected, 'Local array bytes differ')
        value = np.load(io.BytesIO(raw), allow_pickle=False)
    _need(isinstance(value, np.ndarray), 'Expected a NumPy array')
    return value


def _names(stage):
    if stage == 'initialize': return {'plan.json'}
    if stage.startswith('fit-'): return {'model.json', 'report.json', 'receipt.json'}
    if stage.startswith('predict-'): return {'prediction.npy', 'metadata.json'}
    if stage == 'freeze': return {'freeze.json'}
    if stage.startswith('score-'): return {'reference.npy', 'records.json', 'evidence.json', 'access.json'}
    if stage == 'select': return {'selection.json', 'score_receipt.json'}
    raise ValueError('Unknown stage')


def _seal(out, stage):
    folder = _path(out, stage, 'seal.json').parent
    actual = {p.name for p in folder.iterdir()}
    _need(actual == _names(stage), 'Stage outputs are incomplete or unexpected')
    value = {name: _hash(folder/name) for name in sorted(actual)}
    _json_write(folder/'seal.json', value)
    return dict(artifact_sha256={'stage_seal': _hash(folder/'seal.json')}, execution_verified=False)


def _sealed(out, stage, receipt):
    stages._receipt(receipt)
    _need(set(receipt['artifact_sha256']) == {'stage_seal'}, 'Unexpected stage receipt')
    path = _path(out, stage, 'seal.json')
    _need(_hash(path) == receipt['artifact_sha256']['stage_seal'], 'Stage seal differs from lifecycle receipt')
    seal = _read(path)
    _need(set(seal) == _names(stage) and {p.name for p in path.parent.iterdir()} == _names(stage)|{'seal.json'},
          'Stage artifact roster differs')
    for name, expected in seal.items():
        _need(_hash(path.parent/name) == expected, 'Stage artifact bytes changed')
    return seal


def _history(out):
    manifest, events, _, index, blocked = stages._load(_location(out), ROOT)
    completed = {}
    for event in events:
        if event['state'] == 'recorded_complete':
            _sealed(out, event['stage'], event['detail'])
            completed[event['stage']] = event['detail']
    return manifest, events, index, blocked, completed


def _context():
    ctx = inputs._evidence()
    old = inputs._json('models/pff-research.json', inputs.dev.PARENT_BUNDLE_SHA)
    rows = [r for r in old['cache_manifests'] if r['match_id'] in IDS]
    inner = {r['match_id']: r for r in rows}
    _need(len(rows) == 8 and set(inner) == set(IDS), 'Inner observation manifest cohort differs')
    for row in rows:
        _need(row['identity']['sources'] == ctx['producer_sources'], 'Inner observation producer differs')
        baseline = next(r for r in ctx['snapshot']['baseline_records'] if r['match_id'] == row['match_id'])
        _need(baseline['reference_sha256'] == row['identity']['events'], 'Comparator reference event identity differs')
    ctx['inner'] = inner
    return ctx


def _plan(ctx):
    return dict(schema_version=1, protocol_sha256=ctx['snapshot']['protocol_sha256'],
                core_protocol_sha256=protocol.fixed_protocol_sha256(),
                parent_sha256=protocol.CLEAN_SHA256, config_sha256=protocol.CONFIG_SHA256,
                inner_validation_ids=list(IDS), fitting_fixture_ids=list(protocol.fit_ids()),
                pff_manifest_sha256={str(r['match_id']):protocol.original.digest(r)
                    for r in [*ctx['pff'].values(), *ctx['inner'].values()]},
                sportec_seals=ctx['sportec_seals'], **FLAGS)


def _metadata(mid, ctx):
    _mid(mid)
    manifest = ctx['inner'][mid]; folder = f'{inputs.PFF_CACHE}/{mid}'
    raw = inputs._path(f'{folder}/complete.json').read_bytes()
    _need(hashlib.sha256(raw).hexdigest() in inputs.dev.old_verifier._serialized_hashes(manifest)
          and json.loads(raw) == manifest, 'Inner cache manifest differs')
    item = inputs._json(f'{folder}/item.json', manifest['files']['item.json'])
    _need(set(item) == inputs.PFF_ITEM_KEYS and item['match'] == f'pff{mid}'
          and item['reference_fields_used_for_inference'] is False, 'Inner observation metadata differs')
    inputs._period_metadata(item['blocks'], item['usable'], item['fps'], manifest['rows'])
    return item


def _observations(mid, ctx):
    _mid(mid); item = _metadata(mid, ctx)
    manifest = ctx['inner'][mid]; folder = f'{inputs.PFF_CACHE}/{mid}'
    item['x'] = inputs._array(f'{folder}/x.npy', manifest['files']['x.npy'])
    item['z'] = inputs._array(f'{folder}/z.npy', manifest['files']['z.npy'])
    inputs._validate_features(item['x'], item['z'], manifest['rows'])
    return item


def _reference(mid, ctx):
    _mid(mid)
    manifest = ctx['inner'][mid]; folder = f'{inputs.PFF_CACHE}/{mid}'
    alignment = inputs._json(f'{folder}/alignment.json', manifest['files']['alignment.json'])
    _need(str(alignment.get('match')) == str(mid) and alignment.get('fps') == 25 and
          alignment.get('reference_fields_used_for_inference') is False and
          len(alignment.get('source_frames', [])) == manifest['rows'], 'Inner reference alignment differs')
    events = inputs._json(f'data/pff/Event Data/{mid}.json', manifest['identity']['events'])
    y, _ = inputs.prior.pff_adapter.references(events, alignment)
    _need(y.shape == (manifest['rows'],) and np.isin(y, list('HADX')).all(), 'Invalid inner references')
    return y


def _config():
    value = inputs._json(f'{inputs.REBUILD}/refit/config.json', protocol.CONFIG_SHA256)
    return Config(**value).validate()


def _candidate_receipt(cid, model_hash, snapshot):
    return dict(candidate_id=cid, model_sha256=model_hash,
        protocol_sha256=snapshot['protocol_sha256'], core_protocol_sha256=protocol.fixed_protocol_sha256(),
        phase='inner', parent_sha256=protocol.CLEAN_SHA256, config_sha256=protocol.CONFIG_SHA256,
        recipe_sha256=protocol.original.digest(protocol.recipe()), fitting_fixture_ids=list(protocol.fit_ids()),
        normalization_fixture_ids=list(protocol.fit_ids()),
        new_dead_fit_fixture_ids=list(protocol.fit_ids()) if cid == 'B' else [],
        upstream_dead=snapshot['candidates'][cid]['upstream_dead'], upstream_stacking='in-sample-fit-only')


def _model(out, cid, ctx):
    _need(cid in protocol.CANDIDATES, 'Unknown candidate')
    path = _path(out, f'fit-{cid}', 'model.json'); model = _read(path)
    receipt = _read(_path(out, f'fit-{cid}', 'receipt.json'))
    expected = _candidate_receipt(cid, _hash(path), ctx['snapshot'])
    _need(receipt == expected, 'Fit receipt differs from actual model')
    protocol.validate_candidate_receipt(receipt, cid, ctx['snapshot'])
    learned.validate_model(model)
    parent = ctx['parent']; kernel.validate_parent(parent)
    _need(model['robust_control_model'] == parent['robust_control_model'] and
          model['decoder_mode'] == parent['decoder_mode'] and model['candidate_id'] == f'clean-pff-adaptation-{cid}-inner',
          'Fixed control or decoder changed')
    _need(set(model) == set(parent) | ({'training'} if cid == 'B' else set()), 'Unexpected candidate component')
    if cid == 'A':
        _need(all(model[k] == parent[k] for k in kernel.DEAD_PARAMETERS), 'Candidate A upstream D changed')
    meta = model['metadata']['adaptation']
    _need(meta['candidate'] == cid and meta['phase'] == 'inner' and meta['training_matches'] == list(protocol.fit_ids())
          and meta['protocol_sha256'] == protocol.fixed_protocol_sha256()
          and meta['clean_parent_content_sha256'] == kernel.PARENT_CONTENT_SHA256
          and meta['old_joint_used'] is False and meta['external_only'] is True
          and meta['final_accessed'] is False and meta['promoted'] is False and meta['submission_eligible'] is False,
          'Model training ancestry differs')
    _need(model['joint_model']['metadata'] == meta and model['joint_model']['training']['epochs'] == protocol.recipe()['epochs']
          and model['joint_model']['training']['seed'] == protocol.recipe()['seed'], 'Joint training recipe differs')
    report = _read(_path(out, f'fit-{cid}', 'report.json'))
    _need(all(report[k] == v for k,v in meta.items()) and report['recipe'] == protocol.recipe()
          and report['fitting_fixtures'] == len(protocol.fit_ids()) and set(report['fixture_support']) == set(protocol.fit_ids())
          and report['new_joint_training'] == model['joint_model']['training'], 'Model/report fitting evidence differs')
    if cid == 'B':
        _need(report['new_dead_training'] == model['training'] and
              model['training']['l2'] == protocol.recipe()['dead_l2'] and
              1 <= model['training']['iterations'] <= protocol.recipe()['dead_max_iter'],
              'Refitted D-head training recipe differs')
    return model, receipt


def _predict(item, model):
    home = robust_control.predict_home(model['robust_control_model'], item['x'])
    dead = learned.predict_dead(item['z'], model)
    probs = learned.joint_probabilities(dead, home, item['x'], item['z'], item['blocks'], item['fps'],
                                       model['joint_model'], np.zeros(len(dead)))
    result = np.full(len(dead), 'X', dtype='<U1'); config = _config()
    for (_, a, b), usable in zip(item['blocks'], item['usable']):
        if usable: result[a:b+1] = np.array(list('HAD'))[decode(probs[a:b+1], item['fps'], config)]
    return result


def _prediction(out, cid, mid, ctx, receipt):
    _mid(mid); stage = f'predict-{cid}-{mid}'
    path = _path(out, stage, 'prediction.npy'); identity = _hash(path)
    meta = _read(_path(out, stage, 'metadata.json'))
    _need(meta == dict(candidate_id=cid, match_id=mid, model_sha256=receipt['model_sha256'],
                      prediction_sha256=identity, cache_manifest_sha256=protocol.original.digest(ctx['inner'][mid]),
                      reference_fields_used_for_inference=False, references_accessed=False), 'Prediction provenance differs')
    labels = _array(path, identity)
    _need(labels.shape == (ctx['inner'][mid]['rows'],) and np.isin(labels, list('HADX')).all(), 'Invalid prediction array')
    return labels, identity


def _freeze_value(out, ctx):
    candidates = {cid:_model(out,cid,ctx)[1] for cid in protocol.CANDIDATES}
    records = []
    for cid in protocol.CANDIDATES:
        for mid in IDS:
            _, identity = _prediction(out,cid,mid,ctx,candidates[cid])
            records.append(dict(candidate_id=cid,match_id=mid,prediction_sha256=identity))
    value = dict(schema_version=1,protocol_sha256=ctx['snapshot']['protocol_sha256'],
        model_sha256={cid:r['model_sha256'] for cid,r in candidates.items()},
        fit_receipt_sha256={cid:protocol.original.digest(r) for cid,r in candidates.items()},
        records=records,references_accessed=False,all_predictions_complete=True)
    protocol._freeze(value,ctx['snapshot'],candidates)
    return value,candidates


def _check_freeze(out,ctx):
    value,candidates=_freeze_value(out,ctx)
    _need(_read(_path(out,'freeze','freeze.json')) == value,'Frozen prediction cohort changed')
    return value,candidates


def _measure(reference, prediction, blocks):
    result=pm.evaluate(pm.label_codes(reference),pm.label_codes(prediction),bootstrap_n=0)
    temporal=inputs.prior.diagnostic(reference,prediction,25,blocks)
    duration=inputs.prior.duration_diagnostic(reference,prediction,25)
    row=dict(spellF1=result['spell_f1_mean'],macroF1=result['macro_f1'],
        dead_precision=result['per_class']['D']['precision'],dead_recall=result['per_class']['D']['recall'],
        temporal_diagnostics=temporal,duration_precision_recall=duration)
    metrics=inputs.prior.metrics(row);metrics['short_recall']=metrics.pop('short')
    short=diagnosis.short_false_dead(reference,{n:prediction for n in diagnosis.NAMES},blocks,25)
    sd=short['models']['full'];bins=duration['bins']
    counts=dict(predicted_short_spells=sum(bins[k]['n_pred'] for k in ('<0.4s','0.4-1s','1-2s')),
        matched_predicted_short_spells=sum(bins[k]['n_matched_pred'] for k in ('<0.4s','0.4-1s','1-2s')),
        short_spells_with_dead=sd['spells_with_any_dead'],short_spells_erased_by_dead=sd['spells_erased_by_dead'],
        short_false_dead_frames=sd['false_dead_frames'])
    support=dict(known_frames=int(np.isin(reference,list('HAD')).sum()),reference_live=int(np.isin(reference,list('HA')).sum()),
        reference_dead=int((reference=='D').sum()),reference_dead_segments=temporal['dead_segments']['n_gt'],
        reference_short_spells=short['n_gt'],reference_short_frames=short['reference_frames'])
    return metrics,counts,support


def _score_records(out,mid,ctx,reference,candidates):
    item=_metadata(mid,ctx)
    base=next(r for r in ctx['snapshot']['baseline_records'] if r['match_id']==mid)
    records=[]
    for cid in protocol.CANDIDATES:
        pred,identity=_prediction(out,cid,mid,ctx,candidates[cid])
        metrics,short,support=_measure(reference,pred,item['blocks'])
        _need(support == base['reference_support'],'Reconstructed reference support differs from pinned comparator')
        records.append(dict(candidate_id=cid,match_id=mid,condition='native',protocol_sha256=ctx['snapshot']['protocol_sha256'],
            model_sha256=candidates[cid]['model_sha256'],fit_receipt_sha256=protocol.original.digest(candidates[cid]),
            prediction_sha256=identity,baseline_sha256=protocol.CLEAN_SHA256,
            baseline_prediction_sha256=base['baseline_prediction_sha256'],reference_sha256=base['reference_sha256'],
            reference_support=support,metrics=metrics,short_diagnostics=short))
    return records


def _selection(out,ctx,verify_references=True):
    freeze,candidates=_check_freeze(out,ctx);records=[]
    for mid in IDS:
        stage=f'score-{mid}'
        access=_read(_path(out,stage,'access.json'))
        _need(access == dict(match_id=mid,prediction_freeze_sha256=protocol.original.digest(freeze),
            reference_access_after_complete_cohort=True),'Scoring access marker differs')
        evidence=_read(_path(out,stage,'evidence.json'))
        _need(evidence == dict(event_sha256=ctx['inner'][mid]['identity']['events'],
            alignment_sha256=ctx['inner'][mid]['files']['alignment.json']), 'Reference evidence identity differs')
        reference=_array(_path(out,stage,'reference.npy'),_hash(_path(out,stage,'reference.npy')))
        if verify_references:
            _need(np.array_equal(reference,_reference(mid,ctx)),'Saved labels differ from authenticated provider reconstruction')
        expected=_score_records(out,mid,ctx,reference,candidates)
        _need(_read(_path(out,stage,'records.json')) == expected,'Saved metrics differ from actual predictions/references')
        records.extend(expected)
    receipt=dict(schema_version=1,protocol_sha256=ctx['snapshot']['protocol_sha256'],
        prediction_freeze_sha256=protocol.original.digest(freeze),records_sha256=protocol.original.digest(records),
        scored_match_ids=list(IDS),condition='native',reference_access_started_after_prediction_freeze=True,
        both_candidates_scored=True,development_accessed=False,final_accessed=False)
    selected=protocol.select_candidate(records,ctx['snapshot'],candidates=candidates,prediction_freeze=freeze,score_receipt=receipt)
    return selected,receipt


def _dispatch(stage,out,ctx):
    folder=_path(out,stage,'seal.json').parent;folder.mkdir(parents=True,exist_ok=False)
    if stage=='initialize':
        _json_write(folder/'plan.json',_plan(ctx))
    elif stage.startswith('fit-'):
        cid=stage[4:]
        factory=inputs.make_inner_factory(fixture_ids=protocol.fit_ids(),protocol_sha256=protocol.fixed_protocol_sha256())
        model,report=kernel.fit_candidate(cid,factory,fixture_ids=protocol.fit_ids(),parent_model=factory.parent_model(),
                                         protocol_sha256=protocol.fixed_protocol_sha256())
        _json_write(folder/'model.json',model);_json_write(folder/'report.json',report)
        _json_write(folder/'receipt.json',_candidate_receipt(cid,_hash(folder/'model.json'),ctx['snapshot']))
        _model(out,cid,ctx)
    elif stage.startswith('predict-'):
        _,cid,mid=stage.split('-');mid=_mid(int(mid));model,receipt=_model(out,cid,ctx)
        labels=_predict(_observations(mid,ctx),model)
        _array_write(folder/'prediction.npy',labels)
        _json_write(folder/'metadata.json',dict(candidate_id=cid,match_id=mid,model_sha256=receipt['model_sha256'],
            prediction_sha256=_hash(folder/'prediction.npy'),cache_manifest_sha256=protocol.original.digest(ctx['inner'][mid]),
            reference_fields_used_for_inference=False,references_accessed=False))
        _prediction(out,cid,mid,ctx,receipt)
    elif stage=='freeze':
        value,_=_freeze_value(out,ctx);_json_write(folder/'freeze.json',value)
    elif stage.startswith('score-'):
        mid=_mid(int(stage[6:]));freeze,candidates=_check_freeze(out,ctx)
        _json_write(folder/'access.json',dict(match_id=mid,prediction_freeze_sha256=protocol.original.digest(freeze),
                                            reference_access_after_complete_cohort=True))
        reference=_reference(mid,ctx);_array_write(folder/'reference.npy',reference)
        _json_write(folder/'evidence.json',dict(event_sha256=ctx['inner'][mid]['identity']['events'],
                                              alignment_sha256=ctx['inner'][mid]['files']['alignment.json']))
        _json_write(folder/'records.json',_score_records(out,mid,ctx,reference,candidates))
    elif stage=='select':
        selected,receipt=_selection(out,ctx)
        _json_write(folder/'selection.json',selected);_json_write(folder/'score_receipt.json',receipt)
    return _seal(out,stage)


def execute_stage(stage,out):
    """Consume exactly one fixed attempt; subprocess operator supplies outer bounds."""
    _stage(stage);out=_location(out)
    with stages.attempt(out,stage,root=ROOT) as handle:
        _,events,index,_,completed=_history(out)
        _need(events[-1]['stage']==stage and events[-1]['state']=='started' and stages.STAGES[index]==stage,
              'Matching active lifecycle attempt required')
        ctx=_context()
        if stage!='initialize':_need(_read(_path(out,'initialize','plan.json'))==_plan(ctx),'Input evidence changed since initialization')
        receipt=_dispatch(stage,out,ctx)
        _sealed(out,stage,receipt)
        handle.complete(receipt)
    return receipt


def verify_stage(stage,out):
    """Verify completed stage bytes against lifecycle binding; never opens references."""
    _stage(stage);out=_location(out)
    with stages.storage.locked(out):
        _,_,_,_,completed=_history(out)
        _need(stage in completed,'Stage is not recorded complete')
        return completed[stage]


def verify_completed(out):
    """Authenticate all29 lifecycle/artifact bindings and recompute numerical selection.

    References are reconstructed only after the complete frozen/scored inner cohort
    has been authenticated. This is not independent chronology or dataset-rights proof.
    """
    out=_location(out)
    with stages.storage.locked(out):
        _,events,index,blocked,completed=_history(out)
        _need(not blocked and index==len(stages.STAGES) and set(completed)==set(stages.STAGES),'Complete inner lifecycle required')
        ctx=_context();_need(_read(_path(out,'initialize','plan.json'))==_plan(ctx),'Input evidence changed')
        selected,receipt=_selection(out,ctx)
        _need(_read(_path(out,'select','selection.json'))==selected and
              _read(_path(out,'select','score_receipt.json'))==receipt,'Selection artifact differs')
        return dict(selection=selected,selection_sha256=protocol.original.digest(selected),artifacts_verified=True,
                    lifecycle_complete=True,protocol_sha256=ctx['snapshot']['protocol_sha256'],
                    inner_ledger_tip=events[-1]['event_sha256'],**FLAGS)

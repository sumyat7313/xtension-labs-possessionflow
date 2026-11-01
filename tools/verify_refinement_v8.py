"""Verify isolated v8 research and that the released submission is untouched."""
from pathlib import Path
import json,math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from possession.io import sha256
from tools.verify_release import same_evidence,protocol_matches_hash
from tools.train_refinement_v8 import select,scientific_hashes,BASELINE,BASELINE_SHA256,CONFIG_SHA256,FOLDS,REGIMES,PRIMARY_REGIMES,CANDIDATES,GATE
from tools.train_control import CATALOG
from research import precision_v8,stoppage_v8

CSV_SHA256='1ddbe062fb0ff0bbb2f47fccda472953fee399629b3b88af9b23554b5f7a3ceb'

def verify_fold_models(evidence):
    folds=evidence['fold_models']
    if not isinstance(folds,dict) or set(folds)!={str(i) for i in range(len(FOLDS))}:raise ValueError('V8 complete fold models required')
    for index,held in enumerate(FOLDS):
        models=folds[str(index)];expected_train=sorted(set(CATALOG)-set(held))
        if set(models)!={'precision','stoppage'}:raise ValueError('V8 both fold heads required')
        precision_v8.validate_model(models['precision']);stoppage_v8.validate_model(models['stoppage'])
        for model in models.values():
            meta=model['metadata']
            if (meta['protocol_sha256']!=evidence['protocol_sha256'] or type(meta['fold']) is not int or meta['fold']!=index
                or meta['training_matches']!=expected_train or meta['held_out_matches']!=list(held)
                or meta['training_regimes']!=list(REGIMES)):
                raise ValueError('V8 outer fold training/held-out provenance differs')
    expected_fold={match:index for index,held in enumerate(FOLDS) for match in held}
    for row in evidence['records']:
        if type(row.get('fold')) is not int or row['fold']!=expected_fold[row['match']]:raise ValueError('V8 scored fold identity differs')

def verify_sparse_reports(records,folder):
    """Sparse positive timings cannot claim dense precision or fresh validation."""
    from tools.data_intake_v8 import MATCHES,validated_points
    support={}
    for match in MATCHES:
        document=json.loads((folder/f'{match}-points.json').read_text());ranges=document['frame_ranges']
        blocks=[(int(k),v[2],v[2]+v[1]-v[0]) for k,v in sorted(ranges.items())]
        points=validated_points(document,match=match,fps=25,blocks=blocks,n_frames=max(b for _,_,b in blocks)+1)
        first={period:a for period,a,_ in blocks};unique={k:set() for k in ('restart','ball_out')}
        for point in points:
            if point['kind'] in unique and point['index']!=first[point['period']]:
                unique[point['kind']].add((point['period'],point['team'] if point['kind']=='restart' else None,point['index']))
        support[match]={k:len(v) for k,v in unique.items()}
    for row in records:
        if row['match'] not in support:
            if 'human_sparse_timing' in row:raise ValueError('V8 sparse labels claimed for unannotated fixture')
            continue
        report=row['human_sparse_timing']
        if (report['reference_scope']!='sparse corrected human event timing on exposed development matches'
            or any(report[k] is not False for k in ('fresh_match','dense_had_truth','unlisted_frames_are_negatives'))):
            raise ValueError('V8 sparse timing scope differs')
        for kind,count in support[row['match']].items():
            value=report[kind];n=value['n_reference_points'];hits=value['n_matched']
            if (type(n) is not int or n!=count or type(hits) is not int or not 0<=hits<=n
                or value['precision'] is not None or value['f1'] is not None
                or isinstance(value['tolerance_seconds'],bool) or value['tolerance_seconds']!=1.
                or not same_evidence(value['recall_within_1s'],hits/n if n else None)):
                raise ValueError('V8 sparse timing support/rates differ')
            absolute,signed=value['mean_absolute_error_matched_seconds'],value['mean_signed_error_matched_seconds']
            if hits==0:
                if absolute is not None or signed is not None:raise ValueError('V8 unmatched points have no timing error')
            elif (any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in (absolute,signed))
                  or not 0<=absolute<=1.+1e-12 or abs(signed)>absolute+1e-12):
                raise ValueError('V8 sparse timing error out of bounds')

def verify(evidence,experiment_path):
    selection=select(evidence['records']);protocol=evidence['protocol']
    if not same_evidence(selection,evidence['selection']):raise ValueError('V8 selection differs')
    expected={'schema_version':8,'baseline_sha256':BASELINE_SHA256,'config_sha256':CONFIG_SHA256,
      'folds':FOLDS,'regimes':REGIMES,'primary_regimes':PRIMARY_REGIMES,'candidates':[BASELINE,*CANDIDATES],
      'gate':GATE,'precision_policy':precision_v8.POLICY,'stoppage_policy':stoppage_v8.POLICY,'combination_order':['precision','stoppage']}
    expected=json.loads(json.dumps(expected))
    if any(not same_evidence(protocol[k],v) for k,v in expected.items()):raise ValueError('V8 fixed policy differs')
    if not protocol_matches_hash(protocol,evidence['protocol_sha256']):raise ValueError('V8 protocol hash differs')
    if evidence['source_hashes']!=protocol['source_hashes'] or evidence['source_hashes']!=scientific_hashes():raise ValueError('V8 scientific sources differ')
    for value in (protocol,evidence):
        for key in ('competition_labels_used','independent_test_claimed'):
            if value[key] is not False:raise ValueError('V8 development cannot claim independent truth')
    if protocol['inference_receives_annotations'] is not False or protocol['fresh_audit_used_for_selection'] is not False:raise ValueError('V8 reference separation differs')
    if evidence['incumbent_historical_predictions_reproduced'] is not True:raise ValueError('V8 comparator not reproduced')
    verify_fold_models(evidence)
    old=json.loads((ROOT/'models/joint-selection.json').read_text())
    originals={(r['match'],r['regime']):r['prediction_sha256'] for r in old['records'] if r['candidate']=='joint-state'}
    for row in evidence['records']:
        if row['candidate']==BASELINE and row['prediction_sha256']!=originals[row['match'],row['regime']]:raise ValueError('V8 historical comparator changed')
    folder=ROOT/'research/annotations/elastic';points={p.name:sha256(p) for p in sorted(folder.glob('*-points.json'))}
    if len(points)!=3 or points!=protocol['annotation_hashes']:raise ValueError('V8 human annotation identity changed')
    intake=json.loads((folder/'summary.json').read_text())
    if intake['intake_source_sha256']!=sha256(ROOT/'tools/data_intake_v8.py') or intake['promotion_eligible'] is not False or intake['fresh_stoppage_validation_available'] is not False:raise ValueError('V8 intake provenance differs')
    for match,item in intake['summary'].items():
        if points[match+'-points.json']!=item['points_sha256']:raise ValueError('V8 published annotation hash differs')
    verify_sparse_reports(evidence['records'],folder)
    if sha256(experiment_path)!=evidence['experimental_model_sha256']:raise ValueError('V8 experimental model bytes differ')
    model=json.loads(Path(experiment_path).read_text())
    if model['schema_version']!=8 or model['kind']!='isolated_refinement' or model['base_model_sha256']!=BASELINE_SHA256 or model['candidate_id']!=selection['audit_candidate']:raise ValueError('V8 experiment identity differs')
    precision_v8.validate_model(model['precision']);stoppage_v8.validate_model(model['stoppage'])
    for name in ('precision','stoppage'):
        meta=model[name]['metadata']
        if meta['protocol_sha256']!=evidence['protocol_sha256'] or meta['training_matches']!=sorted(CATALOG) or meta['training_regimes']!=list(REGIMES) or meta['stage']!='all-seven-refit' or meta['selected']!=selection['audit_candidate']:raise ValueError('V8 experimental fit provenance differs')
    decision=evidence['release_decision']
    if decision['promoted'] is not False or decision['model_sha256']!=BASELINE_SHA256 or not decision['reason']:raise ValueError('V8 insufficient fresh evidence must retain v1.5')
    for name,expected in [('models/submission.json',BASELINE_SHA256),('configs/default.json',CONFIG_SHA256),('submission/Xtension Labs.csv',CSV_SHA256)]:
        if sha256(ROOT/name)!=expected:raise ValueError('Active v1.5 artifact changed: '+name)
    return selection

def main():
    evidence=json.loads((ROOT/'models/refinement-selection-v8.json').read_text())
    selection=verify(evidence,ROOT/'models/experiments/v8-candidate.json')
    print('V8 isolated research verified; selected='+selection['selected']+'; released v1.5 remains unchanged.')

if __name__=='__main__':main()

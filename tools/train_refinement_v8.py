"""Frozen whole-match precision/stoppage study; never overwrites the submission."""
from pathlib import Path
import argparse, itertools, json, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from possession.config import load_config
from possession.decoder import decode
from possession.io import sha256,write_json
from possession.learned import load_model,joint_probabilities
from tools import train_joint_campaign as v5,train_temporal_v6 as v6,train_boundary_campaign_v7 as v7
from tools.train_control import CATALOG
from tools.train_stoppages import measure
from tools.control_diagnostics import diagnostic as control_diagnostic
from tools.temporal_diagnostics_v6 import diagnostic
from tools.train_transition_v6 import diagnostic as duration_diagnostic

BASELINE=v6.BASELINE;BASELINE_SHA256=v6.BASELINE_SHA256;CONFIG_SHA256=v6.CONFIG_SHA256
FOLDS=v5.FOLDS;REGIMES=v5.REGIMES;PRIMARY_REGIMES=v5.PRIMARY_REGIMES
CANDIDATES=('precision','stoppage','combined')
GATE={'minimum_primary_spellF1_gain':.005,'minimum_short_precision_gain':0.,'minimum_short_recall_gain':0.,
 'minimum_dead_segment_gain':0.,'minimum_dead_boundary_gain':0.,'minimum_dead_precision_gain':0.,
 'minimum_primary_macroF1_gain':-.001,'maximum_cell_spellF1_drop':.01,'maximum_cell_dead_recall_drop':.005,
 'required_either_short_precision_or_dead_boundary_gain':.01,'minimum_match_bootstrap_lower_5pct':0.,
 'fresh_requirement':'Independent, adequately labelled unseen full matches are required. Sparse annotations on exposed matches and phase-only audits cannot authorize promotion.'}

def scientific_hashes():
    names=('tools/train_refinement_v8.py','tools/train_precision_v8.py','tools/train_stoppage_v8.py','tools/data_intake_v8.py','tools/sparse_timing_v8.py')
    return dict(v7.scientific_hashes(),**{p.relative_to(ROOT).as_posix():sha256(p) for p in [*sorted((ROOT/'research').glob('*.py')),*[ROOT/n for n in names]]})

def select(records):
    cells={(m,r) for f in FOLDS for m in f for r in REGIMES};primary={c for c in cells if c[1] in PRIMARY_REGIMES}
    groups={n:{} for n in (BASELINE,*CANDIDATES)}
    for row in records:
        n=row['candidate'];c=row['match'],row['regime']
        if n not in groups or c not in cells or c in groups[n]:raise ValueError('Unknown or duplicate v8 cell')
        values=v6.metrics(row)
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or not 0<=v<=1 for v in values.values()):raise ValueError('Invalid v8 metric')
        groups[n][c]=values
    if any(set(g)!=cells for g in groups.values()):raise ValueError('Every v8 procedure needs all20 cells')
    cells=sorted(cells);primary=sorted(primary);means={};gates={};base=groups[BASELINE]
    matches=sorted({c[0] for c in cells});resamples=np.array(list(itertools.product(range(len(matches)),repeat=len(matches))))
    for n,rows in groups.items():
        delta={k:float(np.mean([rows[c][k]-base[c][k] for c in (primary if k in ('spell','macro') else cells)])) for k in v6.metrics(records[0])}
        means[n]={k:float(np.mean([rows[c][k] for c in (primary if k in ('spell','macro') else cells)])) for k in delta}
        match_delta=np.array([np.mean([rows[m,r]['spell']-base[m,r]['spell'] for r in PRIMARY_REGIMES]) for m in matches])
        lower,upper=np.quantile(match_delta[resamples].mean(axis=1),[.05,.95])
        worst_spell=min(rows[c]['spell']-base[c]['spell'] for c in cells);worst_dead=min(rows[c]['dead_recall']-base[c]['dead_recall'] for c in cells)
        passed=(delta['spell']>=.005 and delta['short_precision']>=0 and delta['short']>=0 and delta['dead_segment']>=0
          and delta['dead_boundary']>=0 and delta['dead_precision']>=0 and delta['macro']>=-.001 and worst_spell>=-.01
          and worst_dead>=-.005 and max(delta['short_precision'],delta['dead_boundary'])>=.01 and lower>0)
        gates[n]={'passed':bool(passed),'mean_gains':delta,'minimum_cell_spellF1_gain':float(worst_spell),'minimum_cell_dead_recall_gain':float(worst_dead),
          'match_bootstrap_5_95':[float(lower),float(upper)],'match_spell_gains':dict(zip(matches,match_delta.tolist()))}
    ranking=sorted(groups,key=lambda n:(-means[n]['spell'],-means[n]['short_precision'],-means[n]['dead_boundary'],n))
    selected=next(n for n in ranking if n==BASELINE or gates[n]['passed'])
    return {'selected':selected,'audit_candidate':selected if selected!=BASELINE else next(n for n in ranking if n!=BASELINE),
      'ranking':ranking,'means':means,'gates':gates,'fresh_audit_used_for_selection':False}

def make_inputs(previous,cache,base,joint,config,match,regime,dest):
    x,z,y,blocks=v5.cached(previous,cache,match,regime);dead,home=v5.baseline_components(base,x,z,blocks)
    p=joint_probabilities(dead,home,x,z,blocks,25,joint,np.zeros(len(y)));labels=np.full(len(y),'X',dtype='<U1')
    usable=json.loads((previous/'cache'/match/f'{regime}-meta.json').read_text())['usable']
    for (_,a,b),use in zip(blocks,usable):
        if use:labels[a:b+1]=np.array(list('HAD'))[decode(p[a:b+1],25,config)]
    dest.parent.mkdir(parents=True,exist_ok=True);np.savez(dest,p=p,labels=labels)
    return {'match':match,'regime':regime,'sha256':sha256(dest),'rows':len(y)}

def load_item(args,folder,match,regime,with_annotations=False):
    x,z,y,blocks=v5.cached(args.previous,args.cache,match,regime)
    with np.load(folder/match/f'{regime}.npz',allow_pickle=False) as data:p=data['p'];labels=data['labels']
    item={'match':match,'regime':regime,'x':x,'z':z,'y':y,'blocks':blocks,'fps':25,'p':p,'labels':labels}
    pointfile=args.annotations/f'{match}-points.json'
    if with_annotations and pointfile.is_file():item['sparse_points']=json.loads(pointfile.read_text())
    return item

def apply_models(models,item,candidate):
    from research import precision_v8,stoppage_v8
    labels=item['labels'].copy();args=tuple(item[k] for k in ('p','x','z','blocks','fps'))
    if candidate in ('precision','combined'):labels=precision_v8.refine(models['precision'],labels,*args)
    if candidate in ('stoppage','combined'):labels=stoppage_v8.refine(models['stoppage'],labels,*args)
    return labels

def score(dest,match,candidate,regime,y,labels,blocks,sparse_points=None):
    row=measure(dest,match,candidate,regime,y,labels,0)
    row['control_diagnostics']=control_diagnostic(y,labels,25);row['temporal_diagnostics']=diagnostic(y,labels,25,blocks)
    row['duration_precision_recall']=duration_diagnostic(y,labels,25)
    if sparse_points is not None:
        from tools.sparse_timing_v8 import diagnostic as sparse_diagnostic
        row['human_sparse_timing']=sparse_diagnostic(labels,blocks,25,sparse_points)
    return row

def run(args):
    from research import precision_v8,stoppage_v8
    from tools import train_precision_v8,train_stoppage_v8
    if args.out.exists():raise ValueError('Use a new immutable v8 output directory')
    history,proof=v6.verify_inputs(args);config=load_config(args.config);base_final=load_model(args.baseline)
    frozen=scientific_hashes();annotation_hashes={p.name:sha256(p) for p in sorted(args.annotations.glob('*-points.json'))}
    if len(annotation_hashes)!=3:raise ValueError('Require allthree provenance-validated ELASTIC development annotations')
    history_hashes=v6.frozen_history(args.previous_v5);older_hashes=v5.historical_hashes(args.previous)
    protocol={'schema_version':8,'source_hashes':frozen,'baseline_sha256':BASELINE_SHA256,'config_sha256':CONFIG_SHA256,
      'folds':FOLDS,'regimes':REGIMES,'primary_regimes':PRIMARY_REGIMES,'candidates':[BASELINE,*CANDIDATES],'gate':GATE,
      'precision_policy':precision_v8.POLICY,'stoppage_policy':stoppage_v8.POLICY,'annotation_hashes':annotation_hashes,
      'v5_historical_sha256':history_hashes,'v4_historical_sha256':older_hashes,
      'training':'Upstream models exclude the entire outer evaluation fold; refinement heads use outer-training in-sample upstream predictions. Human sparse annotations supplied only for those training matches.',
      'combination_order':['precision','stoppage'],'competition_labels_used':False,'fresh_audit_used_for_selection':False,
      'exposure':'Allseven Sportec matches exposed; ELASTIC supplies new human annotations on three already exposed matches, not fresh matches.',
      'inference_receives_annotations':False,'independent_test_claimed':False}
    args.out.mkdir(parents=True);write_json(args.out/'protocol.json',protocol);ph=sha256(args.out/'protocol.json');records=[];cache_records={};fold_models={}
    def unchanged():
        if scientific_hashes()!=frozen or v6.frozen_history(args.previous_v5)!=history_hashes or v5.historical_hashes(args.previous)!=older_hashes:raise ValueError('Frozen v8 source/history changed')
        if {p.name:sha256(p) for p in sorted(args.annotations.glob('*-points.json'))}!=annotation_hashes:raise ValueError('Frozen human annotations changed')
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index,held in enumerate(FOLDS):
            matches=sorted(set(CATALOG)-set(held));base=v5.fold_base(args.previous,load_model(ROOT/'models/baselines/v1.4.json'),index)
            joint=json.loads((args.previous_v5/'fold-models'/str(index)/'joint.json').read_text())
            if joint['metadata']['training_matches']!=matches or joint['metadata']['held_out_matches']!=list(held):raise ValueError('Upstream fold leakage')
            folder=args.out/'inputs'/str(index);print('Preparing originalv1.5 predictions for fold '+str(index),flush=True)
            jobs=[pool.submit(make_inputs,args.previous,args.cache,base,joint,config,m,r,folder/m/f'{r}.npz') for m in sorted(CATALOG) for r in REGIMES]
            cache_records[str(index)]=[j.result() for j in as_completed(jobs)];write_json(args.out/'input-manifest.json',cache_records)
            items=[load_item(args,folder,m,r,True) for m in matches for r in REGIMES]
            meta={'protocol_sha256':ph,'training_matches':matches,'held_out_matches':list(held),'fold':index,'training_regimes':list(REGIMES)}
            print('Fitting both refinement heads for fold '+str(index),flush=True)
            models={'precision':train_precision_v8.fit(items,metadata=meta),'stoppage':train_stoppage_v8.fit(items,metadata=meta)};del items
            fold_models[str(index)]=models
            for n,model in models.items():write_json(args.out/'fold-models'/str(index)/(n+'.json'),model)
            for m in held:
                for r in REGIMES:
                    item=load_item(args,folder,m,r,False)
                    pointfile=args.annotations/f'{m}-points.json'
                    sparse=json.loads(pointfile.read_text()) if pointfile.is_file() else None
                    jobs=[pool.submit(score,args.out/'cv'/m/r/n,m,n,r,item['y'],apply_models(models,item,n),item['blocks'],sparse) for n in (BASELINE,*CANDIDATES)]
                    for j in as_completed(jobs):row=j.result();row['fold']=index;records.append(row)
            unchanged();write_json(args.out/f'fold-{index}-records.json',records)
        old={(r['match'],r['regime']):r for r in history['records'] if r['candidate']=='joint-state'}
        for r in records:
            if r['candidate']==BASELINE and r['prediction_sha256']!=old[r['match'],r['regime']]['prediction_sha256']:raise ValueError('Original baseline prediction bytes differ')
        selection=select(records);write_json(args.out/'selection-before-any-fresh-access.json',{'protocol_sha256':ph,**selection})
        print('Selection frozen: '+json.dumps(selection),flush=True)
        folder=args.out/'inputs'/'full'
        jobs=[pool.submit(make_inputs,args.previous,args.cache,base_final,base_final['joint_model'],config,m,r,folder/m/f'{r}.npz') for m in sorted(CATALOG) for r in REGIMES]
        cache_records['full']=[j.result() for j in as_completed(jobs)];write_json(args.out/'input-manifest.json',cache_records)
    items=[load_item(args,folder,m,r,True) for m in sorted(CATALOG) for r in REGIMES]
    meta={'protocol_sha256':ph,'training_matches':sorted(CATALOG),'training_regimes':list(REGIMES),'stage':'all-seven-refit','selected':selection['audit_candidate']}
    experiment={'schema_version':8,'kind':'isolated_refinement','candidate_id':selection['audit_candidate'],'base_model_sha256':BASELINE_SHA256,
      'precision':train_precision_v8.fit(items,metadata=meta),'stoppage':train_stoppage_v8.fit(items,metadata=meta)}
    write_json(args.out/'experimental-model.json',experiment);unchanged()
    summary={'protocol':protocol,'protocol_sha256':ph,'source_hashes':frozen,'cache_provenance':proof,
      'records':sorted(records,key=lambda r:(r['match'],r['regime'],r['candidate'])),'selection':selection,'fold_models':fold_models,
      'incumbent_historical_predictions_reproduced':True,'competition_labels_used':False,'independent_test_claimed':False,
      'experimental_model_sha256':sha256(args.out/'experimental-model.json'),'input_manifest_sha256':sha256(args.out/'input-manifest.json'),
      'release_decision':{'promoted':False,'model_sha256':BASELINE_SHA256,'reason':'Fresh independently labelled full-match validation is unavailable; research refinements are not deployed.'}}
    write_json(args.out/'summary.json',summary);print('Completed v8; submission unchanged.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n,v in [('previous','runs/multimodal-campaign-v4'),('previous-v5','runs/joint-campaign-v5'),('cache','runs/joint-cache-v5'),
      ('baseline','models/baselines/v1.5.json'),('archive','models/baselines/v1.5-source.zip'),('config','configs/baselines/v1.5.json'),
      ('annotations','research/annotations/elastic'),('out','runs/refinement-campaign-v8')]:p.add_argument('--'+n,type=Path,default=ROOT/v)
    p.add_argument('--workers',type=int,default=2,choices=(1,2));run(p.parse_args())

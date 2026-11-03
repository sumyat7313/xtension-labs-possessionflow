"""Isolated PFF study with fixed fixtures and prediction-before-reference scoring."""
from pathlib import Path
import argparse,json,sys,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from possession.io import sha256,write_json
from tools import pff_adapter,pff_training,pff_validation
from tools import train_refinement_v8 as v8,train_joint_campaign as v5
from tools.train_control import CATALOG
from tools.train_stoppages import measure
from tools.temporal_diagnostics_v6 import diagnostic
from tools.train_transition_v6 import diagnostic as duration_diagnostic
from tools.train_temporal_v6 import metrics

ARRAYS=('x','z','base_home','base_dead')
def sources():
    return dict(v8.scientific_hashes(),**{n:sha256(ROOT/n) for n in ('tools/pff_campaign.py','tools/pff_adapter.py','tools/pff_training.py','tools/pff_validation.py')})

def protocol():
    p=json.loads((ROOT/'models/pff-protocol.json').read_text(encoding='utf-8'))
    pff_validation.verify_protocol(p)
    if sha256(ROOT/'models/submission.json')!=pff_validation.BASELINE_SHA256:raise ValueError('Incumbent changed')
    for m,h in p['metadata_sha256'].items():
        if sha256(ROOT/'data/pff/Metadata'/f'{m}.json')!=h:raise ValueError('Metadata changed')
    return p

def freeze(out,p):
    out.mkdir(parents=True,exist_ok=True)
    value={'split_protocol_sha256':p['protocol_sha256'],'source_hashes':sources(),
      'candidate':'one mixed joint head; 40 PFF + seven clean Sportec fixtures',
      'training_policy':pff_training.POLICY,'observation_policy':pff_adapter.POLICY,
      'final_reference_access':'only after fixed development gate; inadequate provenance retains v1.5',
      'raw_data_and_fitted_weights_local_only':True,'automatic_promotion':False}
    dest=out/'run-protocol.json'
    if dest.exists():
        if json.loads(dest.read_text())!=value:raise ValueError('Scientific sources changed; preserve prior run and choose new output')
    else:write_json(dest,value)
    return value

def prepare_one(out,mid):
    out=Path(out);folder=out/'cache'/str(mid);folder.mkdir(parents=True,exist_ok=True)
    base=ROOT/'data/pff';meta=base/'Metadata'/f'{mid}.json';track=base/'Tracking Data'/f'{mid}.jsonl.bz2'
    identity={'metadata':sha256(meta),'tracking':sha256(track),
      'events':sha256(base/'Event Data'/f'{mid}.json'),'sources':sources()}
    if (folder/'complete.json').exists():
        old=json.loads((folder/'complete.json').read_text())
        if old['identity']!=identity:raise ValueError('Stale PFF cache')
        for name,h in old['files'].items():
            if sha256(folder/name)!=h:raise ValueError('PFF cache changed')
        return old
    tracking,blocks,pitch,alignment=pff_adapter.read_observations(meta,track)
    item,labels,p=pff_training.prepare_cache(tracking,blocks,pitch,alignment['fps'])
    for key in ARRAYS:np.save(folder/f'{key}.npy',item.pop(key),allow_pickle=False)
    np.save(folder/'baseline.npy',labels,allow_pickle=False)
    write_json(folder/'item.json',dict(item,match=f'pff{mid}',pitch=list(pitch)))
    write_json(folder/'alignment.json',alignment)
    names=[*(f'{k}.npy' for k in ARRAYS),'baseline.npy','item.json','alignment.json']
    result={'match_id':mid,'identity':identity,'files':{n:sha256(folder/n) for n in names},
      'rows':len(labels),'visible_ball_fraction':alignment['visible_ball_fraction'],
      'seconds':len(labels)/alignment['fps']}
    write_json(folder/'complete.json',result);return result

def item_at(out,mid):
    folder=out/'cache'/str(mid)
    complete=json.loads((folder/'complete.json').read_text())
    if complete['identity']['sources']!=sources():raise ValueError('Cache sources changed')
    for name,h in complete['files'].items():
        if sha256(folder/name)!=h:raise ValueError('Cache artifact changed: '+name)
    item=json.loads((folder/'item.json').read_text())
    for key in ARRAYS:item[key]=np.load(folder/f'{key}.npy',mmap_mode='r',allow_pickle=False)
    return item

def reference(out,mid):
    folder=out/'cache'/str(mid);complete=json.loads((folder/'complete.json').read_text())
    event=ROOT/'data/pff/Event Data'/f'{mid}.json'
    if sha256(folder/'alignment.json')!=complete['files']['alignment.json'] or sha256(event)!=complete['identity']['events']:
        raise ValueError('Reference source or alignment changed')
    alignment=json.loads((folder/'alignment.json').read_text())
    return pff_adapter.read_references(event,alignment)

def old_items():
    model=pff_training.baseline_model()
    for mid in sorted(CATALOG):
        x,z,y,blocks=v5.cached(ROOT/'runs/multimodal-campaign-v4',ROOT/'runs/joint-cache-v5',mid,'clean')
        dead,home=v5.baseline_components(model,x,z,blocks)
        yield {'match':'sportec'+mid,'x':x,'z':z,'y':y,'blocks':blocks,'fps':25,
          'base_dead':dead,'base_home':home,'reference_fields_used_for_inference':False}

def fit(out,p):
    if (out/'candidate.json').exists():raise ValueError('Candidate already fitted; no overwrite')
    support={}
    training_matches=['pff'+str(mid) for mid in p['split']['train']]+['sportec'+mid for mid in sorted(CATALOG)]
    def items():
        for mid in p['split']['train']:
            item=item_at(out,mid);item['y'],support[str(mid)]=reference(out,mid)
            yield item
        yield from old_items()
    candidate=pff_training.fit_candidate(items(),training_matches=training_matches,
      held_out_matches=['pff'+str(m) for part in ('development','final') for m in p['split'][part]],
      protocol_sha256=p['protocol_sha256'])
    write_json(out/'candidate.json',candidate)
    write_json(out/'fit.json',{'candidate_sha256':sha256(out/'candidate.json'),'training_reference_support':support,
      'training_matches':training_matches,'held_out_references_used':False,'source_hashes':sources()})

def require_final_eligible(out,p):
    path=out/'development-summary.json'
    if not path.exists():raise ValueError('Development gate has not passed; final fixtures stay untouched')
    development=json.loads(path.read_text())
    gate=pff_validation.development_gate(development['records'],p)
    if (development['gate']!=gate or not gate['eligible_for_final_audit']
        or development['source_hashes']!=sources() or development['protocol_sha256']!=p['protocol_sha256']
        or development['candidate_sha256']!=sha256(out/'candidate.json')):
        raise ValueError('Development gate or its frozen provenance is invalid')
    return gate

def predict_split(out,p,part):
    from possession.learned import load_model
    if part not in ('development','final'):raise ValueError('Only held-out partitions may be predicted here')
    if (out/(part+'-reference-access.json')).exists():raise ValueError('References already consumed; cannot regenerate predictions')
    if part=='final':require_final_eligible(out,p)
    model=load_model(out/'candidate.json');records=[]
    for mid in p['split'][part]:
        folder=out/part/str(mid);folder.mkdir(parents=True,exist_ok=True);item=item_at(out,mid)
        labels,_=pff_training.predict_cache(item,model)
        np.save(folder/'candidate.npy',labels,allow_pickle=False)
        base=np.load(out/'cache'/str(mid)/'baseline.npy',allow_pickle=False)
        np.save(folder/'baseline.npy',base,allow_pickle=False)
        records.append({'match_id':mid,'candidate_prediction_sha256':sha256(folder/'candidate.npy'),
          'baseline_prediction_sha256':sha256(folder/'baseline.npy')})
        print(json.dumps({'predicted':mid,'part':part}),flush=True)
    result={'part':part,'candidate_sha256':sha256(out/'candidate.json'),'baseline_sha256':pff_validation.BASELINE_SHA256,
      'split_protocol_sha256':p['protocol_sha256'],'source_hashes':sources(),'records':records,
      'references_accessed':False}
    write_json(out/(part+'-predictions-before-references.json'),result);return result

def score_one(out,part,mid,which,y,labels,blocks):
    row=measure(out/part/str(mid)/which,str(mid),which,'native',y,labels)
    row['temporal_diagnostics']=diagnostic(y,labels,25,blocks)
    row['duration_precision_recall']=duration_diagnostic(y,labels,25)
    values=metrics(row);values['short_recall']=values.pop('short')
    return row,values

def evaluate_split(out,p,part):
    if part not in ('development','final'):raise ValueError('Training fixtures cannot be reported as fresh evaluation')
    if (out/(part+'-summary.json')).exists():raise ValueError('No repeated held-out scoring')
    access_path=out/(part+'-reference-access.json')
    if access_path.exists():raise ValueError('Held-out references already consumed, including incomplete scoring attempts')
    if part=='final':require_final_eligible(out,p)
    pred_path=out/(part+'-predictions-before-references.json')
    frozen=json.loads(pred_path.read_text()) if pred_path.exists() else predict_split(out,p,part)
    if (frozen['part']!=part or frozen['baseline_sha256']!=pff_validation.BASELINE_SHA256
        or frozen['split_protocol_sha256']!=p['protocol_sha256'] or frozen['references_accessed'] is not False
        or frozen['source_hashes']!=sources() or frozen['candidate_sha256']!=sha256(out/'candidate.json')):
        raise ValueError('Prediction freeze changed')
    if [r['match_id'] for r in frozen['records']]!=p['split'][part]:raise ValueError('Prediction split differs')
    # Authenticate the entire cohort before opening its first reference file.
    for saved in frozen['records']:
        folder=out/part/str(saved['match_id'])
        for which in ('baseline','candidate'):
            if sha256(folder/(which+'.npy'))!=saved[which+'_prediction_sha256']:raise ValueError('Saved prediction changed')
    marker={'part':part,'references_consumed':True,'prediction_freeze_sha256':sha256(pred_path),
      'candidate_sha256':frozen['candidate_sha256'],'protocol_sha256':p['protocol_sha256'],
      'source_hashes':sources(),'rule':'Exclusive marker written before first reference read; failed attempts remain consumed.'}
    try:
        with access_path.open('x',encoding='utf-8') as handle:json.dump(marker,handle,sort_keys=True,indent=2)
    except FileExistsError:
        raise ValueError('Concurrent or repeated reference access refused') from None
    rows=[];details=[];supports={}
    for saved in frozen['records']:
        mid=saved['match_id'];folder=out/part/str(mid);item=item_at(out,mid)
        for which in ('baseline','candidate'):
            if sha256(folder/(which+'.npy'))!=saved[which+'_prediction_sha256']:raise ValueError('Saved prediction changed')
        y,supports[str(mid)]=reference(out,mid)
        pair={}
        for which in ('baseline','candidate'):
            row,pair[which]=score_one(out,part,mid,which,y,np.load(folder/(which+'.npy'),allow_pickle=False),item['blocks'])
            details.append(row)
        d=row['temporal_diagnostics']
        rows.append({'match_id':mid,'condition':'native','baseline_sha256':pff_validation.BASELINE_SHA256,
          'candidate_sha256':frozen['candidate_sha256'],**pair,
          'reference_dead_segments':d['dead_segments']['n_gt'],'reference_short_spells':d['short_possessions']['n_gt']})
    evidence={'dense_truth_validated':False,'independent_observations':True,'identity_checked':True,'usage_permitted':False,
      'development_passed':False,'observations_only':True,'predictions_frozen_before_references':True,'adequate_label_support':False}
    if part=='development':gate=pff_validation.development_gate(rows,p)
    else:
        evidence['development_passed']=require_final_eligible(out,p)['eligible_for_final_audit']
        gate=pff_validation.promotion_gate(rows,p,evidence)
    summary={'part':part,'records':rows,'details':details,'reference_support':supports,'gate':gate,
      'evidence':evidence,'source_hashes':sources(),'protocol_sha256':p['protocol_sha256'],
      'prediction_freeze_sha256':sha256(pred_path),'candidate_sha256':frozen['candidate_sha256'],
      'reference_access_sha256':sha256(access_path),
      'reference_quality':'provider event-derived silver; no independent dense human validation',
      'promoted':False,'retained_model_sha256':pff_validation.BASELINE_SHA256}
    write_json(out/(part+'-summary.json'),summary);print(json.dumps(gate),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','fit','evaluate']);parser.add_argument('--split',choices=['train','development','final'],default='train')
    parser.add_argument('--out',type=Path,default=ROOT/'runs/pff-campaign-v1');parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args();out=args.out.resolve();p=protocol();freeze(out,p)
    if args.split=='final':
        require_final_eligible(out,p)
    if args.stage=='prepare':
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for f in as_completed([pool.submit(prepare_one,out,m) for m in p['split'][args.split]]):print(json.dumps(f.result()),flush=True)
    elif args.stage=='fit':fit(out,p)
    elif args.split=='train':raise ValueError('Training fixtures cannot be reported as fresh evaluation')
    else:evaluate_split(out,p,args.split)
    freeze(out,p)

if __name__=='__main__':main()

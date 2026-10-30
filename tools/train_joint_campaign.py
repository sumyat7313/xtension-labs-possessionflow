"""Fixed v5 joint-state/boundary development campaign; fresh audits stay separate."""
from pathlib import Path
import argparse,copy,json,sys,zipfile,hashlib
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from possession import temporal,robust_control,boundary
from possession.config import load_config
from possession.decoder import decode
from possession.io import sha256,write_json
from possession.learned import load_model,validate_model,predict_dead,fuse_control_probabilities,joint_probabilities
from tools import train_joint,train_boundary
from tools.control_diagnostics import training_spell_weights,diagnostic
from tools.train_control import CATALOG
from tools.train_stoppages import measure

BASELINE='baseline-v1.4'
CANDIDATES=('joint-state','boundary','joint-boundary')
REGIMES=('clean','degraded','camera','calibrated');PRIMARY_REGIMES=REGIMES[:3]
FOLDS=(('J03WN1','J03WMX'),('J03WPY','J03WQQ'),('J03WR9',))
BASELINE_SHA256='7f72b108801a9c6029a2543ceaccd2ef9f27947c256ab59265df47c316688653'
ARCHIVE_SHA256='04c5341978660a52fc32148221508ddb3db291921ff19d8fb1bfbf3e5cafeeb9'
CONFIG_SHA256='2755c140130f4ef9821e7909f8da794d6e4c8d4a7691b75de77eada7c866612b'

def scientific_hashes():
    names=('tools/train_joint_campaign.py','tools/prepare_joint_cache.py','tools/train_joint.py','tools/train_boundary.py',
           'tools/train_sequence.py','tools/calibrated_camera.py','tools/camera_augmentation.py','tools/train_control.py',
           'tools/control_diagnostics.py','tools/train_stoppages.py','tools/external_benchmark.py','evaluation/possession_metrics.py')
    return {p.relative_to(ROOT).as_posix():sha256(p) for p in [*sorted((ROOT/'possession').glob('*.py')),*[ROOT/n for n in names]]}

def select(records):
    names=(BASELINE,*CANDIDATES);cells=sorted((m,r) for fold in FOLDS for m in fold for r in REGIMES)
    grouped={n:{} for n in names};primary=[c for c in cells if c[1] in PRIMARY_REGIMES]
    for row in records:
        name=row['candidate'];cell=(row['match'],row['regime'])
        if name not in grouped or cell not in cells or cell in grouped[name]:raise ValueError('Invalid or duplicate candidate/cell')
        scores=[row[k] for k in ('spellF1','macroF1','dead_recall','dead_precision')]
        scores += [row['control_diagnostics']['transitions']['recall'],row['control_diagnostics']['spell_recall_by_duration']['<2s']['recall']]
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or not 0<=v<=1 for v in scores):raise ValueError('Invalid required score')
        grouped[name][cell]=row
    if any(set(g)!=set(cells) for g in grouped.values()):raise ValueError('Each candidate requires all 20 fixed cells')
    base=grouped[BASELINE];gates={};means={}
    for name,rows in grouped.items():
        gains={f'{m}/{r}':rows[m,r]['spellF1']-base[m,r]['spellF1'] for m,r in cells}
        mean=float(np.mean([rows[c]['spellF1']-base[c]['spellF1'] for c in primary]))
        death=min(rows[c]['dead_recall']-base[c]['dead_recall'] for c in cells)
        precision=float(np.mean([rows[c]['dead_precision']-base[c]['dead_precision'] for c in cells]))
        macro=float(np.mean([rows[c]['macroF1']-base[c]['macroF1'] for c in primary]))
        transition=float(np.mean([rows[c]['control_diagnostics']['transitions']['recall']-base[c]['control_diagnostics']['transitions']['recall'] for c in cells]))
        short=float(np.mean([rows[c]['control_diagnostics']['spell_recall_by_duration']['<2s']['recall']-base[c]['control_diagnostics']['spell_recall_by_duration']['<2s']['recall'] for c in cells]))
        gates[name]={'passed':bool(mean>=.01 and min(gains.values())>=-.02 and death>=-.01 and precision>=0 and macro>=0 and transition>=0 and short>=0),
          'primary_mean_spellF1_gain':mean,'cell_gains':gains,'minimum_dead_recall_delta':float(death),'mean_dead_precision_gain':precision,
          'primary_macroF1_gain':macro,'mean_transition_recall_gain':transition,'mean_short_spell_recall_gain':short}
        means[name]={'primary_spellF1':float(np.mean([rows[c]['spellF1'] for c in primary])),
          'primary_macroF1':float(np.mean([rows[c]['macroF1'] for c in primary])),
          'all_conditions_spellF1':float(np.mean([rows[c]['spellF1'] for c in cells]))}
    ranking=sorted(names,key=lambda n:(-means[n]['primary_spellF1'],-means[n]['primary_macroF1'],n))
    selected=next(n for n in ranking if n==BASELINE or gates[n]['passed'])
    audit_candidate=selected if selected!=BASELINE else next(n for n in ranking if n!=BASELINE)
    return {'selected':selected,'audit_candidate':audit_candidate,'ranking':ranking,'gates':gates,'means':means,'fresh_audit_used_for_selection':False}

def logit(p):
    p=np.clip(p,1e-8,1-1e-8);return np.log(p)-np.log1p(-p)

def baseline_components(model,x,z,blocks):
    head=model['robust_control_model'];base=(x/np.asarray(head['scale']))@np.asarray(head['coefficients'])
    home=temporal.predict_home(base,x,blocks,25,model['temporal_model'],model['expert_gate'])
    return predict_dead(z,model),home

def cached(previous,cache,match,regime):
    old=previous/'cache'/match
    x=np.load(old/f'{regime}-features.npy',mmap_mode='r',allow_pickle=False)
    z=np.load(cache/match/f'{regime}-dead-features.npy',mmap_mode='r',allow_pickle=False)
    y=np.load(old/'reference.npy',allow_pickle=False)
    blocks=json.loads((old/'metadata.json').read_text())['blocks']
    return x,z,y,blocks

def joint_weights(y,blocks):
    weights=training_spell_weights(y,25);weights[y=='D']=1.
    for _,a,b in blocks:
        for index in range(a+1,b+1):
            if y[index]!='X' and y[index-1]!='X' and y[index]!=y[index-1]:
                left,right=max(a,index-10),min(b+1,index+11)
                # Bounded emphasis; overlapping neighborhoods never compound.
                weights[left:right]=np.maximum(weights[left:right],1.5)
    weights[y=='X']=0
    active=np.zeros(len(y),bool)
    for _,a,b in blocks:active[a:b+1]=True
    weights[~active]=0
    return weights

def fit_joint(previous,cache,matches,base,metadata):
    sequences=[]
    for match in sorted(matches):
        for regime in REGIMES:
            x,z,y,blocks=cached(previous,cache,match,regime)
            dead,home=baseline_components(base,x,z,blocks);weights=joint_weights(y,blocks)
            for _,a,b in blocks:
                indices=np.arange(a,b+1,5)
                sequences.append({'X':np.array(x[indices]),'Z':np.array(z[indices]),'y':y[indices],'w':weights[indices],
                                  'base_home':logit(home[indices]),'base_dead':logit(dead[indices])})
    print('Fitting joint H/A/D on '+','.join(sorted(matches)),flush=True)
    return train_joint.fit(sequences,seed=10527,epochs=8,metadata=dict(metadata,training_matches=sorted(matches),training_regimes=list(REGIMES)))

def fit_boundary(previous,cache,matches,base,joint,metadata):
    items=[]
    for match in sorted(matches):
        for regime in REGIMES:
            x,z,y,blocks=cached(previous,cache,match,regime);dead,home=baseline_components(base,x,z,blocks)
            p=(fuse_control_probabilities(dead,home,np.zeros(len(y))) if joint is None else
               joint_probabilities(dead,home,x,z,blocks,25,joint,np.zeros(len(y))))
            items.append({'X':boundary.feature_matrix(x,p,blocks,25),'y':y,'blocks':blocks,'fps':25,'w':training_spell_weights(y,25)})
    print('Fitting local boundaries on '+','.join(sorted(matches)),flush=True)
    return train_boundary.fit(items,metadata=dict(metadata,training_matches=sorted(matches),training_regimes=list(REGIMES)),stride=5,l2=.01)

def score(destination,match,candidate,regime,y,p,blocks,config,usable,costs):
    labels=np.full(len(y),'X',dtype='<U1')
    for (_,a,b),use in zip(blocks,usable):
        if use:labels[a:b+1]=np.array(list('HAD'))[decode(p[a:b+1],25,config,None if costs is None else costs[a:b+1])]
    row=measure(destination,match,candidate,regime,y,labels,0)
    row['control_diagnostics']=diagnostic(y,labels,25)
    write_json(destination/'control-diagnostics.json',row['control_diagnostics'])
    return row

def fold_base(previous,baseline,index):
    result=copy.deepcopy(baseline);folder=previous/'fold-models'/str(index)
    result['robust_control_model']=robust_control.load_model(folder/'calibrated-control.json')
    result['temporal_model']=json.loads((folder/'temporal.json').read_text())
    result['expert_gate']=json.loads((folder/'gate.json').read_text())
    train=sorted(set(CATALOG)-set(FOLDS[index]))
    if (result['robust_control_model']['metadata']['training_matches']!=train
        or result['temporal_model']['metadata']['training_matches']!=train
        or result['expert_gate']['metadata']['gate_training_matches']!=train[:1]
        or result['expert_gate']['metadata']['expert_training_matches']!=train[1:]):
        raise ValueError('Historical fold experts or gate include held-out matches')
    ph=json.loads((previous/'summary.json').read_text())['protocol_sha256']
    for name in ('robust_control_model','temporal_model','expert_gate'):
        meta=result[name]['metadata']
        if meta['held_out_matches']!=list(FOLDS[index]) or meta['protocol_sha256']!=ph:
            raise ValueError('Historical fold identity or protocol differs')
    return validate_model(result)

def historical_hashes(previous):
    files=[*sorted((previous/'fold-models').rglob('*.json')),*sorted((previous/'cache').rglob('*-meta.json'))]
    return {p.relative_to(previous).as_posix():sha256(p) for p in files}

def verify_inputs(args):
    if sha256(args.baseline)!=BASELINE_SHA256 or sha256(args.config)!=CONFIG_SHA256 or sha256(args.archive)!=ARCHIVE_SHA256:raise ValueError('v1.4 artifact hashes differ')
    history=json.loads((args.previous/'summary.json').read_text());committed=json.loads((ROOT/'models/multimodal-selection.json').read_text());committed.pop('release_decision')
    if history!=committed or history['release_candidate_model_sha256']!=BASELINE_SHA256:raise ValueError('Historical v4 evidence differs')
    with zipfile.ZipFile(args.archive) as z:
        for name,digest in history['source_hashes'].items():
            if hashlib.sha256(z.read(name)).hexdigest()!=digest:raise ValueError('Historical source archive differs')
    proof=json.loads((args.cache/'provenance.json').read_text())
    if set(proof)!=set(CATALOG):raise ValueError('Incomplete verified feature cache')
    for match,p in proof.items():
        old=args.previous/'cache'/match
        if set(p['files'])!=set(REGIMES):raise ValueError('Incomplete regime feature cache')
        for regime,fields in p['files'].items():
            if sha256(old/f'{regime}-features.npy')!=fields['control_sha256'] or sha256(args.cache/match/f'{regime}-dead-features.npy')!=fields['dead_features_sha256']:raise ValueError('Input feature cache changed')
        if sha256(old/'reference.npy')!=p['reference_sha256'] or sha256(old/'metadata.json')!=p['metadata_sha256']:raise ValueError('Reference cache changed')
    return history,proof

def run(args):
    if args.out.exists():raise ValueError('Use new immutable campaign directory')
    history,proof=verify_inputs(args);baseline=load_model(args.baseline);config=load_config(args.config)
    args.out.mkdir(parents=True);frozen=scientific_hashes();historical_frozen=historical_hashes(args.previous)
    protocol={'schema_version':5,'source_hashes':frozen,'baseline_sha256':BASELINE_SHA256,'config_sha256':CONFIG_SHA256,'archive_sha256':ARCHIVE_SHA256,
      'folds':FOLDS,'candidates':[BASELINE,*CANDIDATES],'regimes':REGIMES,'primary_regimes':PRIMARY_REGIMES,'historical_fold_sha256':historical_frozen,
      'joint_training':{'seed':10527,'epochs':8,'sample_hz':5,'weights':'H/A inverse-sqrt spell weights; D=1; +/-0.4s known-state transitions have minimum1.5; X=0'},
      'boundary_training':{'stride':5,'l2':.01,'target_radius_seconds':.16,'maximum_switch_cost_reduction':.75},
      'development_gate':{'primary_spellF1_gain':.01,'maximum_cell_spellF1_drop':.02,'maximum_cell_dead_recall_drop':.01,
          'minimum_mean_dead_precision_gain':0,'minimum_primary_macroF1_gain':0,'minimum_transition_gain':0,'minimum_short_spell_gain':0},
      'baseline_comparison':'Frozen v1.4 historical out-of-fold control, temporal and gate models; no all-seven comparator on scored folds.',
      'fresh_audit_policy':'Freeze one eligible candidate, or the highest-ranked experiment if none eligible, before fresh reference access. Both development and fresh gates are required for promotion. MetricaGame3 sparse provider-event accuracy and balancedrecall nondecreasing; annotation method unverified. Two reserved SkillCorner automatic-phase matches mean accuracy nondecreasing, maximumindividualdrop .02. Gaps unknown; no fresh human denseHAD claim. Failure retainsv1.4; no runner-up or retuning.',
      'exposure':'All seven Sportec matches exposed; development CV only. Neither fresh source enters training or selection.',
      'competition_labels_used':False,'fresh_audit_used_for_selection':False}
    write_json(args.out/'protocol.json',protocol);ph=sha256(args.out/'protocol.json')
    fold_hashes={};records=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index,held in enumerate(FOLDS):
            train=sorted(set(CATALOG)-set(held));base=fold_base(args.previous,baseline,index)
            if base['robust_control_model']['metadata']['training_matches']!=train:raise ValueError('Comparator contains heldout training match')
            dest=args.out/'fold-models'/str(index);meta={'protocol_sha256':ph,'fold':index,'held_out_matches':list(held)}
            write_json(dest/'baseline.json',base);fold_hashes[str(index)]=sha256(dest/'baseline.json')
            joint=fit_joint(args.previous,args.cache,train,base,meta);write_json(dest/'joint.json',joint)
            bd=fit_boundary(args.previous,args.cache,train,base,None,meta);write_json(dest/'boundary.json',bd)
            jb=fit_boundary(args.previous,args.cache,train,base,joint,meta);write_json(dest/'joint-boundary.json',jb)
            for match in held:
                for regime in REGIMES:
                    x,z,y,blocks=cached(args.previous,args.cache,match,regime);dead,home=baseline_components(base,x,z,blocks)
                    p=fuse_control_probabilities(dead,home,np.zeros(len(y)));q=joint_probabilities(dead,home,x,z,blocks,25,joint,np.zeros(len(y)))
                    usable=json.loads((args.previous/'cache'/match/f'{regime}-meta.json').read_text())['usable']
                    choices={BASELINE:(p,None),'joint-state':(q,None),'boundary':(p,boundary.switch_costs(bd,x,p,blocks,25,config.team_switch_cost)),
                             'joint-boundary':(q,boundary.switch_costs(jb,x,q,blocks,25,config.team_switch_cost))}
                    jobs=[pool.submit(score,args.out/'cv'/match/regime/name,match,name,regime,y,prob,blocks,config,usable,costs) for name,(prob,costs) in choices.items()]
                    for job in as_completed(jobs):
                        row=job.result();row['fold']=index;records.append(row)
            write_json(args.out/f'fold-{index}-records.json',records)
    if scientific_hashes()!=frozen or historical_hashes(args.previous)!=historical_frozen:raise ValueError('Scientific source or comparator changed during CV')
    historical={(r['match'],r['regime']):r for r in history['records'] if r['candidate']=='gated-temporal'}
    for row in records:
        if row['candidate']==BASELINE:
            old=historical[row['match'],row['regime']]
            if row['prediction_sha256']!=old['prediction_sha256'] or any(not np.isclose(row[k],old[k],atol=1e-12,rtol=0) for k in ('spellF1','macroF1','dead_recall','dead_precision')):raise ValueError('Historical v1.4 predictions differ')
    decision=select(records);write_json(args.out/'selection-before-fresh-audit.json',{'protocol_sha256':ph,**decision})
    chosen=decision['audit_candidate'];release=copy.deepcopy(baseline)
    if chosen!=BASELINE:
        train=sorted(CATALOG);meta={'protocol_sha256':ph,'stage':'all-seven-refit','development_cv_selection':chosen}
        joint=fit_joint(args.previous,args.cache,train,baseline,meta) if chosen in ('joint-state','joint-boundary') else None
        if joint is not None:release['joint_model']=joint
        if chosen in ('boundary','joint-boundary'):release['boundary_model']=fit_boundary(args.previous,args.cache,train,baseline,joint,meta)
        release['candidate_id']=chosen
    validate_model(release);write_json(args.out/'release-candidate-model.json',release)
    if chosen==BASELINE:(args.out/'release-candidate-model.json').write_bytes(args.baseline.read_bytes())
    (args.out/'release-candidate-config.json').write_bytes(args.config.read_bytes())
    if scientific_hashes()!=frozen or historical_hashes(args.previous)!=historical_frozen:raise ValueError('Scientific source or comparator changed during refit')
    summary={'protocol':protocol,'protocol_sha256':ph,'source_hashes':frozen,'records':sorted(records,key=lambda r:(r['match'],r['regime'],r['candidate'])),
      'selection':decision,'cache_provenance':proof,'historical_provenance':{'summary_sha256':sha256(args.previous/'summary.json'),'fold_model_sha256':fold_hashes},
      'incumbent_historical_predictions_reproduced':True,'development_candidate_selected':decision['selected']!=BASELINE,
      'fresh_audit_status':'pending-separate-audit','competition_labels_used':False,'independent_test_claimed':False,
      'release_candidate_model_sha256':sha256(args.out/'release-candidate-model.json')}
    write_json(args.out/'summary.json',summary);print(json.dumps(decision),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous',type=Path,default=ROOT/'runs/multimodal-campaign-v4')
    p.add_argument('--cache',type=Path,default=ROOT/'runs/joint-cache-v5')
    p.add_argument('--out',type=Path,default=ROOT/'runs/joint-campaign-v5')
    p.add_argument('--baseline',type=Path,default=ROOT/'models/baselines/v1.4.json')
    p.add_argument('--config',type=Path,default=ROOT/'configs/baselines/v1.4.json')
    p.add_argument('--archive',type=Path,default=ROOT/'models/baselines/v1.4-source.zip')
    p.add_argument('--workers',type=int,default=2,choices=(1,2))
    run(p.parse_args())

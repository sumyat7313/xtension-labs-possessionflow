"""Fixed v6 development ablations against authentic v1.5 out-of-fold models."""
from pathlib import Path
import argparse, copy, json, sys, hashlib, zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from possession import boundary, transition
from possession.config import load_config
from possession.decoder import decode
from possession.io import sha256,write_json
from possession.learned import load_model,validate_model,joint_probabilities
from tools import train_joint_campaign as v5,train_stoppage_v6,train_transition_v6
from tools.control_diagnostics import diagnostic as control_diagnostic,training_spell_weights
from tools.temporal_diagnostics_v6 import diagnostic
from tools.train_control import CATALOG
from tools.train_stoppages import measure

BASELINE='baseline-v1.5'
CANDIDATES=('stoppage-loss','transition','combined')
FOLDS=v5.FOLDS;REGIMES=v5.REGIMES;PRIMARY_REGIMES=v5.PRIMARY_REGIMES
BASELINE_SHA256='3af2dea11261e0454c54ce6791f3c5f0f55f488962da91579addb32a9aea24f0'
ARCHIVE_SHA256='9ff7b8103355d8fdf3160112f78019fc01b001e34fb438a581d23193ad0964fb'
CONFIG_SHA256=v5.CONFIG_SHA256
GATE={'minimum_primary_spellF1_gain':0.,'maximum_cell_spellF1_drop':.02,
      'maximum_cell_dead_recall_drop':.01,'minimum_mean_dead_precision_gain':-.002,
      'minimum_mean_dead_segment_f1_gain':-.002,'minimum_mean_dead_boundary_f1_gain':-.002,
      'minimum_primary_macroF1_gain':-.002,'minimum_mean_short_recall_gain':0.,
      'minimum_mean_short_precision_gain':-.005,
      'required_either_short_or_boundary_gain':.01,
      'ranking':'equal mean of primary spellF1, all-condition short recall and all-condition D boundary F1, then primary spellF1, then candidate name'}


def scientific_hashes():
    names=('tools/train_temporal_v6.py','tools/train_stoppage_v6.py','tools/train_transition_v6.py',
           'tools/temporal_diagnostics_v6.py')
    return dict(v5.scientific_hashes(),**{n:sha256(ROOT/n) for n in names})


def metrics(row):
    d=row['temporal_diagnostics']
    bins=row['duration_precision_recall']['bins']
    short_pred=sum(bins[k]['n_pred'] for k in ('<0.4s','0.4-1s','1-2s'))
    short_hits=sum(bins[k]['n_matched_pred'] for k in ('<0.4s','0.4-1s','1-2s'))
    return {'spell':row['spellF1'],'macro':row['macroF1'],'dead_recall':row['dead_recall'],
            'dead_precision':row['dead_precision'],'dead_segment':d['dead_segments']['f1'],
            'dead_boundary':(d['dead_onsets']['f1']+d['dead_offsets']['f1'])/2,
            'short':d['short_possessions']['recall'],'short_precision':short_hits/short_pred if short_pred else 0.}


def select(records):
    cells=sorted((m,r) for fold in FOLDS for m in fold for r in REGIMES)
    primary=[c for c in cells if c[1] in PRIMARY_REGIMES]
    groups={name:{} for name in (BASELINE,*CANDIDATES)}
    for row in records:
        name=row['candidate'];cell=row['match'],row['regime']
        if name not in groups or cell not in cells or cell in groups[name]:raise ValueError('Unknown or duplicate v6 cell')
        values=metrics(row)
        if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not np.isfinite(x) or not 0<=x<=1 for x in values.values()):
            raise ValueError('Invalid v6 metric or empty required duration bin')
        groups[name][cell]=values
    if any(set(group)!=set(cells) for group in groups.values()):raise ValueError('Every v6 candidate requires all 20 cells')
    means={};gates={};base=groups[BASELINE]
    for name,rows in groups.items():
        delta={k:float(np.mean([rows[c][k]-base[c][k] for c in (primary if k in ('spell','macro') else cells)])) for k in metrics(records[0])}
        means[name]={k:float(np.mean([rows[c][k] for c in (primary if k in ('spell','macro') else cells)])) for k in delta}
        means[name]['objective']=(means[name]['spell']+means[name]['short']+means[name]['dead_boundary'])/3
        cell_spell=min(rows[c]['spell']-base[c]['spell'] for c in cells)
        cell_dead=min(rows[c]['dead_recall']-base[c]['dead_recall'] for c in cells)
        passed=(delta['spell']>=0 and cell_spell>=-.02 and cell_dead>=-.01
                and delta['dead_precision']>=-.002 and delta['dead_segment']>=-.002
                and delta['dead_boundary']>=-.002 and delta['macro']>=-.002 and delta['short']>=0 and delta['short_precision']>=-.005
                and max(delta['short'],delta['dead_boundary'])>=.01)
        gates[name]={'passed':bool(passed),'mean_gains':delta,'minimum_cell_spellF1_gain':float(cell_spell),
                     'minimum_cell_dead_recall_gain':float(cell_dead)}
    ranked=sorted(groups,key=lambda n:(-means[n]['objective'],-means[n]['spell'],n))
    eligible=[n for n in ranked if n==BASELINE or gates[n]['passed']]
    selected=eligible[0]
    audit_candidate=selected if selected!=BASELINE else next(n for n in ranked if n!=BASELINE)
    return {'selected':selected,'audit_candidate':audit_candidate,'ranking':ranked,'means':means,'gates':gates,
            'fresh_audit_used_for_selection':False}


def sequences(previous,cache,matches,base):
    result=[]
    for match in sorted(matches):
        for regime in REGIMES:
            x,z,y,blocks=v5.cached(previous,cache,match,regime)
            dead,home=v5.baseline_components(base,x,z,blocks);w=v5.joint_weights(y,blocks)
            for _,a,b in blocks:
                ii=np.arange(a,b+1,5)
                result.append({'X':np.array(x[ii]),'Z':np.array(z[ii]),'y':y[ii],'w':w[ii],
                               'base_home':v5.logit(home[ii]),'base_dead':v5.logit(dead[ii])})
    return result


def fit_joint(previous,cache,matches,base,meta):
    print('Fitting v6 stoppage loss on '+','.join(sorted(matches)),flush=True)
    return train_stoppage_v6.fit(sequences(previous,cache,matches,base),seed=10527,epochs=8,
        metadata=dict(meta,training_matches=sorted(matches),training_regimes=list(REGIMES)))


def fit_transition(previous,cache,matches,base,joint,meta):
    items=[]
    for match in sorted(matches):
        for regime in REGIMES:
            x,z,y,blocks=v5.cached(previous,cache,match,regime);dead,home=v5.baseline_components(base,x,z,blocks)
            p=joint_probabilities(dead,home,x,z,blocks,25,joint,np.zeros(len(y)))
            items.append({'X':boundary.feature_matrix(x,p,blocks,25),'y':y,'blocks':blocks,'fps':25,'w':training_spell_weights(y,25)})
    print('Fitting v6 transitions on '+','.join(sorted(matches)),flush=True)
    return train_transition_v6.fit(items,metadata=dict(meta,training_matches=sorted(matches),training_regimes=list(REGIMES)),stride=5,l2=.01)


def score(dest,match,candidate,regime,y,p,blocks,config,usable,signals):
    labels=np.full(len(y),'X',dtype='<U1')
    for (_,a,b),use in zip(blocks,usable):
        if use:
            costs=None if signals is None else signals['team_switch_costs'][a:b+1]
            early=None if signals is None else signals['early_exit_allowed'][a:b+1]
            labels[a:b+1]=np.array(list('HAD'))[decode(p[a:b+1],25,config,costs,early_exit_allowed=early)]
    row=measure(dest,match,candidate,regime,y,labels,0)
    row['control_diagnostics']=control_diagnostic(y,labels,25)
    row['temporal_diagnostics']=diagnostic(y,labels,25,blocks)
    row['duration_precision_recall']=train_transition_v6.diagnostic(y,labels,25)
    row['early_exit_eligible_frames']=0 if signals is None else int(signals['early_exit_allowed'].sum())
    write_json(dest/'temporal-diagnostics.json',row['temporal_diagnostics'])
    return row


def frozen_history(previous_v5):
    return {p.relative_to(previous_v5).as_posix():sha256(p) for p in [previous_v5/'summary.json',*sorted((previous_v5/'fold-models').rglob('*.json'))]}


def verify_inputs(args):
    if sha256(args.baseline)!=BASELINE_SHA256 or sha256(args.archive)!=ARCHIVE_SHA256 or sha256(args.config)!=CONFIG_SHA256:
        raise ValueError('v1.5 immutable baseline artifacts changed')
    history=json.loads((args.previous_v5/'summary.json').read_text());committed=json.loads((ROOT/'models/joint-selection.json').read_text());committed.pop('release_decision')
    if history!=committed or history['selection']['selected']!='joint-state' or history['release_candidate_model_sha256']!=BASELINE_SHA256:
        raise ValueError('v1.5 campaign evidence differs')
    with zipfile.ZipFile(args.archive) as z:
        if any(hashlib.sha256(z.read(n)).hexdigest()!=h for n,h in history['source_hashes'].items()):
            raise ValueError('v1.5 source archive differs')
    legacy=argparse.Namespace(previous=args.previous,cache=args.cache,baseline=ROOT/'models/baselines/v1.4.json',
        config=ROOT/'configs/baselines/v1.4.json',archive=ROOT/'models/baselines/v1.4-source.zip')
    _,proof=v5.verify_inputs(legacy)
    return history,proof


def run(args):
    if args.out.exists():raise ValueError('Use a new immutable campaign directory')
    history,proof=verify_inputs(args);base_final=load_model(args.baseline);config=load_config(args.config)
    frozen=scientific_hashes();historical=frozen_history(args.previous_v5);older=v5.historical_hashes(args.previous)
    protocol={'schema_version':6,'source_hashes':frozen,'baseline_sha256':BASELINE_SHA256,'archive_sha256':ARCHIVE_SHA256,
        'config_sha256':CONFIG_SHA256,'folds':FOLDS,'regimes':REGIMES,'primary_regimes':PRIMARY_REGIMES,
        'candidates':[BASELINE,*CANDIDATES],'gate':GATE,'stoppage_weighting':train_stoppage_v6.WEIGHTING,
        'transition_policy':transition.POLICY,'training':{'seed':10527,'epochs':8,'sample_hz':5,'boundary_stride':5,'boundary_l2':.01},
        'v5_historical_sha256':historical,'v4_historical_sha256':older,
        'baseline_comparison':'v1.5 original out-of-fold joint residual plus historical v1.4 fold experts, never all-seven final model on scored folds',
        'exposure':'All seven Sportec matches previously exposed development. Fresh labels never used for training or candidate selection.',
        'fresh_audit_used_for_selection':False,'competition_labels_used':False,
        'fresh_audit_policy':'Audit one frozen experimental winner even if development fails; promotion requires both development and new fresh gates. No runner-up or retuning.'}
    args.out.mkdir(parents=True);write_json(args.out/'protocol.json',protocol);ph=sha256(args.out/'protocol.json');records=[]
    def unchanged():
        if scientific_hashes()!=frozen or frozen_history(args.previous_v5)!=historical or v5.historical_hashes(args.previous)!=older:
            raise ValueError('Frozen scientific source or historical model changed')
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index,held in enumerate(FOLDS):
            train=sorted(set(CATALOG)-set(held));meta={'protocol_sha256':ph,'fold':index,'held_out_matches':list(held)}
            base=v5.fold_base(args.previous,load_model(ROOT/'models/baselines/v1.4.json'),index)
            old_joint=json.loads((args.previous_v5/'fold-models'/str(index)/'joint.json').read_text())
            if old_joint['metadata']['training_matches']!=train or old_joint['metadata']['held_out_matches']!=list(held):
                raise ValueError('v1.5 joint fold contains evaluation matches')
            dest=args.out/'fold-models'/str(index)
            joint=fit_joint(args.previous,args.cache,train,base,meta);write_json(dest/'stoppage-joint.json',joint)
            simple=fit_transition(args.previous,args.cache,train,base,old_joint,meta);write_json(dest/'transition.json',simple)
            combined=fit_transition(args.previous,args.cache,train,base,joint,meta);write_json(dest/'combined-transition.json',combined)
            for match in held:
                for regime in REGIMES:
                    x,z,y,blocks=v5.cached(args.previous,args.cache,match,regime);dead,home=v5.baseline_components(base,x,z,blocks)
                    p=joint_probabilities(dead,home,x,z,blocks,25,old_joint,np.zeros(len(y)))
                    q=joint_probabilities(dead,home,x,z,blocks,25,joint,np.zeros(len(y)))
                    usable=json.loads((args.previous/'cache'/match/f'{regime}-meta.json').read_text())['usable']
                    choices={BASELINE:(p,None),'stoppage-loss':(q,None),
                        'transition':(p,transition.evidence(simple,x,p,blocks,25,config.team_switch_cost)),
                        'combined':(q,transition.evidence(combined,x,q,blocks,25,config.team_switch_cost))}
                    jobs=[pool.submit(score,args.out/'cv'/match/regime/name,match,name,regime,y,prob,blocks,config,usable,signals)
                          for name,(prob,signals) in choices.items()]
                    for job in as_completed(jobs):
                        row=job.result();row['fold']=index;records.append(row)
            write_json(args.out/f'fold-{index}-records.json',records)
    unchanged()
    previous={(r['match'],r['regime']):r for r in history['records'] if r['candidate']=='joint-state'}
    for r in records:
        if r['candidate']==BASELINE:
            old=previous[r['match'],r['regime']]
            if r['prediction_sha256']!=old['prediction_sha256'] or any(not np.isclose(r[k],old[k],atol=1e-12,rtol=0) for k in ('spellF1','macroF1','dead_recall','dead_precision')):
                raise ValueError('v1.5 historical baseline predictions did not reproduce')
    decision=select(records);write_json(args.out/'selection-before-fresh-audit.json',{'protocol_sha256':ph,**decision})
    chosen=decision['audit_candidate'];release=copy.deepcopy(base_final)
    meta={'protocol_sha256':ph,'stage':'all-seven-refit','development_cv_selection':chosen}
    if chosen in ('stoppage-loss','combined'):release['joint_model']=fit_joint(args.previous,args.cache,sorted(CATALOG),base_final,meta)
    if chosen in ('transition','combined'):release['transition_model']=fit_transition(args.previous,args.cache,sorted(CATALOG),base_final,release['joint_model'],meta)
    release['candidate_id']=chosen;validate_model(release);unchanged()
    write_json(args.out/'release-candidate-model.json',release);(args.out/'release-candidate-config.json').write_bytes(args.config.read_bytes())
    summary={'protocol':protocol,'protocol_sha256':ph,'source_hashes':frozen,'cache_provenance':proof,
        'records':sorted(records,key=lambda r:(r['match'],r['regime'],r['candidate'])),'selection':decision,
        'incumbent_historical_predictions_reproduced':True,'development_candidate_selected':decision['selected']!=BASELINE,
        'competition_labels_used':False,'independent_test_claimed':False,'fresh_audit_status':'pending-separate-audit',
        'release_candidate_model_sha256':sha256(args.out/'release-candidate-model.json')}
    write_json(args.out/'summary.json',summary);print(json.dumps(decision),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name,value in [('previous','runs/multimodal-campaign-v4'),('previous-v5','runs/joint-campaign-v5'),('cache','runs/joint-cache-v5'),
                       ('baseline','models/baselines/v1.5.json'),('archive','models/baselines/v1.5-source.zip'),
                       ('config','configs/baselines/v1.5.json'),('out','runs/temporal-campaign-v6')]:
        p.add_argument('--'+name,type=Path,default=ROOT/value)
    p.add_argument('--workers',type=int,choices=(1,2),default=2)
    run(p.parse_args())

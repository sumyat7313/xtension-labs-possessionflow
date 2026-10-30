"""Fixed typed-boundary/contact ablations; fresh labels never select a model."""
from pathlib import Path
import argparse,copy,json,sys
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from possession import typed_boundary as tb,contact_evidence
from possession.config import load_config
from possession.decoder import decode
from possession.io import sha256,write_json
from possession.learned import load_model,validate_model,joint_probabilities
from tools import train_temporal_v6 as v6,train_joint_campaign as v5,train_typed_boundary_v7 as training
from tools.control_diagnostics import diagnostic as control_diagnostic
from tools.temporal_diagnostics_v6 import diagnostic
from tools.train_transition_v6 import diagnostic as duration_diagnostic
from tools.train_stoppages import measure
from tools.train_control import CATALOG

BASELINE=v6.BASELINE;BASELINE_SHA256=v6.BASELINE_SHA256;CONFIG_SHA256=v6.CONFIG_SHA256
FOLDS=v6.FOLDS;REGIMES=v6.REGIMES;PRIMARY_REGIMES=v6.PRIMARY_REGIMES
CANDIDATES=('typed-boundary','typed-contact')
PREVIOUS_SOURCE_SHA256='ee6da5f6ba376ca70bdf4c1f34f53b8452925b2b81009382cb02e40879afd536'
GATE={'minimum_primary_spellF1_gain':.005,'minimum_mean_dead_segment_gain':0.,
      'minimum_mean_dead_boundary_gain':0.,'minimum_mean_short_recall_gain':0.,
      'minimum_mean_dead_precision_gain':-.001,'minimum_mean_short_precision_gain':-.002,
      'minimum_primary_macroF1_gain':-.001,'maximum_cell_spellF1_drop':.01,'maximum_cell_dead_recall_drop':.005,
      'required_either_dead_boundary_or_short_recall_gain':.01,
      'ranking':'primary spellF1 then mean D boundary F1 then short recall then name',
      'fresh_requirement':'Explicit sufficient fresh stoppage/possession truth plus nonregression; phase-only audit cannot authorize promotion.'}


def scientific_hashes():
    return dict(v6.scientific_hashes(),**{n:sha256(ROOT/n) for n in ('tools/train_boundary_campaign_v7.py','tools/train_typed_boundary_v7.py')})


def select(records):
    cells=sorted((m,r) for fold in FOLDS for m in fold for r in REGIMES);primary=[c for c in cells if c[1] in PRIMARY_REGIMES]
    groups={name:{} for name in (BASELINE,*CANDIDATES)}
    for row in records:
        name=row['candidate'];cell=row['match'],row['regime']
        if name not in groups or cell not in cells or cell in groups[name]:raise ValueError('Invalid or duplicate v7 cell')
        values=v6.metrics(row)
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or not 0<=v<=1 for v in values.values()):raise ValueError('Invalid required v7 metric')
        groups[name][cell]=values
    if any(set(g)!=set(cells) for g in groups.values()):raise ValueError('Every v7 candidate needs all 20 cells')
    means={};gates={};base=groups[BASELINE]
    for name,rows in groups.items():
        gains={k:float(np.mean([rows[c][k]-base[c][k] for c in (primary if k in ('spell','macro') else cells)])) for k in v6.metrics(records[0])}
        means[name]={k:float(np.mean([rows[c][k] for c in (primary if k in ('spell','macro') else cells)])) for k in gains}
        spell=min(rows[c]['spell']-base[c]['spell'] for c in cells);death=min(rows[c]['dead_recall']-base[c]['dead_recall'] for c in cells)
        passed=(gains['spell']>=.005 and gains['dead_segment']>=0 and gains['dead_boundary']>=0 and gains['short']>=0
            and gains['dead_precision']>=-.001 and gains['short_precision']>=-.002 and gains['macro']>=-.001
            and spell>=-.01 and death>=-.005 and max(gains['dead_boundary'],gains['short'])>=.01)
        gates[name]={'passed':bool(passed),'mean_gains':gains,'minimum_cell_spellF1_gain':float(spell),'minimum_cell_dead_recall_gain':float(death)}
    ranking=sorted(groups,key=lambda n:(-means[n]['spell'],-means[n]['dead_boundary'],-means[n]['short'],n))
    selected=next(n for n in ranking if n==BASELINE or gates[n]['passed'])
    audited=selected if selected!=BASELINE else next(n for n in ranking if n!=BASELINE)
    return {'selected':selected,'audit_candidate':audited,'ranking':ranking,'gates':gates,'means':means,'fresh_audit_used_for_selection':False}


def prepare_items(previous,cache,matches,base,joint):
    items=[]
    for match in sorted(matches):
        for regime in REGIMES:
            x,z,y,blocks=v5.cached(previous,cache,match,regime);dead,home=v5.baseline_components(base,x,z,blocks)
            p=joint_probabilities(dead,home,x,z,blocks,25,joint,np.zeros(len(y)))
            # Append contact once; base candidate uses the exact shared prefix.
            matrix=tb.feature_matrix(x,z,p,blocks,25,contact=True)
            items.append({'X':matrix,'y':y,'blocks':blocks,'fps':25})
    return items


def fit_items(items,contact,metadata):
    width=len(tb.feature_names(contact))
    return training.fit([dict(item,X=item['X'][:,:width]) for item in items],contact=contact,metadata=metadata,stride=5,l2=.01)


def score(dest,match,candidate,regime,y,p,blocks,config,usable,signals):
    labels=np.full(len(y),'X',dtype='<U1')
    for (_,a,b),use in zip(blocks,usable):
        if use:
            kwargs={} if signals is None else {k:signals[k][a:b+1] for k in ('team_switch_costs','stop_costs','restart_costs')}
            labels[a:b+1]=np.array(list('HAD'))[decode(p[a:b+1],25,config,**kwargs)]
    row=measure(dest,match,candidate,regime,y,labels,0)
    row['control_diagnostics']=control_diagnostic(y,labels,25);row['temporal_diagnostics']=diagnostic(y,labels,25,blocks)
    row['duration_precision_recall']=duration_diagnostic(y,labels,25)
    write_json(dest/'temporal-diagnostics.json',row['temporal_diagnostics']);return row


def run(args):
    if args.out.exists():raise ValueError('Use a new immutable v7 campaign directory')
    history,proof=v6.verify_inputs(args)
    if sha256(ROOT/'models/baselines/v1.5.1-source.zip')!=PREVIOUS_SOURCE_SHA256:raise ValueError('v1.5.1 source archive changed')
    baseline=load_model(args.baseline);config=load_config(args.config)
    frozen=scientific_hashes();h5=v6.frozen_history(args.previous_v5);h4=v5.historical_hashes(args.previous)
    protocol={'schema_version':7,'source_hashes':frozen,'baseline_sha256':BASELINE_SHA256,'config_sha256':CONFIG_SHA256,
        'previous_release_source_sha256':PREVIOUS_SOURCE_SHA256,'v5_source_sha256':v6.ARCHIVE_SHA256,
        'folds':FOLDS,'regimes':REGIMES,'primary_regimes':PRIMARY_REGIMES,'candidates':[BASELINE,*CANDIDATES],
        'gate':GATE,'typed_policy':tb.POLICY,'contact_policy':contact_evidence.POLICY,
        'training':{'stride':5,'all_positive_native_frames':True,'negative_sampling_weight':'exact per-half full-negative-count / sampled-negative-count','l2':.01},
        'v5_historical_sha256':h5,'v4_historical_sha256':h4,'competition_labels_used':False,
        'fresh_audit_used_for_selection':False,'contact_truth_claimed':False,'video_evidence_active':False,
        'exposure':'All Sportec matches exposed development. Old Metrica, WCSF and prior SkillCorner audits consumed; no fresh sources enter training.',
        'audit_rule':'Freeze one experimental winner before new reference access. Phase-only agreement is insufficient for promotion of a stoppage-changing model.'}
    args.out.mkdir(parents=True);write_json(args.out/'protocol.json',protocol);ph=sha256(args.out/'protocol.json');records=[]
    def unchanged():
        if scientific_hashes()!=frozen or v6.frozen_history(args.previous_v5)!=h5 or v5.historical_hashes(args.previous)!=h4:raise ValueError('Frozen v7 source or historical comparator changed')
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index,held in enumerate(FOLDS):
            matches=sorted(set(CATALOG)-set(held));base=v5.fold_base(args.previous,load_model(ROOT/'models/baselines/v1.4.json'),index)
            joint=json.loads((args.previous_v5/'fold-models'/str(index)/'joint.json').read_text())
            if joint['metadata']['training_matches']!=matches or joint['metadata']['held_out_matches']!=list(held):raise ValueError('Joint baseline fold includes scored match')
            meta={'protocol_sha256':ph,'training_matches':matches,'training_regimes':list(REGIMES),'held_out_matches':list(held),'fold':index}
            print('Preparing typed-boundary training fold '+str(index),flush=True)
            items=prepare_items(args.previous,args.cache,matches,base,joint);models={}
            for name,contact in zip(CANDIDATES,(False,True)):
                print('Fitting '+name+' fold '+str(index),flush=True)
                models[name]=fit_items(items,contact,meta);write_json(args.out/'fold-models'/str(index)/(name+'.json'),models[name])
            del items
            for match in held:
                for regime in REGIMES:
                    x,z,y,blocks=v5.cached(args.previous,args.cache,match,regime);dead,home=v5.baseline_components(base,x,z,blocks)
                    p=joint_probabilities(dead,home,x,z,blocks,25,joint,np.zeros(len(y)))
                    usable=json.loads((args.previous/'cache'/match/f'{regime}-meta.json').read_text())['usable']
                    choices={BASELINE:None,**{name:tb.evidence(model,x,z,p,blocks,25,config) for name,model in models.items()}}
                    jobs=[pool.submit(score,args.out/'cv'/match/regime/name,match,name,regime,y,p,blocks,config,usable,signals) for name,signals in choices.items()]
                    for job in as_completed(jobs):
                        r=job.result();r['fold']=index;records.append(r)
            write_json(args.out/f'fold-{index}-records.json',records)
    unchanged();old={(r['match'],r['regime']):r for r in history['records'] if r['candidate']=='joint-state'}
    for r in records:
        if r['candidate']==BASELINE and r['prediction_sha256']!=old[r['match'],r['regime']]['prediction_sha256']:raise ValueError('Historical v1.5 prediction bytes differ')
    selection=select(records);write_json(args.out/'selection-before-fresh-audit.json',{'protocol_sha256':ph,**selection})
    chosen=selection['audit_candidate'];print('Refitting frozen experimental winner: '+chosen,flush=True)
    meta={'protocol_sha256':ph,'training_matches':sorted(CATALOG),'training_regimes':list(REGIMES),'stage':'all-seven-refit','development_cv_selection':chosen}
    items=prepare_items(args.previous,args.cache,sorted(CATALOG),baseline,baseline['joint_model'])
    model=copy.deepcopy(baseline);model['typed_boundary_model']=fit_items(items,chosen=='typed-contact',meta);model['candidate_id']=chosen
    validate_model(model);unchanged();write_json(args.out/'release-candidate-model.json',model)
    (args.out/'release-candidate-config.json').write_bytes(args.config.read_bytes())
    summary={'protocol':protocol,'protocol_sha256':ph,'source_hashes':frozen,'cache_provenance':proof,
        'records':sorted(records,key=lambda r:(r['match'],r['regime'],r['candidate'])),'selection':selection,
        'incumbent_historical_predictions_reproduced':True,'development_candidate_selected':selection['selected']!=BASELINE,
        'competition_labels_used':False,'independent_test_claimed':False,'fresh_audit_status':'pending-separate-audit',
        'release_candidate_model_sha256':sha256(args.out/'release-candidate-model.json')}
    write_json(args.out/'summary.json',summary);print(json.dumps(selection),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name,value in [('previous','runs/multimodal-campaign-v4'),('previous-v5','runs/joint-campaign-v5'),('cache','runs/joint-cache-v5'),
        ('baseline','models/baselines/v1.5.json'),('archive','models/baselines/v1.5-source.zip'),('config','configs/baselines/v1.5.json'),('out','runs/boundary-campaign-v7')]:
        p.add_argument('--'+name,type=Path,default=ROOT/value)
    p.add_argument('--workers',type=int,default=2,choices=(1,2));run(p.parse_args())

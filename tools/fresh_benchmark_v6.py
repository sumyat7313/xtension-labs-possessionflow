"""One-shot v6 audit on new dense provider flags and two automatic phase sources.

No final match is training or adapter-development data. WCSF labels are explicit
provider flags, with undocumented annotation method: never called human gold.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from urllib.request import urlopen
from zipfile import ZipFile
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from possession.io import Tracking,sha256,write_json,write_submission
from possession.config import load_config
from tools import fresh_benchmark as old
from tools.temporal_diagnostics_v6 import diagnostic
from evaluation import possession_metrics as pm

WCSF_COMMIT='1c9ba798f98a9fffb16081a5d3965774d4e5662c'
SC_COMMIT=old.SC_COMMIT
WCSF={
 'positions_FIRST_HALF.csv.zip':(22704565,'d7de2a38f5650150684b1008aa7f4fd28b85d777'),
 'positions_SECOND_HALF.csv.zip':(23746663,'66fc3ee2966aaf3392d4a327ba3f49385e9c7032')}
SC={
 1925299:{'match.json':(28396,'62b69aba5be11f795ccdbcf65c6d07f662832b66'),
 'phases_of_play.csv':(131104,'20c06b378f9a7181ab0e02de7c4a37779316386a'),
 'tracking_extrapolated.jsonl':(97091519,'48528be40ad82fa3e2b232e1af8d311705f7d6d119925ef199d2b59ea36fd3c9')},
 1927964:{'match.json':(33400,'e42fb7edbd49c81e087d0d73cc885cc3095d6a20'),
 'phases_of_play.csv':(117263,'d1960e21564d61372824bf8a844c68b391e35abb'),
 'tracking_extrapolated.jsonl':(91049828,'858c7a756ca3d33017f864775af5999d2c0f364e58ab60ae040c8b9ebab17827')}}
SPLIT={'schema_version':1,'wcsf_commit':WCSF_COMMIT,'skillcorner_commit':SC_COMMIT,
 'final_dense_provider_matches':['wcsf2023'],'final_silver_matches':[1925299,1927964],
 'adapter_development':'Synthetic WCSF fixtures and already exposed SkillCorner1874553 only. No final reference values parsed before model/source/policy freeze.'}
POLICY={'dense_maximum_drop':.005,'dense_dead_segment_maximum_drop':.01,
 'dense_dead_boundary_mean_maximum_drop':.01,'dense_short_recall_maximum_drop':.01,
 'silver_mean_accuracy_minimum_delta':0.,'silver_maximum_match_accuracy_drop':.02,
 'require_strict_dense_boundary_or_short_recall_improvement':True,
 'require_dense_H_A_D_and_short_spells_and_dead_boundaries':True,
 'rule':'Audit one frozen development-selected experiment versus v1.5. Both development and fresh gates are required for promotion; no runner-up, retuning, or repeated audit.'}
OBS_COLUMNS=['x','y','group_id','timestamp_ms','x_ball','y_ball','player_id']
REF_COLUMNS=['timestamp_ms','AliveFlag','PossessionFlag']
GEOMETRY_TOLERANCE_METRES=.01


def freeze_split(folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    for name,value in [('split-before-reference-access.json',SPLIT),('policy-before-reference-access.json',POLICY)]:
        path=folder/name
        if path.exists():
            if json.loads(path.read_text(encoding='utf-8-sig'))!=value: raise ValueError('Fresh split or policy changed')
        else: write_json(path,value)
    return folder/'split-before-reference-access.json'


def catalog(folder):
    folder=Path(folder);rows=[]
    for name,expected in WCSF.items():
        rows.append((folder/name,expected,f'https://raw.githubusercontent.com/MatKempeGroningen/WCSF_SolutionRoom/{WCSF_COMMIT}/{name}'))
    for match,files in SC.items():
        for suffix,expected in files.items():
            name=f'{match}_{suffix}';host='media.githubusercontent.com/media' if suffix.endswith('jsonl') else 'raw.githubusercontent.com'
            rows.append((folder/name,expected,f'https://{host}/SkillCorner/opendata/{SC_COMMIT}/data/matches/{match}/{name}'))
    return rows


def download(folder):
    freeze_split(folder);manifest=[]
    for path,expected,url in catalog(folder):
        if not path.exists():
            temporary=path.with_name(path.name+'.partial');received=0
            with urlopen(url,timeout=120) as response,temporary.open('wb') as out:
                while block:=response.read(1024*1024):
                    received+=len(block)
                    if received>expected[0]: raise ValueError('Download exceeded catalog size')
                    out.write(block)
            old.check_file(temporary,expected);temporary.replace(path)
        old.check_file(path,expected)
        manifest.append({'file':path.name,'bytes':expected[0],'sha256':sha256(path),'source_identity':expected[1],'url':url})
        print('Verified '+path.name,flush=True)
    write_json(Path(folder)/'download-manifest.json',{'files':manifest,'reference_values_parsed':False})


def _read_wcsf_columns(path,columns):
    """Only selected columns are parsed; reference values never enter inference."""
    with ZipFile(path) as archive:
        members=[n for n in archive.namelist() if '/' not in n and n.endswith('.csv')]
        if len(members)!=1: raise ValueError('Expected one root CSV in WCSF ZIP')
        with archive.open(members[0]) as handle:
            return pd.read_csv(handle,usecols=columns)


def _timeline(frame):
    t=np.asarray(frame['timestamp_ms'],dtype=float)
    if not len(t) or not np.isfinite(t).all() or not np.allclose(t,np.rint(t),atol=1e-5,rtol=0):
        raise ValueError('Invalid WCSF timestamps')
    t=np.rint(t).astype(np.int64);values=np.unique(t)
    if np.any((values-values[0])%100): raise ValueError('WCSF timeline is not 10 Hz')
    full=np.arange(values[0],values[-1]+1,100,dtype=np.int64)
    if len(full)>50000: raise ValueError('Unexpected WCSF half duration')
    return full,((t-full[0])//100).astype(int)


def _wcsf_geometry(frame,indices):
    """Resolve documented mirroring using duplicated ball XY only, never labels.

    Four predeclared conventions: ball in team frame or player-relative offset,
    and mirrored y or unchanged y. Require <=1cm within-frame disagreement on
    all available ball copies, plus two frames with both teams. Distinct valid
    coordinate reconstructions are ambiguous and rejected.
    """
    team=np.asarray(frame['group_id'],float)
    if not np.isin(team,[1,2]).all(): raise ValueError('Unknown WCSF team')
    player=np.asarray(frame[['x','y']],float);ball=np.asarray(frame[['x_ball','y_ball']],float)
    if np.isinf(player).any() or np.isinf(ball).any(): raise ValueError('Infinite WCSF coordinate')
    signs=np.where(team==1,1.,-1.);valid=[]
    observed=pd.DataFrame({'frame':indices,'team':team,'finite':np.isfinite(ball).all(axis=1)})
    both=observed[observed.finite].groupby('frame').team.nunique()
    if int((both==2).sum())<2: raise ValueError('Insufficient cross-team ball copies to resolve geometry')
    for relative in (False,True):
        for mirror_y in (True,False):
            scale=np.column_stack((signs,signs if mirror_y else np.ones(len(team))))
            points=player*scale;balls=(ball+player if relative else ball)*scale
            # A relative-offset interpretation can lose ball copies when player
            # coordinates are missing. Require support after reconstruction too.
            supported=pd.DataFrame({'frame':indices,'team':team,'finite':np.isfinite(balls).all(axis=1)})
            paired=supported[supported.finite].groupby('frame').team.nunique()
            if int((paired==2).sum())<2: continue
            copies=pd.DataFrame({'frame':indices,'bx':balls[:,0],'by':balls[:,1]})
            group=copies.groupby('frame')[['bx','by']]
            spread=(group.max()-group.min()).to_numpy();error=float(np.nanmax(spread))
            if error<=GEOMETRY_TOLERANCE_METRES+1e-8:
                valid.append((points,balls,{'ball_coordinates':'player_offset' if relative else 'team_frame',
                    'away_y_mirrored':mirror_y,'max_ball_copy_disagreement_metres':error}))
    if not valid: raise ValueError('No predeclared WCSF geometry convention has consistent ball copies')
    first=valid[0]
    for other in valid[1:]:
        if not (np.allclose(first[0],other[0],atol=1e-8,equal_nan=True) and np.allclose(first[1],other[1],atol=1e-8,equal_nan=True)):
            raise ValueError('Ambiguous WCSF geometry convention')
    return first


def wcsf_observations(frames):
    """Two dataframes containing ONLY OBS_COLUMNS; metadata/flags are forbidden."""
    if len(frames)!=2: raise ValueError('Need both WCSF halves')
    all_ids={}
    for frame in frames:
        if set(frame.columns)!=set(OBS_COLUMNS): raise ValueError('Observation table must use exact whitelist')
        for ident,team in zip(frame['player_id'],frame['group_id']):
            key=str(ident)
            if key in all_ids and all_ids[key]!=int(team): raise ValueError('WCSF player changes team')
            all_ids[key]=int(team)
    ids=sorted(all_ids);lookup={p:i for i,p in enumerate(ids)};parts=[];ballparts=[];blocks=[];times=[];geometry=[];offset=0
    for half,frame in enumerate(frames,1):
        timeline,indices=_timeline(frame)
        if frame[['timestamp_ms','player_id']].duplicated().any(): raise ValueError('Duplicate WCSF player/frame')
        points,balls,convention=_wcsf_geometry(frame,indices);geometry.append(convention)
        n=len(timeline);players=np.full((n,len(ids),2),np.nan)
        slots=np.array([lookup[str(p)] for p in frame['player_id']])
        players[indices,slots]=points/[105.,68.]*[1,-1]+.5
        ball=np.full((n,2),np.nan)
        grouped=pd.DataFrame({'frame':indices,'x':balls[:,0],'y':balls[:,1]}).groupby('frame')[['x','y']].mean()
        ball[grouped.index.to_numpy()]=grouped.to_numpy()/[105.,68.]*[1,-1]+.5
        parts.append(players);ballparts.append(ball);blocks.append((half,offset,offset+n-1));times.append(timeline);offset+=n
    players=np.concatenate(parts);ball=np.concatenate(ballparts);pitch=(105.,68.)
    return Tracking(players,old._speed(players,blocks,pitch,10),ball,old._speed(ball,blocks,pitch,10),
        np.array([all_ids[p]-1 for p in ids],np.int8),('WCSF home','WCSF away')),blocks,pitch,{
            'fps':10,'half_timestamps':[v.tolist() for v in times],'geometry':geometry,
            'observation_policy':'Only x,y,x_ball,y_ball,group_id,timestamp_ms,player_id; provider speeds, flags and game phases excluded. Speeds derived after de-mirroring.'}


def read_wcsf(folder):
    frames=[]
    for name,identity in WCSF.items():
        path=Path(folder)/name;old.check_file(path,identity);frames.append(_read_wcsf_columns(path,OBS_COLUMNS))
    return wcsf_observations(frames)


def wcsf_references(folder,alignment):
    labels=[];counts={'contradictory_frames_excluded':0,'missing_frames_unknown':0}
    for half,(name,identity) in enumerate(WCSF.items()):
        path=Path(folder)/name;old.check_file(path,identity);frame=_read_wcsf_columns(path,REF_COLUMNS)
        timeline,indices=_timeline(frame)
        if timeline.tolist()!=alignment['half_timestamps'][half]: raise ValueError('Reference and observation timelines differ')
        alive=np.asarray(frame['AliveFlag'],float);team=np.asarray(frame['PossessionFlag'],float)
        if not np.isin(alive[np.isfinite(alive)],[0,1]).all(): raise ValueError('Unexpected AliveFlag')
        values=np.full(len(frame),'X',dtype='<U1');values[alive==0]='D'
        values[(alive==1)&(team==1)]='H';values[(alive==1)&(team==2)]='A'
        table=pd.DataFrame({'frame':indices,'label':values});group=table.groupby('frame').label
        unique=group.nunique();first=group.first();conflict=unique>1
        result=np.full(len(timeline),'X',dtype='<U1');result[first.index]=first.to_numpy();result[unique.index[conflict]]='X'
        counts['contradictory_frames_excluded']+=int(conflict.sum())
        counts['missing_frames_unknown']+=int(len(timeline)-len(unique))
        if conflict.sum()/max(len(unique),1)>.01: raise ValueError('More than one percent contradictory WCSF references')
        labels.append(result)
    reference=np.concatenate(labels);counts['class_counts']={c:int(np.sum(reference==c)) for c in 'HADX'}
    counts['scored_frames']=int(np.sum(reference!='X'));counts['annotation_method']='undocumented_provider_flags'
    return reference,counts


def read_skillcorner(folder,match):
    folder=Path(folder)
    for suffix in ('match.json','tracking_extrapolated.jsonl'): old.check_file(folder/f'{match}_{suffix}',SC[match][suffix])
    metadata=json.loads((folder/f'{match}_match.json').read_text(encoding='utf-8'))
    with (folder/f'{match}_tracking_extrapolated.jsonl').open(encoding='utf-8') as f: rows=[json.loads(line) for line in f]
    return old._skillcorner_observations(metadata,rows)


def dense_score(reference,prediction,blocks,fps):
    # Insert unknown separators for official fps-independent metrics; no half joining.
    gt=np.concatenate([np.concatenate((reference[a:b+1],['X'])) for _,a,b in blocks])
    pred=np.concatenate([np.concatenate((prediction[a:b+1],['X'])) for _,a,b in blocks])
    scores=pm.evaluate(pm.label_codes(gt),pm.label_codes(pred),taus=(round(.5*fps),round(fps)))
    return {'n':int(np.sum(reference!='X')),'reference_quality':'dense_provider_flags_annotation_unverified',
        'dense_human_had_truth':False,'fps':fps,'class_counts':{c:int(np.sum(reference==c)) for c in 'HAD'},
        'spellF1':scores['spell_f1_mean'],'macroF1':scores['macro_f1'],
        'dead_precision':scores['per_class']['D']['precision'],'dead_recall':scores['per_class']['D']['recall'],
        'dead_f1':scores['per_class']['D']['f1'],'temporal':diagnostic(reference,prediction,fps,blocks)}


def decide(scores):
    required={'wcsf2023',*[f'skillcorner{m}' for m in SPLIT['final_silver_matches']]}
    if set(scores)!=required: raise ValueError('Every frozen fresh match is required')
    for pair in scores.values():
        if set(pair)!={'baseline','candidate'} or pair['baseline']['n']<=0 or pair['baseline']['n']!=pair['candidate']['n']:
            raise ValueError('Invalid paired reference support')
    dense=scores['wcsf2023'];baseline,candidate=dense['baseline'],dense['candidate']
    paths={'spellF1':lambda r:r['spellF1'],'macroF1':lambda r:r['macroF1'],
        'dead_precision':lambda r:r['dead_precision'],'dead_recall':lambda r:r['dead_recall'],
        'dead_segments_f1':lambda r:r['temporal']['dead_segments']['f1'],
        'dead_boundary_mean_f1':lambda r:np.mean([r['temporal'][k]['f1'] for k in ('dead_onsets','dead_offsets')]),
        'short_recall':lambda r:r['temporal']['short_possessions']['recall']}
    supported=all(baseline['class_counts'].get(c,0)>0 for c in 'HAD') and all(baseline['temporal'][k]['n_gt']>0 for k in ('short_possessions','dead_onsets','dead_offsets'))
    delta={}
    for key,read in paths.items():
        a,b=read(baseline),read(candidate)
        if a is None or b is None:
            if key!='short_recall': raise ValueError('Missing required metric')
            delta[key]=None;supported=False;continue
        if any(isinstance(v,bool) or not np.isfinite(v) or not 0<=v<=1 for v in (a,b)): raise ValueError('Invalid metric')
        delta[key]=float(b-a)
    silver={}
    for match in SPLIT['final_silver_matches']:
        pair=scores[f'skillcorner{match}'];a,b=[pair[k]['accuracy'] for k in ('baseline','candidate')]
        if any(isinstance(v,bool) or not np.isfinite(v) or not 0<=v<=1 for v in (a,b)): raise ValueError('Invalid silver metric')
        silver[str(match)]=b-a
    tolerances={**{k:POLICY['dense_maximum_drop'] for k in ('spellF1','macroF1','dead_precision','dead_recall')},
        'dead_segments_f1':POLICY['dense_dead_segment_maximum_drop'],
        'dead_boundary_mean_f1':POLICY['dense_dead_boundary_mean_maximum_drop'],
        'short_recall':POLICY['dense_short_recall_maximum_drop']}
    passed=supported and all(delta[k] is not None and delta[k]>=-v for k,v in tolerances.items())
    improved=delta['dead_boundary_mean_f1']>0 or (delta['short_recall'] is not None and delta['short_recall']>0)
    passed=passed and improved and np.mean(list(silver.values()))>=0 and min(silver.values())>=-.02
    return {'passed':bool(passed),'dense_support_sufficient':bool(supported),'strict_weakness_improvement':bool(improved),
        'dense_candidate_minus_baseline':delta,'silver_candidate_minus_baseline':silver,
        'silver_mean_accuracy_delta':float(np.mean(list(silver.values()))),'policy':POLICY}


def evaluate_pair(folder,candidate_model,baseline_model,config_path,out,baseline_config_path=None):
    from possession.learned import load_model
    from possession.pipeline import infer
    folder,out=Path(folder),Path(out);split=freeze_split(folder);marker=folder/'final-reference-access.json'
    if marker.exists(): raise ValueError('Fresh final matches already consumed')
    if out.exists() and any(out.iterdir()): raise ValueError('Use a new empty audit output directory')
    out.mkdir(parents=True,exist_ok=True)
    artifacts={'baseline':(Path(baseline_model),Path(baseline_config_path or config_path)),'candidate':(Path(candidate_model),Path(config_path))}
    names=['tools/fresh_benchmark_v6.py','tools/fresh_benchmark.py','tools/independent_benchmark.py',
        'tools/temporal_diagnostics_v6.py','tools/control_diagnostics.py','evaluation/possession_metrics.py']
    sources={p.relative_to(ROOT).as_posix():sha256(p) for p in [*[ROOT/n for n in names],*sorted((ROOT/'possession').glob('*.py'))]}
    protocol={'schema_version':1,'split':SPLIT,'split_sha256':sha256(split),'policy':POLICY,'sources':sources,
        'artifacts':{k:{'model_sha256':sha256(m),'config_sha256':sha256(c)} for k,(m,c) in artifacts.items()},
        'selection_frozen_before_reference_access':True,'no_training_or_retuning':True,
        'reference_order':'All six prediction CSVs saved before explicit WCSF flag or SkillCorner phase scoring. Access marker precedes reading files with embedded ignored labels.',
        'dense_human_had_validation':False,'geometry_tolerance_metres':GEOMETRY_TOLERANCE_METRES}
    write_json(out/'frozen-before-reference-access.json',protocol)
    def unchanged():
        if any(sha256(ROOT/n)!=v for n,v in sources.items()): raise ValueError('Frozen source changed')
        for k,(m,c) in artifacts.items():
            if sha256(m)!=protocol['artifacts'][k]['model_sha256'] or sha256(c)!=protocol['artifacts'][k]['config_sha256']: raise ValueError('Frozen artifacts changed')
    with marker.open('x',encoding='utf-8') as f: json.dump({'protocol_sha256':sha256(out/'frozen-before-reference-access.json'),'audit_directory':str(out.resolve()),'status':'access_started'},f,sort_keys=True)
    predictions={};alignments={};blocks_by_name={};warnings={}
    readers=[('wcsf2023',lambda:read_wcsf(folder)),*[(f'skillcorner{m}',lambda m=m:read_skillcorner(folder,m)) for m in SPLIT['final_silver_matches']]]
    for name,reader in readers:
        print('Inferring frozen pair on '+name,flush=True);unchanged();tracking,blocks,pitch,alignment=reader()
        alignments[name]=alignment;blocks_by_name[name]=blocks;predictions[name]={};warnings[name]={}
        for key,(model,config) in artifacts.items():
            unchanged();labels,*_,warn=infer(tracking,[],blocks,pitch,alignment['fps'],load_config(config),load_model(model))
            predictions[name][key]=labels;warnings[name][key]=warn;write_submission(out/f'{name}-{key}.csv',labels)
    unchanged();scores={};counts={};references={}
    for name,alignment in alignments.items():
        if name=='wcsf2023':
            reference,counts[name]=wcsf_references(folder,alignment)
            references[name]={n:sha256(folder/n) for n in WCSF}
            scores[name]={k:dense_score(reference,v,blocks_by_name[name],alignment['fps']) for k,v in predictions[name].items()}
        else:
            match=int(name.removeprefix('skillcorner'));path=folder/f'{match}_phases_of_play.csv'
            old.check_file(path,SC[match]['phases_of_play.csv']);reference,counts[name]=old.phase_references(path,alignment)
            references[name]={'sha256':sha256(path),'quality':'provider_silver'}
            scores[name]={k:old.score_reference(reference,v,'provider_silver') for k,v in predictions[name].items()}
    unchanged();result={'protocol':protocol,'scores':scores,'gate':decide(scores),'reference_counts':counts,'references':references,
        'prediction_sha256':{p.name:sha256(p) for p in sorted(out.glob('*.csv'))},'warnings':warnings,
        'wcsf_geometry':alignments['wcsf2023'].get('geometry'),'reference_fields_used_for_inference':False,
        'reference_match_read_count':3,'dense_human_had_validation':False}
    write_json(out/'summary.json',result);print(json.dumps(result['gate'],indent=2),flush=True);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('download','evaluate-pair'))
    p.add_argument('--data-dir',type=Path,default=ROOT/'runs/fresh-data-v6');p.add_argument('--model',type=Path)
    p.add_argument('--baseline-model',type=Path);p.add_argument('--config',type=Path,default=ROOT/'configs/default.json')
    p.add_argument('--baseline-config',type=Path);p.add_argument('--out',type=Path,default=ROOT/'runs/fresh-audit-v6')
    a=p.parse_args()
    if a.command=='download': download(a.data_dir)
    else:
        if a.model is None or a.baseline_model is None: p.error('Both frozen model paths are required')
        evaluate_pair(a.data_dir,a.model,a.baseline_model,a.config,a.out,a.baseline_config)

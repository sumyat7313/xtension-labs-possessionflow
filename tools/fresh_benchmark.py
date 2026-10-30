"""One-shot fresh-match audit: provider sparse Metrica + automatic SkillCorner phases.

No data here is used for training. Phase gaps are UNKNOWN, never dead-ball truth.
All three sets of paired predictions are saved before separate reference scoring.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
from urllib.request import urlopen
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from possession.io import Tracking, sha256, write_json, write_submission, xml_root
from possession.config import load_config
from tools.independent_benchmark import verify_file, score_points

SC_COMMIT='4340d274572876239c154c90bc507a9b3250a656'
METRICA_COMMIT='e706dd506b360d69d9d123d5b8026e7294b13996'
SC={
 1874553:{'match.json':(33730,'51ef142ceba0f85700b376faf1fdbc4e4a728629'),
 'phases_of_play.csv':(115978,'a566fbbaf11cf47838cdf36e6d7fe17904b85f59'),
 'tracking_extrapolated.jsonl':(90729279,'ea97f58f8eaad925feaeacc6395ec24860dd80027af8a89450276adebd29d265')},
 1886347:{'match.json':(27541,'10779035f95d26a30adf8330c351e4cc4d16a513'),
 'phases_of_play.csv':(121864,'2a37e6981af6f9b6e1f1878969927cd4234a549d'),
 'tracking_extrapolated.jsonl':(89280839,'3577e2803da95390f8b2f85d47829eb55fbbfee6211c94adc65db0dc78e46b41')},
 1899585:{'match.json':(27810,'42ca159d85dc59476bf5278961ccbda6633a3303'),
 'phases_of_play.csv':(119735,'c8df923b064575d49179301f93cedf8727106951'),
 'tracking_extrapolated.jsonl':(85550410,'90b6289ec2181192d1793ec6240fbbe8225ff0219c7956710c6278c31a661100')}}
METRICA={
 'metadata.xml':(81806,'74b55a80f6aeeb8f8df7fd26e4b3a94e2cc1738d'),
 'tracking.txt':(53443115,'7062b7ceaa44d53345a4625b8f447efc74895b1d'),
 'events.json':(1709486,'992b4cb502f0b0bc1244174f014f3f7ecb6d38ef')}
SPLIT={'schema_version':1,'skillcorner_commit':SC_COMMIT,'metrica_commit':METRICA_COMMIT,
 'adapter_development':[1874553],'final_silver_matches':[1886347,1899585],'final_sparse_metrica':[3],
 'rule':'No training on these matches. Freeze one selected candidate and baseline before final reference use. SkillCorner labels are automatic provider silver; Metrica pass origins are sparse.'}
POLICY={'metrica_accuracy_minimum_delta':0.,'metrica_balanced_recall_minimum_delta':0.,
 'silver_mean_accuracy_minimum_delta':0.,'silver_maximum_match_accuracy_drop':.02,
 'no_dense_human_or_dead_ball_validation':True,
 'rule':'Audit one frozen development-selected experimental candidate (eligible winner or highest-ranked experiment if no eligible winner) versus v1.4. Promotion requires both development gates and this audit. On failure retain v1.4; no runner-up or tuning on these references.'}


def freeze_split(folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    path=folder/'split-before-reference-access.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8-sig'))!=SPLIT: raise ValueError('Fresh split changed')
    else: write_json(path,SPLIT)
    policy=folder/'policy-before-reference-access.json'
    if policy.exists():
        if json.loads(policy.read_text(encoding='utf-8-sig'))!=POLICY: raise ValueError('Fresh audit policy changed')
    else: write_json(policy,POLICY)
    return path


def check_file(path,expected):
    size,digest=expected
    if len(digest)==64:
        if Path(path).stat().st_size!=size or sha256(path)!=digest: raise ValueError('LFS content identity differs')
    else: verify_file(path,size,digest)


def catalog(folder):
    folder=Path(folder); rows=[]
    for match,files in SC.items():
        for suffix,expected in files.items():
            name=f'{match}_{suffix}';host='media.githubusercontent.com/media' if suffix.endswith('jsonl') else 'raw.githubusercontent.com'
            rows.append((folder/name,expected,f'https://{host}/SkillCorner/opendata/{SC_COMMIT}/data/matches/{match}/{name}'))
    for suffix,expected in METRICA.items():
        name=f'Sample_Game_3_{suffix}'
        rows.append((folder/name,expected,f'https://raw.githubusercontent.com/metrica-sports/sample-data/{METRICA_COMMIT}/data/Sample_Game_3/{name}'))
    return rows


def download(folder):
    freeze_split(folder);rows=[]
    for path,expected,url in catalog(folder):
        if not path.exists():
            temporary=path.with_name(path.name+'.partial');received=0
            with urlopen(url,timeout=120) as response,temporary.open('wb') as out:
                while block:=response.read(1024*1024):
                    received+=len(block)
                    if received>expected[0]: raise ValueError('Download exceeded catalog size')
                    out.write(block)
            check_file(temporary,expected);temporary.replace(path)
        check_file(path,expected)
        rows.append({'file':path.name,'bytes':expected[0],'sha256':sha256(path),'source_identity':expected[1],'url':url})
        print('Verified '+path.name,flush=True)
    write_json(Path(folder)/'download-manifest.json',{'files':rows,'reference_files_parsed':False})


def _blocks(periods):
    periods=np.asarray(periods);blocks=[]
    for half in (1,2):
        indices=np.flatnonzero(periods==half)
        if not len(indices) or not np.all(np.diff(indices)==1): raise ValueError('Need contiguous regulation halves')
        blocks.append((half,int(indices[0]),int(indices[-1])))
    if blocks[0][2]>=blocks[1][1]: raise ValueError('Half ordering differs')
    return blocks


def _speed(points,blocks,pitch,fps):
    result=np.full(points.shape[:-1],np.nan);step=max(1,round(.08*fps))
    for _,a,b in blocks:
        if b-a>=2*step:
            result[a+step:b-step+1]=np.linalg.norm((points[a+2*step:b+1]-points[a:b-2*step+1])*np.asarray(pitch),axis=-1)*fps/(2*step)
    return result


def _skillcorner_observations(metadata,rows):
    """Whitelist geometry/detection fields only. Never access possession/phase/state."""
    ids=[p['id'] for p in metadata['players']]
    if len(set(ids))!=len(ids): raise ValueError('Duplicate player IDs')
    side_ids=[metadata['home_team']['id'],metadata['away_team']['id']]
    teams=np.array([side_ids.index(p['team_id']) for p in metadata['players']],dtype=np.int8)
    pitch=(float(metadata['pitch_length']),float(metadata['pitch_width']))
    if not all(45<=v<=130 for v in pitch): raise ValueError('Unexpected pitch dimensions')
    lookup={value:i for i,value in enumerate(ids)};n=len(rows)
    players=np.full((n,len(ids),2),np.nan);ball=np.full((n,2),np.nan);periods=np.zeros(n,dtype=int)
    for i,row in enumerate(rows):
        if row['frame']!=i: raise ValueError('SkillCorner frame IDs must be contiguous from zero')
        if row['period'] not in (None,1,2): raise ValueError('Unexpected period')
        periods[i]=row['period'] or 0
        seen=set()
        for observation in row['player_data']:
            pid=observation['player_id']
            if pid not in lookup or pid in seen: raise ValueError('Unknown/duplicate player observation')
            seen.add(pid)
            if observation.get('is_detected') is True:
                xy=np.asarray([observation['x'],observation['y']],dtype=float)
                if np.isfinite(xy).all(): players[i,lookup[pid]]=xy/np.asarray(pitch)*[1,-1]+.5
        observation=row['ball_data']
        if observation.get('is_detected') is True:
            xy=np.asarray([observation['x'],observation['y']],dtype=float)
            if np.isfinite(xy).all(): ball[i]=xy/np.asarray(pitch)*[1,-1]+.5
    blocks=_blocks(periods)
    return Tracking(players,_speed(players,blocks,pitch,10),ball,_speed(ball,blocks,pitch,10),teams,
        (metadata['home_team']['name'],metadata['away_team']['name'])),blocks,pitch,{
            'frame_ids':np.arange(n),'periods':periods,'team_ids':side_ids,'fps':10,
            'observation_policy':'Detected XY only; extrapolated, possession, phase, ball-state, event and metadata playing-time fields excluded.'}


def read_skillcorner(folder,match):
    folder=Path(folder)
    for suffix in ('match.json','tracking_extrapolated.jsonl'): check_file(folder/f'{match}_{suffix}',SC[match][suffix])
    metadata=json.loads((folder/f'{match}_match.json').read_text(encoding='utf-8'))
    # JSON records contain possession fields, but the whitelisted adapter never accesses them.
    with (folder/f'{match}_tracking_extrapolated.jsonl').open(encoding='utf-8') as f: rows=[json.loads(line) for line in f]
    return _skillcorner_observations(metadata,rows)


def phase_references(path,alignment):
    """Provider silver team phases; gaps, contradictions, period edges remain X."""
    periods=alignment['periods'];refs=np.full(len(periods),'X',dtype='<U1');conflict=np.zeros(len(refs),bool)
    side={int(ident):label for ident,label in zip(alignment['team_ids'],'HA')};count=0;excluded=0
    with Path(path).open(encoding='utf-8-sig',newline='') as handle:
        for row in csv.DictReader(handle):
            a,b,half=int(row['frame_start']),int(row['frame_end']),int(row['period'])
            team=int(row['team_in_possession_id'])
            if team not in side or a<0 or b<=a or half not in (1,2): raise ValueError('Invalid phase reference')
            count+=1
            # The development file has one phase straddling halftime. Exclude the
            # entire invalid interval, rather than moving or relabelling its edge.
            if b>len(refs) or not np.all(periods[a:b]==half): excluded+=1;continue
            label=side[team];old=refs[a:b]
            conflict[a:b]|=(old!='X')&(old!=label)
            refs[a:b]=label
    refs[conflict]='X'
    if excluded/max(count,1)>.01: raise ValueError('More than one percent of phase intervals are misaligned')
    return refs,{'phase_intervals':count,'contradictory_frames_excluded':int(conflict.sum()),
        'period_or_range_misaligned_intervals_excluded':excluded,
        'scored_frames':int(np.sum(refs!='X')),'gap_frames_are_unknown':True,'dead_reference_frames':0}


def _metrica_schema(path):
    root=xml_root(path);score=next(root.iter('Score'))
    team_ids=[score.get('idLocalTeam'),score.get('idVisitingTeam')]
    names={t.get('id'):t.findtext('Name') for t in root.iter('Team')}
    players=list(root.iter('Player'));ids=[p.get('id') for p in players];lookup={p:i for i,p in enumerate(ids)}
    teams=np.array([team_ids.index(p.get('teamId')) for p in players],dtype=np.int8)
    channels={p.get('id'):(p.get('playerId'),p.get('channelId')) for p in root.iter('PlayerChannel')}
    formats=[]
    for spec in root.iter('DataFormatSpecification'):
        groups=spec.findall('SplitRegister')
        slots=[]
        for pair in groups[0].findall('SplitRegister'):
            refs=[channels[c.get('playerChannelId')] for c in pair.findall('PlayerChannelRef')]
            if len(refs)!=2 or refs[0][0]!=refs[1][0] or [r[1] for r in refs]!=['x','y']: raise ValueError('Unexpected EPTS channels')
            slots.append(lookup[refs[0][0]])
        formats.append((int(spec.get('startFrame')),int(spec.get('endFrame')),slots))
    formats.sort();parameters={p.findtext('Name'):p.findtext('Value') for p in root.iter('ProviderParameter')}
    blocks=[(half,int(parameters[prefix+'_start'])-1,int(parameters[prefix+'_end'])-1) for half,prefix in [(1,'first_half'),(2,'second_half')]]
    size=next(root.iter('FieldSize'));pitch=(float(size.findtext('Width')),float(size.findtext('Height')))
    if int(root.findtext('.//FrameRate'))!=25: raise ValueError('Unexpected Metrica frame rate')
    return formats,blocks,pitch,teams,tuple(names[i] for i in team_ids),team_ids


def read_metrica3(folder):
    folder=Path(folder)
    for suffix in ('metadata.xml','tracking.txt'): check_file(folder/f'Sample_Game_3_{suffix}',METRICA[suffix])
    formats,blocks,pitch,teams,names,team_ids=_metrica_schema(folder/'Sample_Game_3_metadata.xml')
    n=blocks[-1][2]+1;players=np.full((n,len(teams),2),np.nan);ball=np.full((n,2),np.nan);periods=np.zeros(n,dtype=int)
    for half,a,b in blocks: periods[a:b+1]=half
    index=0;seen=0
    with (folder/'Sample_Game_3_tracking.txt').open(encoding='utf-8-sig') as f:
        for row,line in enumerate(f,1):
            parts=line.strip().split(':')
            if len(parts)!=3 or int(parts[0])!=row or row>n: raise ValueError('Metrica tracking frame mismatch')
            while index<len(formats) and row>formats[index][1]: index+=1
            if index>=len(formats) or row<formats[index][0]: raise ValueError('Missing EPTS frame specification')
            slots=formats[index][2];xy=np.fromstring(parts[1].replace(';',','),sep=',')
            bxy=np.fromstring(parts[2],sep=',')
            if xy.size!=2*len(slots) or bxy.size!=2 or np.isinf(xy).any() or np.isinf(bxy).any(): raise ValueError('Malformed EPTS coordinates')
            players[row-1,slots]=xy.reshape(-1,2);ball[row-1]=bxy;seen=row
    if seen!=n: raise ValueError('Incomplete Metrica tracking')
    return Tracking(players,_speed(players,blocks,pitch,25),ball,_speed(ball,blocks,pitch,25),teams,names),blocks,pitch,{
        'frame_ids':np.arange(1,n+1),'periods':periods,'team_ids':team_ids,'fps':25,
        'observation_policy':'EPTS XY observations only. Event file is not opened by this adapter.'}


def metrica3_references(path,alignment):
    records=json.loads(Path(path).read_text(encoding='utf-8'))['data']
    lookup={int(value):i for i,value in enumerate(alignment['frame_ids'])}
    sides=dict(zip(alignment['team_ids'],'HA'));refs=np.full(len(lookup),'X',dtype='<U1');conflicts=set();counts={'pass_events':0,'duplicates':0,'conflicting_frames':0,'unmapped_frames':0}
    for event in records:
        if event['type']['id'] not in (1,7,6,16): continue
        counts['pass_events']+=1;frame=event['start']['frame'];team=event['team']['id']
        if type(frame) is not int or team not in sides: raise ValueError('Unexpected Metrica event frame/team schema')
        i=lookup.get(frame)
        if i is None: counts['unmapped_frames']+=1;continue
        if alignment['periods'][i]!=event['period']: raise ValueError('Metrica event and tracking periods differ')
        label=sides[team]
        if refs[i]==label: counts['duplicates']+=1
        elif refs[i]!='X': conflicts.add(i)
        refs[i]=label
    for i in conflicts: refs[i]='X'
    counts['conflicting_frames']=len(conflicts);counts['scored_frames']=int(np.sum(refs!='X'))
    if counts['unmapped_frames'] or not counts['scored_frames']: raise ValueError('Unmapped or absent Metrica pass origins')
    return refs,counts


def score_reference(reference,prediction,quality):
    result=score_points(reference,prediction)
    result['metric']='phase_present_team_accuracy' if quality=='provider_silver' else 'sparse_pass_origin_team_accuracy'
    result['reference_quality']=quality;result['dense_had_truth']=False
    return result


def decide(scores):
    required={'metrica3','skillcorner1886347','skillcorner1899585'}
    if set(scores)!=required: raise ValueError('Fresh audit requires every frozen match')
    deltas={}
    for match,values in scores.items():
        if set(values)!={'baseline','candidate'} or values['baseline']['n']!=values['candidate']['n']: raise ValueError('Paired points differ')
        for value in values.values():
            if value['n']<=0 or value['dense_had_truth'] is not False: raise ValueError('Invalid reference scope')
            for metric in ('accuracy','balanced_team_recall'):
                if isinstance(value[metric],bool) or not np.isfinite(value[metric]) or not 0<=value[metric]<=1: raise ValueError('Invalid audit score')
        deltas[match]={metric:values['candidate'][metric]-values['baseline'][metric] for metric in ('accuracy','balanced_team_recall')}
    human=deltas['metrica3'];silver=[deltas['skillcorner'+str(m)]['accuracy'] for m in SPLIT['final_silver_matches']]
    passed=human['accuracy']>=0 and human['balanced_team_recall']>=0 and np.mean(silver)>=0 and min(silver)>=-.02
    return {'passed':bool(passed),'policy':POLICY,'candidate_minus_baseline':deltas,'silver_mean_accuracy_delta':float(np.mean(silver))}


def evaluate_pair(folder,candidate_model,baseline_model,config_path,out,baseline_config_path=None):
    from possession.learned import load_model
    from possession.pipeline import infer
    folder,out=Path(folder),Path(out);split=freeze_split(folder)
    marker=folder/'final-reference-access.json'
    if marker.exists(): raise ValueError('Fresh final matches already consumed; cannot claim an untouched audit again')
    if out.exists() and any(out.iterdir()): raise ValueError('Use a new empty output directory')
    out.mkdir(parents=True,exist_ok=True)
    artifacts={'baseline':(Path(baseline_model),Path(baseline_config_path or config_path)),'candidate':(Path(candidate_model),Path(config_path))}
    sources={p.relative_to(ROOT).as_posix():sha256(p) for p in [Path(__file__),ROOT/'tools/independent_benchmark.py',*sorted((ROOT/'possession').glob('*.py'))]}
    protocol={'schema_version':1,'split':SPLIT,'split_sha256':sha256(split),'policy':POLICY,
        'artifacts':{k:{'model_sha256':sha256(m),'config_sha256':sha256(c)} for k,(m,c) in artifacts.items()},
        'sources':sources,'selection_frozen_before_reference_access':True,'no_training_or_retuning':True,
        'reference_order':'All six paired prediction CSVs saved before phase/event scoring. Embedded SkillCorner possession fields are parsed as JSON but never accessed by the observation adapter.',
        'dense_human_had_validation':False}
    write_json(out/'frozen-before-reference-access.json',protocol)
    def unchanged():
        if any(sha256(ROOT/name)!=value for name,value in sources.items()): raise ValueError('Frozen inference source changed')
        for key,(m,c) in artifacts.items():
            if sha256(m)!=protocol['artifacts'][key]['model_sha256'] or sha256(c)!=protocol['artifacts'][key]['config_sha256']: raise ValueError('Frozen artifact changed')
    # Exclusively consume before reading any final tracking JSON, whose records contain ignored possession fields.
    with marker.open('x',encoding='utf-8') as f: json.dump({'protocol_sha256':sha256(out/'frozen-before-reference-access.json'),'audit_directory':str(out.resolve()),'status':'access_started'},f,sort_keys=True)
    predictions={};alignments={};warnings={}
    for name,reader in [('metrica3',lambda:read_metrica3(folder)),*[(f'skillcorner{m}',lambda m=m:read_skillcorner(folder,m)) for m in SPLIT['final_silver_matches']]]:
        print('Inferring frozen pair on '+name,flush=True);tracking,blocks,pitch,alignment=reader();alignments[name]=alignment;predictions[name]={};warnings[name]={}
        for key,(model,config) in artifacts.items():
            unchanged();labels,*_,warn=infer(tracking,[],blocks,pitch,alignment['fps'],load_config(config),load_model(model))
            predictions[name][key]=labels;warnings[name][key]=warn;write_submission(out/f'{name}-{key}.csv',labels)
    unchanged();scores={};counts={};references={}
    for name,alignment in alignments.items():
        if name=='metrica3':
            path=folder/'Sample_Game_3_events.json';check_file(path,METRICA['events.json']);reference,counts[name]=metrica3_references(path,alignment);quality='provider_sparse_events'
        else:
            match=int(name.removeprefix('skillcorner'));path=folder/f'{match}_phases_of_play.csv';check_file(path,SC[match]['phases_of_play.csv']);reference,counts[name]=phase_references(path,alignment);quality='provider_silver'
        references[name]={'sha256':sha256(path),'quality':quality}
        scores[name]={key:score_reference(reference,value,quality) for key,value in predictions[name].items()}
    unchanged();decision=decide(scores)
    result={'protocol':protocol,'scores':scores,'gate':decision,'reference_counts':counts,'references':references,
        'prediction_sha256':{p.name:sha256(p) for p in sorted(out.glob('*.csv'))},'warnings':warnings,
        'reference_fields_used_for_inference':False,'reference_read_count':3,'dense_human_had_validation':False}
    write_json(out/'summary.json',result);print(json.dumps(decision,indent=2),flush=True);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('download','evaluate-pair'))
    p.add_argument('--data-dir',type=Path,default=ROOT/'runs/fresh-data-v5');p.add_argument('--model',type=Path)
    p.add_argument('--baseline-model',type=Path);p.add_argument('--config',type=Path,default=ROOT/'configs/default.json')
    p.add_argument('--baseline-config',type=Path);p.add_argument('--out',type=Path,default=ROOT/'runs/fresh-audit-v5')
    a=p.parse_args()
    if a.command=='download': download(a.data_dir)
    else:
        if a.model is None or a.baseline_model is None: p.error('Both frozen model paths are required')
        evaluate_pair(a.data_dir,a.model,a.baseline_model,a.config,a.out,a.baseline_config)

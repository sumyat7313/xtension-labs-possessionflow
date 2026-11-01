"""Pinned improved development events and strict future reviewed-label intake.

Sparse event/reception points never become dense possession or stoppage truth.
"""
from __future__ import annotations
import argparse, hashlib, json, math, re, sys
from pathlib import Path
from urllib.request import urlopen
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from possession.io import sha256,write_json

ELASTIC_COMMIT='e245224aad7450961eca40954ae2bc85e1e6e696'
CATALOG={
 'README.md':(8949,'243a89adbd6fe285a4ef0b17bdb674658bd273bd'),
 'experiments/benchmark.py':(16957,'22226a0dfa0c485b27d1c6c0e0876947308dda63'),
 'sync/config.py':(2205,'767c410350227d0c308be88718e36b9abbbc2123'),
 'sync/utils.py':(12961,'5ee078e9b635054902a1df7fe56cfcbc27cb35a1'),
 'benchmark/gt/J03WMX.parquet':(40537,'d70d1ce21778ab0511c22a055e3cc4e183360e36'),
 'benchmark/gt/J03WN1.parquet':(32285,'6c2ffe8cddc54c22823886a711cf41da49d9cc18'),
 'benchmark/gt/J03WPY.parquet':(37462,'1125bb08c765f0570814adb07fdf01b4622c2b0a')}
MATCHES=('J03WMX','J03WN1','J03WPY')
SOURCE_SHA256={'J03WMX':'3bf246033e8df7028563f8b90ecf338212063b6aed9ac6772741d429feded52b',
 'J03WN1':'20cf64b5169099ed3577c30a6bf23eda316fea6f5927e1061496f5f6dccf7681',
 'J03WPY':'8ba1df509b68f1e2081b520f87f43ad39218436538023de2abeb31287bfc6dad'}
RESTART_TYPES=frozenset(('throw_in','freekick_short','freekick_crossed','goalkick','corner_crossed','corner_short','shot_penalty','shot_freekick'))
EVENT_TYPES=RESTART_TYPES|frozenset(('pass','interception','clearance','tackle','bad_touch','dispossessed','cross','shot','foul','ball_recovery','shot_block','keeper_pick_up','keeper_save','keeper_claim','keeper_punch'))
PLAYER=re.compile(r'^(home|away)_[0-9]+$')
OUT=frozenset(('out_bottom','out_top','out_left','out_right'))
GOAL=frozenset(('goal_left','goal_right'))
EXPOSED=frozenset(('J03WMX','J03WN1','J03WPY','J03WOH','J03WOY','J03WQQ','J03WR9','Metrica1','Metrica2','Metrica3','WCSF2023','skillcorner1874553','skillcorner1886347','skillcorner1899585','skillcorner1925299','skillcorner1927964','skillcorner1953632','skillcorner1959846'))

def check_blob(path,expected):
    data=Path(path).read_bytes();size,blob=expected
    if len(data)!=size or hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()!=blob:
        raise ValueError('Pinned ELASTIC source identity differs: '+Path(path).name)

def download(folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);files={}
    for name,expected in CATALOG.items():
        path=folder/name.replace('/','_');url=f'https://raw.githubusercontent.com/hyunsungkim-ds/elastic/{ELASTIC_COMMIT}/{name}'
        if not path.exists():
            with urlopen(url,timeout=60) as response:data=response.read(expected[0]+1)
            partial=path.with_suffix(path.suffix+'.partial');partial.write_bytes(data);check_blob(partial,expected);partial.replace(path)
        check_blob(path,expected);files[name]={'bytes':expected[0],'git_blob_sha1':expected[1],'sha256':sha256(path),'url':url}
    manifest={'dataset':'ELASTIC corrected Sportec events','commit':ELASTIC_COMMIT,'files':files,
      'matches':list(MATCHES),'role':'exposed-development-only','data_license':'CC-BY-4.0',
      'code_license':'MPL-2.0','labels_parsed_by_download':False,'fresh_holdout':False}
    write_json(folder/'download-manifest.json',manifest);return manifest

def integer(value,name):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,float,np.integer,np.floating)) or not math.isfinite(value) or int(value)!=value:
        raise ValueError('Invalid integer '+name)
    return int(value)

def nullable(value):return value is None or isinstance(value,(float,np.floating)) and math.isnan(value)

def frame_index(frame,period,ranges):
    frame=integer(frame,'frame');period=integer(period,'period')
    if str(period) not in ranges:raise ValueError('Unknown half')
    start,end,offset=[integer(x,'frame range') for x in ranges[str(period)]]
    if start>end or offset<0 or not start<=frame<=end:raise ValueError('Event outside its tracking half')
    return offset+frame-start

def normalize_events(rows,match,ranges):
    """Exact source frame -> cached timeline mapping; no temporal interpolation."""
    if match not in MATCHES:raise ValueError('ELASTIC match is not in the exposed pinned catalog')
    points=[];previous=None
    for event_id,row in enumerate(rows):
        period=integer(row['period_id'],'period');frame=integer(row['frame_id'],'frame')
        index=frame_index(frame,period,ranges);player=row['player_id'];kind=row['spadl_type']
        if not isinstance(player,str) or not PLAYER.fullmatch(player) or kind not in EVENT_TYPES:raise ValueError('Unexpected event/player schema')
        if previous is not None and (period,frame)<previous:raise ValueError('Event order reversed')
        previous=period,frame
        point={'event_id':event_id,'period':period,'frame_id':frame,'index':index,'kind':'restart' if kind in RESTART_TYPES else 'event',
               'event_type':kind,'actor':player,'team':'H' if player.startswith('home_') else 'A'}
        points.append(point)
        receiver,receive=row['receiver_id'],row['receive_frame_id']
        if nullable(receiver)!=nullable(receive):raise ValueError('Receiver and receive frame must be jointly present')
        if not nullable(receive):
            receive=integer(receive,'receive frame');receive_index=frame_index(receive,period,ranges)
            if receive<frame:raise ValueError('Reception precedes event')
            if not isinstance(receiver,str):raise ValueError('Invalid receiver identity')
            role='reception' if PLAYER.fullmatch(receiver) else 'ball_out' if receiver in OUT else 'goal' if receiver in GOAL else None
            if role is None:raise ValueError('Unexpected terminal event')
            points.append({'event_id':event_id,'period':period,'frame_id':receive,'index':receive_index,'kind':role,'event_type':kind,'actor':receiver,
                           'team':('H' if receiver.startswith('home_') else 'A') if role=='reception' else None})
    return {'schema_version':1,'match':match,'commit':ELASTIC_COMMIT,'fps':25,'role':'exposed-development-only',
            'fresh_holdout':False,'dense_had_truth':False,'unlisted_frames':'unknown; not negative events or dead-ball labels',
            'annotation_method':'provider-reported median of three human timestamp annotations',
            'individual_rater_uncertainty_available':False,'points':points}

def compile_elastic(folder,cache,out):
    import pyarrow.parquet as pq  # Optional intake dependency; inference needs no Parquet engine.
    folder,cache,out=Path(folder),Path(cache),Path(out)
    if out.exists():raise ValueError('Use a new intake output directory')
    manifest=download(folder);out.mkdir(parents=True);summary={}
    for match in MATCHES:
        path=folder/f'benchmark_gt_{match}.parquet';meta=cache/match/'metadata.json'
        metadata=json.loads(meta.read_text());ranges=metadata['alignment']['frame_ranges']
        original=pq.read_table(path).to_pylist();accepted=[];quarantine=[];source_ids=[]
        for source_id,row in enumerate(original):
            try:normalize_events([row],match,ranges)
            except (ValueError,KeyError,TypeError) as error:
                quarantine.append({'source_event_id':source_id,'reason':str(error)});continue
            accepted.append(row);source_ids.append(source_id)
        result=normalize_events(accepted,match,ranges)
        for point in result['points']:point['source_event_id']=source_ids[point['event_id']]
        result['quarantined_rows']=quarantine
        result['source_sha256']=sha256(path);result['tracking_cache_metadata_sha256']=sha256(meta)
        result['frame_ranges']=ranges;result['tracking_input_provenance']=metadata['inputs']
        write_json(out/f'{match}-points.json',result)
        summary[match]={'source_rows':len(original),'quarantined_rows':quarantine,'events':sum(p['kind'] in ('event','restart') for p in result['points']),
          'point_counts':{kind:sum(p['kind']==kind for p in result['points']) for kind in ('event','restart','reception','ball_out','goal')},
          'points_sha256':sha256(out/f'{match}-points.json'),'source_sha256':sha256(path)}
    proof={'schema_version':1,'intake_source_sha256':sha256(Path(__file__)),'dataset':manifest,
           'summary':summary,'fresh_stoppage_validation_available':False,'promotion_eligible':False}
    write_json(out/'summary.json',proof);return proof

def validated_points(document,*,match,fps,blocks,n_frames):
    """Validate the full provenance wrapper and every exact half-local point.

    The campaign also freezes file hashes. Schema checks alone cannot establish
    that arbitrary supplied points truly came from the declared source bytes.
    """
    if match not in MATCHES or document.get('match')!=match or document.get('commit')!=ELASTIC_COMMIT:
        raise ValueError('Sparse annotation match/source identity differs')
    if (document.get('schema_version')!=1 or document.get('role')!='exposed-development-only'
        or document.get('fresh_holdout') is not False or document.get('dense_had_truth') is not False
        or document.get('individual_rater_uncertainty_available') is not False
        or document.get('source_sha256')!=SOURCE_SHA256[match]
        or isinstance(fps,bool) or fps!=25 or document.get('fps')!=25):
        raise ValueError('Sparse annotation scope/fps/provenance differs')
    n_frames=integer(n_frames,'n_frames');ranges=document['frame_ranges'];sections={}
    for period,a,b in blocks:
        period,a,b=integer(period,'period'),integer(a,'block'),integer(b,'block')
        if period in sections or not 0<=a<=b<n_frames:raise ValueError('Invalid annotation target blocks')
        sections[period]=(a,b)
    if set(ranges)!={str(k) for k in sections}:raise ValueError('Annotation halves differ')
    for period,(a,b) in sections.items():
        first,last,base=[integer(v,'frame range') for v in ranges[str(period)]]
        if first>last or base!=a or base+last-first!=b:raise ValueError('Annotation frame range differs from block')
    metadata_sha=document.get('tracking_cache_metadata_sha256')
    if not isinstance(metadata_sha,str) or not re.fullmatch('[0-9a-f]{64}',metadata_sha):raise ValueError('Missing tracking identity')
    inputs=document.get('tracking_input_provenance',{})
    if not isinstance(inputs,dict) or not inputs:raise ValueError('Missing tracking provenance')
    for name,value in inputs.items():
        if not name.startswith(match+'_') or not isinstance(value,dict) or not re.fullmatch('[0-9a-f]{64}',value.get('sha256','')) or integer(value.get('bytes'),'input bytes')<=0:
            raise ValueError('Malformed tracking provenance')
    excluded={integer(row['source_event_id'],'quarantined event') for row in document.get('quarantined_rows',[])}
    points=document['points']
    if not isinstance(points,list):raise ValueError('Points must be a list')
    for point in points:
        event_id=integer(point['event_id'],'event id');source_id=integer(point['source_event_id'],'source event id')
        if min(event_id,source_id)<0 or source_id in excluded:raise ValueError('Invalid or quarantined source event')
        index=integer(point['index'],'point index');period=integer(point['period'],'period')
        if index!=frame_index(point['frame_id'],period,ranges) or not 0<=index<n_frames:raise ValueError('Point frame/index mismatch')
        role,actor,kind,team=(point[k] for k in ('kind','actor','event_type','team'))
        if kind not in EVENT_TYPES or not isinstance(actor,str):raise ValueError('Invalid point type')
        if role in ('event','restart','reception'):
            if not PLAYER.fullmatch(actor) or team!=('H' if actor.startswith('home_') else 'A'):raise ValueError('Point actor/team mismatch')
            if role in ('event','restart') and (role=='restart')!=(kind in RESTART_TYPES):raise ValueError('Restart point semantics differ')
        elif role in ('ball_out','goal'):
            if actor not in (OUT if role=='ball_out' else GOAL) or team is not None:raise ValueError('Terminal point semantics differ')
        else:raise ValueError('Unknown point role')
    return points

def reviewed_intervals(document):
    """Validate externally reviewed dense labels; claims of human review stay declarations.

    A downstream frozen match-level evaluation policy must independently assess
    provenance and support. This importer never authorizes model promotion.
    """
    if document.get('schema_version')!=1 or document.get('annotation_origin')!='human':raise ValueError('Reviewed human annotation schema required')
    reviewers=document.get('reviewer_ids',[])
    if not isinstance(reviewers,list) or len(reviewers)<2 or any(not isinstance(v,str) or not v.strip() for v in reviewers) or len(set(reviewers))!=len(reviewers):raise ValueError('Two distinct reviewers required')
    if document.get('reviewers_blinded_to_predictions') is not True or document.get('disagreements_adjudicated') is not True:raise ValueError('Blind review and adjudication declarations required')
    if not isinstance(document.get('source_video_sha256'),str) or not re.fullmatch('[0-9a-f]{64}',document['source_video_sha256']):raise ValueError('Video identity required')
    n=integer(document['n_frames'],'n_frames');fps=document['fps']
    if n<=0 or isinstance(fps,bool) or not isinstance(fps,(int,float)) or not math.isfinite(fps) or fps<=0:raise ValueError('Invalid video timeline')
    label=np.full(n,'X',dtype='<U1');occupied=np.zeros(n,bool);blocks={}
    for period,a,b in document['blocks']:
        period,a,b=integer(period,'period'),integer(a,'block'),integer(b,'block')
        if period in blocks or not 0<=a<=b<n or occupied[a:b+1].any():raise ValueError('Invalid or overlapping half')
        blocks[period]=(a,b);occupied[a:b+1]=True
    assigned=np.zeros(n,bool)
    for row in document['intervals']:
        a,b,period=[integer(row[k],k) for k in ('start','end','period')];uncertain=integer(row['boundary_uncertainty_frames'],'uncertainty')
        if period not in blocks or row['label'] not in ('H','A','D','X') or uncertain<0:raise ValueError('Invalid reviewed interval')
        left,right=blocks[period]
        if not left<=a<=b<=right or assigned[a:b+1].any():raise ValueError('Overlapping or cross-half annotations')
        assigned[a:b+1]=True
        if a+uncertain<=b-uncertain:label[a+uncertain:b-uncertain+1]=row['label']
    return label,{'human_review_declared':True,'human_review_independently_verified':False,
      'match':document['match'],'role':'exposed-development-only' if document['match'] in EXPOSED else 'candidate-holdout-pending-provenance-review',
      'unknown_frames':int((label=='X').sum()),'promotion_eligible':False}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('download-elastic','compile-elastic','validate-reviewed'))
    p.add_argument('--data-dir',type=Path,default=ROOT/'runs/fresh-data-v8/elastic');p.add_argument('--cache',type=Path,default=ROOT/'runs/multimodal-campaign-v4/cache')
    p.add_argument('--out',type=Path,default=ROOT/'runs/elastic-intake-v8');p.add_argument('--annotations',type=Path)
    a=p.parse_args()
    if a.command=='download-elastic':result=download(a.data_dir)
    elif a.command=='compile-elastic':result=compile_elastic(a.data_dir,a.cache,a.out)
    else:
        if a.annotations is None:p.error('--annotations required')
        _,result=reviewed_intervals(json.loads(a.annotations.read_text()))
    print(json.dumps(result,indent=2))

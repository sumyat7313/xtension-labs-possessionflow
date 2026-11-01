"""Positive-only held-out ELASTIC timing checks; sparse points are not dense truth."""
import numpy as np
from evaluation import possession_metrics as pm
from tools.data_intake_v8 import ELASTIC_COMMIT,frame_index,RESTART_TYPES,OUT

def diagnostic(labels,blocks,fps,document):
    labels=np.asarray(labels)
    if labels.ndim!=1 or not np.isin(labels,list('HADX')).all():raise ValueError('Invalid sparse timing predictions')
    if fps!=25 or document.get('fps')!=25 or document.get('commit')!=ELASTIC_COMMIT or document.get('fresh_holdout') is not False or document.get('dense_had_truth') is not False:
        raise ValueError('Unexpected sparse annotation provenance')
    spans={};active=np.zeros(len(labels),bool)
    for half,a,b in blocks:
        if half in spans or not 0<=a<=b<len(labels) or active[a:b+1].any():raise ValueError('Invalid sparse timing blocks')
        spans[half]=(a,b);active[a:b+1]=True
    points={'restart':{},'ball_out':{}}
    for point in document['points']:
        kind=point['kind']
        if kind not in points:continue
        i=frame_index(point['frame_id'],point['period'],document['frame_ranges'])
        if i!=point['index'] or point['period'] not in spans:raise ValueError('Sparse timing alignment differs')
        a,b=spans[point['period']]
        if not a<=i<=b:raise ValueError('Sparse timing point outside half')
        if i==a:continue  # No preceding frame in this half, so no observable edge.
        if kind=='restart':
            if point['event_type'] not in RESTART_TYPES or point['team'] not in ('H','A'):raise ValueError('Invalid restart semantics')
            group=point['period'],point['team']
        else:
            if point['actor'] not in OUT:raise ValueError('Invalid ball-out semantics')
            group=point['period'],None
        points[kind].setdefault(group,set()).add(i)
    result={'reference_scope':'sparse corrected human event timing on exposed development matches',
      'fresh_match':False,'dense_had_truth':False,'unlisted_frames_are_negatives':False}
    for kind,groups in points.items():
        total=hits=0;errors=[]
        for (half,team),indices in groups.items():
            a,b=spans[half];edges=[]
            for i in np.flatnonzero(labels[a+1:b+1]!=labels[a:b])+a+1:
                before,after=labels[i-1],labels[i]
                good=(before=='D' and after==team) if kind=='restart' else (before in ('H','A') and after=='D')
                if good:edges.append((int(i),0,1))
            truth=[(int(i),0,1) for i in sorted(indices)]
            matched,_,_=pm.match_transitions(truth,edges,fps,typed='none')
            total+=len(truth);hits+=len(matched);errors.extend((p[0]-g[0])/fps for g,p in matched)
        result[kind]={'n_reference_points':total,'n_matched':hits,'recall_within_1s':hits/total if total else None,
          'mean_absolute_error_matched_seconds':float(np.abs(errors).mean()) if errors else None,
          'mean_signed_error_matched_seconds':float(np.mean(errors)) if errors else None,
          'precision':None,'f1':None,'tolerance_seconds':1.}
    return result

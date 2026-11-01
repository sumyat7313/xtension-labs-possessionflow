"""Experimental topology-preserving D-boundary relocation; never creates D."""
import json
import numpy as np
from possession.boundary import _blocks
from possession.learned import BASE_NAMES

POLICY={'maximum_shift_seconds':2.,'candidate_step_seconds':.08,'target_radius_seconds':.12,
        'minimum_score_gain':float(np.log(2)),'distance_penalty_per_second':.1,
        'minimum_observed_fraction':.5,'minimum_player_coverage':.4,
        'minimum_live_seconds':.4,'minimum_dead_seconds':.6,
        'segment_slack_fraction':.5,'training_segment_iou':.5,
        'sparse_human_policy':'single aligned same-team restart or ball-out point in legal window of matched D segment; ambiguity excludes group; absence retains dense target'}
RADII=(.2,.6,1.5)
SIGNALS=('dead_probability','ball_log_speed','player_median_speed','outside_support','stationary_support','ball_observed')
FEATURE_NAMES=('dead_probability',)+BASE_NAMES+tuple(f'{s}_{side}_{r:g}s' for r in RADII for s in SIGNALS for side in ('past','future','change'))


def _inputs(labels,p,x,z,blocks,fps):
    labels=np.asarray(labels);p=np.asarray(p,float);x=np.asarray(x,float);z=np.asarray(z,float)
    if labels.ndim!=1 or not np.isin(labels,list('HADX')).all():raise ValueError('Invalid stoppage labels')
    n=len(labels)
    if isinstance(fps,bool) or not np.isfinite(fps) or fps<=0:raise ValueError('Invalid stoppage FPS')
    if p.shape!=(n,3) or x.shape!=(n,98) or z.shape!=(n,58):raise ValueError('Invalid stoppage input shapes')
    if not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(axis=1),1):raise ValueError('Invalid stoppage probabilities')
    if not np.isfinite(x).all() or not np.isfinite(z).all():raise ValueError('Nonfinite stoppage observations')
    sections,active=_blocks(blocks,n);known=[]
    for a,b in sections:
        valid=labels[a:b+1]!='X';edges=np.diff(np.r_[False,valid,False].astype(int))
        known.extend((a+int(i),a+int(j)) for i,j in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)))
    return labels,p,x,z,known


def feature_matrix(labels,p,x,z,blocks,fps):
    labels,p,x,z,known=_inputs(labels,p,x,z,blocks,fps);n=len(labels)
    matrix=np.zeros((n,len(FEATURE_NAMES)));observed=np.zeros(n,bool)
    signal=np.column_stack((p[:,2],z[:,2],z[:,11],z[:,6],z[:,17],z[:,0]))
    for a,b in known:
        count=b-a;idx=np.arange(count);columns=[p[a:b,2:3],z[a:b,:19]];complete=np.ones(count,bool)
        prefix=np.vstack((np.zeros((1,6)),np.cumsum(signal[a:b],axis=0)))
        for seconds in RADII:
            radius=max(1,round(seconds*fps));left=np.maximum(0,idx-radius);right=np.minimum(count,idx+radius)
            nl=idx-left;nr=right-idx
            past=np.divide(prefix[idx]-prefix[left],nl[:,None],out=np.zeros((count,6)),where=nl[:,None]>0)
            future=np.divide(prefix[right]-prefix[idx],nr[:,None],out=np.zeros((count,6)),where=nr[:,None]>0)
            complete&=(nl==radius)&(nr==radius)
            columns.append(np.column_stack([v for j in range(6) for v in (past[:,j],future[:,j],future[:,j]-past[:,j])]))
            if seconds==.6:
                coverage=np.r_[0,np.cumsum(z[a:b,9])]
                player=(coverage[right]-coverage[left])/np.maximum(1,right-left)
                complete&=(past[:,5]>=POLICY['minimum_observed_fraction'])&(future[:,5]>=POLICY['minimum_observed_fraction'])
                complete&=player>=POLICY['minimum_player_coverage']
        matrix[a:b]=np.column_stack(columns);observed[a:b]=complete
    return matrix,observed


def segments(labels,known):
    out=[]
    for a,b in known:
        edges=np.r_[a,np.flatnonzero(labels[a+1:b]!=labels[a:b-1])+a+1,b]
        out.append([(int(i),int(j),str(labels[i])) for i,j in zip(edges[:-1],edges[1:])])
    return out


def proposals(labels,p,x,z,blocks,fps):
    """Return fixed legal candidate edges with disjoint half-slack budgets."""
    labels,p,x,z,known=_inputs(labels,p,x,z,blocks,fps)
    matrix,observed=feature_matrix(labels,p,x,z,blocks,fps);result=[]
    radius=max(1,round(POLICY['maximum_shift_seconds']*fps));step=max(1,round(POLICY['candidate_step_seconds']*fps))
    for span in segments(labels,known):
        for left,right in zip(span[:-1],span[1:]):
            if ('D' in (left[2],right[2])) and left[2]!=right[2]:
                edge=right[0];kind='stop' if right[2]=='D' else 'restart'
                if not observed[edge]:continue
                def slack(segment):
                    floor=POLICY['minimum_dead_seconds'] if segment[2]=='D' else POLICY['minimum_live_seconds']
                    return max(0,segment[1]-segment[0]-max(1,round(floor*fps)))//2
                low=edge-min(radius,slack(left));high=edge+min(radius,slack(right))
                indices=edge+step*np.arange(-((edge-low)//step),(high-edge)//step+1);indices=indices[observed[indices]]
                if len(indices)<2:continue
                dead=right if kind=='stop' else left
                result.append({'kind':kind,'edge':edge,'indices':indices,'X':matrix[indices],
                               'left':left,'right':right,'dead_segment':dead,'known_span':(span[0][0],span[-1][1])})
    return result


def validate_model(model):
    if not isinstance(model,dict) or model.get('kind')!='stoppage_boundary_ranker_v8' or model.get('schema_version')!=1:raise ValueError('Invalid stoppage ranker')
    if model.get('feature_names')!=list(FEATURE_NAMES) or model.get('policy')!=POLICY:raise ValueError('Stoppage schema/policy differs')
    if set(model.get('heads',{}))!={'stop','restart'}:raise ValueError('Both stoppage heads required')
    for head in model['heads'].values():
        for key in ('mean','scale','coefficients'):
            value=np.asarray(head.get(key),float)
            if value.shape!=(len(FEATURE_NAMES),) or not np.isfinite(value).all():raise ValueError('Invalid stoppage '+key)
            if key=='scale' and (value<=0).any():raise ValueError('Invalid stoppage scale')
    json.dumps(model,allow_nan=False);return model


def refine(model,labels,p,x,z,blocks,fps):
    validate_model(model);result=np.asarray(labels).copy()
    for proposal in proposals(labels,p,x,z,blocks,fps):
        head=model['heads'][proposal['kind']];indices=proposal['indices'];edge=proposal['edge']
        matrix=np.clip((proposal['X']-np.asarray(head['mean']))/np.asarray(head['scale']),-8,8)
        scores=matrix@np.asarray(head['coefficients']);original=int(np.flatnonzero(indices==edge)[0])
        gains=scores-scores[original]-POLICY['distance_penalty_per_second']*np.abs(indices-edge)/fps
        chosen=int(np.argmax(gains));target=int(indices[chosen])
        if gains[chosen]<POLICY['minimum_score_gain']:continue
        # Independent half-slack budgets ensure this cannot erase/merge a segment,
        # even if both of its endpoints move towards one another.
        if target<edge:result[target:edge]=proposal['right'][2]
        elif target>edge:result[edge:target]=proposal['left'][2]
    return result

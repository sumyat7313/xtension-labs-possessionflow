"""Training-only boundary ranking; whole-match exclusion belongs to the caller."""
import numpy as np
from research import stoppage_v8 as sv


def _sparse_points(item):
    from tools.data_intake_v8 import ELASTIC_COMMIT,RESTART_TYPES,OUT,integer,frame_index,validated_points
    document=item.get('sparse_points')
    if document is None:return []
    if not isinstance(document,dict):raise ValueError('Full sparse provenance document required')
    validated_points(document,match=item.get('match'),fps=item['fps'],blocks=item['blocks'],n_frames=len(item['labels']))
    if (not isinstance(document,dict) or document.get('schema_version')!=1
        or document.get('commit')!=ELASTIC_COMMIT or not isinstance(item.get('match'),str)
        or not item['match'] or document.get('match')!=item['match']
        or document.get('role')!='exposed-development-only' or document.get('fresh_holdout') is not False
        or document.get('dense_had_truth') is not False or document.get('fps')!=item['fps']
        or document.get('annotation_method')!='provider-reported median of three human timestamp annotations'):
        raise ValueError('Invalid sparse human annotation provenance')
    ranges=document.get('frame_ranges');blocks={str(period):(a,b) for period,a,b in item['blocks']}
    if not isinstance(ranges,dict) or set(ranges)!=set(blocks):raise ValueError('Sparse annotation halves differ')
    for period,values in ranges.items():
        start,end,offset=[integer(v,'annotation frame range') for v in values]
        if (offset,offset+end-start)!=blocks[period]:raise ValueError('Sparse annotation timeline differs')
    points=[];seen=set()
    for point in document['points']:
        index=integer(point['index'],'annotation index');period=integer(point['period'],'annotation period')
        if index!=frame_index(point['frame_id'],period,ranges) or not 0<=index<len(item['labels']):raise ValueError('Sparse annotation frame alignment differs')
        kind=point['kind']
        if kind not in ('restart','ball_out'):continue
        if kind=='restart' and (point['event_type'] not in RESTART_TYPES or point['team'] not in ('H','A')):raise ValueError('Invalid restart semantics')
        if kind=='ball_out' and (point['actor'] not in OUT or point['team'] is not None):raise ValueError('Invalid ball-out semantics')
        key=(kind,index,point['team'],integer(point['source_event_id'],'source event'))
        if key in seen:raise ValueError('Duplicate sparse human boundary point')
        seen.add(key);points.append(point)
    return points


def _matches(labels,y,known):
    predicted=[s for span in sv.segments(labels,known) for s in span if s[2]=='D']
    truth=[s for span in sv.segments(y,known) for s in span if s[2]=='D']
    pairs=[]
    for pi,p in enumerate(predicted):
        for gi,g in enumerate(truth):
            overlap=max(0,min(p[1],g[1])-max(p[0],g[0]));union=max(p[1],g[1])-min(p[0],g[0])
            iou=overlap/union
            if iou>=sv.POLICY['training_segment_iou']:pairs.append((-iou,pi,gi))
    usedp=set();usedg=set();result={}
    for _,pi,gi in sorted(pairs):
        if pi not in usedp and gi not in usedg:result[predicted[pi]]=truth[gi];usedp.add(pi);usedg.add(gi)
    return result


def training_groups(items):
    groups={'stop':[],'restart':[]};counts={'unknown_context_excluded':0,'matched_targets':0,'retention_targets':0,
        'human_stop_targets':0,'human_restart_targets':0,'human_ambiguous_groups_excluded':0,'human_absent_dense_fallback':0}
    for item in items:
        labels,p,x,z,known=sv._inputs(item['labels'],item['p'],item['x'],item['z'],item['blocks'],item['fps'])
        y=np.asarray(item['y']);fps=item['fps']
        if y.shape!=labels.shape or not np.isin(y,list('HADX')).all():raise ValueError('Invalid stoppage training reference')
        points=_sparse_points(item);matched=_matches(labels,y,known);context=max(1,round(max(sv.RADII)*fps))
        for q in sv.proposals(labels,p,x,z,item['blocks'],fps):
            indices=q['indices'];lo=max(q['known_span'][0],int(indices.min())-context);hi=min(q['known_span'][1],int(indices.max())+context+1)
            if np.any(y[lo:hi]=='X'):counts['unknown_context_excluded']+=1;continue
            target=q['edge'];gt=matched.get(q['dead_segment'])
            if gt is not None:
                actual=gt[0] if q['kind']=='stop' else gt[1]
                live=q['left'][2] if q['kind']=='stop' else q['right'][2]
                flank=actual-1 if q['kind']=='stop' else actual
                if lo<=flank<hi and y[flank]==live and indices.min()<=actual<=indices.max():
                    target=actual;counts['matched_targets']+=1
                else:counts['retention_targets']+=1
            else:counts['retention_targets']+=1
            # Sparse points refine timestamp targets only. No interval expansion,
            # new segment, whistle-from-foul label, or missing-event negative.
            wanted='ball_out' if q['kind']=='stop' else 'restart'
            human=[v for v in points if v['kind']==wanted and indices.min()<=v['index']<=indices.max()]
            if gt is not None and len(human)>1:
                counts['human_ambiguous_groups_excluded']+=1;continue
            if gt is not None and len(human)==1 and (wanted=='ball_out' or human[0]['team']==q['right'][2]):
                target=human[0]['index'];counts['human_stop_targets' if q['kind']=='stop' else 'human_restart_targets']+=1
            else:counts['human_absent_dense_fallback']+=1
            distance=np.abs(indices-target)
            desired=distance<=max(1,round(sv.POLICY['target_radius_seconds']*fps)) if target!=q['edge'] else indices==target
            if not desired.any():desired=distance==distance.min()
            weight=desired.astype(float)/desired.sum()
            groups[q['kind']].append((q['X'],weight))
    return groups,counts


def _fit_head(groups,l2):
    from scipy.optimize import minimize
    if not groups:raise ValueError('Stoppage fitting requires stop and restart candidates')
    sizes=np.array([len(x) for x,_ in groups]);starts=np.r_[0,np.cumsum(sizes)[:-1]]
    x=np.concatenate([x for x,_ in groups]);target=np.concatenate([t for _,t in groups]);ng=len(groups)
    sample_weight=np.repeat(1/(ng*sizes),sizes);mean=np.sum(x*sample_weight[:,None],axis=0)
    scale=np.sqrt(np.sum((x-mean)**2*sample_weight[:,None],axis=0));scale[scale<1e-8]=1
    z=np.clip((x-mean)/scale,-8,8)
    def objective(beta):
        score=z@beta;maximum=np.maximum.reduceat(score,starts);exp=np.exp(score-np.repeat(maximum,sizes))
        denom=np.add.reduceat(exp,starts);prob=exp/np.repeat(denom,sizes)
        loss=(np.sum(maximum+np.log(denom))-target@score)/ng+.5*l2*(beta@beta)
        grad=z.T@(prob-target)/ng+l2*beta
        return float(loss),grad
    result=minimize(objective,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':250,'ftol':1e-11,'gtol':1e-7})
    if not result.success or not np.isfinite(result.x).all():raise ValueError('Stoppage ranking optimizer failed: '+str(result.message))
    return {'mean':mean.tolist(),'scale':scale.tolist(),'coefficients':result.x.tolist(),
            'training':{'groups':ng,'rows':len(x),'l2':l2,'iterations':int(result.nit),'objective':float(result.fun)}}


def fit(items,metadata=None,l2=.01):
    if isinstance(l2,bool) or not np.isfinite(l2) or l2!=.01:raise ValueError('Fixed stoppage L2 must be .01')
    if metadata and any(item.get('match') in metadata.get('held_out_matches',[]) for item in items):raise ValueError('Held-out match entered stoppage training')
    groups,counts=training_groups(items)
    model={'kind':'stoppage_boundary_ranker_v8','schema_version':1,'feature_names':list(sv.FEATURE_NAMES),
           'policy':dict(sv.POLICY),'heads':{k:_fit_head(v,l2) for k,v in groups.items()},
           'metadata':metadata or {},'training':dict(counts,reference_scope='dense provider D segments; not physical contacts',
             upstream_layer_training='Caller must exclude the complete outer evaluation fold; upstream scores within training matches may be in-sample.')}
    return sv.validate_model(model)

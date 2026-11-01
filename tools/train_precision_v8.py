"""Fold-only segment-quality supervision; ambiguous reference islands abstain."""
import numpy as np
from research import precision_v8 as precision


def examples(items):
    matrices=[];targets=[];counts={'eligible_islands':0,'safe_merge':0,'preserve':0,'ambiguous_or_unknown':0,'sparse_conflict_excluded':0}
    for item in items:
        labels=np.asarray(item['labels']);y=np.asarray(item['y'])
        if y.shape!=labels.shape or not np.isin(y,list('HADX')).all():raise ValueError('Invalid precision reference')
        points=[]
        if 'sparse_points' in item:
            from tools.data_intake_v8 import validated_points
            if not isinstance(item['sparse_points'],dict):raise ValueError('Full sparse annotation document required')
            points=validated_points(item['sparse_points'],match=item['match'],fps=item['fps'],blocks=item['blocks'],n_frames=len(labels))
        rows,matrix=precision.segments(labels,item['p'],item['x'],item['z'],item['blocks'],item['fps'])
        counts['eligible_islands']+=len(rows)
        for row,feature in zip(rows,matrix):
            a,b=row['start'],row['end'];ref=y[a:b];context=y[row['context_start']:row['context_end']]
            if not np.isin(context,list('HA')).all():counts['ambiguous_or_unknown']+=1;continue
            if np.all(ref==row['replacement']):
                # A human reception is not dense control truth. It can, however,
                # make a supposedly certain all-flank merge label questionable.
                if any(v['kind']=='reception' and v['team']==row['team'] and a<=v['index']<b for v in points):
                    counts['sparse_conflict_excluded']+=1;continue
                target=1;counts['safe_merge']+=1
            elif np.mean(ref==row['team'])>=.5:target=0;counts['preserve']+=1
            else:counts['ambiguous_or_unknown']+=1;continue
            matrices.append(feature);targets.append(target)
    return np.asarray(matrices,float).reshape(-1,len(precision.FEATURE_NAMES)),np.asarray(targets,float),counts


def fit(items,metadata=None,l2=.01):
    from scipy.optimize import minimize
    if isinstance(l2,bool) or not np.isfinite(l2) or l2<=0:raise ValueError('Invalid precision L2')
    x,y,counts=examples(items)
    if len(np.unique(y))!=2:
        width=len(precision.FEATURE_NAMES)
        return precision.validate_model({'schema_version':1,'kind':'short_island_quality_v8',
            'feature_names':list(precision.FEATURE_NAMES),'policy':dict(precision.POLICY),
            'mean':[0.]*width,'scale':[1.]*width,'coefficients':[0.]*width,'intercept':-35.,'metadata':metadata or {},
            'training':dict(counts,rows=len(y),supported=False,reason='Both preserve and safe-merge training classes are required; explicit no-op',
                l2=float(l2),calibrated_probability=False,class_reweighting=False,threshold_search=False)})
    mean=x.mean(axis=0);scale=x.std(axis=0);scale[scale<1e-8]=1
    z=np.column_stack((np.clip((x-mean)/scale,-8,8),np.ones(len(x))))
    def objective(beta):
        eta=z@beta;p=1/(1+np.exp(-np.clip(eta,-35,35)))
        loss=np.mean(np.logaddexp(0,eta)-y*eta)+.5*l2*np.sum(beta[:-1]**2)
        grad=z.T@(p-y)/len(y);grad[:-1]+=l2*beta[:-1]
        return float(loss),grad
    initial=np.zeros(z.shape[1]);initial[-1]=np.log(y.mean()/(1-y.mean()))
    result=minimize(objective,initial,jac=True,method='L-BFGS-B',options={'maxiter':300,'ftol':1e-12,'gtol':1e-8})
    if not result.success or not np.isfinite(result.x).all():raise ValueError('Precision fit failed: '+result.message)
    model={'schema_version':1,'kind':'short_island_quality_v8','feature_names':list(precision.FEATURE_NAMES),
        'policy':dict(precision.POLICY),'mean':mean.tolist(),'scale':scale.tolist(),
        'coefficients':result.x[:-1].tolist(),'intercept':float(result.x[-1]),'metadata':metadata or {},
        'training':dict(counts,rows=len(y),supported=True,l2=float(l2),iterations=int(result.nit),objective=float(result.fun),
            calibrated_probability=False,targets='all island frames equal flank team => merge; >=50% own team => preserve; others excluded',
            sparse_conflicts='safe-merge target excluded if exact validated human reception by island team lies within island',
            class_reweighting=False,threshold_search=False)}
    return precision.validate_model(model)

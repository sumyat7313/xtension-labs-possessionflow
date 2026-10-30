"""Training-only four-class typed-boundary softmax, with prior correction."""
import numpy as np
from possession import typed_boundary as tb
from possession.boundary import _blocks


def targets(labels,blocks,fps):
    """Nearest typed edge within +/-0.12s; no edge crosses X or a half.

    Outputs integer class 0..3 and usability. A single frame cannot be both a
    stop and a restart. Ties belong to the earlier edge deterministically.
    """
    y=np.asarray(labels)
    if y.ndim!=1 or not np.isin(y,list('HADX')).all():raise ValueError('Invalid typed boundary labels')
    if isinstance(fps,bool) or not np.isfinite(fps) or fps<=0:raise ValueError('Invalid FPS')
    sections,active=_blocks(blocks,len(y));usable=active&(y!='X');result=np.zeros(len(y),np.int8)
    distance=np.full(len(y),np.inf);radius=max(1,round(tb.POLICY['target_radius_seconds']*fps))
    for a,b in sections:
        usable[a]=False
        for i in range(a+1,b+1):
            if y[i-1]=='X':usable[i]=False
            if y[i]=='X' or y[i-1]=='X' or y[i]==y[i-1]:continue
            kind=2 if y[i]=='D' else 3 if y[i-1]=='D' else 1
            left,right=i,i
            while left>max(a,i-radius) and y[left-1]!='X':left-=1
            while right<min(b,i+radius) and y[right+1]!='X':right+=1
            ii=np.arange(left,right+1);d=np.abs(ii-i);use=d<distance[ii]
            result[ii[use]]=kind;distance[ii[use]]=d[use]
    result[~usable]=0
    return result,usable


def fit(items,*,contact=False,metadata=None,stride=5,l2=.01):
    """Items contain X=typed features, y=reference, blocks, fps; no H/A weights.

    Every positive neighborhood is retained at native rate. Negative sampling
    starts independently inside each half and preserves that half's exact original
    negative mass. Balanced multiclass fitting is corrected to the empirical prior.
    The caller excludes outer evaluation matches from every training item.
    """
    from scipy.optimize import minimize
    names=tb.feature_names(contact)
    if not items or type(stride) is not int or stride<1 or isinstance(l2,bool) or not np.isfinite(l2) or l2<=0:
        raise ValueError('Invalid typed boundary training settings')
    matrices=[];labels=[];weights=[]
    for item in items:
        x=np.asarray(item['X'],float);y=np.asarray(item['y'])
        if x.shape!=(len(y),len(names)) or not np.isfinite(x).all():raise ValueError('Invalid typed training matrix')
        if 'w' in item:raise ValueError('Typed boundary fit uses frame priors, not possession spell weights')
        target,usable=targets(y,item['blocks'],item['fps']);sections,_=_blocks(item['blocks'],len(y))
        sampled=np.zeros(len(y),bool);item_weight=np.ones(len(y))
        for a,b in sections:
            sampled[a+1:b+1:stride]=True
            negatives=np.flatnonzero(usable[a:b+1]&(target[a:b+1]==0))+a
            if len(negatives):
                picked=negatives[sampled[negatives]]
                if not len(picked):sampled[negatives[0]]=True;picked=negatives[:1]
                item_weight[picked]=len(negatives)/len(picked)
        sampled|=target>0;indices=np.flatnonzero(sampled&usable)
        if len(indices):
            matrices.append(x[indices]);labels.append(target[indices]);weights.append(item_weight[indices])
    if not matrices:raise ValueError('No usable typed training observations')
    x,y,original=np.concatenate(matrices),np.concatenate(labels),np.concatenate(weights)
    masses=np.bincount(y,weights=original,minlength=4)
    if (masses<=0).any():raise ValueError('Typed fitting requires all four classes')
    normalized=original/original.sum();mean=np.sum(x*normalized[:,None],axis=0)
    scale=np.sqrt(np.sum((x-mean)**2*normalized[:,None],axis=0));scale[scale<1e-8]=1
    x=np.column_stack((np.clip((x-mean)/scale,-8,8),np.ones(len(x))))
    weight=original/(4*masses[y]);target=np.eye(4)[y,1:]
    def objective(flat):
        c=flat.reshape(x.shape[1],3);eta=x@c
        logits=np.column_stack((np.zeros(len(x)),eta));maximum=logits.max(axis=1)
        exp=np.exp(logits-maximum[:,None]);denom=exp.sum(axis=1);probability=exp/denom[:,None]
        loss=np.sum(weight*(maximum+np.log(denom)-logits[np.arange(len(y)),y]))+.5*l2*np.sum(c[:-1]**2)
        gradient=x.T@(weight[:,None]*(probability[:,1:]-target));gradient[:-1]+=l2*c[:-1]
        return float(loss),gradient.ravel()
    result=minimize(objective,np.zeros(x.shape[1]*3),jac=True,method='L-BFGS-B',options={'maxiter':300,'ftol':1e-11,'gtol':1e-7})
    if not result.success or not np.isfinite(result.x).all():raise ValueError('Typed boundary optimizer failed: '+result.message)
    c=result.x.reshape(x.shape[1],3);priors=masses/masses.sum()
    model={'schema_version':1,'kind':'typed_boundary_softmax','classes':list(tb.CLASSES),
        'contact_features':contact,'feature_names':list(names),'policy':dict(tb.POLICY),
        'mean':mean.tolist(),'scale':scale.tolist(),'coefficients':c[:-1].tolist(),
        'intercepts':(c[-1]+np.log(priors[1:]/priors[0])).tolist(),'priors':priors.tolist(),
        'metadata':metadata or {},'training':{'sample_stride':stride,'rows':len(y),
            'class_rows':np.bincount(y,minlength=4).tolist(),'weighted_class_mass':masses.tolist(),
            'l2':l2,'objective':float(result.fun),'iterations':int(result.nit),
            'positive_neighborhoods_all_native_frames':True,
            'negative_sampling_weight':'per-half full negative count / sampled negative count; exact class mass preserved',
            'balanced_softmax_prior_corrected':True,'calibration_claimed':False}}
    return tb.validate_model(model)

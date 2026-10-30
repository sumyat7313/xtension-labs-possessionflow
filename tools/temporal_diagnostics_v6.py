"""Half-local stoppage and short-possession diagnostics; never used by inference."""
import numpy as np
from evaluation import possession_metrics as pm
from tools.control_diagnostics import _codes, _fps


def _rates(hits, actual, predicted):
    precision = hits / predicted if predicted else 0.
    recall = hits / actual if actual else 0.
    return {'n_gt': int(actual), 'n_pred': int(predicted), 'n_matched': int(hits),
            'precision': precision, 'recall': recall,
            'f1': 2 * hits / (actual + predicted) if actual + predicted else 0.}


def diagnostic(reference, prediction, fps=25, blocks=None):
    """Raw D segments and typed live/D boundaries, without filling abstentions.

    Segments use the official deterministic one-to-one IoU=.5 matcher. Boundary
    matching is one-to-one within +/-1s, separately inside each known half/run.
    Short H/A recall follows official abstention filling, at IoU=.5. Missing GT
    bins return None; empty classes cannot silently establish fresh accuracy.
    """
    fps = _fps(fps); gt, pred = _codes(reference), _codes(prediction)
    if gt.shape != pred.shape: raise ValueError('Reference/prediction lengths differ')
    if blocks is None: blocks = [(1, 0, len(gt)-1)] if len(gt) else []
    active = np.zeros(len(gt), bool); runs = []
    for _, start, last in blocks:
        if (not isinstance(start, (int, np.integer)) or not isinstance(last, (int, np.integer))
                or not 0 <= start <= last < len(gt) or active[start:last+1].any()):
            raise ValueError('Invalid or overlapping diagnostic blocks')
        active[start:last+1] = True
        runs.extend((start+a, start+b) for a,b in pm.valid_runs(gt[start:last+1]))
    actual = [(a,b,c) for a,b,c in pm.segments(gt,runs) if c == pm.D]
    predicted = [(a,b,c) for a,b,c in pm.segments(pred,runs) if c == pm.D]
    flags = pm.matched_gt_flags(pm.spell_overlaps(actual,predicted),len(actual),.5)
    result = {'dead_segments': dict(_rates(int(flags.sum()),len(actual),len(predicted)), iou=.5)}
    duration = np.array([(b-a)/fps for a,b,_ in actual])
    result['dead_duration_recall'] = {}
    for name, low, high in (('<2s',0,2),('2-5s',2,5),('5-15s',5,15),('>=15s',15,np.inf)):
        mask = (duration>=low)&(duration<high); n = int(mask.sum()); hits = int(flags[mask].sum())
        result['dead_duration_recall'][name] = {'n_gt':n,'n_matched':hits,'recall':hits/n if n else None}
    for name, onset in (('dead_onsets',True),('dead_offsets',False)):
        pairs=[]; total_gt=total_pred=0
        def wanted(t):
            return (t[1] in (pm.H,pm.A) and t[2]==pm.D) if onset else (t[1]==pm.D and t[2] in (pm.H,pm.A))
        for run in runs:
            a=[t for t in pm.transitions(gt,[run]) if wanted(t)]
            b=[t for t in pm.transitions(pred,[run]) if wanted(t)]
            matched,_,_=pm.match_transitions(a,b,fps,typed='none')
            pairs.extend(matched); total_gt+=len(a); total_pred+=len(b)
        errors=np.array([(b[0]-a[0])/fps for a,b in pairs])
        result[name]=dict(_rates(len(pairs),total_gt,total_pred),tolerance_seconds=1.,
            mean_absolute_timing_error_seconds=float(np.abs(errors).mean()) if len(errors) else None,
            mean_signed_timing_error_seconds=float(errors.mean()) if len(errors) else None)
    a=pm.possession_spells(gt,runs); b=pm.possession_spells(pm.fill_abstentions(pred,runs),runs)
    flags=pm.matched_gt_flags(pm.spell_overlaps(a,b),len(a),.5)
    duration=np.array([(stop-start)/fps for start,stop,_ in a]); mask=duration<2
    count=int(mask.sum()); hits=int(flags[mask].sum())
    result['short_possessions']={'n_gt':count,'n_matched':hits,'recall':hits/count if count else None,
        'n_pred_short':sum((stop-start)/fps<2 for start,stop,_ in b),'iou':.5,'duration_exclusive_seconds':2.}
    result['short_duration_recall']={}
    for name,lo,hi in (('<0.4s',0,.4),('0.4-1s',.4,1),('1-2s',1,2)):
        mask=(duration>=lo)&(duration<hi); n=int(mask.sum()); hits=int(flags[mask].sum())
        result['short_duration_recall'][name]={'n_gt':n,'n_matched':hits,'recall':hits/n if n else None}
    return result

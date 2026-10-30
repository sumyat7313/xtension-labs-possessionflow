"""Training wrapper for the fixed, bounded v6 H/A transition experiment."""
from possession.transition import from_boundary
from tools import train_boundary


def fit(items, *, metadata=None, stride=5, l2=.01):
    """Fit on training matches only, using boundary.feature_matrix inputs.

    Each item supplies X, y, blocks, fps and optional weights. The target and
    prior correction retain the v5 boundary semantics; only inference's mapping
    of rare-event evidence and gated short-team exits change.
    """
    classifier = train_boundary.fit(items, metadata=metadata, stride=stride, l2=l2)
    return from_boundary(classifier, metadata)


def diagnostic(reference, prediction, fps=25):
    """One-to-one IoU=.5 precision/recall by true and predicted spell duration.

    A match is assigned globally before stratifying either side by duration;
    matching independently inside each bucket would hide oversized predictions.
    The evaluator's known-run and abstention semantics are preserved.
    """
    import numpy as np
    from evaluation import possession_metrics as pm
    from tools.control_diagnostics import _codes, _fps
    fps = _fps(fps)
    gt, pred = _codes(reference), _codes(prediction)
    if gt.shape != pred.shape: raise ValueError("Reference and prediction lengths differ")
    runs = pm.valid_runs(gt)
    actual = pm.possession_spells(gt, runs)
    estimated = pm.possession_spells(pm.fill_abstentions(pred, runs), runs)
    hits_gt, hits_pred = np.zeros(len(actual), bool), np.zeros(len(estimated), bool)
    for iou, gi, pi in pm.spell_overlaps(actual, estimated):
        if iou < .5: break
        if not hits_gt[gi] and not hits_pred[pi]: hits_gt[gi] = hits_pred[pi] = True
    gd = np.asarray([(b-a)/fps for a,b,_ in actual])
    pd = np.asarray([(b-a)/fps for a,b,_ in estimated])
    bins = {}
    for name, low, high in (("<0.4s",0,.4),("0.4-1s",.4,1),("1-2s",1,2),
                             ("2-5s",2,5),(">=5s",5,np.inf)):
        gs, ps = (gd >= low) & (gd < high), (pd >= low) & (pd < high)
        ng, npred = int(gs.sum()), int(ps.sum())
        mg, mp = int(hits_gt[gs].sum()), int(hits_pred[ps].sum())
        bins[name] = {"n_gt":ng,"n_pred":npred,"n_matched_gt":mg,"n_matched_pred":mp,
                      "recall":mg/ng if ng else None,"precision":mp/npred if npred else None}
    return {"iou":.5,"matching":"global one-to-one before duration stratification", "bins":bins}

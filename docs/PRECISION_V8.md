# V8 short-possession precision research

This optional refinement lives in `research/precision_v8.py`. It does not modify the production v1.5 inference package, model, configuration or submission. A fixed whole-match experiment must establish any gain; implementation tests do not establish football accuracy.

## What went wrong in v7

The completed v7 comparison increased mean short-possession recall from 0.2784 to 0.2899 but reduced mean short-possession precision from 0.3962 to 0.3849. Across its 20 correlated match/observation cells, v1.5 produced 1,665 short predicted spells with 650 matched; typed boundaries produced 1,801 with 681 matched. That is a net increase of 136 predictions and only 31 matched predictions. These are differences of globally matched counts, not a claim that 31 particular new spells were correct. The four observation conditions do not create independent matches.

Discounting local transition costs could create a short H/A island inside an otherwise continuous opposing-team possession. More boundary sensitivity therefore did not automatically give better possession segmentation. V8 targets unsupported islands conservatively, preserving stoppage decisions for a separate experiment.

## Candidate and fixed policy

Eligible patterns are A-H-A or H-A-H, with an interior island shorter than two seconds. Both flanks must supply a complete 0.4-second context inside the same half and the same flanking label. The island and each flank independently require at least 50% directly observed ball frames. Two seconds is the diagnostic short-spell category, not a new minimum possession duration. Duration alone never triggers a merge.

An 18-feature segment classifier combines conditional H/A posterior margin, margin consistency, duration, nearest-team and contact geometry, observation reliability, player coverage and the opposing flanks. All signed geometry and margins are oriented by the island's team, so exchanging H and A leaves the quality features unchanged. These are geometric/model observations, not verified physical touches.

A merge requires a score of at least 0.8 and mean signed nearest-team support no greater than 0.1. The fixed 0.8 threshold expresses a conservative decision policy; it is not a claim of externally calibrated 80% correctness. Proposals are generated once. Where their evidence windows overlap, the highest-scoring proposal wins, with earlier time breaking ties. Accepted proposals do not generate another round of proposals.

The module never changes D or X frames, never extends through a half boundary, and never deletes an endpoint spell. Missing-only evidence cannot authorize a merge. It cannot invent missed short possessions or fix a missed stoppage; those limits are intentional and must appear in the measured results.

## Training and label quality

The classifier uses natural class proportions and L2 0.01, with deterministic L-BFGS and training-only feature normalization. It does not rebalance the classes or search thresholds against the evaluation fold.

For each eligible training island:

- All island reference frames equal the flanking team: a provisional safe-merge target.
- At least half of its reference frames equal its current island team: a preserve target.
- Any unknown/dead reference in the complete local context, or an ambiguous mixture: exclude the example.

New ELASTIC human reception annotations are sparse, on already exposed matches. They are never expanded into dense H/A/D labels. If a provisional safe-merge island contains an exact annotated reception by its island team, the training example is excluded as a label conflict. A reception does not automatically mean sustained control, so it is not turned into a preserve target either. The full annotation document is validated against match identity, source provenance, frame mapping and half boundaries before use. A bare point list is insufficient. This auxiliary evidence enters training only, never inference.

If an outer training fold has no supported examples or lacks either class, the fit explicitly returns a no-op model with `training.supported=false`. It does not loosen evidence requirements to manufacture support.

The caller excludes the entire scored outer match fold from every upstream and refinement fit. Within the outer training set, refinement examples use upstream in-sample predictions; that distribution mismatch is disclosed, not described as cross-fitting. The scored outer fold remains excluded throughout. Only whole-match evaluation can determine whether this quality head generalizes.

## Integration

`tools/train_precision_v8.fit(items, metadata=None, l2=.01)` accepts items containing `labels`, `p`, `x`, `z`, `y`, `blocks` and `fps`. With auxiliary evidence, supply `match` and the full `sparse_points` document. Here `p` has three H/A/D columns, `x` has the frozen 98 control features and `z` has 58 dead-state features.

`research.precision_v8.refine(model, labels, p, x, z, blocks, fps)` returns a new label array. `return_diagnostics=True` also returns eligible/merged counts and accepted intervals. Reference labels and human annotations are not accepted by this API. The combined parent experiment applies precision refinement first, then the independent stoppage-boundary refiner, and evaluates that exact order.

Tests cover team symmetry, missing-observation abstention, complete-context requirements, overlapping proposals, D/X preservation, half isolation, deterministic fitting, unknown-reference exclusion, unsupported-fold no-ops and strict annotation provenance.

```powershell
python -m unittest discover -s tests -p test_precision_v8.py -v
```

The parent campaign reports short recall and precision using global one-to-one IoU matching before duration stratification, alongside possession spell F1 and stoppage metrics. Sparse human events and repeatedly exposed development matches cannot substitute for an adequately labelled fresh full-match test when deciding whether to replace v1.5.

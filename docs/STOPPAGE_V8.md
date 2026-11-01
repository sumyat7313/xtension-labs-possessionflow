# V8 stoppage boundary refinement

This is an isolated research candidate in `research/stoppage_v8.py`. It does not change the deployed v1.5 model or inference package. Whole-match development results and adequate fresh validation are required before deployment; engineering tests alone are not evidence of improved football accuracy.

## Failure addressed

The v7 typed-boundary model increased predicted D segments from 2,154 to 2,181 across the 20 correlated match/observation-condition cells. Mean D segment F1 fell from 0.91631 to 0.91335 and D boundary F1 from 0.79794 to 0.79665. Discounting transition costs could add fragmentation even when a local event score looked useful. The geometric restart audit also failed: no hard contact or video rule is enabled here.

V8 instead learns where an existing stoppage starts or finishes while preserving the complete sequence of H/A/D segments. It cannot invent a new D spell, delete a stoppage, merge two stoppages, change an H/A-only transition, cross an unknown gap, or cross a half. Consequently it cannot recover an entirely missed stoppage; that limitation is explicit.

## Fixed method

Separate stop and restart listwise ranking heads score candidate timestamps around each existing D edge. Seventy-four team-symmetric features combine the original D probability, 19 observed geometry/motion features, and past/future/difference summaries over 0.2, 0.6 and 1.5 seconds. Input player identity, event annotations and reference labels are never inference features.

Candidates lie on a baseline-edge-anchored 0.08-second grid, at most two seconds from the original edge. The original edge is always present. Both past and future 0.6-second windows require at least 50% directly observed ball frames, average player coverage at least 40%, and complete half/known-run context. Missing ball is an abstention condition for relocation, never a new dead-ball label.

Each adjacent segment donates at most half its duration slack above the original 0.4-second live or 0.6-second D minimum, rounded to native frames. Even if both endpoints move inward, their combined budgets cannot erase or shorten the segment below its original minimum. Already shorter truncated segments donate no slack. This local constraint preserves topology without clipping a conflicting result after prediction.

A relocation needs a learned score gain of at least `log(2)` after a fixed movement penalty of 0.1 per second. These scores are ranking evidence, not calibrated physical-event probabilities. Normalization is learned on training groups only; groups are weighted equally. The deterministic SciPy L-BFGS fit uses L2 0.01, maximum 250 iterations, and fails if the optimizer does not converge. Policy is serialized in the model and validated exactly. No thresholds may be changed after campaign scoring.

## Labels and separation

Baseline and reference D segments are matched one-to-one at IoU at least 0.5. An unambiguous matched reference edge inside the legal proposal window supplies a positive timestamp neighborhood of ±0.12 seconds, provided its adjacent live team agrees. Otherwise the supervised target stays at the original edge. Any candidate neighborhood containing unknown reference frames is excluded entirely.

Optional ELASTIC human-corrected event points are supplied **only for outer-training matches** through the full provenance document in `item['sparse_points']`. The trainer checks match identity, pinned source commit, development-only role, native rate, half mapping, source frame ID versus cached timeline index, event semantics and duplicates. It rejects a training item listed in the supplied metadata's held-out matches. The parent campaign separately authenticates annotation-file hashes and upstream folds.

For a matched existing D segment, exactly one human restart in its legal window can replace the dense offset target when the event's team agrees with the following live team. Exactly one `ball_out` terminal point can replace an onset target when its actor explicitly names a pitch exit. These are points, not dense intervals. Multiple same-kind human points make the training group ambiguous and exclude it. Missing points preserve the independently supplied dense target; they are never treated as negative events. Fouls, goals, ordinary passes and reception points do not become whistle or restart labels.

The three annotation files contain 282 restart and 150 ball-out points before this model's observation, matching and ambiguity gates. They belong to already exposed WMX/WN1/WPY matches and do not create a new holdout. The provider describes timestamp medians from three human raters, but individual-rater uncertainty is unavailable. See [label provenance](LABELLED_DATA_V8.md). Model training records `human_stop_targets`, `human_restart_targets`, `human_ambiguous_groups_excluded`, `human_absent_dense_fallback` and other support counts; repeated observation regimes are not independent human annotations.

The upstream models exclude the complete outer evaluation fold. Upstream predictions on outer-training matches may be in-sample for those upstream layers; the parent campaign records this stacking limitation. Sparse human points and dense references are not accepted by `refine()` and cannot reach prediction. The combined experiment applies the separate precision correction first, then this stoppage correction.

## Interface and tests

```python
from tools.train_stoppage_v8 import fit
from research.stoppage_v8 import refine, validate_model
model = fit(items, metadata=fold_metadata)
prediction = refine(model, labels, p, x, z, blocks, fps)
```

Each training item requires `labels`, `p` (n×3), `x` (n×98), `z` (n×58), `y`, `blocks` and `fps`; human-annotated items additionally carry `match` and the full `sparse_points` document. Inference retains `x` for the common experimental interface but never uses team-asymmetric control columns.

The focused suite checks deterministic learning, neutral identity, input nonmutation, half/gap isolation, H/A exchange symmetry, missing-ball abstention, topology/minimum-duration preservation, one-to-one matching, unknown-label exclusion, human target overrides and ambiguity, provenance/alignment/semantic rejection, and held-out training rejection. Run `python -m unittest discover -s tests -p test_stoppage_v8.py -v` with the pinned SciPy training dependency installed.

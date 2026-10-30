# V6 short-possession hypothesis

The v5 development predictions expose two distinct problems: the ordinary 0.4-second minimum duration blocks very short internal possessions, while lowering switch costs everywhere increases fragmentation. The v6 experiment therefore combines a shorter, locally enabled exit with a stricter evidence gate on every cost reduction.

## Evidence from existing development matches

The following counts pool the five scored Sportec matches across four observation conditions. Each match appears four times; these are correlated development observations, not independent fresh games. Matching uses the official one-to-one IoU 0.5 criterion globally before stratifying true and predicted durations. Thus a true short spell may match a somewhat longer prediction, and matched counts need not agree within a duration bucket.

| True duration | V5 joint-state recall | V5 joint-boundary recall | Reference count across conditions |
| --- | ---: | ---: | ---: |
| Under 0.4s | 0.52% | 0.52% | 384 |
| 0.4–1s | 19.78% | 23.26% | 804 |
| 1–2s | 37.89% | 40.69% | 1,536 |
| 2–5s | 57.38% | 58.11% | 2,588 |
| At least 5s | 83.49% | 83.51% | 3,500 |

V5 joint-state emits no sub-0.4-second possessions. Its two matches in that bucket come from longer predictions overlapping short reference spells. The old boundary variant raises 0.4–1-second recall but lowers precision among predictions of that duration from **30.34% to 26.09%**. Simply reducing all switch penalties further is therefore not a justified improvement.

In the joint-state predictions, 373 of the 804 reference 0.4–1-second spells have an opposing-team prediction on more than half their frames; 83 have a dead prediction on more than half their frames. For 1–2-second spells the corresponding counts are 509 and 64. These categories describe frame-majority confusion, not an exhaustive or disjoint causal classification of every unmatched spell.

The held-out J03WN1 clean example illustrates rare-boundary confidence. The fitted boundary classifier's weighted prior is 0.0601. Its probability quartiles at 394 true direct H/A transitions are 0.0982, 0.4247 and 0.8075. Absolute probability alone does not reflect how surprising an observation is relative to that prior. Nevertheless, supporting context is necessary to avoid amplifying false boundaries.

The full local diagnostic is `runs/short-spell-v6-diagnosis/summary.json`; provider observations and prediction runs are excluded from the repository.

## Fixed candidate, chosen before the v6 campaign

The transition model wraps the same training-only boundary-neighborhood classifier as v5. Training targets, class-prior correction, L2 0.01 and stride five retain their earlier definitions. The new inference policy is fixed:

1. Convert classifier probability to log-odds evidence relative to its training prior. Strength is zero at or below prior and saturates at an eightfold odds ratio. This is evidence strength, not an externally calibrated transition hazard.
2. Require complete, half-local 0.2-second past and future contexts with opposite home/away margins of at least 0.2 in magnitude. Current and both context dead probabilities must be at most 0.25.
3. Only at those supported crossings, reduce H/A switch cost by at most 75%. Elsewhere preserve the ordinary cost exactly.
4. Where strength is at least 0.75, permit an early H/A-to-opposite-H/A exit after at least 0.16 seconds, with an additional cost of 1.0. The decoder retains its ordinary 0.4-second minimum elsewhere. Dead exits and restart penalties are unchanged.

The context gate applies to cost reductions as well as early exits. On the J03WN1 example, only 0.444% of frames receive any reduction and 0.431% receive an early-exit flag. These are design diagnostics, not measured candidate accuracy gains. Predictions and full candidate metrics must come from the parent campaign before any promotion decision.

All features are observations or model probabilities; inference accepts no reference labels. H/A exchange leaves costs and flags unchanged. Context never crosses half boundaries or inactive gaps. Unknown and dead labels remain excluded from boundary training; missing ball is never converted directly into a dead label.

## Interfaces and checks

`possession.transition.evidence(model, control_matrix, probabilities, blocks, fps, base_cost)` returns `team_switch_costs` and `early_exit_allowed`. Separate `switch_costs` and `short_exit_allowed` wrappers are available. The model kind is `evidence_gated_team_transitions` with schema version 1, a nested boundary classifier and a validated fixed policy. `tools.train_transition_v6.fit` accepts the same boundary-feature training items as v5.

Nine focused tests cover neutral/subprior evidence, rare-event odds, supported exits, weak/dead context rejection, team symmetry, half isolation, malformed inputs, single-frame blocks and duration diagnostics that expose fragmentation. The parent decoder integration separately tests the 0.16-second floor, extra early-exit penalty and unchanged dead transitions.

This is one predeclared candidate, not a threshold sweep. No fresh references were used in its design. Its development outcome and any fresh audit are recorded by the v6 campaign; this document does not claim that the hypothesis won.

# Clean-candidate component diagnosis: fixed method

This method implements the bounded investigation in [CLEAN_PFF_TRAINING_DIAGNOSIS.md](CLEAN_PFF_TRAINING_DIAGNOSIS.md). The choices below were made from source-code semantics and existing aggregate reports, before this diagnosis accessed reference labels. They define a training-partition diagnosis, not a new performance claim, parameter search, or promotion procedure.

## Scope and controls

Use exactly the original forty PFF training fixtures in `models/pff-protocol.json`. Reject every development or final fixture ID before reading observations, references or predictions. Authenticate the fixed clean candidate `7d79dc50cfc16a7f64c37b65908be90e588d3bfdec35f599071b25180089bf47`, its configuration and lineage, and the existing observation-feature caches. Recompute the clean control and dead probabilities from `x` and `z`; the historical cached v1.5 `base_home` and `base_dead` arrays are not valid clean upstream inputs.

Freeze the following three ablations, their code and configuration before reference access. Save all three predictions for the complete forty-match cohort before opening diagnosis references. Keep raw provider records, reconstructed references, frame predictions and fitted artifacts local. The published output contains aggregate counts, metrics, methods and identities only.

| Ablation | Conditional H:A evidence | Dead emission |
| --- | --- | --- |
| `base` | Clean upstream control, without joint residual | Recomputed frozen dead head |
| `ha_only` | Full clean model's conditional H:A distribution | Exactly the same recomputed frozen dead head |
| `full` | Existing clean joint inference path | Existing clean joint inference path |

All three use the same period blocks, observation-support handling and resolved duration/transition decoder configuration. No tactical tags or provider reference fields are added as inference evidence.

## H:A-only construction and required identities

Let `q` be the full model's H/A/D emission matrix and `d0` the recomputed upstream dead probability. Define `c = q[:, :2] / q[:, :2].sum(axis=1, keepdims=True)`. Construct `ha_only[:, :2] = (1 - d0[:, None]) * c` and assign `ha_only[:, 2] = d0` directly. Require a finite positive live denominator and `0 <= d0 < 1`; fail rather than invent a fallback distribution. The frozen upstream sigmoid never produces exactly one. No additional clipping or renormalization should alter the assigned dead column.

The implementation must check:

- The H:A-only dead column is exactly equal to `d0`.
- Its conditional home share agrees with the full model within absolute tolerance `1e-12` and zero relative tolerance. Conditional team argmax uses the same deterministic tie convention.
- Full-model probabilities and decoded labels reproduce the existing clean inference path. Inputs are not mutated.
- Shapes, finite normalized emissions, period barriers and unsupported regions retain their existing validation.

This removes only the D emission correction. It does **not** imply that H:A-only and base have identical decoded D masks: the joint decoder couples team switches, durations and dead transitions. A change in decoded D can therefore be mediated by H:A evidence even when D emissions are equal. Report emissions and decoded outcomes distinctly.

## Fixed observation strata

Use `possession.learned.FEATURE_NAMES` to locate columns instead of relying on unnamed column numbers. Construct observation strata before applying any reference mask. Apply the same stratum membership to all three ablations, so paired comparisons describe the same frames.

| Grouping | Fixed definition | Interpretation |
| --- | --- | --- |
| Base dead confidence | `[0, .2)`, `[.2, .4)`, `[.4, .6)`, `[.6, .8)`, `[.8, 1]` | Bins of the recomputed **base** dead emission for every ablation. The last bin includes 1. These are diagnostic bins, not selected operating thresholds or a probability-calibration claim. |
| Ball availability | `observed`: `ball_observed == 1`; `interpolated`: unobserved and `ball_reliability > 0`; `unavailable`: unobserved and reliability zero | `ball_observed` is an accepted, cleaned ball observation. A raw visible point rejected as implausible or a spike is not observed here. The frozen interpolation weight is positive, so the second category identifies interpolation support. |
| Valid player-motion count | `0`, `1–5`, `6–11`, `12–17`, `18–22` | Recover the capped count from `player_coverage * 22`, validating its integer consistency. This feature requires valid coordinates **and** finite speed in `[0, 13]` m/s. It is not the raw number of visible players. Coverage is clipped at 22. |
| Missing-ball run duration | `observed`, `(0, .6] s`, `(.6, 2] s`, `(2, 5] s`, `(5, 8] s`, `>8 s` | Every unobserved frame receives the length of its entire contiguous unobserved run divided by FPS. Reset at each period boundary. Interpolated frames remain unobserved for this definition. |
| Missing-run edge censoring | Neither period edge, left edge only, right edge only, both edges | A run touching a period boundary may extend beyond the available period. Its measured duration is censored. Keep this flag distinct from the duration bin. |

The duration cuts are fixed time scales: `.6` and `2` seconds are existing dead-feature smoothing scales; `2` and `5` seconds also occur in missing-ball context decay; `8` seconds is the maximum context-anchor horizon. They are not fitted to error distributions. At 25 Hz the inclusive upper limits correspond to 15, 50, 125 and 200 frames.

`missing_support` is a smoothed fraction of unobserved ball frames. It is **not** a duration and cannot substitute for run-length calculation. Reference X intervals must not change observation-run construction. Construct runs across all observation frames within the period, then restrict scoring to known reference frames. Conversely, reference X intervals must remain barriers when constructing reference possession spells.

Each individual grouping must partition its eligible observations exactly once. Publish its support counts and reconcile summed frame counts with the unstratified totals. Do not add outcome-selected subgroups or retrospectively merge bins to improve a result. Reference-side restart-distance grouping is omitted from this bounded diagnosis; it cannot enter inference features.

## Per-fixture error accounting

Use the same active-period, known-reference H/A/D mask for all ablations. For each fixture and fixed stratum, retain counts rather than only rounded rates:

- True D: reference D and prediction D.
- False D: reference H/A and prediction D.
- Missed D: reference D and prediction other than D, including predicted X.
- Known live/D support and predicted-X counts.
- D precision from true D divided by all predicted D on known references; D recall from true D divided by reference D.

For team control, separate three quantities. **Conditional-emission team accuracy** is the H/A probability argmax accuracy on all known live frames, irrespective of decoded D. **Unconditional decoded live accuracy** is the correct H/A decoded label divided by all known live frames; D and X predictions are errors. If conditional decoded accuracy among H/A predictions is also shown, report its live-prediction coverage and denominator alongside it. Otherwise, abstaining on hard frames can misleadingly raise the reported accuracy.

Publish all forty per-fixture records and paired changes for `ha_only` versus `base` and `full` versus `ha_only`. For each contrast, report newly introduced false-D frames, corrected false-D frames, recovered true-D frames and lost true-D frames. Report both pooled frame counts/rates and equal-fixture means with clear labels; these are different aggregations. Empty diagnostic denominators should be explicit rather than presented as evidence of perfect performance. Preserve existing evaluator conventions unchanged in the official-style metric fields.

## Short-possession diagnostics

Retain the existing evaluator and duration diagnostic: team-typed, one-to-one greedy IoU matching at `0.5`, with the existing `<0.4 s`, `0.4–1 s`, and `1–2 s` bins. A short spell has duration **strictly less than two seconds**; a two-second spell is excluded. Keep the existing prediction-abstention filling convention for these evaluator-derived metrics and report its use. Do not invent a new matching threshold or redefine precision to improve the apparent result.

For each short reference spell, add complementary counts on **raw decoded labels before abstention filling**:

- False-D frames within short H/A spells, and the total short-spell frame support.
- Short spells with at least one decoded D frame, divided by all supported short spells.
- Short spells completely erased by D: every frame in the spell is decoded D.

Complete D erasure is narrower than failure to recover the correct team; an all-X or opposite-team spell must not count as complete D erasure. If correct-team absence is reported separately, name and count it separately. Per-frame false-D fraction and per-spell interruption/erasure rates answer different questions and must retain their respective denominators.

Reference spells must never bridge a period boundary, D interval or unknown-reference X gap. Spells touching a period edge or the edge of a known-reference run are boundary-censored: a short observed fragment may belong to a longer true possession. Preserve all evaluator-included spells for comparability, report the censored support separately, and clearly distinguish any fully observed subset from the main metric. Censoring is reference-side reporting only; it must not modify inference or the common reference mask.

Synthetic tests should cover exact two-second exclusion, each duration boundary, an X gap, adjacent period blocks, all-D erasure, partial-D interruption, all-X prediction, opposite-team prediction and zero short-spell support. Counts must remain nonnegative and reconcile with the reference spell/frame support.

## Interpretation and next decision

This diagnosis can attribute errors to fixed components on the training partition. It cannot establish generalization, causality of missing observations, or a winning competition score. PFF event-derived labels remain silver references; their timing and dense validity limitations remain material.

No fit, threshold sweep or normalization update occurs here. The twelve exposed development matches and twelve reserved final matches remain outside this task. Do not lower the existing development gates, change masks, or reuse the final cohort to choose a model. A later adaptation family must be separately predeclared with at most two candidates and its own external-training-only selection procedure. PFF-trained artifacts remain local while delivery rights are unresolved. No automatic promotion or submission follows from the diagnosis.

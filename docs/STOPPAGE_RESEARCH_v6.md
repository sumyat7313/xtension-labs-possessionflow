# V6 stoppage loss ablation

The v5 joint model already detects most long stoppages with complete tracking. The
remaining development errors concentrate around missing-ball intervals and restart
boundaries. This experiment reallocates training attention to those boundaries
and short stopped spells while preserving the existing model architecture.

## Evidence used to choose the experiment

Only the existing v5 out-of-fold joint-state predictions and the previously exposed
Sportec development references were examined. Across the five scored matches,
there are 318,020 reference D frames (12,720.8 seconds), including 498 stopped
intervals. Only three D intervals last under two seconds; this sample cannot
support a strong claim about very short stoppages.

| Observation condition | False D frames on known live references | Missed D frames |
| --- | ---: | ---: |
| clean | 4,176 | 4,514 |
| degraded | 15,315 | 11,338 |
| camera | 7,410 | 6,169 |
| calibrated camera | 21,867 | 13,044 |

An exploratory best-overlap interval diagnostic found much larger onset/offset
errors in the degraded and calibrated-camera conditions. The local artifact
`runs/stoppage-v6-diagnostics/baseline-exploration.json` lists examples. Its
best-overlap interval counts are descriptive only; the v6 campaign's independent
one-to-one interval and typed boundary metrics determine the result.

These four conditions are correlated views of five matches, not twenty fresh
matches. No new-match reference, competition label, or target-match outcome
informed this proposal.

## Fixed training change

`tools/train_stoppage_v6.fit` accepts the same per-half 5 Hz sequence dictionaries
as `tools/train_joint.fit`: `X`, `Z`, `y`, `w`, `base_home`, `base_dead`. It returns
the unchanged `joint_model` JSON schema, so inference still uses the existing
NumPy implementation without a second head or an inference-time label input.

The architecture, seed 10527, eight CPU epochs, optimizer, original feature
normalization, and conditional H/A binary-cross-entropy term remain unchanged.
Only the D binary-cross-entropy term receives separate training weights:

1. Begin with the v5 frame weights supplied by the campaign.
2. For each training D spell of duration d, multiply by sqrt(5 seconds/d).
3. Multiply both sides of a known live/D transition within one second by two.
   Neighborhoods are half-open and symmetric around the transition; overlapping
   neighborhoods never multiply repeatedly.
4. Normalize separately within the D and live classes of each half. A fixed
   80-step monotone solve preserves each class's original total weighted mass
   while bounding the final multiplier to [0.5, 3].
5. X and zero-weight observations split the valid runs. No transition or D spell
   is inferred across such a gap. Their final loss weights are zero.

The objective is `d_weight * BCE(dead) + w * live * BCE(home_given_live)`.
The H/A loss and feature normalization use original w. This reduces the risk of
improving D recall simply by shifting the aggregate D/live training balance.
Shared hidden features can still affect both tasks; only whole-match evaluation
can establish whether that tradeoff helps.

## Validation and limits

Nine targeted tests cover bounded weights, preservation of weighted class masses,
short-stop emphasis, time reversal and team symmetry, unknown and zero-weight
barriers, noncompounding neighborhoods, invalid inputs, deterministic training,
and exact reproduction of v5 training in the constant-weight limit. Two training
tests require optional PyTorch; production inference needs NumPy only.

This file describes the preregistered candidate, not a claimed improvement.
The parent campaign compares it with the exact historical v5 out-of-fold model
under fixed gates before any new-match audit. A failed candidate must not replace
the current release. Results and any promotion decision belong to that campaign.
The unchanged 5 Hz representation still has a 0.2-second sampling interval and
cannot by itself recover unobservable events or validate unlabelled stoppages.

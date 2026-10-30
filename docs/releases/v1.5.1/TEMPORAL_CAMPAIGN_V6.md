# V6 stoppage boundaries and short possessions

Retained v1.5: experimental combined did not pass both fixed gates (development=False, fresh=False).

Development gate: **fail**. Fresh audit gate: **fail**. Deployed model: **v1.5 joint-state**.

## What changed

The stoppage-loss ablation keeps the v5 joint network and eight deterministic CPU epochs. It reallocates D-loss weight toward short stops and +/-1-second live/dead boundaries, preserves each class's weighted mass, and leaves conditional H/A loss weights unchanged.

The transition ablation converts rare-boundary probabilities into evidence relative to their training prior. Only confident, locally supported H/A reversals receive a switch-cost discount. Early H/A exits require at least 0.16 seconds (rounded up to whole frames), strong live reversal evidence and an additional penalty. D entry/exit constraints stay fixed. These are offline half-local context features, not a causal streaming model or calibrated probabilities.

The combined candidate uses both changes. No fresh references or challenge labels enter training.

## Fixed development comparison

Four procedures each scored in 20 cells: five evaluation matches, four observation conditions. Each new fit excludes its complete outer evaluation fold. The v1.5 comparator uses original out-of-fold models; all historical prediction bytes reproduced. WOH/WOY remain training-only for the historical D classifier. No all-seven final fit is scored as a held-out comparator.

| Candidate | Spell F1 (primary) | Short recall | Short precision | D segment F1 | D boundary F1 | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| combined | 0.6216 | 0.2970 | 0.3978 | 0.9031 | 0.7947 | fail |
| transition | 0.6203 | 0.2912 | 0.3810 | 0.9163 | 0.7979 | fail |
| stoppage-loss | 0.6239 | 0.2877 | 0.4163 | 0.9031 | 0.7950 | fail |
| baseline-v1.5 | 0.6233 | 0.2784 | 0.3962 | 0.9163 | 0.7979 | incumbent |


Primary spell F1 uses clean/degraded/camera conditions; other table metrics average all four, including calibrated-camera corruption. D boundary F1 averages separate one-to-one onset/offset F1 within +/-1 second. Short recall is official IoU .5 recall for true <2-second spells; short precision matches predicted <2-second spells globally before duration stratification.

Promotion requires nondecreasing primary spell F1 and short recall, at least .01 improvement in short recall or D boundary F1, and fixed regression limits for per-cell spell F1/D recall, mean D precision/segment/boundary F1, macro F1 and short precision. Eligible candidates rank by the equally weighted mean of primary spell F1, short recall and D boundary F1, then primary spell F1 and name. Full numeric rules are embedded in the protocol; none changed after scoring.

Frozen audit candidate: **combined**. Only this candidate was tested on fresh references, even if development rejected it. No runner-up was tried after the audit.

## New dense provider audit

The WCSF sample contains **57,264** scored frames, **104** D segments and **120** true short possessions. It supplies explicit dense provider references at 10 Hz, enabling stoppage and short-possession evaluation that the previous sparse audit could not support.

| Dense provider metric | v1.5 | Frozen experiment | Change |
| --- | ---: | ---: | ---: |
| Possession spell F1 | 0.7091 | 0.7222 | +0.0132 |
| Frame macro F1 | 0.9428 | 0.9445 | +0.0017 |
| D precision | 0.9889 | 0.9884 | -0.0005 |
| D recall | 0.9693 | 0.9760 | +0.0068 |
| D segment F1, IoU .5 | 0.9032 | 0.8909 | -0.0123 |
| D onset F1, +/-1s | 0.7487 | 0.7701 | +0.0214 |
| D offset F1, +/-1s | 0.8272 | 0.8677 | +0.0405 |
| Short H/A recall, <2s | 0.4750 | 0.4917 | +0.0167 |


| Fresh phase match | Scored frames | v1.5 accuracy | Experiment accuracy |
| --- | ---: | ---: | ---: |
| skillcorner1925299 | 35,640 | 84.26% | 82.82% |
| skillcorner1927964 | 37,111 | 87.21% | 85.49% |


All six prediction CSVs were saved before reference scoring. Observation adapters whitelist geometry, exclude state/possession fields, derive velocities within halves, and authenticate source bytes. The WCSF coordinate convention was chosen by a fixed observation-only duplicated-ball consistency test, never by label agreement. See [fresh provenance](FRESH_DATA_V6.md).

All seven Sportec matches are previously exposed development data. WCSF is an unused, anonymized provider sample with explicit AliveFlag/PossessionFlag references; annotation method and cross-provider fixture independence cannot be verified. Its two halves count as one match. SkillCorner phases are automatic labels and may bridge short disruptions; missing phases remain unknown, never D. The three audited sources are now consumed and cannot be reused as untouched final tests. No human-gold or hidden competition score is claimed.

## Verification and reproduction

The combined experiment improved dense-provider boundary timing and short recall,
but failed the D-segment gate and reduced agreement on both new SkillCorner
matches. In those matches its predicted-D fraction at H/A phase-reference points
rose by roughly 1.8 percentage points. Provider phases may bridge disruptions,
but this does not establish that the new predictions are correct; synchronized
independent annotation is needed. The original fixed gates remain in force.
See [next experiments](NEXT_EXPERIMENTS.md) for the resulting research priorities.

290 engineering tests pass locally, including decoder exhaustive-path optimality, production dispatcher integration, gap/half isolation, deterministic training and fail-closed release evidence. Complete clean-source verification is recorded separately in [verification.json](verification.json).

Inference uses the packaged frozen model with NumPy on CPU. Optional training requires requirements-training.txt and the original, checksum-verified v4/v5 local caches/fold artifacts. Raw provider data and caches are excluded. See SOLUTION.md and the two training design notes for commands and constraints.

Protocol SHA-256: `c5d150203d71056973074be7dd9e60db2b9fbf694011aeb38fd9f41b91e78739`.

Audited model SHA-256: `b0ac0d9f9b1bc42026eb4074544e27ba44500c188ff7210b55d7637a9809b052`.

Deployed model SHA-256: `3af2dea11261e0454c54ce6791f3c5f0f55f488962da91579addb32a9aea24f0`.

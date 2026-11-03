# Initial Challenge clarification and release hold

Recorded 2026-11-04 after the user supplied the organiser's overriding clarification. This is a technical provenance review, not an organiser ruling. It supersedes earlier statements that the retained package is ready for submission. Historical experiment reports remain intact.

**Current decision: preserve v1.5 unchanged as a historical comparator; hold it and its PFF descendants from submission.** No replacement has been promoted. Pause the proposed combined-correction study until an external-only training lineage is established. The final twelve PFF fixtures remain unopened for preparation, prediction and scoring.

## Controlling clarification supplied by the user

The organiser prohibits checking output against the provided video, manually created data on the provided game, uploading provided data to AI services, and training/fitting/calibrating on the ground-truth possession data or provided video. It allows automatic checks and inference using the supplied automatic data/video, subject to those restrictions. External datasets require appropriate rights.

The concluding instruction states: “If you need to train (or retrain), you MUST use only external data.” It also requires algorithm parameters to be tuned on external data. Consequently, absence of target possession labels alone does not establish an acceptable fitting lineage.

## Findings from code and existing reports

1. **Target-derived camera fitting enters v1.5 training.** `docs/MULTIMODAL_CAMPAIGN.md` describes eight field-of-view footprints and duration/player-count statistics fitted from unlabeled challenge observations. `tools/calibrated_camera.py:91` implements `fit_profile`; `tools/train_multimodal.py:223` uses the fitted profile to generate augmented external training observations. `tools/train_joint_campaign.py:19` includes the calibrated regime and `:92` fits the joint model on it. Its final refit also inherits the previous model. Removing a profile file at inference would not remove its influence from fitted weights.
2. **External possession supervision is documented, but is insufficient to clear the above dependency.** The v1.5 joint fit uses seven external Sportec fixtures. The audited paths do not establish challenge ground-truth fitting. This is not a certification that every historical action complied.
3. **PFF descendants inherit the concern.** `docs/PFF_RESULTS.md` records that upstream heads and decoder remain v1.5. Fitting a new PFF correction does not independently remove the earlier target-derived training influence.
4. **Video pilots require separate historical review.** `docs/VIDEO_RECOVERY.md` records local challenge-video experiments and confidence calibration. Video recovery is inactive in submission inference. Inactivity does not retrospectively establish compliance of the experiments; no further target-video calibration or output/video comparison should be run.
5. **Historical AI exposure is unverified.** Tracked-path inspection found a supplied public possession-reference CSV in `evaluation/gt/`; its contents were not opened for this review. File presence alone does not establish fitting or AI-service upload. A code/docs audit cannot certify past tool-context exposure. Do not send challenge inputs, labels, video, crops or frame-level records into AI tools. Keep any authorized automatic processing local and report only necessary aggregate check results.
6. **PFF submission rights remain unresolved.** The earlier bounded analytical-use interpretation is not a dataset-specific permission to submit fitted weights. See `PFF_TERMS_AND_LABELS.md`. The clarification does not supply missing provider rights.

## Verified research status

All scores below are means over the twelve already-used PFF development fixtures, against event-derived provider references. They are not hidden challenge scores or untouched-final results.

| Metric | v1.5 | Full context correction |
| --- | ---: | ---: |
| Possession spell F1 | 0.435538 | 0.534267 |
| Short-possession precision | 0.085490 | 0.143209 |
| Short-possession recall | 0.106151 | 0.136102 |
| Stoppage boundary F1 | 0.339145 | 0.553321 |
| Stoppage precision | 0.933314 | 0.933632 |
| Stoppage recall | 0.890216 | 0.961832 |

Eight global penalties and six context blends were evaluated. None passed every development gate. Full context passed the original numerical gates and all seven 90%-gain-retention checks; it failed only the additional stoppage-precision margin. It improved precision by 0.03184 percentage points, below the required 0.2-point gain. The development precision threshold is 93.53138%; achieved precision is 93.36322%.

The completed suite recorded **580 passing tests**. These establish engineering checks, not predictive accuracy or rule compliance. The combined study has preliminary model/validation code but no runner, frozen experiment or results. New research work is local and uncommitted at this checkpoint.

Directory/access-marker inspection and aggregate evidence confirm no final preparation/prediction/scoring artifacts; all twelve final fixtures remain reserved. Their raw records and labels were not opened during this review.

The model and CSV hashes were rechecked locally and match the preserved release:

- Model: `3af2dea11261e0454c54ce6791f3c5f0f55f488962da91579addb32a9aea24f0`.
- CSV: `1ddbe062fb0ff0bbb2f47fccda472953fee399629b3b88af9b23554b5f7a3ceb`.

## Required next result

Rebuild all affected upstream and joint components using only appropriately licensed external observations, labels and parameter-selection data. An earlier synthetic-camera approach may provide a starting point, but must itself pass lineage review. Do not reuse target-profile-derived training caches or fitted descendants. Keep the target game for permitted frozen inference and automatic structural checks only.

Preserve existing acceptance criteria: at least a 0.2-point stoppage-precision gain over the historical comparator, at least 90% of the original PFF candidate's gains in the other seven metrics, and all original safeguards. Do not lower gates because of this rebuild. Freeze the eligible external-development winner before any final-fixture access; run the final comparison once. Resolve reference validity, dataset rights and the new candidate's provenance before any submission replacement. A compliant fallback, if needed, is a separate eligibility decision and must not be described as a performance improvement without evidence.

No organiser message, submission, history rewrite or new GitHub push was performed as part of this review.

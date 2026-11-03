# PFF dataset access and fresh-match plan

Prepared 2026-11-01; updated 2026-11-03 for Xtension Labs. Status: **all 261 files covering 64 fixtures downloaded; the 40/12/12 fixture split is frozen; the isolated training/evaluation campaign is complete; v1.5 is retained**. The user supplied the contact details for the access request; they are not stored in this repository. The active v1.5 model and submission remain unchanged. The [PFF results report](PFF_RESULTS.md) records the completed 12-match comparison: substantial aggregate gains, but a stoppage-precision regression fails the fixed development gate. The final 12 matches remain untouched.

## Access result

The form displayed **You're Almost There** and an immediate Google Drive access link. The supplied folder opened without an additional access request. Its complete 261-file release is now stored locally: per-game event, metadata, roster and compressed tracking files for 64 fixtures, plus the specifications and supporting files. Provider assets and delivery details remain outside git.

The [terms and label review](PFF_TERMS_AND_LABELS.md) distinguishes the release's analytical invitation from unverified rights to redistribute data or fitted artifacts. No separate dataset agreement was presented or accepted during the observed form flow. Local research proceeds within the documented analytical-use interpretation; judge delivery and broader release permissions remain unresolved. The prepared support enquiry remains unsent.

The delivered specifications and the preassigned training pilot were inspected. Event-derived smoothed ball locations are excluded from inference. The adapter uses raw visible observations and reconstructs provider-silver reference states separately, preserving unknown intervals. These labels are not independently reviewed dense human truth. See [the training and adapter design](PFF_TRAINING.md) and [the frozen validation protocol](PFF_VALIDATION.md) for the actual safeguards.

## Verified access route

Open the [official World Cup dataset page](https://www.blog.fc.pff.com/blog/pff-fc-release-2022-world-cup-data) and use the embedded **Get Access to Our 2022 World Cup Dataset** form. Its live fields were checked in the browser on the date above:

| Form field | Prepared value or information needed |
| --- | --- |
| First Name | Provided by user and submitted; omitted from git |
| Last Name | Provided by user and submitted; omitted from git |
| Email | Provided by user and submitted; omitted from git |
| Organization | Xtension Labs |

The dataset form's submit button reads **SIGN ME UP!**. Separate PFF/Gradient newsletter popups also appear; these are distinct from the dataset form. No project statement, phone, payment details or password was requested in the observed dataset form. Submission supplied the working delivery link directly. This access flow did not establish a dataset-specific licence grant. Do not enter personal contact details into this repository.

The official announcement describes 64 games of tracking and events, but it does not establish dense human-reviewed possession/stoppage truth. Access does not itself establish permission for every proposed use. The page lists **fchelp@pff.com** as support. A ready-to-send coverage/use-rights enquiry is in [PFF_ACCESS_REQUEST.txt](PFF_ACCESS_REQUEST.txt); it has not been sent.

## Application text, if requested later

Xtension Labs is developing PossessionFlow for the More Than A Hack 2027 football analytics challenge. We estimate home possession, away possession, stoppages and unknown periods from observed player and ball tracking. We request access to the 2022 World Cup release to train and independently evaluate possession boundaries and stoppage detection on complete unseen fixtures. Our development repository is private. We will keep raw data and restricted references local and share only outputs permitted by the supplied terms. We would appreciate confirmation of permitted hackathon/model-training use, frame coverage, annotation provenance and any available synchronized video for independent review.

## Intake completed and remaining evidence

1. **Delivery:** all release files were downloaded through the supplied route into ignored local storage. Original provider files, documentation and access details stay local. No raw data, reconstructed labels or fitted PFF weights are cleared for repository publication by the access flow alone.
2. **Schema and observations:** the delivered per-game JSON layout, centred-metre coordinates, native clock and duplicate-frame behavior were checked using documentation and training-only pilot 3812. The adapter uses raw `VISIBLE` players and balls, excludes event-derived smoothing and estimated positions, preserves missing observations, and aligns a fixed 25-Hz timeline with period separators. Exact pilot comparisons verified unchanged baseline outputs after a memory-only storage optimization.
3. **Fixture identities:** the metadata audit excludes overlap with previously consumed Sportec, Metrica, WCSF, SkillCorner and challenge fixtures. WCSF's exact identity remains anonymized, but its documented European-league competition class excludes this national-team World Cup release. The [validation report](PFF_VALIDATION.md) records the evidence and limits.
4. **References:** the current event-file annotations are read separately from observations. Explicit control and restart events carry team ownership; OUT/END begin stoppages. Missing-video and initial-non-event annotations remain unknown. This produces a bounded provider-silver study, not independent human adjudication of every possession or whistle boundary.
5. **Remaining conditions:** establish dataset-specific judge-delivery/reporting rights and adequate independent dense reference quality before promotion. If needed, use permitted synchronized video and two blinded reviewers under the [reviewed-label workflow](LABELLED_DATA_V8.md#build-reviewed-labels-from-permitted-video). No independent reviewer work or separately authorized video review has been completed. The support enquiry remains available for these unresolved points.

## Frozen split and promotion protocol

The [immutable metadata-only reservation](../models/pff-protocol.json) assigns **40 training, 12 development and 12 locked final-test fixtures**, including both halves and extra time. It was frozen before PFF tracking or event records were opened. Stage-stratified deterministic ranking defines the split; recurring teams are disclosed. Exact IDs, identity checks and the protocol hash are in [PFF_VALIDATION.md](PFF_VALIDATION.md). A fresh-fixture reservation does not certify label quality or prove that a model wins.

The completed campaign used one fixed mixed-data joint-head candidate, with upstream heads and decoding retained from v1.5. It fitted on training fixtures and applied the fixed criteria to development fixtures. Final reference access requires a passing development gate; this candidate did not pass. Both prediction sets must be frozen before references are scored. Comparisons use exact v1.5, identical inputs and masks. The frozen criteria are:

- Primary possession-spell F1 gain at least 0.005.
- Nondecreasing short-possession precision/recall and stoppage segment F1, boundary F1 and precision.
- At least 0.01 improvement in short-possession precision or stoppage boundary F1.
- Macro F1 decline no greater than 0.001; no fixture/condition spell-F1 decline greater than 0.01 or stoppage-recall decline greater than 0.005.
- A positive lower 5th percentile of paired match-bootstrap spell-F1 gains, using 20,000 fixture-level resamples and seed 20261101; frames or corrupted copies are not independent resamples.

Report per-match metrics and support, actual live/dead and sub-two-second spell counts, observation coverage, onset/restart errors and uncertainty. Retain unknown reference gaps. Check whether confidence intervals and event support are sufficient; do not use aggregate improvement to hide inadequate truth. Retain v1.5 if any gate fails, provenance remains unresolved or validation is insufficient. After final evaluation, those fixtures are consumed; further tuning needs a new untouched test.

Frozen active model SHA-256: `3af2dea11261e0454c54ce6791f3c5f0f55f488962da91579addb32a9aea24f0`.

Frozen submission CSV SHA-256: `1ddbe062fb0ff0bbb2f47fccda472953fee399629b3b88af9b23554b5f7a3ceb`.

Verify retention with `python tools/verify_release.py` and `python tools/verify_refinement_v8.py`. The [PFF results report](PFF_RESULTS.md) distinguishes completed fitting, development results and the retained deployment. No final audit was opened because the development gate failed.

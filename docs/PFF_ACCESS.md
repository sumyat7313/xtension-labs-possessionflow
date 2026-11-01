# PFF dataset access and fresh-match plan

Prepared 2026-11-01 for Xtension Labs. Status: **dataset form submitted; official delivery folder opened successfully**. The user supplied the contact details for this request; they are not stored in this repository. No separate dataset agreement was presented or accepted during the observed form flow. Usage rights and reference quality remain to be established. The active v1.5 model and submission remain unchanged.

## Access result

The form displayed **You're Almost There** and an immediate Google Drive access link. The supplied folder opened without an additional access request and contains Event Data, Metadata, Rosters and Tracking Data subfolders, plus competitions.csv, players.csv and a PFF FC Change Log. The tracking listing exposes per-game compressed JSONL archives. Only folder/file names and the change log were inspected; match labels were not opened, no split has been frozen and no PFF training or scoring has occurred.

The change log records initial-touch events and possession-sequence identifiers added in May 2025, along with later field changes. Their definitions, completeness and annotation provenance still need validation. The folder root and change log did not establish the dataset licence. The prepared support enquiry remains unsent. The official delivery URL and confirmation screenshot are retained locally rather than adding the gated delivery link or contact details to git.

## Verified access route

Open the [official World Cup dataset page](https://www.blog.fc.pff.com/blog/pff-fc-release-2022-world-cup-data) and use the embedded **Get Access to Our 2022 World Cup Dataset** form. Its live fields were checked in the browser on the date above:

| Form field | Prepared value or information needed |
| --- | --- |
| First Name | Provided by user and submitted; omitted from git |
| Last Name | Provided by user and submitted; omitted from git |
| Email | Provided by user and submitted; omitted from git |
| Organization | Xtension Labs |

The dataset form's submit button reads **SIGN ME UP!**. Separate PFF/Gradient newsletter popups also appear; these are distinct from the dataset form. No project statement, phone, payment details or password was requested in the observed dataset form. Later delivery/consent screens have not been inspected, so this does not establish their requirements or the dataset licence. Do not enter personal contact details into this repository.

The official announcement describes 64 games of tracking and events, but it does not establish dense human-reviewed possession/stoppage truth. Access does not itself establish permission for every proposed use. The page lists **fchelp@pff.com** as support. A ready-to-send coverage/use-rights enquiry is in [PFF_ACCESS_REQUEST.txt](PFF_ACCESS_REQUEST.txt); it has not been sent.

## Application text, if requested later

Xtension Labs is developing PossessionFlow for the More Than A Hack 2027 football analytics challenge. We estimate home possession, away possession, stoppages and unknown periods from observed player and ball tracking. We request access to the 2022 World Cup release to train and independently evaluate possession boundaries and stoppage detection on complete unseen fixtures. Our development repository is private. We will keep raw data and restricted references local and share only outputs permitted by the supplied terms. We would appreciate confirmation of permitted hackathon/model-training use, frame coverage, annotation provenance and any available synchronized video for independent review.

## When access arrives

1. Save the actual licence, specifications, delivery date and source identity locally. Confirm challenge training, derived model distribution to judges, and permitted reporting/attribution. Do not assume that a free download permits redistribution. An expiring delivery URL or access token belongs in local ignored storage, not git.
2. Download through the official supplied route into `data/pff/` (already excluded by the repository's `data/*` rule). Inventory file names, sizes and SHA-256 hashes. Keep the original bytes unchanged. Record incompleteness instead of silently treating missing files as an empty match.
3. Inspect documentation and a training-only pilot fixture to establish schema, native timestamps/frame rate, half boundaries, coordinate units/directions and team/player mapping. The announcement lists CSV metadata, while the [Kloppy guide](https://kloppy.pysport.org/user-guide/loading-data/pff/) shows per-game metadata/roster JSON. Resolve the delivered layout before writing the adapter. The guide exposes an in-play filter: explicitly retain dead frames (`only_alive=False`) in an adapter that supports it, and verify actual returned coverage. This flag cannot restore frames absent from the source.
4. Identify actual fixtures by teams, date and competition, across providers. Compare against all previously consumed Sportec, Metrica, WCSF and SkillCorner sources. An anonymized prior source with unresolved identity is an unresolved independence risk; changing provider IDs does not resolve it. Exclude known overlaps and resolve ambiguous identities before claiming a fresh holdout.
5. Verify how ball position, ball state and possession were produced. Keep provider state/ownership, event types and annotation-derived signals out of inference features. Interpolated or event-derived ball tracks cannot independently validate the events that generated them. Report observed-ball and player coverage, camera cuts and missing periods.
6. Establish reference quality. Sparse events alone cannot provide dense negative examples or every possession interval. If necessary, use legally available synchronized video, two reviewers blinded to predictions, adjudication and explicit uncertain intervals. The [reviewed-label workflow](LABELLED_DATA_V8.md#build-reviewed-labels-from-permitted-video) and `reviewed_intervals` importer are ready; they validate declarations, not the truth of a claimed review. No reviewer work or video access has yet been obtained.

## Proposed split and promotion protocol

This is a preparation plan, not a completed PFF benchmark or a claim of fresh data. Freeze exact fixture IDs and a protocol hash after metadata/duplicate checks, **before examining held-out labels or predictions**. If all 64 fixtures qualify, target 40 training, 12 development and 12 locked final-test matches. Otherwise document a revised allocation before training; do not silently shrink the test until a candidate wins. Partition whole fixtures, including both halves and extra time. Balance stages and observation coverage using metadata only, and record team overlap. Twelve test matches is a planning target, not a guarantee of statistical power.

Use training fixtures for fitting and development fixtures for all threshold/model choices. Freeze one candidate and both prediction sets before opening final-test references. Compare against the exact v1.5 artifact, using identical inputs and masks. Retain the existing development checks and predeclare the final-match version of the v8 gate:

- Primary possession-spell F1 gain at least 0.005.
- Nondecreasing short-possession precision/recall and stoppage segment F1, boundary F1 and precision.
- At least 0.01 improvement in short-possession precision or stoppage boundary F1.
- Macro F1 decline no greater than 0.001; no fixture/condition spell-F1 decline greater than 0.01 or stoppage-recall decline greater than 0.005.
- A positive lower 5th percentile of paired match-bootstrap spell-F1 gains; resample fixtures, not frames or corrupted copies. Predeclare the deterministic resampling method for the actual test size.

Report per-match metrics and support, actual live/dead and sub-two-second spell counts, observation coverage, onset/restart errors and uncertainty. Retain unknown reference gaps. Check whether confidence intervals and event support are sufficient; do not use aggregate improvement to hide inadequate truth. Retain v1.5 if any gate fails, provenance remains unresolved or validation is insufficient. After final evaluation, those fixtures are consumed; further tuning needs a new untouched test.

Frozen active model SHA-256: `3af2dea11261e0454c54ce6791f3c5f0f55f488962da91579addb32a9aea24f0`.

Frozen submission CSV SHA-256: `1ddbe062fb0ff0bbb2f47fccda472953fee399629b3b88af9b23554b5f7a3ceb`.

Verify retention with `python tools/verify_release.py` and `python tools/verify_refinement_v8.py`. No production code changes are needed to request access.

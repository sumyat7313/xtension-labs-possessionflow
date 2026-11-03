# PFF fresh-fixture research

The v1.5 model, production code, configuration and 199,319-row submission remain unchanged. This study downloads the current PFF release, trains one fixed mixed-data candidate and compares it on reserved fixtures. It does not claim a hidden competition score or independently adjudicated human ground truth. Fitted PFF weights and raw provider files remain local.

## Data obtained and checked

The current release contains 261 downloaded files (3.928 GB), covering all 64 World Cup fixtures. All 64 tracking archives passed full BZip2 decompression/CRC/EOF checks. JSON signatures and file hashes were checked; syntax and semantics are checked by the adapter only on the authorized partitions. Historical March/May event snapshots were listed but excluded in favor of the current release. Download interruptions were resumed without discarding completed files. [Download evidence](pff-download-verification.json).

The delivered specifications establish that `ballsSmoothed` uses event-labelled carriers and interpolation. This field is completely excluded. The adapter uses raw VISIBLE player/ball geometry, retains missing observations, collapses only consistent duplicate frames, and aligns native video timestamps to 25 Hz without bridging gaps. Period separators prevent cross-half spells. Provider events enter a separate reference reconstruction step and never become inference features. [Adapter and training method](PFF_TRAINING.md).

The references carry explicit OTB/kickoff team ownership and OUT/END state, preserving uncertain/video-missing/initial-non-event game intervals as X. These are **provider-event-derived references**, not independently adjudicated dense human labels. Initial touch is not automatically controlled acquisition. [Terms and label review](PFF_TERMS_AND_LABELS.md).

## Frozen experiment

The metadata-only partition reserves 40 training, 12 development and 12 final fixtures. Competition/date provenance shows that the previously consumed Sportec, SkillCorner, Metrica and WCSF samples are different fixtures. Team overlap is reported; this is a new-fixture comparison, not a claim of unseen-team or unseen-league generalization. [Protocol and identity checks](PFF_VALIDATION.md).

One joint H/A/D head was fitted for eight epochs with a fixed seed, using the 40 PFF training fixtures plus seven clean Sportec fixtures and equal per-fixture sampled weight. Upstream heads and decoder remain v1.5. No threshold search or runner-up selection was performed. Both models' complete cohort prediction files were saved and hashed before the first held-out reference was opened. Gates and a 20,000-resample paired match bootstrap were fixed before scoring.

The initial preparation-only run was replaced before any fit or score by a lossless memory optimization. A full training-pilot reread reproduced identical observations, alignment, features, upstream probabilities and all 154,201 v1.5 labels. No split, training policy or promotion threshold changed. A subsequent fit attempt stopped on nullable uncertainty flags before producing a model or opening held-out labels. Training-only inspection found 13 affected FOUL rows in nine fixtures. Those rows represent 12 distinct foul records that reuse on-ball event IDs. The corrected reference parser retains distinct event types under reused IDs, still rejects conflicts within one ID/type, and conservatively marks null certainty unknown. The prior run is preserved; only authenticated observation caches whose observation semantics are unchanged were reused for the new frozen run. [Preparation migration proof](PFF_PREPARATION_MIGRATION.md).

## Development

| Metric | v1.5 | PFF mixed candidate | Change |
| --- | ---: | ---: | ---: |
| Possession spell F1 | 0.4355 | 0.5317 | +0.0962 |
| Macro F1 | 0.8328 | 0.8719 | +0.0391 |
| Short-possession precision | 0.0855 | 0.1428 | +0.0574 |
| Short-possession recall | 0.1062 | 0.1329 | +0.0267 |
| Stoppage segment F1 | 0.7063 | 0.8165 | +0.1102 |
| Stoppage boundary F1 | 0.3391 | 0.5442 | +0.2051 |
| Stoppage precision | 0.9333 | 0.9268 | -0.0065 |
| Stoppage recall | 0.8902 | 0.9624 | +0.0722 |

Failed checks: no_target_regression.

Paired match-bootstrap 5th/95th percentile spell-F1 gain: [0.08400167029515579, 0.10886281347268721].

### Per-match development comparison

All values are on the same provider-reference masks. Changes below are in percentage points; the aggregate table above uses the 0–1 metric scale.

| Match | v1.5 spell F1 | Candidate spell F1 | Stoppage precision change |
| --- | ---: | ---: | ---: |
| 3813 | 0.4755 | 0.5476 | -1.30 |
| 3816 | 0.3787 | 0.4670 | +0.35 |
| 3818 | 0.4415 | 0.5292 | -2.72 |
| 3823 | 0.4726 | 0.5834 | -0.83 |
| 3831 | 0.4487 | 0.5756 | -0.72 |
| 3844 | 0.4339 | 0.5073 | -1.29 |
| 3850 | 0.4976 | 0.5900 | -1.61 |
| 3854 | 0.4869 | 0.5736 | +0.23 |
| 3858 | 0.4485 | 0.5906 | -0.90 |
| 10502 | 0.3280 | 0.4638 | +2.21 |
| 10514 | 0.4695 | 0.5538 | -0.25 |
| 10517 | 0.3450 | 0.3988 | -0.95 |

The scored references contain 1,152 stoppage segments and 1,088 under-two-second possession spells. Short-possession precision remains only 0.1428 and recall 0.1329: the gain does not solve this weakness. Unknown-reference gaps can truncate spells; these counts and metrics describe the reconstructed reference intervals. The next development priority is reducing false stoppage predictions while retaining the new boundary/recall gains, followed by independent review of short-spell and restart timings. No thresholds were retuned after this result.

Stoppage frame precision fell in nine of twelve matches; the largest drops were Mexico–Poland (3818, −2.72 percentage points) and Poland–Argentina (3850, −1.61 points). Short-possession recall also regressed on Japan–Spain (3854): 10/69 matched reference short spells became 7/69, despite the mean improvement.

Across all twelve matches, false-dead frames rose from 47,831 to 57,867 (+10,036 frames, or 401.44 seconds at 25 Hz), while correctly recovered dead frames increased by 55,665. The candidate predicted fewer stoppage segments (1,440 → 1,375) and matched more (917 → 1,031). This points to interval extent and incorrect dead intervals as useful development targets; these aggregates do not establish an early/late timing cause.

## Deployment decision

**Retain v1.5.** Development numeric gate passed: **False**. Final fixtures evaluated: **0**. The final cohort remains untouched because the fixed development gate did not pass.

The official release invites analysis and sharing findings, supporting the bounded private experiment described in the terms review. The supplied PDFs do not establish unrestricted dataset or fitted-model redistribution rights. Challenge submission of PFF-trained weights needs dataset-specific clarification. Neither a numerical gain nor a private repository resolves that question. The prepared support enquiry remains unsent.

## Verification and reproduction

477 engineering tests passed. Original release integrity and separate PFF research verifiers authenticate the retained deployment, fixed split/source hashes, model membership, prediction-before-reference records, aggregate gates and local artifact bytes. The aggregate repository bundle excludes fitted PFF weights and raw data. Local verification also checks the candidate and caches; portable CI recomputes evidence consistency without provider files.

```powershell
python -m pip install -r requirements-training.txt
python tools/pff_inventory.py --workers 2
python tools/pff_campaign.py prepare --split train --workers 1 --out runs/pff-reproduction
python tools/pff_campaign.py prepare --split development --workers 1 --out runs/pff-reproduction
python tools/pff_campaign.py fit --out runs/pff-reproduction
python tools/pff_campaign.py evaluate --split development --out runs/pff-reproduction
python tools/verify_pff_research.py --run runs/pff-reproduction
python tools/verify_release.py
python tools/verify_refinement_v8.py
```

Reproduction requires the authenticated original PFF files and existing Sportec caches; raw assets remain ignored by git. A final audit is separately guarded by the recomputed development gate. A failed/partial held-out reference pass cannot be silently rerun. Reproducing already inspected fixtures does not restore their freshness.

Active model SHA-256: `3af2dea11261e0454c54ce6791f3c5f0f55f488962da91579addb32a9aea24f0`.

Local experimental candidate SHA-256: `0ab133743c593466d8ae92ea94937e8c6bf22510f6b64053e3e69c7b467dcf3f`.

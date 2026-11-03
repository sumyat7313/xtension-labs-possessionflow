# First external-only rebuild: completed

Trial `external-rebuild-001` completed all eleven stages at 2026-11-03 20:14:46 UTC (2026-11-04 in Singapore). The seven external-match cache preparations, three grouped development folds and selected all-development refit completed without failed or repeated stages. The selected procedure is **external-joint**. This is a successful reconstruction milestone, **not a demonstrated improvement over v1.5 or a submission promotion**.

## Development results

Means below cover fifteen paired cells: five previously exposed Sportec matches under clean, degraded and fixed synthetic-camera conditions. The regimes are correlated observations of the same fixtures, not fifteen independent matches.

| Metric | Newly fitted external control | Newly fitted external joint |
| --- | ---: | ---: |
| Possession spell F1 | 0.477627 | 0.616227 |
| Macro F1 | 0.854818 | 0.912156 |
| Stoppage precision | 91.7728% | 96.9079% |
| Stoppage recall | 94.6372% | 98.0159% |
| Transition recall | 37.3513% | 51.0955% |
| Short-possession recall | 14.3390% | 27.8167% |

All seven predeclared rebuild diagnostic checks passed: spell improvement, per-cell spell and stoppage-recall safeguards, and nondecreasing mean stoppage precision, macro F1, transition recall and short-possession recall. Selection was recorded before the all-seven-match refit.

These diagnostic checks compare two newly rebuilt procedures. They do not replace the stronger v1.5/PFF promotion gates. Both new control and joint fits exclude each scored fold, but the historical dead-head and decoder-configuration selection had already exposed some of these fixtures. This remains development evidence, not an unbiased fresh test. The final refitted model has no untouched Sportec holdout.

## Provenance and verification

Only the exact historical external-data dead head was inherited; the previous control model and all later target-profile-derived learned components were discarded. Fourteen pinned external files were verified, and new observation caches used only external tracking and fixed synthetic camera stress. External references remain separate from inference inputs. See [method and lineage](EXTERNAL_REBUILD.md).

The root verifier authenticated the harness recipe, transitive source hashes, protocol, every completed stage's actual cache/model bytes, recorded summary, model bindings, and recomputed selection. An independent agent separately checked ledger consistency, stage-manifest identities, aggregate fold/refit/summary agreement, lineage and actual model bytes. The independent review did not itself rehash all cache payloads; the root check did.

- Candidate model SHA-256: `7d79dc50cfc16a7f64c37b65908be90e588d3bfdec35f599071b25180089bf47`.
- Protocol SHA-256: `0cd152ae7105f4561b5fa6f97f23fc71f1dd2afde49fe32301fb7d57dca6f31f`.
- Frozen recipe SHA-256: `8977f28bbbf455943419e1e105a2e3d188b9099c998584b0d24c85e945722f8d`.
- Final ledger event SHA-256: `4257b163b032f2631db612465cc55fab564f9c0a25629c9ad9148b6824eade98`.
- [Aggregate records, protocol and verification evidence](../models/external-rebuild-001-research.json) contain all thirty candidate/cell records; raw records, predictions and fitted weights remain local.

The existing **656-test engineering suite** and Windows/Linux CI passed for implementation commit `4bcf7ae`. This completion report introduces no scientific-code changes and does not claim a new test run. Verification hashes still match that frozen implementation.

## Decision and next experiment

**No promotion.** Original v1.5 model and submission CSV hashes remain unchanged, and its submission hold remains in force. The twelve reserved PFF final fixtures still have no prepared-cache overlap or final-access/evaluation artifacts. No target data, video or frame-level output was sent to AI tools during this review.

The next bounded task is to implement and freeze a development-only evaluator for this exact clean candidate on the twelve already-exposed PFF development fixtures, paired with the frozen v1.5 comparator and original PFF gain-retention anchors. It is a diagnostic comparison, not a fresh test. Do not run another fit or a parameter search before obtaining that comparison.

Only observation-derived cache features `x`, `z`, period blocks, FPS and observation-support metadata may be reused after verifying their source/configuration compatibility. **Do not reuse cached `base_home` or `base_dead`, or call the old PFF prediction helper that consumes them**: those probabilities came from v1.5. Recompute control and dead probabilities from the new candidate, apply its joint residual and unchanged decoder, and freeze the complete twelve-match prediction cohort before accessing separate development references. A new evaluator must reject every final fixture at its entry points and retain all current numerical, provenance and evidence gates. PFF reference-quality and usage-scope limitations remain unresolved; this step does not clear a submission.

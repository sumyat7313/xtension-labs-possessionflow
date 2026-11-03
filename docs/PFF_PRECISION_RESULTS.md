# PFF false-stoppage precision study

**Retain v1.5.** This follow-up implements a bounded correction to the trained PFF candidate, tests the complete declared development grid and preserves the original submission. No fixed-grid setting passed development; the final 12 fixtures remain untouched.

## Training diagnosis and correction

The [40-match training audit](PFF_FALSE_STOPPAGE_DIAGNOSTICS.md) found that lower dead-play confidence separates added false stoppages better than missing-ball or contact flags. Missing support and apparent contact occur at nearly equal rates in added errors and correctly recovered dead frames. A blanket veto based on either would risk losing gains. The audit is in-sample agreement with provider-event-derived references, not independently reviewed truth.

The correction multiplies D probability by `exp(-penalty)` and renormalizes all three classes. This subtracts the penalty from D log-odds, preserves conditional H:A odds, and retains the trained joint model, decoder, period boundaries and observation-support handling. It performs no new feature learning and reads no references during inference. Eight settings were frozen before scoring: `0, .125, .25, .375, .5, .75, 1, 1.25`.

Zero penalty reproduced the parent's probabilities and all 154,201 labels on the training pilot, without reading labels. Full development zero-control predictions and metric records also had to reproduce the previously frozen parent. Existing scientific code and production artifacts are unchanged.

## Fixed development experiment

The 12 development matches were previously inspected and are now tuning data. These results are not a new fresh confirmation. Every setting must pass all original v1.5 gates, exceed its mean stoppage frame precision by at least 0.2 percentage points, and retain at least 90% of the parent's positive mean gains in seven other metrics. Selection among passing settings uses spell F1, then boundary F1, then declared grid order. [Protocol](PFF_PRECISION_PROTOCOL.md).

| D logit penalty | Spell F1 | D precision | D recall | D boundary F1 | Short precision | Pass |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 0 | 0.5317 | 0.9268 | 0.9624 | 0.5442 | 0.1428 | False |
| 0.125 | 0.5318 | 0.9306 | 0.9590 | 0.5363 | 0.1390 | False |
| 0.25 | 0.5315 | 0.9350 | 0.9561 | 0.5320 | 0.1370 | False |
| 0.375 | 0.5299 | 0.9382 | 0.9525 | 0.5286 | 0.1348 | False |
| 0.5 | 0.5285 | 0.9418 | 0.9490 | 0.5276 | 0.1296 | False |
| 0.75 | 0.5250 | 0.9483 | 0.9405 | 0.5169 | 0.1210 | False |
| 1 | 0.5178 | 0.9548 | 0.9300 | 0.4840 | 0.1139 | False |
| 1.25 | 0.5104 | 0.9606 | 0.9165 | 0.4491 | 0.1068 | False |


Selected candidate: **None**. Development eligible for final audit: **False**. No unreported settings or post-result threshold changes were used.


## Evidence and deployment

PFF references remain carried provider-event-derived silver labels. Unknown reference gaps can truncate short spells; recurring teams and one tournament limit generalization. Development bootstrap intervals after tuning are diagnostic, not fresh significance claims. Independent dense review and permission to distribute fitted PFF models remain unverified. The numerical and evidence gates are separate; neither this study nor a favorable score resolves those limitations.

The v1.5 model, production configuration and 199,319-row CSV are unchanged. No PFF fitted model or raw records were published. The repository contains original code, aggregate results, arbitrary grid settings and hashes only. Final fixtures evaluated: **0**.

532 engineering tests passed locally. The research verifier recomputes all gates, derives records from detailed diagnostics, authenticates protocol/source/artifact identities and checks prediction-before-reference evidence. Its local mode also authenticates saved model/cache/prediction bytes. Original release and PFF research verifiers continue to pass.

```powershell
python tools/verify_pff_precision.py
python tools/verify_pff_precision.py --run runs/pff-precision-v1
python tools/verify_pff_research.py
python tools/verify_release.py
```

Reproduction requires the original authenticated data and the local frozen PFF parent model/caches. Choose a fresh output directory, initialize the protocol, predict the complete development grid, then score it. `prepare-final`, final prediction and final scoring each independently require a recomputed passing development selection. Exclusive reference markers keep interrupted scoring attempts consumed. Reproduction does not restore the freshness of already inspected fixtures.

Study protocol SHA-256: `555656e96cdaf5039b42180ca185abf72d2a5d17b37797a507204c20d66237bb`.

Retained model SHA-256: `3af2dea11261e0454c54ce6791f3c5f0f55f488962da91579addb32a9aea24f0`.

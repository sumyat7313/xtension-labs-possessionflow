# Clean-model component diagnosis: completed training40 result

**No replacement or new fit.** All 83 stages completed for the original forty PFF training fixtures. The diagnosis supports adapting the upstream stoppage classifier while retraining a compatible joint correction. It also identifies excessive unmatched short predictions as a separate concern. These are training-partition findings against provider-event-derived references, not a fresh validation result or evidence that v1.5 was beaten.

The historical v1.5 model and submission remain unchanged and on submission hold. No development or final fixture inputs were accessed in this study. The final twelve remain reserved.

## What was compared

The same externally rebuilt model, observation support and decoder were used in three fixed ablations:

- **Base:** clean control and frozen stoppage heads, without the joint correction.
- **H:A-only:** full-model conditional team evidence, with the exact base stoppage emission.
- **Full:** the unchanged clean joint model.

All forty fixtures' predictions were frozen before the first reference read. No thresholds, masks or model parameters were fitted or selected. The [fixed method](CLEAN_PFF_DIAGNOSIS_METHOD.md) describes the observation groups, period/unknown barriers and denominators.

## Complete mean comparison

These are equal-fixture means over forty matches; rates use the 0–1 scale.

| Metric | Base | H:A-only | Full |
| --- | ---: | ---: | ---: |
| Possession spell F1 | 0.405798 | 0.397409 | 0.419285 |
| Macro F1 | 0.829726 | 0.829649 | 0.836648 |
| Short-possession precision | 0.090520 | 0.078027 | 0.075814 |
| Short-possession recall | 0.083369 | 0.089499 | 0.091234 |
| Stoppage segment F1 | 0.603589 | 0.605544 | 0.643749 |
| Stoppage boundary F1 | 0.308241 | 0.309845 | 0.324192 |
| Stoppage precision | 0.868376 | 0.871137 | 0.895755 |
| Stoppage recall | 0.925098 | 0.922429 | 0.910210 |

The [complete aggregate evidence](../models/clean-pff-training-diagnosis-research.json) retains every fixture, model, metric, observation group and failed tradeoff. There is no candidate selection or promotion gate in this diagnostic study.

## False stoppages: the correction already helps

Pooled counts cover the same 5,824,045 known-reference frames: 3,319,653 live and 2,504,392 stopped. These counts and their pooled rates are distinct from the equal-fixture means above.

| Count | Base | H:A-only | Full |
| --- | ---: | ---: | ---: |
| False stoppage frames | 350,501 | 340,956 | 262,398 |
| Correctly predicted stoppage frames | 2,315,407 | 2,308,629 | 2,277,184 |
| Missed stoppage frames | 188,985 | 195,763 | 227,208 |

Base to full removes **88,103 net false stoppage frames (25.1%)**, at a cost of 38,223 fewer correctly predicted stoppage frames. Isolating the D correction, H:A-only to full removes 108,473 false-D frames and introduces 29,915, for a net reduction of **78,558**; it recovers 37,823 true-D frames and loses 69,268, for a net recall cost of **31,445**. Stoppage precision improves on 38/40 fixtures and spell F1 on 36/40 in that latter comparison.

Removing the joint D correction is therefore not supported as the next fix. This is a fixed-output ablation: it does not establish the effect of retraining the network. Base to H:A-only can also change decoded D labels because the decoder couples team and stoppage decisions, despite identical D emissions.

## Where residual errors concentrate

The following are separate, overlapping marginal groups for the full model. They cannot be added together or interpreted as independent causes.

| Observation group | False-D frames | Share of all false D | False-D / reference-live frames in group |
| --- | ---: | ---: | ---: |
| Base D emission at least 0.8 | 199,024 | 75.85% | 80.69% |
| Zero valid player-motion observations | 203,848 | 77.69% | 72.73% |
| Ball unavailable | 222,570 | 84.82% | 21.90% |

The player-motion group requires valid coordinates and usable speed; it is not a raw visible-player count. Ball unavailability means no accepted observation or interpolation support. Neither condition is itself a stoppage label.

A blanket missingness veto would remove substantial provider-labelled stoppage evidence: the zero-player-motion group includes **2,018,537 reference-D frames**, with **99.79% D recall**. The result motivates external-domain adaptation and careful precision/recall checks, not a rule that missing observations imply live play. High base confidence on false-D frames also argues against assuming that confidence is calibrated on PFF.

## Short possessions: less D erasure, more unmatched predictions

The common support contains **3,404 reference short spells** and 100,588 frames. Of those spells, **412** touch a period/known-reference-run edge and may be censored; their observed durations are not guaranteed to be full possession durations. The original scoring mask is unchanged.

| Short-possession count | Base | H:A-only | Full |
| --- | ---: | ---: | ---: |
| Reference spells with any decoded D | 937 | 899 | 726 |
| Reference spells completely erased by D | 435 | 412 | 339 |
| False-D frames inside reference short spells | 17,918 | 17,068 | 12,925 |
| Predicted short spells | 3,163 | 4,133 | 4,145 |
| Matched predicted short spells | 289 | 323 | 316 |

Full-model interruption falls from 27.53% to 21.33% and complete D erasure from 12.78% to 9.96%. However, it produces **982 additional short predictions for only 27 additional matched short predictions**: 955 extra unmatched predictions under the fixed provider-reference matcher. H:A-only accounts for 970 of those 982 added predictions. This supports monitoring oversegmentation separately from stoppage erasure; decoder coupling prevents attributing every difference solely to team-transition decisions.

Conditional team accuracy improves by only 0.109 percentage points in equal-fixture means. Full and H:A-only have identical conditional team evidence, as intended. A future D-precision improvement must not be accepted by ignoring short precision, spell F1 or recall losses.

## Reproducibility and verification

Before execution, **752 engineering tests passed locally** in 141.421 seconds. The implementation commit `ee38fabfafa31a5831f947a79bc0b9cc9c68d258` passed Windows and Linux CI. The completed-run [independent audit](clean-pff-diagnosis-audit.json) authenticated 83 stages, 63 source files, all forty-by-three aggregate records, artifact identities and the operator ledger; 47,666 consistency assertions passed. This audit reconciled saved aggregate evidence, rather than recomputing frame-level scientific results.

The bounded operator used one worker, one BLAS thread, a 2,700-second total budget and a 300-second stage timeout. It invoked the following fixed command 83 times, with every command, exit code and console hash recorded locally:

```powershell
python -m tools.clean_pff_training_diagnosis run --out runs/clean-pff-training-diagnosis-v1 --stage next
```

Inspection without rerunning consumed stages:

```powershell
python -m tools.clean_pff_training_diagnosis status --out runs/clean-pff-training-diagnosis-v1
```

- Candidate SHA-256: `7d79dc50cfc16a7f64c37b65908be90e588d3bfdec35f599071b25180089bf47`.
- Protocol canonical digest: `f92a95fc3a59c8daa3adc3997b4f85708a5baa25e79daf0bad37290241957d33`.
- Complete aggregate file SHA-256: `c3f3ff3804ecde020d0c389f6e76c396100d19366f737857e7da76ec15ece7b4`.

Raw records, predictions, references and fitted artifacts remain local. The audit confirms consistency, not independent tamper-proof timestamps, dense human truth or dataset-delivery rights. Those evidence limitations and the submission hold remain unresolved.

## Next bounded action

Prepare the [matched two-candidate external adaptation family](CLEAN_PFF_ADAPTATION_PLAN.md). Retain the clean control head and decoder; compare a newly trained joint correction with versus without an upstream D-head refit. Freeze the exact training-only split and recipe before any fitting. Preserve the original development gates, additional +0.002 stoppage-precision margin and 90% gain-retention requirements. No fitting or new development comparison was performed in this reporting checkpoint.

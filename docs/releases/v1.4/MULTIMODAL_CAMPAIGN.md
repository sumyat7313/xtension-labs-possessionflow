# V4 calibrated-camera and temporal development campaign

Promoted **gated-temporal** after the fixed development gates and separate sparse audit. Promoted gated-temporal after passing the fixed development gates and one-shot sparse independent audit.

## Fixed experiment

Five Sportec matches are scored in three whole-match folds: WN1/WMX, WPY/WQQ, and WR9. Each control fit excludes its entire scored fold. WOH/WOY helped train the unchanged dead head and are never scored. All seven matches are previously exposed development data. The v1.3 comparator uses its original out-of-fold heads, not its all-seven release refit.

Each candidate is scored under clean, degraded, original camera corruption and target-calibrated corruption: 20 cells per candidate, 80 total. Source and protocol hashes are frozen before fitting. The old cache files were not hashed in v3; the new campaign honestly records hashes now, verifies them against subsamples, rereads raw references and requires exact reproduction of the incumbent's historical predictions and scores.

## Candidates

- `baseline-v1.3`: frozen historical camera-context out-of-fold heads.
- `calibrated-control`: refit the symmetric 98-feature control head with the extra calibrated augmentation regime.
- `temporal`: add a bias-free, team-symmetric temporal residual. Half-local 5 Hz inputs pass through 16 hidden units and three kernel-three convolution layers with dilations 1/3/9. The exported inference implementation uses NumPy.
- `gated-temporal`: combine control and temporal expert probabilities using unsigned observation-reliability features. The gate is fitted on the lexicographically first outer-training match while its two experts are fitted on the remaining inner-training matches, then the experts are refitted on all outer-training matches.

The temporal training uses eight fixed CPU epochs, deterministic seed 9207, training-only feature scales, spell weights and masked chunk halos. No convolution crosses a half boundary. D/X rows have zero supervised H/A loss, although their observation features can provide temporal context. The dead-ball classifier and decoder settings remain fixed across this campaign. Thus this experiment does not directly learn a better stoppage classifier.

Camera calibration uses eight population-mean field-of-view footprints and coarse duration/player-count statistics from unlabeled target observations. The measured ball-observation fraction is approximately 59.2%. Homographies map normalized pitch coordinates to pixels. Individual target trajectories and frame geometry are not published; the aggregate profile stays local under `runs/`.

## Release gates and results

An eligible candidate needs a mean spell-F1 gain of at least 0.01 on the original three conditions, no loss above 0.02 in any of the 20 cells, no dead-recall loss above 0.01 in any cell, and nondecreasing mean transition and under-two-second spell recall over all 20 cells. Eligible candidates are ranked by primary-condition spell F1, then macro F1, then name; the incumbent remains eligible. The calibrated condition cannot alone drive a win.

| Candidate | Spell F1: original 3 conditions | Spell F1: all 4 conditions | Transition recall: all 4 | <2s recall: all 4 | Development gate |
| --- | ---: | ---: | ---: | ---: | --- |
| gated-temporal | 0.5767 | 0.5596 | 0.4661 | 0.2237 | pass |
| temporal | 0.5742 | 0.5579 | 0.4840 | 0.2355 | pass |
| baseline-v1.3 | 0.4776 | 0.4676 | 0.3582 | 0.1368 | incumbent |
| calibrated-control | 0.4752 | 0.4656 | 0.3689 | 0.1427 | fail |

Development selection: **gated-temporal**. Deployed model: **gated-temporal**. The temporal residual is active in the deployed model. The learned reliability gate is active.

Protocol SHA-256: `ac19d0766537e16fb30cedb6cc87e32495d8987ca251938e3b7e1609bfda8145`.

Production model SHA-256: `7f72b108801a9c6029a2543ceaccd2ef9f27947c256ab59265df47c316688653`.

The selection is frozen before any separate final audit. A promoted procedure is refitted on all seven matches; development scores do not become an independent evaluation of that final artifact. Full per-cell outcomes, regression checks and source hashes are in `models/multimodal-selection.json`.

## Independent-source check

The single frozen candidate was compared with v1.3 on 964 unique Game 2 pass origins. Exact team accuracy: baseline **88.59%**, candidate **87.66%**; balanced team recall: **88.43%** and **87.60%**. The predeclared sparse-audit gate passed. D/X predictions count as errors. This audit cannot measure dense spell F1 or stoppage recall. Game 2 has now been consumed and must not be reused as an untouched tuning target.

The predeclared policy permits at most a two-percentage-point reduction in both exact pass-origin accuracy and balanced team recall. Audit failure retains v1.3; it does not trigger a runner-up search or tuning on Game 2. The Game 1 development result was 89.10% exact accuracy over 798 unique pass origins, including 53 false dead-ball predictions. Its event references were never passed to inference. Neither sparse check supplies dense H/A/D ground truth.

## Video branch outcome

Video recovery is implemented and tested but inactive. The YOLO pilot accepted 107 of 132 masked held-out frames; 19 accepted observations differed from the automatic tracking reference by more than two metres. This is silver-reference disagreement, not a human-verified error rate. Development confidence calibration found no threshold meeting 95% within-two-metre agreement with sufficient support. Three broadcast tracking-alignment windows all failed acceptance. No recovered ball coordinates or broadcast evidence enter the production model.

## Reproduction and limits

The research environment additionally uses `requirements-training.txt` (pinned CPU PyTorch and SciPy). The calibrated profile requires authorized tactical tracking and `BAR-ATH_tactical_homography.csv`. It remains local under `runs/` and contains aggregate camera footprints, not target possession labels.

```bash
python -m pip install -r requirements-training.txt
python tools/calibrated_camera.py --data-dir data --out runs/calibrated-reproduction/profile.json
python tools/train_multimodal.py --data-dir data/external/idsse --previous runs/robust-campaign-v3 --profile runs/calibrated-reproduction/profile.json --out runs/multimodal-reproduction --workers 2
```

The campaign command requires the verified **original v3 local campaign artifacts**: caches, out-of-fold heads, protocol and summary. They are not included in the source archive because the caches contain raw-data-derived observations. Running the old trainer under modified v4 sources is not an equivalent historical artifact and the provenance check will reject it. The archive preserves v1.3 source and weights; reconstructing the historical campaign inputs is a separate prerequisite. The packaged final prediction needs none of these caches or training dependencies.


The engineering suite passed 181 tests in the research environment; release packaging separately verifies output bytes and the official CSV contract. No competition public reference labels were used to tune or select this campaign. The target has no available dense hidden labels for local validation. Improvements on correlated synthetic regimes cannot establish the competition rank. More independent dense labels, better false-stoppage discrimination, and precise missing-ball evidence remain the principal opportunities.

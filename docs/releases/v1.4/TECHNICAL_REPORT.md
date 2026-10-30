# Xtension Labs — PossessionFlow 1.4.0

Promoted **gated-temporal** after the fixed development gates and separate sparse audit.

## Task and production method

The pipeline produces H/A/D/X for all 199,319 BAR–ATH video frames from tactical tracking and automatic Smart Tagging. It aligns frames, isolates match halves, scales coordinates to metres, rejects implausible ball spikes and limits interpolation to short gaps. A frozen 58-feature live/dead classifier and a symmetric 98-feature control head feed duration-aware joint Viterbi decoding. Possession-change features and coverage-weighted missing-ball geometry augment the original control features without inventing a ball trajectory.

Deployed model: **gated-temporal**. The temporal residual is active in the deployed model. The learned reliability gate is active. Video recovery and broadcast fusion are inactive. Inference is offline and CPU-only using the pinned core requirements.

## New experiments and evidence

The v4 campaign adds target-calibrated camera corruption, a half-local 5 Hz symmetric temporal residual, and a reliability gate trained on an inner held-out match. The calibrated profile uses eight aggregate footprints and about 59.2% observed-ball visibility; no target possession labels or frame-level geometry are published.

Reference labels come from [IDSSE/Sportec](https://doi.org/10.6084/m9.figshare.28196177), Bassek, Rein, Weber and Memmert (2025), CC BY 4.0. Input provenance and scientific hashes are recorded. All seven matches were previously exposed. Five are scored in three whole-match folds, excluding each complete fold from control training; the frozen dead head's two historical training matches are not scored. The incumbent uses its historical out-of-fold heads. Each candidate has 20 match/condition scores. These are development results, not a fresh test or competition score.

| Candidate | Spell F1: original 3 conditions | Spell F1: all 4 conditions | Transition recall: all 4 | <2s recall: all 4 | Development gate |
| --- | ---: | ---: | ---: | ---: | --- |
| gated-temporal | 0.5767 | 0.5596 | 0.4661 | 0.2237 | pass |
| temporal | 0.5742 | 0.5579 | 0.4840 | 0.2355 | pass |
| baseline-v1.3 | 0.4776 | 0.4676 | 0.3582 | 0.1368 | incumbent |
| calibrated-control | 0.4752 | 0.4656 | 0.3689 | 0.1427 | fail |

Development selection: **gated-temporal**. Promotion requires at least 0.01 mean spell-F1 gain on the original conditions, no cell spell-F1 loss above 0.02, no cell dead-recall loss above 0.01, and nondecreasing average transition/short-spell recall. The calibrated condition also participates in regression checks. Selection is frozen before the separate audit and any all-seven refit. See the [full campaign report](https://github.com/likalight/xtension-labs-possessionflow/blob/main/docs/MULTIMODAL_CAMPAIGN.md) and `models/multimodal-selection.json`.

## Independent data and video research

The single frozen candidate was compared with v1.3 on 964 unique Game 2 pass origins. Exact team accuracy: baseline **88.59%**, candidate **87.66%**; balanced team recall: **88.43%** and **87.60%**. Both declined, within the two-percentage-point limit fixed before this audit. The predeclared sparse-audit gate therefore passed; this is not evidence of an independent accuracy improvement. D/X predictions count as errors. This audit cannot measure dense spell F1 or stoppage recall. Game 2 has now been consumed and must not be reused as an untouched tuning target.

Metrica sample Game 1 was used for adapter development: v1.3 correctly classified 89.10% of 798 pass-origin frames, with 53 predictions of D. Pass-origin labels are sparse and do not measure complete possession intervals. Raw Metrica sample data is credited and not redistributed. SoccerTrack v2 was researched but gated data was not downloaded; its event-derived auxiliary ball trajectory would not be a valid independent observation for scoring the same events.

Local YOLO ball recovery accepted 107 of 132 masked held-out frames, with 19 accepted points over two metres from the automatic reference. A development-only confidence screen could not satisfy the prescribed precision/support requirement. All three broadcast-alignment pilot windows were rejected. These modules remain inactive; no possession benefit is claimed.

## Reproduction and verification

```bash
python -m pip install -r requirements-solution.txt
python -m unittest discover -s tests -v
python -m possession predict --data-dir data --config configs/default.json --model models/submission.json --out runs/reproduction
python -m possession validate runs/reproduction/predictions.csv
```

The research environment passed 181 engineering tests. The manifest records input, source, configuration, model and output hashes. The release procedure validates the organizer's CSV contract and repeats inference from a clean source extraction. `docs/verification.json` records the completed checks, actual counts and limitations. The source bundle contains frozen weights and historical v1.2/v1.3 sources; restricted match assets, raw training observations and local feature caches are excluded. Full training campaign replay requires the verified original v3 cache and fold-model artifacts described in `SOLUTION.md`; they are not needed for final CSV reproduction.

Production model SHA-256: `7f72b108801a9c6029a2543ceaccd2ef9f27947c256ab59265df47c316688653`.

CSV SHA-256: `f044c553e8007a3dcffa5931f06f0dcf32beb6453e947b9f5eabdfa594e7827f`.

## Limits and submission status

Sparse independent labels do not validate dense spell F1. Short possessions, false stoppages, aerial balls, occlusions and domain shift remain difficult. The unchanged dead classifier is a separate bottleneck that the H/A temporal residual cannot directly repair. Automatic target tags are not represented by the external benchmarks. Correlated corruption regimes are not independent matches, and all-seven refits have no Sportec holdout.

The repository remains private at the user's request. Prepared files are not a submitted competition form. The organizer's public-source requirement, emailed submission form and team eligibility remain outstanding. Hidden competition scores, ranking and winning probability are unknown.

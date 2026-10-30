# Xtension Labs — PossessionFlow 1.5.0

Promoted **joint-state** after both the fixed development and fresh-match gates passed.

## Production method

The algorithm produces H/A/D/X for all 199,319 BAR–ATH tactical video frames from automatic player/ball tracking and Smart Tagging. It aligns frames, isolates halves, scales pitch coordinates to metres, rejects implausible ball spikes and only interpolates short gaps. A 58-feature live/dead classifier, 98-feature symmetric control model, temporal control residual and expert gate provide the v1.4 base. Duration-aware Viterbi decoding yields coherent states and intervals.

Deployed model: **joint-state**. The joint H/A/D residual is active in the deployed model. The learned local H/A boundary costs are implemented and evaluated, but inactive. Video recovery and broadcast fusion remain inactive. Prediction is offline, deterministic and CPU-only with the pinned inference requirements.

## Joint-state and boundary experiments

The joint model learns bounded home/control and dead-ball residuals with a three-class H/A/D objective. Shared unsigned context supports both a dead head and team-symmetric odd control branch; temporal windows stay within a half. The boundary model learns local H/A transition neighborhoods and may lower H/A switching cost by at most 75%, without changing minimum durations or dead/restart costs. Both are exported to portable NumPy inference. Their implementation does not imply activation or a measured gain.

Training references are from Bassek, Rein, Weber and Memmert (2025), [IDSSE/Sportec](https://doi.org/10.6084/m9.figshare.28196177), CC BY 4.0. All seven matches were previously exposed. Five are scored in three complete-match folds; each new fit excludes its scored fold, and the two historical dead-head training matches are not scored. The comparator uses v1.4 out-of-fold weights. Four candidates have 20 scores each under clean, degraded, camera and calibrated-camera observations. The target camera profile is derived from unlabeled aggregate observations; no target possession reference is used to train, tune or select.

| Candidate | Spell F1: original 3 conditions | Spell F1: all 4 | Turnover recall: all 4 | <2s recall: all 4 | Development gate |
| --- | ---: | ---: | ---: | ---: | --- |
| joint-state | 0.6233 | 0.6076 | 0.5003 | 0.2784 | pass |
| joint-boundary | 0.6205 | 0.6054 | 0.5168 | 0.3035 | pass |
| baseline-v1.4 | 0.5767 | 0.5596 | 0.4661 | 0.2237 | incumbent |
| boundary | 0.5767 | 0.5604 | 0.4864 | 0.2534 | fail |

Development selection: **joint-state**. Frozen audit candidate: **joint-state**. If no new candidate passes the development gate, the highest-ranked experiment is still audited to answer the fresh-match research question, but remains ineligible for promotion. The audit candidate is fixed before any final reference access.

## Fresh-match evidence

One frozen candidate, **joint-state**, was evaluated against v1.4 on three matches not used for training or candidate tuning. The fresh-match gate **passed**. Metrica Game 3 balanced team recall was **86.91% → 90.03%**. All increases and declines are shown below; a successful audit alone cannot override a failed development gate.

| Fresh match and reference type | Scored points | v1.4 team accuracy | Candidate team accuracy | Change (percentage points) |
| --- | ---: | ---: | ---: | ---: |
| Metrica Game 3: sparse provider pass events | 1,467 | 86.98% | 90.12% | +3.14 |
| SkillCorner 1886347: automatic phase labels | 31,988 | 84.61% | 86.34% | +1.73 |
| SkillCorner 1899585: automatic phase labels | 29,514 | 81.06% | 82.18% | +1.12 |

Metrica Game 3 supplies sparse pass-origin labels from provider events; the annotation method is unverified, so these are not described as human ground truth. SkillCorner supplies dense H/A labels only inside its automatic phase intervals. Unlabelled gaps remain X and are never interpreted as dead ball. D/X predictions at known H/A points count as errors. These checks do not establish fresh human-labelled dense H/A/D accuracy, possession-spell F1 or stoppage recall. The three audit matches are now exposed and cannot be reused as untouched tuning targets.

Development promotion requires at least 0.01 mean spell-F1 gain on the original three conditions; no cell spell-F1 drop above 0.02; no cell dead-recall drop above 0.01; and nondecreasing mean dead precision, primary macro F1, turnover recall and under-two-second spell recall. All 20 cells participate in the regression checks. The fresh audit additionally requires nondecreasing Metrica Game 3 accuracy and balanced recall, nondecreasing mean accuracy on the two SkillCorner matches, and no individual SkillCorner accuracy decline greater than two percentage points. Both gates must pass for promotion; no fresh-match result triggers a runner-up search or retuning.

Final decision: Promoted joint-state after passing both the fixed development gates and the one-shot three-match fresh audit.

## Reproduction and verification

```bash
python -m pip install -r requirements-solution.txt
python -m unittest discover -s tests -v
python -m possession predict --data-dir data --config configs/default.json --model models/submission.json --out runs/reproduction
python -m possession validate runs/reproduction/predictions.csv
```

The research environment passed 230 engineering tests. `docs/verification.json` records completed clean-source reproduction and official-format checks. The manifest authenticates inputs, source, configuration, model and output. Final inference requires only the packaged weights and authorized tactical inputs. Full research replay additionally needs the original verified v4 caches and historical fold artifacts described in `SOLUTION.md`; they are not redistributed.

Production model SHA-256: `3af2dea11261e0454c54ce6791f3c5f0f55f488962da91579addb32a9aea24f0`.

CSV SHA-256: `1ddbe062fb0ff0bbb2f47fccda472953fee399629b3b88af9b23554b5f7a3ceb`.

## Limits and submission status

Fresh dense human H/A/D validation remains unavailable. Short possessions, false stoppages, aerial balls, occlusions and provider shift remain difficult. Automatic target tags are absent from the external benchmarks. Prior video pilots failed their precision/alignment requirements, so their recovered points do not alter production tracking. Data from Metrica and SkillCorner is used for evaluation only and is not redistributed. Raw target tracking, footage and local feature caches remain excluded.

The repository remains private under `likalight`, as requested. Prepared files are not a submitted competition form. The organizer's public-source requirement, emailed form and team eligibility remain outstanding. Hidden competition scores, ranking and winning probability are unknown.

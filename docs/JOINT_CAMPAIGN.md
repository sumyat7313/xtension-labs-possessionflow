# V5 joint-state and short-boundary campaign

Promoted **joint-state** after both the fixed development gates and the one-shot three-match fresh audit passed.

## Fixed development experiment

The three outer folds score WN1/WMX, WPY/WQQ and WR9. Their complete match folds are excluded from every new fit. WOH/WOY trained the historical dead classifier and are not scored. The incumbent uses its historical v1.4 out-of-fold control, temporal and gate models, never its all-seven final fit on a scored fold. All seven Sportec matches are previously exposed development data.

Every candidate is scored in clean, degraded, camera and calibrated-camera conditions: 20 cells each, 80 total. Feature preparation reconstructs observation-only dead features and authenticates their original control observations and frozen dead outputs. Scientific sources, cache identities and historical model identities are recorded; historical incumbent prediction bytes and scores must reproduce exactly.

## Candidates

- `baseline-v1.4`: frozen gated-temporal procedure.
- `joint-state`: add a jointly trained temporal correction to H/A control and dead-ball logits. The 58 unsigned dead features and absolute 98 control features feed a 12-channel even branch. A 12-channel odd branch retains H/A exchange symmetry while sharing the even context. Three convolution layers use dilations 1/3/9 at 5 Hz. Both residuals are smoothly bounded to ±4 logits.
- `boundary`: use 53 observation-only, team-invariant features to predict a short neighborhood around direct H/A transitions and lower H/A switch costs locally by at most 75%. Minimum durations, stoppage penalties and restart penalties stay fixed.
- `joint-boundary`: combine the joint residual with a boundary model fitted to its resulting observation probabilities.

Joint training uses the factorized three-class H/A/D negative log likelihood, eight deterministic CPU epochs, seed 10527 and AdamW learning rate .002. Positive-weight H, A and D examples are required. X/zero-weight rows do not influence normalization or supervised loss. H/A spell weights are capped; D receives weight one; known-state transitions have bounded extra emphasis within ±0.4 seconds. Half-local sampling and masked chunk halos prevent cross-half influence. Inference uses NumPy only.

The boundary target covers ±0.16 seconds around direct H/A transitions without crossing D/X barriers or half boundaries. Training uses stride five and fixed L2 .01. Class-balanced fitting is corrected to its training prior; this is not a claim of externally calibrated probabilities. Boundary evidence never enters via reference labels during inference.

## Gates and development outcome

Development promotion requires at least 0.01 mean spell-F1 gain on the original three conditions; no cell spell-F1 drop above 0.02; no cell dead-recall drop above 0.01; and nondecreasing mean dead precision, primary macro F1, turnover recall and under-two-second spell recall. All 20 cells participate in the regression checks. The fresh audit additionally requires nondecreasing Metrica Game 3 accuracy and balanced recall, nondecreasing mean accuracy on the two SkillCorner matches, and no individual SkillCorner accuracy decline greater than two percentage points. Both gates must pass for promotion; no fresh-match result triggers a runner-up search or retuning.

| Candidate | Spell F1: original 3 conditions | Spell F1: all 4 | Turnover recall: all 4 | <2s recall: all 4 | Development gate |
| --- | ---: | ---: | ---: | ---: | --- |
| joint-state | 0.6233 | 0.6076 | 0.5003 | 0.2784 | pass |
| joint-boundary | 0.6205 | 0.6054 | 0.5168 | 0.3035 | pass |
| baseline-v1.4 | 0.5767 | 0.5596 | 0.4661 | 0.2237 | incumbent |
| boundary | 0.5767 | 0.5604 | 0.4864 | 0.2534 | fail |

Development selection: **joint-state**. Frozen audit candidate: **joint-state**. If no new candidate passes the development gate, the highest-ranked experiment is still audited to answer the fresh-match research question, but remains ineligible for promotion. The audit candidate is fixed before any final reference access.

Protocol SHA-256: `6b42774237a87fe6f85d3b72a9185e6f7159a9dfc23ea17514d88abf22081c9f`.

The audit candidate is refitted on all seven training matches only after the selection is frozen. Its final weights therefore have no untouched Sportec holdout. Neither fresh provider enters fitting or tuning.

## Fresh-match audit

One frozen candidate, **joint-state**, was evaluated against v1.4 on three matches not used for training or candidate tuning. The fresh-match gate **passed**. Metrica Game 3 balanced team recall was **86.91% → 90.03%**. All increases and declines are shown below; a successful audit alone cannot override a failed development gate.

| Fresh match and reference type | Scored points | v1.4 team accuracy | Candidate team accuracy | Change (percentage points) |
| --- | ---: | ---: | ---: | ---: |
| Metrica Game 3: sparse provider pass events | 1,467 | 86.98% | 90.12% | +3.14 |
| SkillCorner 1886347: automatic phase labels | 31,988 | 84.61% | 86.34% | +1.73 |
| SkillCorner 1899585: automatic phase labels | 29,514 | 81.06% | 82.18% | +1.12 |

Metrica Game 3 supplies sparse pass-origin labels from provider events; the annotation method is unverified, so these are not described as human ground truth. SkillCorner supplies dense H/A labels only inside its automatic phase intervals. Unlabelled gaps remain X and are never interpreted as dead ball. D/X predictions at known H/A points count as errors. These checks do not establish fresh human-labelled dense H/A/D accuracy, possession-spell F1 or stoppage recall. The three audit matches are now exposed and cannot be reused as untouched tuning targets.

SkillCorner 1874553 is reserved for adapter development; final SkillCorner matches are 1886347 and 1899585. Final Metrica data is Game 3. Source identities and the split/policy are frozen before reference access. The evaluator computes both models' outputs for all three matches before accessing separate scoring references. Provider possession fields do not enter the observation adapter.

## Release decision and limits

Development gate: **pass**. Fresh-match gate: **pass**. Release status: **passed**.

Promoted **joint-state** after both the fixed development gates and the one-shot three-match fresh audit passed.

Deployed model: **joint-state**. The joint H/A/D residual is active in the deployed model. The learned local H/A boundary costs are implemented and evaluated, but inactive.

Raw provider samples, cache matrices, camera profiles and video are excluded from the repository. Scores from correlated corruption regimes are not independent matches. Automatic phase labels can share provider-specific tracking errors and do not replace human validation. The fresh sources do not validate D or dense possession boundaries; this remains a material limit. No hidden competition result or winning probability is known.

## Reproduction

Optional research uses `requirements-training.txt`, which pins CPU PyTorch and SciPy. The v5 campaign requires the verified **original v4 local artifacts**: full feature caches, fold models, protocol, summary and the aggregate calibrated-camera profile. These caches contain raw-data-derived observations and are excluded from the repository. The pinned v1.4 source archive authenticates the historical procedure. Rebuilding those historical inputs is a separate prerequisite; running the old trainer under changed sources is not an equivalent provenance record.

```bash
python -m pip install -r requirements-training.txt
python tools/prepare_joint_cache.py --data-dir data/external/idsse --previous runs/multimodal-campaign-v4 --profile runs/calibrated-camera/profile.json --out runs/joint-cache-reproduction --workers 2
python tools/train_joint_campaign.py --previous runs/multimodal-campaign-v4 --cache runs/joint-cache-reproduction --out runs/joint-campaign-reproduction --workers 2
```

Cache preparation reconstructs the 58 observation-only live/dead features and requires exact agreement with every historical 98-feature control matrix and frozen dead prediction. The campaign compares four candidates in 20 match/condition cells each. Sportec references are used only within their assigned training or evaluation folds; all seven Sportec matches were previously exposed development data. Neither fresh provider enters training.

To reproduce the **already consumed** fresh audit, obtain the pinned provider samples and use the frozen experimental candidate produced above:

```bash
python tools/fresh_benchmark.py download --data-dir runs/fresh-audit-reproduction-data
python tools/fresh_benchmark.py evaluate-pair --data-dir runs/fresh-audit-reproduction-data --model runs/joint-campaign-reproduction/release-candidate-model.json --baseline-model models/baselines/v1.4.json --config runs/joint-campaign-reproduction/release-candidate-config.json --baseline-config configs/baselines/v1.4.json --out runs/fresh-audit-reproduction
```

Creating a new directory does not make these published audit references fresh again. The evaluator freezes models, configuration and source before access, saves all six prediction CSVs before separate scoring, and refuses to reuse a consumed local audit directory. SkillCorner tracking records contain embedded possession fields; the observation adapter whitelists geometry/detection fields and never accesses possession/phase/state for inference. Embedded fields are nevertheless parsed with the JSON records, so the exclusive audit marker is created before final tracking reads.


The research suite passed 230 tests. Release packaging separately checks the official CSV contract and clean-source prediction bytes; `docs/verification.json` records completed checks. The v1.4 fallback source archive SHA-256 is `04c5341978660a52fc32148221508ddb3db291921ff19d8fb1bfbf3e5cafeeb9`.

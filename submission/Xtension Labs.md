# Xtension Labs — PossessionFlow 1.2

Technical submission document · More Than A Hack 2027, Phase 1 · October 29, 2026

## Objective and selected release

Predict H (home), A (away), D (dead ball), or X (unknown/ungraded) for all 199,319
tactical-video frames at 25 Hz. The selected model is **control-spell-balanced-l2-0.001-cost-1**. It combines
the frozen v1.1 live/dead classifier with a learned team-control head and
duration-aware joint Viterbi decoding. It runs locally on CPU without API keys,
network requests, or retraining.

The v2 campaign compared 15 candidates against the exact released v1.1 model.
Selection used two development matches. The frozen winner then passed the
predeclared release gate on three fresh matches in both clean and degraded
tracking. Mean spell F1 rose from **0.3670 to 0.4419**, a gain of
**0.0748**. These are external results, not a competition leaderboard
score or a guarantee of target-match accuracy.

## Challenge assets and alignment

Inputs are the tactical ATD XML metadata, raw tactical tracking, and automatic
Smart Tagging XML. ATD frame IDs 1–199319 map to submission indices 0–199318.
Published half boundaries define independent decoding blocks; other spans are X.
The first metadata team maps to home. Normalized positions are scaled to the
published 105.3 × 68 metre pitch. Tag seconds map by round(seconds × 25).

Player slots need not preserve identity. Geometry is aggregated by team each
frame. Missing frames, duplicate IDs, malformed rows and schema conflicts fail
explicitly. Corrected-tracking offsets are not applied to this tactical stream.
Videos, homographies and broadcast tracking are unused. Restricted inputs stay
local and are excluded from the source package.

## Training data and separation

External data: Bassek, Rein, Weber and Memmert (2025), *An integrated dataset of
spatiotemporal and event data in elite soccer*, [IDSSE/Sportec dataset](https://doi.org/10.6084/m9.figshare.28196177),
CC BY 4.0; [dataset paper](https://doi.org/10.1038/s41597-025-04505-y).
Preserve this attribution for the derived trained weights.

Provider-supplied BallStatus and BallPossession are references only. The upstream
tracking includes provider quality control and identity correction; the paper
does not separately establish whether those two fields were manually assigned or
algorithmically derived. No human annotation or correction of the challenge
match was created. The external event feed is not used.

| Role in v2 | Matches | Permitted use |
| :--- | :--- | :--- |
| Training | J03WOH, J03WOY | Fit control coefficients, RMS scales, and training-only spell weights |
| Development/selection | J03WN1, J03WMX | Compare the fixed 15 candidates |
| Fresh final test | J03WPY, J03WQQ, J03WR9 | Evaluate only frozen winner and v1.1 baseline |

J03WMX was the v1.1 final test; it is explicitly exposed development data for v2.
No final match was used to fit, select, recalibrate, or refit the model. All seven
matches are different from the challenge match. Competition public labels were
not scored or used for training, selection or final evaluation.

The adapter converts centre-origin metres to normalized pitch positions and km/h
to m/s, using explicit home/guest roles. Recording frame ranges, not reference
labels, define half boundaries. BallStatus 0 maps to D; active status with owner
1/2 maps to H/A; unknown codes remain X. Mutation tests confirm reference labels
cannot alter observations, features, decoding, or predictions.

## Observation models

**Live/dead:** the existing 58-feature regularized logistic model stays frozen.
It uses ball visibility/reliability, speed, nearest distance, boundary support,
player coverage and speed summaries, plus half-local temporal means. Its v1.1 fit
used 111,902 known sampled rows from the same two training matches.

**Team control:** 52 signed features represent nearest-team distance margins and
ratios, soft proximity at multiple radii, second/third-nearest support, local
density, geometric control, speed-conditioned control, and ball-velocity-change
evidence for possible contact. Means over 0.3, 0.8 and 2 seconds and directional
context over 0.4 and 1.2 seconds are half-local. Offline future context is allowed;
it does not cross halftime. No field encodes labels, player identity, match ID or
team name.

Swapping team membership negates every control feature. A zero-intercept logistic
model with uncentred training-only RMS scaling therefore complements p(home).
Missing neutral features yield 0.5. D/X and zero-weight samples are excluded
before fitting. The selected control fit uses **59,288 live rows**, L2
**0.001**, with capped inverse-square-root possession-duration weights. Their
final bounds are 0.5–3 and live-frame mean one before stride-5 sampling. D/X and
half boundaries split spells; only training references construct these weights.

Contact/direction-change cues are motivated by [Vidal-Codina et al. (2022)](https://arxiv.org/abs/2202.00804).
Temporal, permutation-invariant aggregation is supported by [Borghesi et al. (2023)](https://doi.org/10.1016/j.eswa.2023.119780).
These are independently implemented compact features, not imported pretrained
models or claims of matching those papers' results. No PathCRF checkpoint was
used: those checkpoints can expose the same seven Sportec matches.

## Tactical evidence and decoding

Existing automatic tactical-tag log odds are added to the control logit. Defensive
tags favor the other side; unknown team tags remain neutral. The unchanged tag
fusion is tested synthetically but has no directly equivalent automatic tag feed
in this external benchmark, so its contribution is not established by these scores.

Emissions are (1-pD)·pH, (1-pD)·(1-pH), and pD. Joint Viterbi uses minimum internal
team duration 0.4 s, dead duration 0.6 s, stop cost 3, restart cost 2.8, and the
selected team-switch cost **1** (v1.1 used 4). Costs scale to 5 effective
observations/second. Half-edge segments can be truncated; each half resets.
Although the pD head is frozen, changed team emissions can change decoded D
intervals, so dead precision/recall are reported explicitly.

The production model includes validated decoder settings and nested control
weights. Inference snapshots the exact model and effective config. Missing or
incompatible control features fail loudly; requested models never silently
fall back to a different method.

## Candidate comparison and final results

The fixed candidates were v1.1, decoder-only costs 2/1, and control
L2 {0.001,0.01} × cost {4,2,1} × weighting {uniform,spell-balanced}.
Selection gates each candidate before ranking by equal-cell mean official
spell F1, then macro F1, then lexical ID. A candidate must gain at least 0.02
on average and cannot drop more than 0.02 in any match/regime cell.

The winner improved development mean spell F1 by 0.1113. Lowering switch cost
alone reached about 0.33, versus 0.4218 for the selected model and 0.3105 for v1.1.
Thus the comparison supports value from learned control beyond the decoder change.

The winner's weights, settings and source were frozen before reading final
references. Only that winner and baseline were evaluated across the following
six cells; the same gate passed. No runner-up search or retuning followed.

| Fresh final match | Tracking | v1.1 spell F1 | v1.2 spell F1 | Gain |
| :--- | :--- | ---: | ---: | ---: |
| J03WPY | clean | 0.4207 | **0.4985** | +0.0778 |
| J03WPY | degraded | 0.2847 | **0.3369** | +0.0522 |
| J03WQQ | clean | 0.4610 | **0.5542** | +0.0932 |
| J03WQQ | degraded | 0.3171 | **0.3974** | +0.0803 |
| J03WR9 | clean | 0.4343 | **0.5245** | +0.0902 |
| J03WR9 | degraded | 0.2844 | **0.3397** | +0.0553 |
| **Equal-cell mean** | **6 conditions** | **0.3670** | **0.4419** | **+0.0748** |

| Match / regime | Macro F1 before → after | D precision before → after | D recall before → after | Turnover recall before → after | <2s spell recall before → after |
| :--- | ---: | ---: | ---: | ---: | ---: |
| J03WPY / clean | 0.8569 → 0.8824 | 0.9437 → 0.9478 | 0.9738 → 0.9761 | 0.1605 → 0.3148 | 0.0403 → 0.0887 |
| J03WPY / degraded | 0.7642 → 0.7961 | 0.8834 → 0.8891 | 0.9612 → 0.9606 | 0.0926 → 0.1698 | 0.0161 → 0.0161 |
| J03WQQ / clean | 0.8480 → 0.8750 | 0.9169 → 0.9209 | 0.9378 → 0.9333 | 0.1987 → 0.4205 | 0.0198 → 0.0891 |
| J03WQQ / degraded | 0.7638 → 0.8030 | 0.8644 → 0.8738 | 0.9257 → 0.9207 | 0.1026 → 0.2616 | 0.0198 → 0.0297 |
| J03WR9 / clean | 0.8518 → 0.8790 | 0.9155 → 0.9180 | 0.9609 → 0.9607 | 0.1942 → 0.3813 | 0.0638 → 0.1277 |
| J03WR9 / degraded | 0.7523 → 0.7773 | 0.8136 → 0.8196 | 0.9420 → 0.9388 | 0.1079 → 0.2050 | 0.0319 → 0.0638 |

Clean/degraded regimes share each match and are correlated. Corruption uses only
observations: seeded ball-loss bursts, missing players, position and speed noise.
No outcome labels choose where corruption occurs. Additional diagnostics use
typed one-to-one H↔A transition matches within one second, and official IoU-0.5
spell recall in fixed duration bins. They explain performance rather than select
models. Paired within-match bootstrap intervals are included in
`models/control-selection.json`; they are not population-level confidence
intervals or a winning probability.

## Target prediction, reproducibility and checks

The target CSV has **199,319 rows** and passes the unchanged organizer format
validator. Its hidden accuracy is unknown. Target inputs were used for inference
only; no target prediction count was optimized toward a presumed true count.

The release has **93 passing engineering tests**, including independent tests of
team symmetry, slot invariance, physical scaling, halftime separation, neutral
missingness, train-only scaling/weighting, deterministic fitting/serialization,
legacy emission compatibility, campaign completeness, per-cell gates, production
equivalence, final-read ordering and data-excluding packaging.

Verified byte-identical from a clean source extraction.

- CSV SHA-256: `005773891ee98672c89b9bb280aa60a03adf3bdbf2eb0194523b635a2b69e590`
- Model SHA-256: `115c192cd4aa71be837e1c4bd2137d91add6dcea4b006a809b13232e62f43c0b`
- Inference source SHA-256: `339df7cf495318261694c62976814c3101a7ff577697d4272309dfaa14100f92`
- Protocol SHA-256: `f6f810eb6522f9db3b36bbc8374df862ec66ba8e3cc0f27a369917c5105a4e24`

Use Python 3.12 and the pinned `requirements-solution.txt`. The verified local
environment is CPython 3.12.14 / NumPy 2.5.4.
Reproduce with:

```bash
python -m pip install -r requirements-solution.txt
python -m unittest discover -s tests -v
python -m possession predict --data-dir data --config configs/default.json --model models/submission.json --out runs/xtension-labs
python -m possession validate runs/xtension-labs/predictions.csv
python -m possession bundle --run-dir runs/xtension-labs --team-name "Xtension Labs" --out dist/xtension-labs
```

## Remaining limitations and submission status

Three fresh matches improve evidence but still use one tracking provider. Their
synthetic missingness does not reproduce every tactical-camera error. Ball height,
purposeful touches and referee decisions are not observed. Short possessions
remain difficult; improved recall does not mean they are solved. Probabilities
are observation support, not calibrated target-match confidence.

The registered team name is Xtension Labs. The repository remains private by
user request. The challenge's required public-source URL, eligibility confirmation
and organizer's emailed submission form remain separate outstanding steps. No
entry has been submitted and no ranking is claimed.

Source code retains the upstream MIT license. Competition assets have separate
restricted terms. External data/derived weights retain CC BY attribution; organizer
logos and trademarks are excluded from the code license.

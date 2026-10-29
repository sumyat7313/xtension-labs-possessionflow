# PossessionFlow improvement research

Research snapshot: October 29, 2026. This document records sources and proposed experiments, not evidence that every proposal improves the submission. The executed campaign protocol and measured results are authoritative for promotion decisions.

## Highest-value next step

The v1.1 external result points to **missed short possessions and team-control transitions** as the main remaining weakness. On its previous final match, the selected model produced 258 clean / 256 degraded possession spells against 511 reference spells, despite much higher dead-ball recall. A model can label long stretches correctly while missing many possession objects. The next experiment should improve control evidence and the decoder's response to sustained changes, while retaining the frozen stoppage model as a stable component.

This is a diagnosis of aggregate external results, not a target possession count. Matching the count alone would not establish correct possessions and must not be an optimization objective.

## What current primary research suggests

| Source | Finding relevant to this project | Concrete application |
| --- | --- | --- |
| [Vidal-Codina et al., Automatic event detection in football using tracking data (2022)](https://arxiv.org/html/2202.00804v3) | Their possession stage distinguishes proximity, duels, and effective contact; local ball-direction/speed changes help distinguish touches from a ball passing near a player. Their reported event-detection results are not directly comparable to our possession spell F1. | Add observable control features and short temporal context. Do not assume the nearest player controls every airborne or travelling ball. |
| [Kim et al., PathCRF (KDD 2026)](https://arxiv.org/abs/2602.12080) | Models possession paths as structured sequences of player interactions, with constraints and Viterbi inference. It can infer events from player trajectories without observing the ball. | Supports richer temporal evidence and sequence consistency. A full neural replacement is a separate experiment, requiring its own data split and transfer validation. |
| [Link et al., Real Time Quantification of Dangerousity in Football (2016)](https://pubmed.ncbi.nlm.nih.gov/28036407/) | Separates ball control from pressure and other spatial context rather than equating possession with proximity. | Represent relative team support and ambiguity; avoid treating proximity or theoretical pitch control as certain actual possession. |
| [Borghesi et al., Using Temporal Convolutional Networks to estimate ball possession in soccer games (2023)](https://www.sciencedirect.com/science/article/pii/S0957417423002816) | Studies temporal trajectory embeddings, permutation-invariant team aggregation, and separate live/dead and team classifiers. Its training data is confidential and its accuracy metric differs from challenge spell F1. | Supports temporal, order-invariant team features and separated heads; it does not supply a directly eligible training dataset or a comparable leaderboard result. |
| [Spearman, Beyond Expected Goals (2018)](https://static.hudl.com/craft/downloads/SSAC2018_Beyond_Expected_Goals.pdf) | Models potential control through player arrival and ball-control processes in physical space. | Motivates physical geometry and competing team support. Our compact team head is not an implementation of the full pitch-control field, and potential future control is distinct from current possession. |

The [official PathCRF implementation](https://github.com/hyunsungkim-ds/pathcrf) includes pretrained models and uses the same seven-match Sportec dataset. Its weights are **not eligible as an untouched-test comparator without checking their training folds**. Loading a convenient checkpoint could leak the proposed holdouts. Its repository identifies MPL-2.0 for its code; copying implementation would require preserving its license obligations. This campaign can implement simple independent features without importing that code or those weights.

The 2022 event paper also uses provider live/dead status in its broader event pipeline. That information is unavailable as a legitimate inference input here, so that part cannot be transplanted as though it were observed sensor data. Our live/dead classifier must continue predicting it from observations.

## Finite candidate plan

Freeze the candidate set and release rules before reading fresh final-test references. A useful bounded matrix is:

1. **Frozen v1.1 release:** exact weights and decoding settings; this is the real comparator, not only the older heuristic.
2. **Decoder-only control:** preserve probabilities and test a small prespecified set of team-switch penalties to isolate temporal over-smoothing. The implementation campaign selected 4.0, 2.0, and 1.0 before fitting.
3. **Learned team-control head:** train home/away probability from signed team-difference features, using a small frozen regularization set and those same decoder settings.

The model should be equivariant: swapping the teams should change `p(home)` into `1-p(home)`. A no-intercept logistic fit on antisymmetric features, with train-only symmetric scaling, is a compact way to enforce this. Features can include nearest and second-nearest team distances, bounded local player support, distance contrasts conditional on ball reliability, and short past/future context. Centered context is legitimate for offline inference, but cannot cross half boundaries. No feature should require stable player identity across tracker slot changes.

The current team evidence attenuates fast or distant-ball observations and uses a team switch cost of 4.0. It is plausible that true brief control intervals do not accumulate enough evidence to overcome two switches. This is an implementation-level hypothesis to test, not a reason to lower the penalty until the desired number of spells appears.

If training remains dominated by long possessions, a later prespecified ablation can use capped inverse-square-root spell-duration weights derived **only from training references**, normalized per match. All candidate-selection and final-test references must remain excluded from fitting, scaling, weighting and calibration.

Keep the learned stoppage head and tactical-tag fusion fixed initially. The external benchmark has no equivalent automatic tactical tags, so it cannot directly establish that a new tag weighting transfers to the target match. Any tag changes require separate synthetic semantic/contradiction checks and a clearly stated evidence limit.

## Fresh independent test matches

The [Sportec/DFL dataset paper](https://www.nature.com/articles/s41597-025-04505-y) describes seven matches: two first-division and five second-division games. The position data is multi-camera tracking, with `BallPossession` identifying the controlling team and `BallStatus` the active/inactive state. These fields are references, not prediction features. The data is CC BY 4.0. Tracking quality and label conventions can differ from the challenge; seven matches do not establish broad generalization.

The existing local inventory contained only J03WOH, J03WOY, J03WN1, and J03WMX. Their previous outcomes are already known. **J03WMX is no longer an untouched test for development informed by v1.1 results.**

The following three remaining matches were identified from [Figshare's public article metadata](https://api.figshare.com/v2/articles/28196177). This research read filenames, sizes, licenses and checksums only; it did not load their reference outcomes.

| Match | Asset | File ID | Bytes | Provider MD5 |
| --- | --- | ---: | ---: | --- |
| J03WPY | Match information | 51643487 | 11,922 | `65a4568e4f374af94d3c1015cbc7c4f5` |
| J03WPY | Tracking | 51643526 | 371,688,499 | `e56ff1336a334f8da51110fbe7cab764` |
| J03WQQ | Match information | 51643484 | 11,932 | `5979209af316a914ce57347d0633df6c` |
| J03WQQ | Tracking | 51643529 | 356,219,423 | `9105bad987191d3c3c362c1fce9a4fcd` |
| J03WR9 | Match information | 51643490 | 11,823 | `5e2d580d3d8ed97e32959de90d04992b` |
| J03WR9 | Tracking | 51643532 | 372,934,140 | `e060bb7d7a5a3bfbd15bbf9bc631024d` |

Downloads use `https://ndownloader.figshare.com/files/{file_id}`. The six files total approximately 1.10 GB. Verify provider checksums on download and record SHA-256 hashes for campaign provenance. MD5 here checks transfer against the public catalog, not cryptographic authenticity.

Suggested separation: keep the old training matches for fitting, use J03WN1 and the now-exposed J03WMX for selection, and reserve all three remaining games for the frozen final comparison. A larger refit using development matches would require freezing that refit procedure before the final stage. Do not choose an alternate model after seeing final failures.

## Evaluation improvements that matter

The [official evaluation specification](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/EVALUATION.md) ranks by one-to-one possession matching over temporal IoU thresholds 0.3–0.7. Macro F1 is secondary. `X` is filled within scoring runs for the temporal metric, so abstention does not erase a difficult transition.

Use the unchanged official evaluator. Compare against v1.1 on each match and each observation regime. Equal-weight match averages prevent the longest match from deciding the result. Freeze at least a mean-gain requirement and a maximum permitted per-match/regime regression before final evaluation. Report all six final match/regime cells, including failures.

Report diagnostics that explain the ranking metric:

- Correct/missed/spurious possession counts and spell F1 at each IoU threshold.
- Transition precision, recall and timing error at the evaluator's fixed tolerances.
- Recall of short, medium and long reference possessions, with bins fixed in advance.
- Dead-ball precision and recall, preserving the existing tradeoff rather than hiding it behind an average.
- Performance on observed-ball and missing-ball intervals as diagnostics, never as label-based feature masks.

Clean and degraded versions of one match are correlated, not two independent matches. Within-match possession bootstrap intervals describe uncertainty conditional on that match. Three fresh matches improve evidence but still do not support a probability of winning the hackathon.

Keep observation corruption input-only and seeded. The existing burst missingness is useful; structured camera viewport loss, duplicate detections and spurious ball points are additional failure modes. Predeclare these stress tests rather than choosing a degradation that favors a candidate. Do not use human observations of the target game to write frame-specific rules.

## Further data and larger approaches

[Metrica's official sample-data repository](https://github.com/metrica-sports/sample-data) offers three anonymized matches with synchronized events/tracking and asks users to acknowledge the source. It is useful for a later cross-provider audit, but the repository does not directly supply the same dense H/A/D possession truth as Sportec. Deriving dense labels from sparse events introduces convention and timestamp choices that must be documented and validated. It should not be silently treated as equivalent ground truth or as CC BY-licensed data.

Automatic video ball recovery, multi-view synchronization, or a PathCRF-like player-trajectory model could address long occlusions. They are substantial independent projects, not guaranteed improvements from adding a pretrained component. Their value should be demonstrated on external matches with comparable missingness before changing the final CSV. A broader search cannot justify implementing mutually incompatible or unevaluated methods into one production pipeline.

## Competition constraints remain binding

The [official rules](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/RULES.md), checked online on the research date, permit algorithmic improvements and labels from other games. They prohibit human labels/corrections for the supplied match, including external human event data about that match. Public five-minute labels are for correctness checks, not model selection. The submitted file must be reproducible exactly from inspectable code and permitted assets. This research did not open or score the competition labels.

Keep restricted match data and reconstructable tracking exports out of the repository. The user's repository remains private by request; that is separate from the challenge's eventual public-source submission requirement. Source and report quality make the entry inspectable, but the ranking depends on its measured predictions.

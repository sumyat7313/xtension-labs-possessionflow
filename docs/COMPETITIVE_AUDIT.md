# Submission and competitive audit

This initial audit preceded the learned-stoppage release. The registered team name
is now confirmed as **Xtension Labs**; eligibility, public repository publication
and form submission remain unconfirmed. Current model results and release evidence
are in [TRAINING_CAMPAIGN.md](TRAINING_CAMPAIGN.md) and [BUILD_STATUS.md](BUILD_STATUS.md).
The public-window score below belongs to the historical heuristic, not the new CSV.

Reviewed October 29, 2026. The official submission, rules, FAQ, and evaluation
documents were checked against the public GitHub repository. Those resources
reported a last modification date of October 23, 2026 and agreed with the local
copies on the requirements below. This document is an audit of readiness and
research opportunities, not a prediction of competition rank.

## Submission completeness

| Requirement | Status at this audit | Evidence |
|---|---|---|
| Full-length possession CSV | Prepared locally; exactly 199,319 rows and official format validation recorded | [Verification](verification.json), [build status](BUILD_STATUS.md) |
| Reproducible source and pinned dependencies | Prepared locally; clean-source reproduction recorded | [Solution guide](../SOLUTION.md), [verification](verification.json) |
| Public source repository URL | Not published by this build; a source ZIP alone does not satisfy this requirement | [Submission requirements](../SUBMISSION.md#2-a-public-code-repository) |
| Short technical PDF or Markdown | Prepared locally | [Technical report](TECHNICAL_REPORT.md) |
| Registered team name and team eligibility | Team name confirmed as Xtension Labs; eligibility still unconfirmed | [Submission instructions](../SUBMISSION.md#how-to-submit), [team rules](../RULES.md#eligibility-and-teams) |
| Official submission form response | Not submitted by this build; the organiser emails its URL to registered participants | [Submission instructions](../SUBMISSION.md#how-to-submit) |

The deadline is **November 17, 2026**. The repository does not specify a cutoff
time or timezone. Use the team name registered by all members for the CSV and
technical-document filenames. One response per team may be edited; the last
response before the deadline is scored. Teams must contain 3–5 people; the rules
specifically exclude teams below three from the Innovation Challenge and Final
Pitch. Confirm the registration details separately from preparing technical files.

The published requirements allow either Markdown or PDF. No dashboard, pitch deck,
demo website, video, or model-training step is required for this Phase 1 submission.

## What determines rank

The ranking is automatic. [EVALUATION.md](../EVALUATION.md#spellf1--right-possessions)
defines spellF1 as one-to-one matching of whole possessions by temporal IoU,
averaged over thresholds 0.3, 0.4, 0.5, 0.6 and 0.7. A dead-ball interval separates
two possessions even when the same team restarts. Consequently, good boundaries
and avoiding fragmentation matter alongside correct team labels.

The actual [scorer](../evaluation/run_evaluation.py), function `rank_key`, sorts
by spellF1, then macroF1, transition F1, lower absolute transition latency, and
finally submission name. The [overview](../OVERVIEW.md#evaluation-criteria) states
that presentation, novelty and LLM use have no independent jury score in Phase 1.

The recorded public check is spellF1 **0.3429** and macroF1 **0.5507**. This is not
evidence that the solution beats the published **full-match** baseline of 0.2433:
the scores concern different windows. The public five minutes contain only 17
possessions, and the organisers demonstrate reversed model rankings between that
window and the full match. Neither winning nor superiority on the hidden match
has been established. Engineering tests and byte-identical reproduction establish
software properties, not real-match predictive quality.

## Priorities for further improvement

1. **Establish independent validation on other matches.** The initial solution lacked external real-match validation; the fixed model
   has now been tested on two other matches, documented in
   [the benchmark report](EXTERNAL_BENCHMARK.md). Broader independent evaluation
   and training remain unfinished. Use accessible, documented other-match
   labels and match-level holdouts with the official temporal metric to select
   improvements. Record provenance, splits and selection decisions. Test data
   from the competition match must not become a tuning set.
2. **Strengthen missing-ball and restart evidence.** The current implementation
   uses tactical tracking and automatic tagging. Videos, broadcast tracking and
   homography are not used. Algorithmic ball recovery or confidence-gated alignment
   of broadcast observations are plausible research avenues, not established
   improvements. False synchronization or correlated tag errors could make them
   worse, so measure them independently before adoption.
3. **Challenge segmentation across different regimes.** Include long occlusions,
   deflections, short controlled possessions, same-team restarts, crowded boxes,
   stationary live play, and missing observations during stoppages. Synthetic
   stress cases should test alternatives to the model's assumptions. They cannot
   replace independent real-match validation.
4. **Preserve a reproducible release candidate.** Freeze code and configuration,
   regenerate the complete CSV, validate that exact file, and reproduce its hash
   from a clean source extraction. Keep assets and reconstructive data exports out
   of the public repository.
5. **Finish submission administration alongside research.** Confirm the team,
   publish the code, and submit the three required fields through the official
   form. Local packaging must continue to report these as outstanding until their
   completion is independently confirmed.

These are opportunities for additional evidence and development. No method has
been selected because it improves the public-window score.

## Boundaries for allowed experiments

- [RULES.md](../RULES.md#training-and-validation-data) permits accessible labelled
  data from other matches, automatic pseudo-labels, and unsupervised or
  self-supervised methods on the provided match. Document all sources and methods.
- The public five-minute labels are for correctness checking only. The
  [evaluation instructions](../EVALUATION.md#the-public-window-is-not-a-leaderboard)
  expressly prohibit using them to choose between two versions.
- No human labels, hand corrections, private ground truth, or third-party
  human-produced event data for this competition match may enter training,
  validation or prediction. Do not add match-specific frame rules from manual
  inspection or published kickoff details; infer states through general algorithms.
- The [FAQ](../FAQ.md#can-i-use-llms-pre-trained-models-external-data) prohibits
  sending competition videos, datasets, restricted materials or their extracts to
  hosted/cloud LLMs. Run restricted-data processing locally.
- Do not publish the restricted assets or derived data that reconstruct them.
  The organiser must be able to inspect and run all necessary code and dependencies.

The core official sources are the
[submission specification](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/SUBMISSION.md),
[rules](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/RULES.md),
[FAQ](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/FAQ.md), and
[evaluation specification](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/EVALUATION.md).
Where prose differs from metric behavior, the organiser's metric code is authoritative.

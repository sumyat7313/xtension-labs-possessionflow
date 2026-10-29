# Independent other-match benchmark

This is the historical, fixed-heuristic baseline experiment. The subsequent
[learned-stoppage campaign](TRAINING_CAMPAIGN.md) uses four matches in total,
leaving three unused, and supplies the current release. Its validation role for
J03WN1 supersedes the initial held-out role described below.

The competition public five minutes are not a tuning or model-selection set.
This benchmark makes it possible to measure progress on other matches without
using those labels. It does not estimate the hidden competition score.

## Data and access

Use the official [IDSSE / Sportec Open DFL dataset](https://doi.org/10.6084/m9.figshare.28196177),
described by Bassek, Rein, Weber and Memmert in
[Scientific Data (2025)](https://doi.org/10.1038/s41597-025-04505-y).
The publisher releases seven Bundesliga matches under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Attribute the authors,
link the dataset and licence, and describe transformations if distributing derivatives.
The XML files are publicly downloadable without registration. No videos are included.

Two matches were chosen and assigned roles before obtaining scores:

| Match | Role | Tracking bytes | Metadata bytes |
|---|---|---:|---:|
| J03WOH: Fortuna Düsseldorf–SSV Jahn Regensburg | Development | 348,945,715 | 11,745 |
| J03WN1: VfL Bochum–Bayer Leverkusen | Held out | 391,722,803 | 11,722 |

The remaining five matches stay unused. A held-out match that has been inspected
cannot subsequently be described as an untouched final test. Any later experiments
must preserve the complete history and choose a fresh external final holdout before
selecting models. Split by whole matches, never neighboring frames.

Files are downloaded to ignored `data/external/idsse/`; source archives must not
contain the approximately 740 MB of raw input files. The adapter pins official
Figshare file IDs, byte counts, and MD5 checksums, then records SHA-256 provenance.

## Mapping and separation of labels

The adapter accepts only those external match IDs and checks metadata and tracking
identity. Team roles come from explicit `home` and `guest` metadata. Referees are
excluded; team membership comes from the player roster. Centre-origin metre
coordinates are converted to normalized pitch coordinates using each match's
published pitch dimensions. Periods come from `FrameSet.GameSection`, not labels.
The adapter checks the 25 Hz timestamp interval and uses complete first and second
half frame ranges, with an ungraded separator between them.
The dataset paper defines `S` as kilometres per hour for both player and ball
frames; the adapter divides by 3.6 to provide metres per second. A supplied speed
field can disagree with finite differences of positions, so this is a documented
provider unit conversion rather than a claim that every speed observation is exact.

The [Kloppy Sportec serializer](https://github.com/PySport/kloppy/blob/master/kloppy/infra/serializers/tracking/sportec/deserializer.py)
establishes the reference field mapping: `BallStatus=1` means alive;
`BallPossession=1` means home and `2` means away. The benchmark uses `BallStatus=0`
as dead, and rejects ambiguity by leaving missing or unknown live-possession codes
ungraded rather than treating them as away. Reference labels are stored in a
separate array and used only by the official evaluator. They never appear in
`Tracking`, features, confidence, half selection, or automatic tactical tags.

No event feed is supplied to inference. Unlike the competition recording, these
external matches have no corresponding automatic Smart Tagging. Consequently this
is a benchmark of the tracking and decoder components, not the complete
tracking-plus-tagging competition system.

## Commands

From the repository root with the core solution environment:

```bash
python tools/external_benchmark.py --self-test
python tools/external_benchmark.py --download --out runs/external-benchmark-v1
```

The self-test checks unit conversion, half isolation, unknown-code handling, and
that changing the reference XML attributes leaves every observation array identical.
The real run records an immutable protocol before scoring, checks source stability,
runs the unchanged inference function over whole halves, and invokes the unmodified
official metrics with 1,000 bootstrap samples. No configuration search occurs.
Use a new output directory for a new benchmark; existing protocols cannot be overwritten.

Each match produces `predictions.csv`, `reference.csv`, `metrics.json`,
`alignment.json`, and `manifest.json`. The top-level `protocol.json` records the
fixed configuration, code and metric hashes, split, and limitations;
`summary.json` contains aggregate scores. These outputs remain in ignored `runs/`.

## Completed baseline evidence, October 29, 2026

Both full-match downloads passed their published checksums. The adapter self-test
and three independently written adapter tests passed, including a direct check
that changing all reference-label attributes leaves inference predictions unchanged.
The fixed original configuration then completed both matches:

| Match | spellF1 | 5–95% bootstrap interval | macroF1 | Predicted/reference possessions |
|---|---:|---|---:|---:|
| J03WOH, development | 0.3463 | 0.3073–0.3859 | 0.6327 | 224 / 402 |
| J03WN1, held out for this run | 0.3150 | 0.2788–0.3524 | 0.6141 | 272 / 500 |

The diagnostic is consistent across these two external games: dead-ball precision
is 0.9905 and 0.9873, while dead-ball recall is only 0.4127 and 0.4493. The model
recognizes many genuine stoppages conservatively, but misses more than half of
their frames and predicts substantially fewer possessions than the references.
This supports prioritizing stoppage detection and interval boundaries in future
development. It does not prove that any proposed change will improve the challenge.

No parameters were changed or models selected from these results. The complete
run record is `runs/external-benchmark-v1/`, including immutable pre-score protocol,
input hashes, unit/frame mapping, full metrics, bootstrap intervals and predictions.

## What these results can and cannot tell us

This dataset is from another provider and uses multi-camera, postprocessed tracking.
Its possession conventions may differ from the competition's corrected Metrica
tracking. Its field mapping is verified; semantic equivalence at every disputed
touch or stoppage is not established. Cleaner tracking also does not reproduce
the competition's ball occlusions, track fragmentation, or inferred tactical tags.

An improvement must therefore survive independent matches and deterministic,
label-independent corruption tests (missing-ball bursts, coordinate noise, slot
permutations). Add such experiments only after declaring their protocol. A gain
here is evidence about generalization under those conditions, not evidence of
winning the hackathon.

## Research priorities

1. Establish this independent baseline and diagnose dead-ball errors, possession
   fragmentation and boundary timing. Use the official spellF1, not just frame accuracy.
2. Model control transitions using physically plausible ball contacts, persistent
   control and subsequent team reception; distinguish these from brief proximity
   during duels. [Vidal-Codina et al. (2022)](https://doi.org/10.1007/s12283-022-00381-6)
   combines control/duel zones, incoming and outgoing ball motion, and restart context.
3. Improve stoppage/restart inference as a separate component, because a missed
   stoppage merges two same-team possessions. Validate on the development match
   and synthetic adversarial cases, then freeze before any new holdout evaluation.
4. Consider learned temporal models only after the external-data evaluation is
   established. [The 2023 TCN possession study](https://www.sciencedirect.com/science/article/pii/S0957417423002816)
   motivates permutation-invariant trajectory embeddings, but its reported scores
   are not comparable to this competition and the research data are confidential.
   Do not claim that its weights or performance are available here.
5. [PathCRF](https://arxiv.org/abs/2602.12080) is a research candidate for player-only
   evidence through ball gaps. Its graph/CRF possession paths need stable trajectory
   information; the competition's identity switches and missing players make direct
   transfer uncertain. It is not an installed or validated part of this solution.

Video ball recovery and whistle/restart recognition remain separate research tracks.
They require the local videos, checked models/weights, and a compute budget; they
have not been implemented or evaluated. Adding a complex video pipeline without
independent evidence is not itself a competitive improvement.

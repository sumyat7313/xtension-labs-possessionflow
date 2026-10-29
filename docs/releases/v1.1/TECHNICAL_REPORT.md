# Xtension Labs — PossessionFlow 1.1

Technical submission document · More than a Hack 2027, Phase 1 · October 29, 2026

## Objective and selected approach

The system emits H, A, D, or X for each of 199,319 tactical-video frames at 25 Hz.
It combines team-control geometry and automatic tactical phases with temporal
segmentation. This release adds a learned live/dead observation model trained on
other matches, and evaluates whether it improves whole-possession segmentation.
The model choice is based on the official spellF1 metric on external matches,
never on the competition's public five-minute labels.

The selected learned variant is **l2-0.001_joint**, using L2 regularization 0.001 and **joint** decoding. Its scaler and weights were fitted on 111,902 known training rows, including clean and degraded observations. The weights were not refitted on validation or final-test data.

## Competition assets and alignment

Inference uses tactical ATD metadata, raw tactical tracking, and automatic Smart
Tagging. Metadata provides team membership; the first ATD metadata team is home.
Track slots are not treated as stable player identities. Normalized positions are
scaled to the published 105.3 by 68 metre pitch.

ATD frame IDs 1–199319 map to submission frames by subtracting one. The separate
corrected-tracking offsets are not applied. Tag seconds map by round(seconds*25).
Published half boundaries define independent decoding blocks; ungraded spans are
X. No rule supplies a match-specific label or manually corrected transition.
Malformed records, missing frames, and metadata disagreements fail explicitly.

Videos, homographies, and broadcast tracking are unused. Broadcast has a different
timeline, so it is not merged without synchronization evidence. No restricted
assets are passed to hosted models. All inference runs locally on CPU, without
API keys or network calls.

## Training sources and separation

Training and external evaluation use the public IDSSE/Sportec dataset:
Bassek, Rein, Weber and Memmert (2025), *An integrated dataset of spatiotemporal
and event data in elite soccer*, https://doi.org/10.6084/m9.figshare.28196177,
licensed under CC BY 4.0. The authors/provider supplied BallStatus and
BallPossession reference fields; Xtension Labs created no hand labels for these
matches or the competition match. Field definitions are documented in the
accompanying paper, https://doi.org/10.1038/s41597-025-04505-y.

The upstream optical tracking includes provider human quality control and identity
correction. The paper does not separately specify whether the two reference fields
were manually assigned or algorithmically derived. They are therefore described
as provider-supplied references, with that provenance limitation explicit. The
human-annotated external event feed is not used by this model.

| Role | Match IDs | Allowed use in this experiment |
|---|---|---|
| Training | J03WOH, J03WOY | Fit normalization and coefficients |
| Validation | J03WN1 | Compare the six declared learned variants and baseline |
| Final test | J03WMX | Test the frozen choice once against baseline |

All four are other matches. XML positions are centre-origin metres; speeds in
km/h are divided by 3.6. Explicit home/guest roles map teams. Complete half spans
come from recording metadata, never reference labels. BallStatus 0 maps to D;
status 1 with possession 1 or 2 maps to H or A. Unknown reference codes remain X.
Reference fields never enter Tracking observations, features, corruption, state
initialization, or decoding boundaries. Independent mutation tests verify this.

Training samples every fifth frame (5 Hz) from both training matches, under both
clean and degraded regimes. Unknown targets and zero-weight rows are excluded
before fitting the scaler. Public dataset IDs, byte counts, checksums, exact
training counts, source hashes, and split provenance are recorded in the model
and `models/selection.json`. No competition labels enter training or selection.

## Features, learning, and possession rules

Ball cleaning rejects isolated physically implausible spikes and interpolates
only gaps up to 0.32 seconds with plausible endpoints, within a half. Interpolated
positions receive reduced support. Longer gaps remain missing.

Home/away evidence comes from the nearest-team distance margin, weakened for
distant, fast, or uncertain ball observations. Offensive tags support the tagged
team; defensive tags support the opposing team. Unknown teams supply no team
preference. Tag evidence is tapered and overlapping support is bounded. Soft
automatic restart evidence can support team ownership; the learned dead-ball
head does not use tactical tags because no equivalent tags exist in training.

The learned head uses 58 observation-only features: ball speed and visibility,
distance to players and pitch boundaries, outside support, player coverage,
collective speed means and quartiles, fractions of slow players, and half-local
context windows of 0.6, 2 and 6 seconds. Features avoid player identities and team
names. A regularized logistic model is fitted deterministically by Newton/IRLS
optimization with a decreasing-objective line search. Training-only means and
scales, coefficients, intercept, and feature schema are shipped in JSON.

The declared candidates use L2 values 0.001, 0.01, and 0.1, each with one of two
decoders. Joint decoding forms H/A/D observation support and uses the original
duration-constrained Viterbi path. Separate decoding first determines live/dead
intervals, then determines team possession within predicted live intervals.
Both reset at halftime. Internal team/live segments have a 0.4-second minimum;
dead segments have a 0.6-second minimum. Half-edge segments may be truncated.
Transition costs remain fixed: team switch 4.0, stop 3.0, restart 2.8, with five
effective observations per second. Exhaustive short-sequence tests check both
decoders against global minimum costs.

Fast loose balls weaken immediate control evidence. Temporal persistence prevents
every short opposing touch becoming a possession change; sustained opposing
evidence can change ownership. Predicted D separates possessions even when the
same team restarts. A half with no usable ball or tactical evidence returns X.
The system cannot perfectly distinguish every deflection, duel, and controlled
touch from proximity observations alone.

## Experiment and selection

The immutable protocol was written before fitting. Input-only degradation uses
fixed seeds and alternates 2–8-second visible ball windows with 0.4–6-second gaps;
removes player slots in 12-second blocks with probability 0.15; and adds Gaussian
position noise of 0.15 metres and speed noise of 0.3 m/s. Reference labels do not
choose where corruption occurs. These are stress conditions, not an asserted
model of the competition camera's exact error process.

Eligible learned candidates must improve mean clean/degraded validation spellF1
by at least 0.02, with no regime dropping more than 0.02. Rank eligible candidates
by mean spellF1, then mean macroF1. Freeze the winner and weight hash before reading
the final test, and apply the same gate there. If it fails, retain baseline;
never try runners-up or retune on that final match. The final weights are the
training-only weights. The official metric implementation is unchanged.

### Validation: J03WN1

| Regime | Version | spellF1 | macroF1 | Dead precision | Dead recall |
|---|---|---:|---:|---:|---:|
| clean | baseline | 0.3150 | 0.6141 | 0.9873 | 0.4493 |
| clean | l2-0.001_joint | 0.3645 | 0.8074 | 0.9197 | 0.9619 |
| degraded | baseline | 0.1610 | 0.4958 | 0.9912 | 0.2908 |
| degraded | l2-0.001_joint | 0.2254 | 0.7274 | 0.8748 | 0.9663 |

### Final untouched test: J03WMX

| Regime | Version | spellF1 | macroF1 | Dead precision | Dead recall |
|---|---|---:|---:|---:|---:|
| clean | baseline | 0.3557 | 0.7199 | 0.9882 | 0.5520 |
| clean | l2-0.001_joint | 0.3938 | 0.8550 | 0.9164 | 0.9663 |
| degraded | baseline | 0.1877 | 0.5969 | 0.9903 | 0.3723 |
| degraded | l2-0.001_joint | 0.2581 | 0.7772 | 0.8591 | 0.9535 |

The release gate result is **passed**.
See `docs/TRAINING_CAMPAIGN.md` and `models/selection.json` for every candidate,
paired bootstrap diagnostics, provenance, and the frozen selection record.

## Limitations

The external provider has cleaner tracking and potentially different edge-case
possession conventions. Automatic tactical tags are absent externally, so these
tests assess tracking and decoding, not every interaction in the competition
pipeline. Two training matches, one validation match, one final match, and one
fixed degradation seed per match cannot establish general performance. Synthetic
gaps do not reproduce all camera cuts, slot duplication, aerial balls, or errors.
Observation support is not calibrated confidence on the competition match.
Local video recovery, broadcast synchronization, and richer controlled-touch
models remain unimplemented. No hidden-match score or winning rank is claimed.

The earlier heuristic's public-window correctness score is historical and is not
a score for a changed submission. The competition public labels were not rescored
or used to select this release.

## Reproduction and submission files

Use CPython 3.12 and the pinned core dependencies. Extract the provided source
archive, place the three tactical assets under `data/tactical/`, and run:

```bash
python -m pip install -r requirements-solution.txt
python -m unittest discover -s tests -v
python -m possession predict --data-dir data --config configs/default.json --model models/submission.json --out runs/xtension-labs
python -m possession validate runs/xtension-labs/predictions.csv
python -m possession bundle --run-dir runs/xtension-labs --team-name "Xtension Labs" --out dist/xtension-labs
```

The supplied model snapshot is sufficient for inference; retraining is not needed
to regenerate the CSV. To reproduce the external experiment, use
`python tools/train_stoppages.py --download --out runs/campaign-reproduction`.
It downloads approximately 1.52 GB of publicly licensed external XML if absent.
Use a fresh run directory to preserve immutable experiment records.

Inference writes deterministic UTF-8/LF CSV, config, model snapshot where used,
diagnostics, segments, and a manifest containing source, input, config, model and
output hashes. Packaging rejects altered CSVs, configs, code, or model snapshots.
The exact release hash and clean-source reproduction result are recorded in
`docs/verification.json`. Restricted competition assets and raw external XML are
excluded from the source archive. Required deliverables are `Xtension Labs.csv`,
this Markdown report, and a public repository URL. Publishing and form submission
are separate steps and are not claimed by local packaging.

# Xtension Labs — PossessionFlow 1.3.0

Selected **camera-context** and refitted its control head on all seven public matches.

## Task and method

The pipeline infers H/A/D/X for all 199,319 frames of BAR–ATH from tactical
tracking and automatic Smart Tagging. Physical coordinates are normalized to
metres; periods are isolated; implausible ball spikes are rejected; only short
gaps are interpolated. A frozen 58-feature live/dead model and a symmetric
learned control head feed duration-aware joint Viterbi decoding. Inference is
offline, CPU-only and deterministic with the pinned Python environment.

The new implementation adds 28 possession-change and 18 missing-ball features
to the existing 52 control features. Camera augmentation clips observations to
a moving field of view and simulates detector gaps, duplicates, false balls and
fragmented slots. Missing-ball evidence decays within each half and does not
fabricate a ball trajectory. Team geometry is weakened under poor coverage.
These new features are active in the selected model.

## Training and evaluation

Other-match references come from [IDSSE/Sportec](https://doi.org/10.6084/m9.figshare.28196177),
Bassek, Rein, Weber and Memmert (2025), CC BY 4.0. File sizes and MD5 are verified;
SHA-256 inputs and scientific source hashes are recorded. References never
enter observation features. No target-match labels or external target events
were used for training, selection or manual corrections.

All seven external matches were exposed in earlier work. The new experiment is
explicitly development cross-validation. Five matches were scored in three
whole-match folds; control fitting excluded the complete scored fold. The two
matches used to train the frozen dead head were not scored. The other matches
trained each candidate with L2 0.001, stride 10 and training-only spell balancing.
All candidates were fixed before scoring. There is no untouched final test.

| Candidate | Mean spell F1 | Transition recall | <2s spell recall | Gate |
| --- | ---: | ---: | ---: | --- |
| release-1.2 | 0.4409 | 0.2976 | 0.0861 | baseline |
| camera-context | 0.4776 | 0.3735 | 0.1434 | pass |
| camera-context-fast | 0.4507 | 0.4281 | 0.1793 | fail |
| camera-control | 0.4442 | 0.3092 | 0.0882 | fail |
| context-control | 0.4746 | 0.3972 | 0.1600 | pass |


Promotion required mean spell-F1 gain at least 0.02, no cell loss above 0.02,
no cell dead-recall loss above 0.01, and nondecreasing mean transition and
under-two-second spell recall. The baseline remained eligible. The full
protocol, ablations, per-cell scores and selection are in
[the campaign report](https://github.com/likalight1/xtension-labs-possessionflow/blob/main/docs/ROBUST_CAMPAIGN.md) and `models/robust-selection.json`.
The selected fixed procedure was then refitted on all seven matches. Its CV score is not an independent score of the final refit.

## Reproduction and verification

Install `requirements-solution.txt` under Python 3.12, then:

```bash
python -m unittest discover -s tests -v
python -m possession predict --data-dir data --config configs/default.json --model models/submission.json --out runs/reproduction
python -m possession validate runs/reproduction/predictions.csv
```

The release records model/config/source/input/output hashes. The source bundle
contains code and weights, including the archived v1.2 source; it excludes raw
challenge assets, training observations and cached feature matrices. The local
release procedure checks all tests, the official CSV contract, and byte-identical
CSV reproduction from a fresh source extraction. See `docs/verification.json`
for completed checks and exact hashes.

## Limits and submission status

Camera corruption is a fixed synthetic stress model, not calibrated target-camera
error. Team geometry can be distorted by visibility changes; short possessions,
ambiguous contacts, aerial balls and long missing-ball intervals remain difficult.
Automatic target tags are not covered by the external benchmark. Correlated
regimes are not independent matches. Hidden competition accuracy and winning
probability remain unknown.

The repository is private at the user's request. This is a prepared submission
package, not a submitted form; the organizer's public-source requirement,
emailed form and team eligibility still need to be addressed before submission.

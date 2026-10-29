# Camera and temporal-context development campaign

This campaign adds camera-aware training, possession-change evidence, and a
missing-ball branch. It compares four fixed alternatives against the exact v1.2
model. It does not use the competition match's reference labels.

All seven public Sportec matches have already been inspected in earlier work.
The results are **development cross-validation**, not an untouched test or an
estimate of the hackathon leaderboard score.

## What is implemented

- A virtual camera with a moving field of view, partial player coverage,
  detector gaps, slot fragmentation, duplicate detections and false ball bursts.
  Its parameters are fixed synthetic stress settings, not a calibration to the
  challenge camera or a reconstruction from its homographies.
- Possession-change features describing contested touches, reach advantage,
  ball acceleration/deceleration and direction changes, with short past/future
  context. These are learned inputs rather than rules forcing a team change.
- Missing-ball features using decaying observations within the same half and
  visible team shape/motion. They do not invent a ball trajectory. Context
  expires after eight seconds, and geometry is downweighted when coverage drops.
- A symmetric logistic control head combining the existing 52 features with
  28 change features and 18 missing-ball features. Swapping teams reverses every
  feature and complements the predicted home probability. The live/dead model
  stays frozen to avoid discarding the previously validated stoppage behavior.

The algorithm is offline: short future context is allowed. Player slot identity
is not assumed to persist. No label or event outcome is an observation feature.

## Fixed comparison

The three scored folds are J03WN1 + J03WMX, J03WPY + J03WQQ, and J03WR9.
For each fold, control training uses all remaining public matches. Neither
J03WOH nor J03WOY is scored because the frozen live/dead head trained on them.
An entire scored match is excluded from that fold's fitting and scaling.

Every scored match is evaluated under clean, legacy degraded and virtual-camera
tracking: 15 cells per candidate. The candidates are:

| Candidate | Features | Training regimes | Team switch cost / minimum duration |
| --- | --- | --- | --- |
| release-1.2 | Frozen 52-feature release | Historical fit | 1 / 0.4 s |
| camera-control | Existing 52, refitted | All three | 1 / 0.4 s |
| context-control | All 98 | Clean + degraded | 1 / 0.4 s |
| camera-context | All 98 | All three | 1 / 0.4 s |
| camera-context-fast | Same head as camera-context | All three | 0.5 / 0.2 s |

All new heads use L2 0.001, stride 10, and training-only spell weights clipped
to [0.5, 3] with live-frame mean one. Their settings are fixed before scoring.
The camera/control comparison also changes the amount of training data relative
to v1.2, so its gain cannot be attributed solely to augmentation. Comparisons
between new candidates use the same folds and data availability.

An eligible replacement must improve equal-cell mean official spell F1 by at
least 0.02, lose no more than 0.02 in any cell, lose no more than 0.01 dead recall
in any cell, and preserve or improve mean transition recall and under-two-second
spell recall. The baseline is always eligible. Candidates are ranked by mean
spell F1, then macro F1, then name. No settings are adjusted after scoring.

If selected, a new head is refitted on all seven public matches at its fixed
settings. The cross-validation result describes the training procedure, not an
independent test of that final refitted artifact. An independent dataset remains
necessary for a stronger generalization claim.

## Reproduce

Obtain the seven filesets from the [IDSSE/Sportec dataset](https://doi.org/10.6084/m9.figshare.28196177),
Bassek, Rein, Weber and Memmert (2025), CC BY 4.0. The historical downloader in
`tools/train_control.py` can obtain them; the v3 runner verifies their registered
sizes and MD5 checksums and records SHA-256 provenance.

```bash
python tools/train_robust.py --data-dir data/external/idsse --out runs/robust-reproduction
```

Use a new output directory. The protocol, scientific source hashes, fold models,
per-cell predictions, official scores, diagnostics and selection are retained.
Caches and raw tracking stay local and are excluded from Git and source bundles.
`models/robust-selection.json` records the completed campaign outcome.

## Completed result

Selected **camera-context** and refitted its control head on all seven public matches.

| Candidate | Mean spell F1 | Transition recall | <2s spell recall | Gate |
| --- | ---: | ---: | ---: | --- |
| release-1.2 | 0.4409 | 0.2976 | 0.0861 | baseline |
| camera-context | 0.4776 | 0.3735 | 0.1434 | pass |
| camera-context-fast | 0.4507 | 0.4281 | 0.1793 | fail |
| camera-control | 0.4442 | 0.3092 | 0.0882 | fail |
| context-control | 0.4746 | 0.3972 | 0.1600 | pass |

Each mean gives equal weight to the same 15 match/regime cells. The camera condition is synthetic; all results are development evidence. The complete individual scores and gate failures are in [robust-selection.json](../models/robust-selection.json).

The new code passed 125 engineering tests and three agent reviews of leakage, symmetry, half isolation, compatibility and the campaign split. Team centroid motion remains susceptible to camera composition changes even when visible-player counts stay constant.

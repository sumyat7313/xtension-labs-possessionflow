# Control improvement campaign, v2

## Protocol fixed before outcomes

The comparator is the exact released PossessionFlow 1.1 learned-stoppage model,
not the weaker original heuristic. Its model and configuration are preserved in
`models/baselines/v1.1.json` and `configs/baselines/v1.1.json`, with exact checksums
enforced by the runner.

Training uses J03WOH and J03WOY. Selection uses J03WN1 and J03WMX. J03WMX was the
previous release's final test and is explicitly considered exposed development
data in this campaign. Fresh final matches are J03WPY, J03WQQ and J03WR9.

Fifteen candidates are declared in advance:

- The unchanged 1.1 release.
- Two decoder-only ablations with team-switch cost 2 or 1, versus the original 4.
- Twelve learned-control combinations: L2 0.001 or 0.01; team-switch cost 4, 2 or 1;
  uniform or capped training-only inverse-square-root possession-duration weights.

The 52 control features combine physical distances, local player density, ball
motion/contact evidence, and short half-local temporal context. Signed team
differences, zero intercept, and training-only RMS scales preserve home/away
exchange symmetry. D/X frames are excluded from control fitting. Duration weights
remain within 0.5 to 3 and have mean one over live frames before stride-5 sampling;
dead/unknown gaps and half boundaries separate spells.

The existing 58-feature live/dead classifier stays frozen. Joint decoding means
its final D decisions may nevertheless change when team evidence changes. Both
dead-ball precision and recall are therefore reported for every evaluation cell.

Each candidate is scored with the official evaluator on clean and deterministically
degraded observations. Selection requires mean spell-F1 improvement of at least
0.02 over the released baseline and no individual match/regime cell more than 0.02
worse. Eligible candidates are ranked by equal-cell mean spell F1, then macro F1,
then lexical ID. Baseline is always an eligible fallback.

Only the frozen winner and baseline enter the three-match final evaluation. The
same gate applies across all six final match/regime cells. A failure retains the
baseline; no runner-up or final-set tuning is permitted. If no validation candidate
is eligible, fresh final references are not opened. The runner records source,
input, protocol, selected-model, configuration and prediction hashes.

Additional diagnostics report typed one-to-one control transitions within one
second, timing error, and official IoU-0.5 recall for possessions shorter than 2 s,
2–5 s, 5–15 s, and at least 15 s. These explain behavior; they are not selection
objectives. Paired within-match bootstrap intervals are descriptive, not evidence
of population-wide performance or hackathon rank.

## Reproduce

```bash
python tools/train_control.py --download --out runs/control-campaign-v2
```

All seven external matches total approximately 2.62 GB of raw XML. Existing valid
downloads are reused. The runner writes candidate artifacts under the run folder
and does not automatically replace the submission model. Whole-match final data
is read only after `selection-frozen-before-final.json` has been written.

## Interpretation

The [research review](RESEARCH_V2.md) documents primary sources and design choices.
External matches use the same tracking provider, and degraded inputs are synthetic;
their success does not establish performance on unseen providers or camera views.
External automatic tactical tags are unavailable, so existing tag fusion is
preserved without claiming this campaign validates its contribution. No pretrained
PathCRF checkpoints, competition labels, manual target annotations, raw target
uploads, or competition-score-driven tuning are used.

See the recorded outcome below and [independent review](REVIEW_V2.md).

## Recorded outcome

**Promotion gate passed for control-spell-balanced-l2-0.001-cost-1.** The equal-cell final mean rose from
0.3670 to 0.4419 (+0.0748). All six individual cell gains
are recorded below. Weights were not refitted after validation or final testing.

| Fresh final match | Tracking | v1.1 spell F1 | v1.2 spell F1 | Gain |
| :--- | :--- | ---: | ---: | ---: |
| J03WPY | clean | 0.4207 | **0.4985** | +0.0778 |
| J03WPY | degraded | 0.2847 | **0.3369** | +0.0522 |
| J03WQQ | clean | 0.4610 | **0.5542** | +0.0932 |
| J03WQQ | degraded | 0.3171 | **0.3974** | +0.0803 |
| J03WR9 | clean | 0.4343 | **0.5245** | +0.0902 |
| J03WR9 | degraded | 0.2844 | **0.3397** | +0.0553 |
| **Equal-cell mean** | **6 conditions** | **0.3670** | **0.4419** | **+0.0748** |


The repository README includes a chart of this frozen final comparison.

### Control and stoppage diagnostics

| Match / regime | Macro F1 before → after | D precision before → after | D recall before → after | Turnover recall before → after | <2s spell recall before → after |
| :--- | ---: | ---: | ---: | ---: | ---: |
| J03WPY / clean | 0.8569 → 0.8824 | 0.9437 → 0.9478 | 0.9738 → 0.9761 | 0.1605 → 0.3148 | 0.0403 → 0.0887 |
| J03WPY / degraded | 0.7642 → 0.7961 | 0.8834 → 0.8891 | 0.9612 → 0.9606 | 0.0926 → 0.1698 | 0.0161 → 0.0161 |
| J03WQQ / clean | 0.8480 → 0.8750 | 0.9169 → 0.9209 | 0.9378 → 0.9333 | 0.1987 → 0.4205 | 0.0198 → 0.0891 |
| J03WQQ / degraded | 0.7638 → 0.8030 | 0.8644 → 0.8738 | 0.9257 → 0.9207 | 0.1026 → 0.2616 | 0.0198 → 0.0297 |
| J03WR9 / clean | 0.8518 → 0.8790 | 0.9155 → 0.9180 | 0.9609 → 0.9607 | 0.1942 → 0.3813 | 0.0638 → 0.1277 |
| J03WR9 / degraded | 0.7523 → 0.7773 | 0.8136 → 0.8196 | 0.9420 → 0.9388 | 0.1079 → 0.2050 | 0.0319 → 0.0638 |


The complete 60 validation records, 12 final records, input/source hashes,
training protocol and paired bootstrap diagnostics are in
[control-selection.json](../models/control-selection.json). The initial experiment
source is preserved in commit `30d8c61`; the frozen protocol's scientific hashes
remain authoritative.

Short possessions and missing-ball behavior remain imperfect. These results
support this release among the declared candidates, not a hidden-match rank.

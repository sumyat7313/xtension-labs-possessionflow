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

Campaign outcome and independent review will be recorded after the frozen run.

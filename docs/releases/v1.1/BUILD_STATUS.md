# Xtension Labs build status

Completed locally on October 29, 2026. The selected release is
**l2-0.001_joint**, PossessionFlow 1.1.0. Its learned live/dead head
passed the predeclared external validation and untouched-final release gates.
The competition submission is rebuilt with the frozen selected weights.

## Verification

- 53 engineering tests passed in the development repository.
- The same 53 tests passed from a fresh extraction of the source ZIP.
- The fresh extraction's Windows launcher loaded the packaged model and
  reproduced all 199,319 prediction rows byte for byte.
- The exact output passed the organiser's CSV-format validator.
- Input, source, configuration, model, selection-evidence and CSV hashes are recorded.
- Installed core dependencies match every pinned version.
- Public competition labels were not rescored or used for training/model selection.
- Restricted match assets, external raw XML, local environments and run caches
  are excluded from the source archive.

CSV SHA-256: `ea2807bbe4b15d25f1d1ec4494fea42a85bee215f2d1f43a417642d5bd60ac96`

Model SHA-256: `3759734a29a07eafc11723029800d30863d00c0a9e478bea3fcafaf6805c2867`

Source SHA-256: `8c69979d0235a1e99d98561e92e353af3a0475a79f1c99b17e705cfe35f1937e`

## Independent results

Training used J03WOH and J03WOY; validation selected the candidate on J03WN1;
the frozen model was then tested on previously untouched J03WMX. The six learned
candidates varied regularization and joint/separate decoding. Clean and
deterministically degraded regimes were both required for selection.

| Final-test regime | Baseline spellF1 | Selected spellF1 | Baseline D recall | Selected D recall |
|---|---:|---:|---:|---:|
| clean | 0.3557 | 0.3938 | 0.5520 | 0.9663 |
| degraded | 0.1877 | 0.2581 | 0.3723 | 0.9535 |

Dead-ball precision decreases as recall improves; the selected model nevertheless
improves whole-possession spellF1 in both regimes. Possession counts still miss
reference transitions. This is the strongest eligible version of the declared
campaign, not proof of the best possible model or a hackathon rank. Provider
differences and camera errors remain transfer risks. See
[the training campaign](TRAINING_CAMPAIGN.md), [independent review](MODEL_REVIEW.md),
and [technical report](TECHNICAL_REPORT.md).

The historical heuristic public-window score does not apply to the rebuilt CSV.
The private full-match competition accuracy remains unknown.

## Final local deliverables

`dist/xtension-labs/` contains `Xtension Labs.csv`, `Xtension Labs.md`,
`Xtension Labs_source.zip`, and the model/configuration/provenance records.
The source archive contains the exact inference model as `models/submission.json`.
Use the commands in `SOLUTION.md` to reproduce it.

The team name **Xtension Labs** is confirmed. Public repository publication and
the organiser's emailed submission form are still outstanding; team eligibility
also remains unconfirmed. No form response or public repository has been created.
The published deadline is November 17, 2026; no cutoff timezone is stated.

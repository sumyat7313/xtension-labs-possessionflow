# Xtension Labs — release 1.2

Selected **control-spell-balanced-l2-0.001-cost-1**, tested against the exact v1.1 release on three fresh
external matches. Final mean spell F1: **0.3670 → 0.4419**.
The predeclared mean-gain and every-cell non-regression gate passed.

- 93 engineering tests pass.
- All 199,319 target rows pass the organizer's CSV validator.
- Verified byte-identical from a clean source extraction.
- Source, config, model, inputs, selection evidence and output hashes are recorded.
- No competition labels were used for model selection or rescored for this release.
- Restricted target data and raw external XML are excluded from Git and bundles.

CSV SHA-256: `005773891ee98672c89b9bb280aa60a03adf3bdbf2eb0194523b635a2b69e590`

Model SHA-256: `115c192cd4aa71be837e1c4bd2137d91add6dcea4b006a809b13232e62f43c0b`

See [technical report](TECHNICAL_REPORT.md), [campaign](CONTROL_CAMPAIGN.md),
[independent review](REVIEW_V2.md), and [verification](verification.json).

The private GitHub repository contains prepared files; it is not a completed form
submission. The public-source requirement, eligibility confirmation and organizer
form remain outstanding. Hidden competition accuracy and ranking are unknown.
The prior release records remain under `docs/releases/v1.1/`.

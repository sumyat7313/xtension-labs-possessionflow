# Submission readiness — November 14, 2026

Team: **Xtension Labs**. The existing `likalight/xtension-labs-possessionflow` repository must remain private. No repository visibility change, public publication, provider contact or competition submission is performed by this checklist.

The organiser's [published submission contract](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/SUBMISSION.md), checked November 14, requires one possession CSV, a public code-repository URL and a short technical document. The published deadline is **November 17, 2026**; the document does not state the cutoff time or timezone. The user supplied the emailed submission-form URL and explicitly instructed **do not submit yet**. The full form URL is retained only in the local checkpoint, not this repository document. The precise cutoff details remain unanswered.

## Candidate separation

| Artifact | Purpose and decision |
| --- | --- |
| Historical v1.5 | Preserve unchanged for comparison. Submission hold remains because of challenge-derived camera-fitting ancestry. |
| Clean Sportec fallback | Exact external-joint reconstruction, model `7d79dc50…89bf47`, config `2755c140…66612b`. Local package prepared and CSV independently reproduced from the extracted source archive. It failed improvement gates and must not be described as stronger than v1.5. |
| A/B PFF adaptation | Separate frozen research campaign. Any selected model requires complete artifact verification, connected lineage, unchanged development checks and separate delivery-rights review. Never replace the fallback merely because fitting finished. |
| Final twelve PFF fixtures | Remain sealed. No preparation, inference or reference access until all declared numerical and substantive review conditions pass. The fallback does not need them. |

The scheduled automation was deleted at the user's request. The current work is interactive, user-authorized execution; this document does not recreate a schedule.

## Prepared local deliverables

`tools/build_compliant_fallback.py` prepared the audited fallback package. Its explicit source allowlist is the publication boundary. The verified portable source contains only required inference code, pinned dependencies, the audited Sportec-trained model/configuration, licensing/attribution, package integrity information and the technical report. It excludes `.git`, supplied match data, challenge ground truth, all target-derived historical models, PFF-fitted weights, provider documents, sensitive download links, model-training caches, frame-level predictions and diagnostics.

The team-named prediction CSV is a **separate form upload**, not a file in the public source archive. It is stored in the ignored local delivery directory with its receipt; the team-named technical document is at the package root. The existing `submission/Xtension Labs.csv` remains the held historical comparator and is not overwritten.

The technical document source is [Xtension_Labs_Technical_Report.md](Xtension_Labs_Technical_Report.md). It covers the fixed fallback only: method, used/dropped assets, training amount and label provenance, possession semantics, limitations, reproduction and attribution. If a different model later becomes eligible, prepare a new model-specific report rather than silently changing what the existing report describes.

## Release checklist

Checks 1–6 passed locally for the fixed fallback on November 14. The two inference executions used separate source/output directories; the second source was extracted independently from the prepared archive. Both CSVs have exactly 199,319 valid ordered rows and identical SHA-256, both receipts match, and source verification remained unchanged. This establishes reproducibility and format correctness, not predictive accuracy or submission authorization.

1. Authenticate the exact clean candidate, configuration, connected ancestry and copied inference-source bytes. Reject target-derived learned components or mismatched model/configuration files.
2. Validate the source-export allowlist and archive contents. No history, raw data, restricted weights, predictions or sensitive URLs may enter the public-source candidate. Confirm the current private repository's visibility remains unchanged.
3. Run frozen local inference on the authorized supplied automatic inputs only. Do not tune, fit, manually label, inspect frame predictions in AI tools, or compare output against challenge video/ground truth.
4. Validate exactly 199,319 rows, monotonically ordered frames `0…199318`, valid `H/A/D/X` labels and the exact header. Structural validity is distinct from predictive quality.
5. Reproduce the CSV from the extracted portable source into a different output directory using the same pinned runtime and inputs. Require matching SHA-256 hashes. Retain the original and reproduced local receipts.
6. Check that the technical report describes the model actually packaged. Record source, model, configuration and CSV hashes. Preserve the explicit failure to beat v1.5 and do not convert engineering-test success into an accuracy claim.
7. Review the concrete sanitized public-source candidate with the user before publication. Use a separate repository or export destination; do not make the existing development repository public.
8. The emailed form URL has been received. Confirm the registered team name and membership, and confirm the cutoff time/timezone. The [published rules](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/RULES.md) specify teams of three to five; GitHub collaborators are not evidence of registration.
9. Submission is expressly not authorized yet. Submit only if the user later authorizes the concrete CSV, public URL and report. Save the form confirmation separately. There is no automatic submission step.

## Rights and label-quality record

Public Figshare metadata was checked on November 14: **An integrated dataset of spatiotemporal and event data in elite soccer**, version 1, DOI `10.6084/m9.figshare.28196177.v1`, authors Manuel Bassek, Robert Rein, Hendrik Weber and Daniel Memmert, with licence URL `https://creativecommons.org/licenses/by/4.0/`. The fallback package must retain attribution and identify its feature/augmentation/model transformations. This source-rights evidence is separate from whether the provider's possession/status references are independently adjudicated truth.

The recorded Sportec source is external to the challenge game. The original provider is DFL and the data paper describes TRACAB tracking and quality control. The exact manual-versus-automatic production of every possession/status field is not separately established. No new manual labels were created. These limitations must remain in the technical document.

PFF local analytical access does not establish permission to deliver PFF-trained model artifacts or redistribute provider files. An A/B winner cannot become a submission candidate while that issue remains unresolved. The fallback's inference package contains no PFF-fitted weights and needs no PFF records to reproduce its CSV.

## Outstanding inputs and decisions

- The precise deadline time/timezone; the submission-form URL is already received.
- Confirmation that all team members registered under **Xtension Labs**, with the required team size.
- User approval of a concrete, separate sanitized public-code destination; existing repository stays private.
- A later explicit instruction to submit; the current instruction is to hold submission.
- If selecting a PFF-trained replacement: actual passing development evidence, connected lineage and resolved delivery rights. No score or eligibility is assumed in advance.

All research failures and consumed experiment attempts remain in the local ledgers. This fallback is an explicit eligibility/reproducibility contingency, not a relaxation of the numerical conditions for claiming an improvement.

## Completed local reproduction evidence

The aggregate checkpoint `runs/submission-preparation-20261114/fallback-reproduction.json` records `passed` at `2026-11-14T05:31:23Z`, independent extracted-source reproduction, identical receipts, and no reference access, final-PFF access, publication or submission. Input-specific hashes and paths remain local and are not reproduced here.

| Artifact | SHA-256 |
| --- | --- |
| Both prediction CSVs | `313f467af967cf1f471d4e37bd85feddfb37041c16e96d48816833f1851ebb26` |
| Sanitized source archive | `26bceb5aaa243884128c66542c94606377fee93761b67de19364ad3cc740cdb2` |
| Source integrity manifest | `9ff9a5400aaee73ff16e4f89562075565798838b9b00875c3a238b5002652db7` |
| Packaged technical document | `c55f06df28f25e23f50b5c997236bf22eacb28ddf8c6c81aee9f169a8a0c57c5` |

Model and configuration identities remain those specified above. The technical document was not edited after packaging. All frame predictions and fitted model files remain outside tracked repository changes.

## Independent export review

Four synthetic integrity tests passed on November 14 after review fixes. They verify the clean allowlisted export and rejection of an extra model file, frame-record file, or changed input-reader source even when the archive, manifest and local receipts are all rehashed consistently. The implementation now authenticates the fixed model/configuration and scientific source bytes independently of those mutable receipts. Runtime Git metadata is permitted only for a cloned public source repository; it remains excluded from the prepared archive. CSV validation enforces the exact header, frame sequence and label values.

The isolated review command was:

```powershell
$env:TEMP = Join-Path (Get-Location) 'runs/fallback-review-test-tmp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null
python -B -m unittest tests.test_fallback_export_integrity -v
```

This used the existing CPython 3.12 project environment. An initial invocation during concurrent builder editing stopped at a syntax error before executing tests; the owner corrected it, and the completed run passed all four tests. The review used invented model bytes and records, original code and non-reconstructive evidence only. It did not perform real-match inference, inspect fitted-weight contents, establish predictive accuracy or resolve historical eligibility.

## Timeline and fixed-configuration provenance review

A bounded review on November 14 compared code, Git history and aggregate campaign evidence; it opened no challenge records, fitted-weight contents or predictions.

The complete local `games.py` equals the [organiser's published source](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/games.py) after converting Windows CRLF line endings to LF. The local SHA-256 is `473bf0502a28d93ff49fe5c20be4e546e30a9d8fa4112ab50b1ecdaf44233064`; the fetched LF source is `3b2122794e9e0476e2b3a871d14880e5fa47c4feaeacf99caef054c2309076ce`. Thus the frame-count, period boundaries, offsets and pitch constants used here are organiser-provided. The comments about video measurement originate in that supplied source; this review found no team-added manual timeline annotation. The comparison initially encountered sandbox DNS failure; the successful public-source fetch and newline-normalized equality resolved the question without accessing any match asset.

The v1.2 configuration differs from the v1.1 configuration in exactly one value: `team_switch_cost` changed from `4.0` to `1.0`. [The v2 campaign](CONTROL_CAMPAIGN.md), `tools/train_control.py` and `models/control-selection.json` document a frozen switch-cost set of `4`, `2`, and `1`, selected on external Sportec matches `J03WN1` and `J03WMX`, followed by the three declared external final matches. The recorded selected candidate is `control-spell-balanced-l2-0.001-cost-1`; no final-set retuning is declared. The v1.1 config is SHA-256 `8d190bec7b92ae61157f6b41fbc839bc6d400e6f626908d4a6d8bdacc7dd76cf`; the fallback retains the resulting v1.2 config `2755c140130f4ef9821e7909f8da794d6e4c8d4a7691b75de77eada7c866612b`.

All other configuration values match the fixed `Config` defaults, whose implementation is unchanged since the earliest tracked v1.1 release, commit `6e7cf13`. That source declares no public-label use for those defaults; the baseline external benchmark records no configuration search. The repository does not preserve a complete pre-v1.1 history explaining each originally chosen physical constant. Historical public correctness checks and the separate AI-exposure concern remain recorded in the rules review; this audit cannot retrospectively certify every earlier action.

The available evidence supports the fallback's documented external-only selection and organiser-supplied timeline. It identifies no additional target-fitted component in these two areas. Missing pre-release history is a limit on certification, not affirmative evidence that these constants were fitted on the target. No formal organiser eligibility ruling is claimed, and the explicit submission hold remains.

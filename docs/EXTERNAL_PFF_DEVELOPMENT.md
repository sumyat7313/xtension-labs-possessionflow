# External-only model: PFF development comparison

This study evaluates the frozen external-only rebuild on the **same twelve PFF development fixtures already used in earlier research**. They are tuning/development data, not fresh final confirmation. It does not train, calibrate or modify the candidate, and cannot open the twelve reserved final fixtures or replace the submission.

The candidate is `external-joint` from `external-rebuild-001`, SHA-256 `7d79dc50cfc16a7f64c37b65908be90e588d3bfdec35f599071b25180089bf47`. Its frozen decoder configuration has SHA-256 `2755c140130f4ef9821e7909f8da794d6e4c8d4a7691b75de77eada7c866612b`. Rebuilding and scoring this candidate are separate experiments. Its earlier Sportec development gains were against a newly fitted external control model; they were not a comparison with v1.5 on PFF.

The lineage document's **file-byte SHA-256** is `3d31772a8821e869b8569563ab43fcca3b7b2afe11159072ebfd76c1620f0e8c`. File hashes and canonical JSON digests are different identities and must not be substituted for one another. Portable verification accepts only the documented LF/CRLF serializations where a saved JSON file identity is required; canonical protocol digests use the single prescribed encoding.

## Frozen comparison

The cohort is 3813, 3816, 3818, 3823, 3831, 3844, 3850, 3854, 3858, 10502, 10514 and 10517. The original metadata-only 40/12/12 reservation remains unchanged.

Each fixture receives three predictions: historical v1.5, the original PFF mixed-data parent, and the frozen external-only candidate. Historical models are comparison controls only. They do not become candidate ancestors or new training inputs. The new candidate must use the original observation features with its own external-only upstream probabilities; it must not reuse cached v1.5 control probabilities as its inference input.

The runner must finish and authenticate **all twelve fixtures' predictions before creating an exclusive reference-access marker**. It then scores the cohort once. A scoring failure consumes that attempt. It must be reported, not hidden by retrying until a successful summary appears. Existing PFF event-derived references are used only by the separate scoring step. No provider labels or frame predictions are published in the aggregate bundle.

Both historical controls must reproduce the original development metrics and reference support. Their frozen prediction identities and observation manifests must match the earlier authenticated campaign. Candidate metric records are independently re-derived from saved aggregate diagnostics. Neither a `passed` flag nor a summary mean alone establishes a passing result.

## Acceptance and limits

The numerical comparison retains every original gate, adds at least **0.002** mean stoppage-precision gain over v1.5, and requires at least **90%** retention of the original PFF parent's positive mean gains in spell F1, macro F1, short-possession precision and recall, stoppage segment F1, stoppage boundary F1, and stoppage recall. The original PFF records are pinned by digest so the reference gains cannot be redefined. All twelve paired development fixtures and adequate per-match support are required.

A numerical development pass is not permission to submit or access final fixtures. PFF dense-reference validity and fitted-model submission rights remain unresolved. The candidate itself was trained externally on Sportec; this comparison still depends on PFF evaluation rights and reference quality. `promoted`, `submission_eligible` and `final_access_authorized` remain false in the verification report regardless of the numerical outcome. The original v1.5 artifacts remain an unchanged historical comparator under their existing submission hold.

## Independent verification

`tools/verify_external_pff_development.py` has two scopes:

- **Portable aggregate verification:** authenticate the published parent/rebuild evidence, fixed candidate and configuration, complete cohort, source/protocol identities, observation manifests, prediction freeze, exclusive access marker, detailed diagnostic support, unchanged historical controls, and recomputed gate. This mode does not require restricted data or fitted PFF weights. It cannot independently establish that archived declarations describe every historical action.
- **Local artifact verification:** additionally authenticate the actual candidate, parent and configuration bytes, lineage bindings, frozen prediction arrays, scored CSV hashes, and saved run evidence. Contents remain local; only compact verification results are printed. No inference, fitting or final-fixture access occurs.

An incomplete run has a separate status result. It must never be converted into a completed evidence bundle or supplied with fabricated metrics. A reference-access marker without a completed score is reported as a consumed incomplete scoring attempt. Partial predictions may be authenticated individually, but they do not satisfy the complete-cohort freeze requirement.

The development driver keeps its run output under `runs/external-pff-development-v1/output`. Use the project Python environment from the repository root:

```powershell
# Authenticate a partial run without publishing completed evidence.
python tools/verify_external_pff_development.py --run runs/external-pff-development-v1/output --allow-in-progress

# Only after all fifteen stages have completed: authenticate actual local bytes
# and create a new aggregate bundle. Existing evidence files cannot be replaced.
python tools/verify_external_pff_development.py --run runs/external-pff-development-v1/output --write models/external-pff-development-research.json

# Portable verification needs the published evidence and tracked source only.
python tools/verify_external_pff_development.py --evidence models/external-pff-development-research.json
```

The status command distinguishes a resumable checkpoint from a consumed incomplete attempt. It does not claim that an apparent in-progress process is still alive; the driver's lock and process evidence must also be checked before starting another worker. `--allow-in-progress` cannot be combined with `--write`. A portable verification result explicitly reports that restricted local artifact bytes were not checked in that mode.

Real results belong in a subsequently generated completion report only after the run and verifier finish. This protocol document intentionally does not claim that the evaluation has completed or that the candidate beat v1.5. Synthetic tamper tests exercise verification behavior; their fixtures are not model-performance evidence.

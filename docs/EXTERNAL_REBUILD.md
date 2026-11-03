# External-only control and joint-state rebuild

This runner replaces the target-camera-derived training path with fresh Sportec observation caches and fixed synthetic camera stress. It is a bounded research procedure, not a submission or final-test command. It does not automatically replace v1.5 or claim to beat it.

## Lineage and data access

Only the seven registered Sportec/IDSSE fixtures are accepted, with their published byte sizes and MD5 checksums checked before parsing and SHA-256 recorded. The source is the [Sportec/IDSSE dataset](https://doi.org/10.6084/m9.figshare.28196177), Bassek, Rein, Weber and Memmert (2025), CC BY 4.0. Metadata/tracking inputs stay local. The only permitted data locations are this repository's `data/external/idsse` and the sibling upstream checkout's same external-data directory. Challenge directories and arbitrary fixture IDs are rejected before reading records.

The only inherited learned component is the exact v1.2 logistic live/dead head, SHA-256 `115c192cd4aa71be837e1c4bd2137d91add6dcea4b006a809b13232e62f43c0b`. Its existing control model is discarded. Its dead-head fit used J03WOH/J03WOY, as documented in `TRAINING_CAMPAIGN.md`; neither is scored. The v1.2 fixed configuration is authenticated separately. Historical external development selection is already exposed; no untouched-validation claim is made for those fixtures or parameters. Source-level lineage review remains necessary beyond hash matching.

No v1.3+ model, target-derived profile, historical feature cache, challenge video, possession ground truth, or PFF data is loaded. The v1.5 hash appears only as the identity of a future historical comparator. All 12 reserved PFF finals remain outside this command's fixture allowlist.

## Training and development comparison

Fresh caches contain 98 control features, 58 live/dead features, separate external references, period blocks and observation-support flags. Clean, legacy degraded and fixed synthetic-camera regimes are rebuilt from raw external observations. Synthetic camera parameters are the declared settings in `tools/camera_augmentation.py`; they are not estimated from the challenge.

The three scored folds are WN1/WMX, WPY/WQQ and WR9. For each fold, both the 98-feature robust control model and joint H/A/D residual are fitted on the other complete fixtures only. Control fitting samples at 2.5 Hz with training-derived spell weights. Joint fitting samples at 5 Hz, keeps each half separate, ignores X/unsupported targets and uses bounded transition weighting. The frozen dead head has external WOH/WOY supervision only.

The joint procedure is compared with the newly fitted external control procedure on all 15 fixture/regime cells. Its development screen requires mean spell-F1 gain >= .01, no cell spell-F1 decline > .02, no cell dead-recall decline > .01, and nondecreasing mean dead precision, macro F1, transition recall and short-spell recall. The control model remains the fallback. These are diagnostic gates for this rebuild, **not replacements for the stronger unchanged v1.5/PFF promotion gates**.

Selection is written before refitting the selected procedure on all seven development fixtures. That artifact has no untouched Sportec holdout. A later compliant external development comparison and the separately protected fresh-final decision are still required.

## Bounded, resumable execution

`plan` authenticates the fixed parent/configuration and returns the recipe without reading match records. `preflight` additionally verifies all registered external input bytes, without parsing their records or writing outputs. `run --stage next` executes exactly one stage: seven individual cache preparations, three grouped-fold fits/evaluations, then one final development refit. Invoke the same immutable recipe/output again to advance.

```powershell
python -m tools.external_rebuild plan --out runs/external-rebuild-v1
python -m tools.external_rebuild preflight --out runs/external-rebuild-v1
python -m tools.external_rebuild run --stage next --out runs/external-rebuild-v1
```

The optional recipe JSON has exactly these fields:

```json
{"schema_version":1,"seed":10527,"epochs":8,"control_l2":0.001,"transition_weight":1.5}
```

Epochs are restricted to 4/8/12, control L2 to .001/.01 and transition weight to 1/1.5/2. This is an admissible parameter space, not authorization to exhaustively search it: the harness must preregister each bounded family and preserve all prior results. Seeds are integers in [0, 2^31). No path, fixture, profile or model overrides can be supplied through the recipe.

Each stage creates an exclusive start marker. Errors/interruption consume that stage and require a new study ID; they are never silently retried. Completed stage files are sealed with hashes. The runner authenticates completed cache/model bytes and scientific sources before the next stage and checks sources again after execution. A source change cannot silently continue an old experiment. `summary.json` records `in_progress` or `complete`, the next stage and explicit false final-access/promotion flags. A completed run emits the fitted candidate, a full dependency manifest and actual model-file bindings checked by the independent lineage policy. All such local artifacts remain under ignored `runs/`.

Run one heavy stage at a time with BLAS/OMP thread counts set to one. Torch also sets one thread. The harness's bounded subprocess timeout is additional protection; a timeout is recorded as a consumed failure, not a successful checkpoint.

## Verification

`python -m unittest tests.test_external_rebuild -v` exercises strict recipes, fixture/path/profile rejection before reads, whole-match splits, pinned parent authentication, removal of old control heads, unknown/period masking, cache tamper detection, complete-cell selection, lineage-policy integration and success/failure checkpoint behavior. These synthetic tests verify safeguards, not model accuracy.

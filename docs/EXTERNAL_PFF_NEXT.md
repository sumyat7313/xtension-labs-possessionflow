# Next bounded study: clean-candidate PFF development comparison

**Planning checkpoint only. No evaluator has been implemented or run for this candidate.** This is the next task after the verified completion of `external-rebuild-001`; do not rerun that completed training trial.

Evaluate one fixed candidate, with no fitting, calibration, parameter grid or new threshold selection. Use exactly the already-exposed development fixtures: 3813, 3816, 3818, 3823, 3831, 3844, 3850, 3854, 3858, 10502, 10514, 10517. Reject all other IDs before I/O. Do not expose final preparation or scoring entry points.

| Identity | SHA-256 |
| --- | --- |
| Candidate | `7d79dc50cfc16a7f64c37b65908be90e588d3bfdec35f599071b25180089bf47` |
| Candidate configuration | `2755c140130f4ef9821e7909f8da794d6e4c8d4a7691b75de77eada7c866612b` |
| Candidate lineage canonical digest | `3d31772a8821e869b8569563ab43fcca3b7b2afe11159072ebfd76c1620f0e8c` |
| Rebuild protocol file | `0cd152ae7105f4561b5fa6f97f23fc71f1dd2afde49fe32301fb7d57dca6f31f` |
| Rebuild summary file | `c605d16d3550b36db281c95ecee964ffe8e97538d096f9dbef6c6e4b8d5cdb34` |
| Original PFF aggregate bundle | `46e49be1e4fcbcc75a6ae7193d73886738a76d0bdd5a622f2f9c8f1c5a980654` |
| Original parent-record canonical digest | `502bfef18b1d693392d9b22d5e22bb0d560eebab178586be6886a78be0d7a39a` |
| Development cache-manifest-map canonical digest | `c1f9e252ff447dfa8f2eb0621f10151a612bf3e41f07926caf57e977bae4587f` |

Candidate location: `runs/continuous-improvement/trials/external-rebuild-001/output/refit/candidate.json`. Original observation caches: `runs/pff-campaign-v3/cache/{id}`. The planning agent found identical configuration and all thirty shared scientific source hashes; the new preflight must independently recompute these checks.

Reuse only authenticated `x.npy`, `z.npy` and observation metadata (blocks, FPS, usable masks, match and pitch). **Never use old `base_home.npy` or `base_dead.npy` as candidate inputs.** Recompute with the clean robust control and dead heads, then use its joint model and the unchanged decoder. The old `pff_training.predict_cache` helper is unsuitable because it trusts supplied upstream probabilities. Reference reconstruction remains separate and unchanged.

Implement new versioned files such as `tools/external_pff_development.py` and `tools/verify_external_pff_development.py`, with focused tests. Do not edit the completed harness/runner's bound sources. A new strictly allowlisted orchestration integration must preserve the original ledger and recipe identity. Stages should freeze initialization, predict individual fixtures, authenticate the complete prediction cohort, perform one exclusive reference-scoring pass, then verify results. Failed or partial scoring remains consumed.

Wrap existing reservation, aggregate-bundle, reference and metric helpers with strict development membership guards. Use `pff_validation.verify_protocol`, rather than the old campaign helper that reads all raw fixture metadata. Recompute clean-candidate inference explicitly. Rescore both historical v1.5 and original PFF controls and require exact metric/support reproduction; never change reference masks to recover agreement. Original controls are evaluation comparators, not training ancestry.

Acceptance still uses `pff_precision_validation.development_gate`: all original checks, +0.002 absolute stoppage precision and 90% retention against the exact original PFF gain anchors. Apply independent lineage and evidence screening separately. PFF is evaluation-only for this experiment, so it must not be inserted into the training DAG. Benchmark-use and reference-quality limitations remain explicit. Always retain `promoted:false`, `submission_eligible:false` and `final_access_authorized:false` in this development tool.

Required tests cover recomputation instead of old upstream probabilities, final-ID rejection before I/O, exact candidate/lineage binding, complete predictions before references, cache/prediction tampering, consumed partial scoring, historical control reproduction and unresolved evidence blocking final access. Use synthetic fixtures; do not put challenge data into tests or AI context.

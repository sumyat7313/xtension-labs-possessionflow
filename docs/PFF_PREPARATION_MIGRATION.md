# PFF observation-cache reuse before fitting

On 2026-11-03, the first v2 training attempt stopped while parsing training references. No fitted candidate or held-out reference access existed. Two training-only schema findings required a correction inside `references()`: explicit null uncertainty flags must mask certainty, and the provider sometimes reuses an event ID for an OTB record and a distinct FOUL record. The corrected parser was checked against all 40 training fixtures before proceeding. Development and final labels were not used to make these corrections.

The v2 run was preserved. `tools/pff_migrate_preparation.py` copied its 52 prepared observation caches into a fresh `runs/pff-campaign-v3` run. These are the 40 training and 12 development fixtures reserved before label inspection. The final 12 fixtures remain outside this migration.

The migration verified every original manifest and all 364 artifact files before creating v3, then verified every copied file. No observations, features, baseline predictions, alignment records or item descriptions were recomputed or altered. Copies are independent files, not hardlinks. Each new manifest explicitly records `inherited_preparation`, including the complete original manifest, its hash, the old and new adapter hashes, the old frozen run identity and the unchanged observation-code AST digest.

All scientific source hashes except `tools/pff_adapter.py` were identical. The helper independently compared the complete adapter AST after replacing only the body of the top-level `references()` function with a placeholder. Imports, helpers, observation handling, the reference function signature, decorators and default arguments therefore had to remain identical. The unchanged v1.5 decoder configuration was also authenticated. Existing fitted models, fit records, held-out predictions or reference-access markers would have blocked reuse.

| Evidence | SHA-256 |
| --- | --- |
| Preserved v2 adapter | `432317243dc656ba14427337eb507808be63cc4c6942aebb6f0fb142c7036a5e` |
| Corrected v3 adapter | `52fbbc13347261230389b048eb1108b801077f7cd84058fdd470353ec9493f15` |
| Adapter AST outside the reference body | `140f2591364414a9249e312f2c57acc7053bf4a83af1afb736e69207b7deb814` |
| Original v2 run protocol | `8b8e2e1365b5a3f7d71bdd708be0a84b359b2a3911de9e048c1c4f0047d060f5` |
| New v3 run protocol | `218cd91cb35b27671d56f344969db097dcb347fdd0d305b1b708c2c85e2afb18` |
| Migration proof | `826a5013153ce19c76918d25724607f5e7b47831d4ebc6bf5418cd85aa062394` |

The detailed proof is local in `runs/pff-campaign-v3/preparation-migration.json`, with completion evidence beside it. The aggregate research verifier embeds this code-only proof, including the original adapter source, in any published evidence bundle that contains inherited caches. It verifies the old source hash, AST equivalence, unchanged non-adapter sources, original manifest hashes, exact copied artifacts and full inherited cohort. Neither this proof nor the aggregate bundle contains provider tracking records or reference labels.

This is a documented parser correction before evaluation, not a new independent experiment on already scored matches. The fixed whole-match reservation, candidate training policy, prediction-before-reference rule and requirement to retain v1.5 unless stronger evidence exists remain in effect. PFF references remain provider-derived silver labels; cache reuse does not strengthen their truth status or resolve deployment rights.

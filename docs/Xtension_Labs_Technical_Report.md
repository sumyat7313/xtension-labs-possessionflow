# Xtension Labs — PossessionFlow external-only fallback

Prepared November 14, 2026, for the More than a Hack Initial Challenge. This document describes the fixed Sportec-trained fallback, identified below. It does not describe a newly selected PFF-trained model. The fallback has not demonstrated an improvement over the historical v1.5 comparator, and no hidden-challenge accuracy or winning probability is claimed.

## Approach and supplied assets

The system assigns `H` (home possession), `A` (away possession), `D` (dead ball), or `X` (no answer) to every tactical-video frame. It combines geometric observations with a learned home/away control classifier, a learned live/dead classifier, a temporal residual model and duration-constrained decoding. This separates team control from stoppage evidence while allowing temporal context to correct both.

Inference reads the supplied tactical tracking metadata, tactical automatic tracking and automatic Smart Tagging. It uses pitch-normalized positions, team membership, player-motion summaries, ball support and the published period/frame alignment. Player slots may disappear or change identity; the method does not assume stable player identifiers. Short supported ball gaps can be interpolated; unsupported gaps remain explicitly represented in the features.

Automatic offensive and defensive tags provide bounded team evidence, with defensive tags referring to the other side's possession. Set-piece tags provide context for a stationary-to-moving ball cue. A tag interval's edge is not copied directly into a possession boundary. The tag rules are fixed; they are not fitted to the supplied game.

The delivered fallback does not read either supplied video, broadcast tracking or standalone homography files. It does not include video recovery, a challenge-fitted camera profile, a typed-boundary model or a separate learned transition model. Earlier experimental components are not dependencies of the selected model's training ancestry.

## External training data and supervision

The fitted components use the public **An integrated dataset of spatiotemporal and event data in elite soccer**, by Manuel Bassek, Robert Rein, Hendrik Weber and Daniel Memmert (2025), version 1, [DOI 10.6084/m9.figshare.28196177.v1](https://doi.org/10.6084/m9.figshare.28196177.v1). Its Figshare metadata lists [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/). The original provider is DFL; positions were collected using TRACAB optical tracking. No raw external match files are included in the source export.

The seven fitting fixtures are `J03WMX`, `J03WN1`, `J03WOH`, `J03WOY`, `J03WPY`, `J03WQQ` and `J03WR9`. Fourteen metadata/tracking XML files are authenticated by the reconstruction protocol. Provider `BallStatus` and `BallPossession` fields supply the reference labels; they are not inference features. The accompanying [dataset paper](https://www.nature.com/articles/s41597-025-04505-y) defines the fields and describes collection quality control, but does not separately establish whether every possession/status value was manually annotated or algorithmically derived. We therefore describe these as provider references, not independently adjudicated human-gold labels. We created no new hand labels and did not use the separate manually annotated event feed for fitting this fallback.

The 58-feature logistic stoppage head is inherited only from the audited external v1.2 ancestor. It was fitted on `J03WOH` and `J03WOY`: 279,750 known reference frames at 25 Hz, sampled to 55,951 unique time samples at 5 Hz; clean and deterministically corrupted copies produced 111,902 fitted rows. This ancestor's previous team-control model was discarded.

The new 98-feature symmetric home/away classifier and joint temporal residual were fitted using the seven fixtures. Clean, degraded and fixed synthetic-camera observation regimes share the external references. The camera-corruption parameters were declared independently of the challenge game. Control fitting uses 2.5 Hz sampling and training-derived spell weights. Joint fitting uses 5 Hz samples within each half, excludes unknown targets from the loss, and increases weight around external reference changes. The declared reconstruction recipe uses seed `10527`, eight joint epochs, control regularization `0.001` and transition weight `1.5`. Scaling is learned from the fitting observations only; inference never refits or normalizes against its prediction batch.

No PFF-fitted weights, challenge-derived training caches, challenge video, challenge possession labels or target-fitted camera statistics enter this fallback's training ancestry. PFF adaptation experiments are separate research and do not change this package automatically.

## Possession rules and temporal decoding

Home/away possession represents team control while the ball is live. A nearest-player observation is evidence, not a hard rule assigning every loose ball to the nearest team. Control geometry, recent observation context and the temporal model inform contested and missing-ball intervals. Temporal persistence limits unsupported switching, but may carry a team label too far through a genuine change.

`D` represents a stoppage, without an owning team. Missing ball or player observations are not by themselves a definitive stoppage rule; their support indicators enter the learned evidence. Restarts combine context with observed motion. The method cannot directly observe a whistle and may confuse slow live play with stopped play.

The fixed decoder finds a joint `H/A/D` sequence separately within each published half, using transition costs and minimum state durations. It does not carry inferred state across the period boundary. Frames outside the defined play blocks remain `X`; a whole block lacking usable ball and tactical evidence also returns `X`. Unknown output is not treated as an accuracy claim. In the official ranking metric, abstentions are filled within the half before possession matching, so they do not remove difficult possessions from evaluation.

## Evaluation and limitations

The external reconstruction compared newly fitted control and joint procedures on five already-exposed Sportec fixtures under three correlated observation regimes. The joint procedure achieved mean spell F1 `0.616227`, versus `0.477627` for the newly fitted control procedure, before the all-seven-fixture refit. Those fifteen cells represent five matches, not fifteen independent matches. The historical stoppage/configuration development had already exposed some of these fixtures; these results are development diagnostics and do not establish fresh-test accuracy.

A later cross-dataset development comparison failed the project's unchanged improvement criteria. The fallback is preserved for its audited external-only ancestry and reproducibility, not presented as a performance winner. It has no untouched Sportec holdout. The twelve reserved PFF final fixtures remain outside the fallback workflow and are not needed to regenerate its CSV.

The principal weaknesses are false stoppages under severe missing observations, over-fragmented short possessions, camera-domain differences, and ambiguity in provider reference definitions. Fixed duration constraints can suppress real brief control changes. The method does not resolve every deflection, challenge or referee stoppage. No output was selected or corrected by comparing it with the challenge video; challenge labels and manually produced annotations are not used for fitting or tuning this package.

## Reproduction and artifact identity

The portable source export includes pinned dependencies, the fixed model/configuration, their provenance, the inference source closure and an input-independent integrity manifest. It excludes Git history, supplied match assets, reference labels, research caches, frame-level diagnostics and PFF model artifacts. Users must obtain the challenge data through their own authorized access.

From the extracted source directory, use CPython 3.12 and run:

```bash
python -m pip install -r requirements.txt
python run_fallback.py --data-dir /path/to/authorized/data --out ../reproduced
python run_fallback.py --validate "../reproduced/Xtension Labs.csv"
```

The data directory must contain the supplied `tactical` folder with the original tracking metadata, tracking and Smart Tagging filenames. The runner performs frozen inference, not training. Output must be outside the extracted source directory, so predictions and receipts cannot enter the public source inventory. It writes `Xtension Labs.csv` with exactly 199,319 ordered frames (`0` through `199318`) and the header `frame,label`. Structural validation does not load ground truth or measure predictive quality. Local reproduction receipts record hashes so the exact deliverable can be compared across independent runs; predictions and input-specific receipts are kept out of the public source export.

The model bytes are pinned to SHA-256 `7d79dc50cfc16a7f64c37b65908be90e588d3bfdec35f599071b25180089bf47`; the configuration is pinned to `2755c140130f4ef9821e7909f8da794d6e4c8d4a7691b75de77eada7c866612b`. Replacing either creates a different candidate and requires an updated report and validation. The submission CSV's exact hash is established by the local build receipt, not guessed in this document.

## Attribution and delivery scope

The original challenge software and documentation are credited to Metrica Sports under the supplied MIT licence. Dataset attribution is to Bassek, Rein, Weber and Memmert and the DFL source described above; the fallback transforms the external observations into features, synthetic missingness regimes and learned model coefficients. No endorsement by the dataset authors or organisers is implied. See the source export's `ATTRIBUTION.md` and `LICENSE` for retained notices.

The team keeps its existing development repository private. A separately reviewed, sanitized public code export is required by the published submission instructions; preparing this document does not publish that export or submit a competition entry. The overriding organiser clarification requiring external-only training/tuning governs the package, even where older repository wording differs.

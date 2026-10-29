# Xtension Labs — PossessionFlow 1.2

Local team-possession inference for More than a Hack 2027 Phase 1. The current
release is **control-spell-balanced-l2-0.001-cost-1**. Model selection used other matches with independent
validation and a final test; it did not use the competition public labels.

Read [the technical report](docs/TECHNICAL_REPORT.md),
[training results](docs/CONTROL_CAMPAIGN.md), and
[independent model review](docs/REVIEW_V2.md).

## Run the submission

Use Python 3.12. No GPU, API key, network call, or retraining is needed for inference.

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements-solution.txt
python -m unittest discover -s tests -v
```

Place these organiser-provided assets under `data/tactical/`:

- `BAR-ATH_tactical_atd.xml`
- `BAR-ATH_tactical_atd.txt`
- `BAR-ATH_tactical_smart_tagging.xml`

Videos, broadcast and homographies are not used by this release. Do not redistribute
competition assets. The source archive excludes them and all raw external data.

```bash
python -m possession doctor --data-dir data
python -m possession predict --data-dir data --config configs/default.json --model models/submission.json --out runs/xtension-labs
python -m possession validate runs/xtension-labs/predictions.csv
python -m possession bundle --run-dir runs/xtension-labs --team-name "Xtension Labs" --out dist/xtension-labs
```

On Windows, `./run-challenge.ps1 -Action Predict` uses the packaged selected model
when `models/submission.json` is present. Supply `-PythonPath` if your environment
is elsewhere. The launcher defaults to `runs/xtension-labs` and team Xtension Labs.

Each run saves predictions, config, model snapshot if used, provenance, aggregate
diagnostics and predicted intervals. Packaging verifies that these agree. Extract
the source ZIP and repeat the command with the same assets/environment to reproduce
the exact CSV. A learned model with a mismatched feature schema fails explicitly.

Omitting `--model` from the Python prediction command reproduces the original
heuristic method. It is retained as a benchmark, not silently selected when a
requested model is missing. `--tracking-only` explicitly disables tactical tags.
`MTAH_DATA_DIR` may supply the directory containing `tactical/`.

## Training and evidence

```bash
python tools/train_control.py --download --out runs/campaign-reproduction
python tools/audit_observations.py --output runs/observation-audit.json
```

External data is the [IDSSE/Sportec dataset](https://doi.org/10.6084/m9.figshare.28196177)
by Bassek, Rein, Weber and Memmert (2025), CC BY 4.0. Its references are used only
as targets or scores, never inference features. See the model and selection JSON
for training provenance and the technical report for transformations and attribution.

The 15-candidate control campaign uses two whole training matches, two development
matches and three fresh final matches. J03WMX is now development data. Clean and deterministic degraded inputs are compared separately.
The public five-minute competition labels are not a tuning or model-selection set.
`python -m possession evaluate` remains available for correctness checks, but it
was not used to select or retune the current release. Historical public scores
must not be presented as scores for changed predictions.

Tests cover parsing/alignment, physical scaling, half isolation, feature symmetry,
unknown-target exclusion, train-only scaling, deterministic fitting and model
serialization, exact temporal decoding, validation selection gates, label leakage,
model snapshots and data-excluding packaging. A full synthetic demonstration is
available through `python -m possession demo`; it cannot be packaged as a submission.

## Submission status

The prepared team name is **Xtension Labs**. Local files do not constitute a form
submission. A public source repository URL and the organiser's emailed Google Form
are still required; team eligibility must also be confirmed. The published deadline
is November 17, 2026, without a time/timezone in the repository.

No hidden-match score or winning rank is claimed. External tracking is cleaner
than the competition recording; occlusions, false tags, aerial duels and referee
decisions remain limitations. The organiser's original licence is retained in
`LICENSE`; the external data and derived trained model require their CC BY attribution.

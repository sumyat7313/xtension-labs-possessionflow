# Xtension Labs — PossessionFlow 1.3.0

Selected **camera-context** and refitted its control head on all seven public matches.

See [the technical report](docs/TECHNICAL_REPORT.md), [current development evidence](docs/ROBUST_CAMPAIGN.md), and [historical v1.2 evidence](docs/CONTROL_CAMPAIGN.md).

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
python tools/train_robust.py --data-dir data/external/idsse --out runs/campaign-reproduction
```

The runner verifies seven public IDSSE/Sportec filesets and records its fixed protocol before fitting. All matches are previously exposed development data. Whole scored matches are excluded from their control fit; the frozen dead head's training matches are not scored. No competition reference labels are loaded. See the campaign document for CC BY attribution, fixed candidates and release gates.

## Submission status

Team: **Xtension Labs**. The repository remains private. Prepared files are not a form submission. The organizer's public-source requirement, emailed submission form and team eligibility remain outstanding. The published deadline is November 17, 2026, without a time/timezone in the repository.

Hidden competition accuracy and ranking are unknown.

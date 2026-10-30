# Xtension Labs — PossessionFlow 1.4.0

Promoted **gated-temporal** after the fixed development gates and separate sparse audit.

## Reproduce the prediction

Use Python 3.12 and the three authorized tactical assets described in the README. Prediction uses no network call, GPU, API key, video or homography file. Homographies are used only by the optional training-profile experiment.

```bash
python -m pip install -r requirements-solution.txt
python -m unittest discover -s tests -v
python -m possession predict --data-dir data --config configs/default.json --model models/submission.json --out runs/reproduction
python -m possession validate runs/reproduction/predictions.csv
```

`python -m possession doctor --data-dir data` checks inputs. On Windows, `./run-challenge.ps1 -Action Predict` selects the packaged model; use `-PythonPath` to choose your environment. `MTAH_DATA_DIR` may supply the directory containing `tactical/`.

Each run saves the model, configuration, diagnostics, prediction intervals and manifest. Bundle a real validated run with:

```bash
python -m possession bundle --run-dir runs/reproduction --team-name "Xtension Labs" --out dist/xtension-labs
```

Omitting `--model` runs the original heuristic. A requested model with an incompatible schema fails explicitly. `--tracking-only` disables automatic tags. The source bundle excludes competition assets, raw external data, video weights and local feature caches.

## Optional training campaign

The research environment additionally uses `requirements-training.txt` (pinned CPU PyTorch and SciPy). The calibrated profile requires authorized tactical tracking and `BAR-ATH_tactical_homography.csv`. It remains local under `runs/` and contains aggregate camera footprints, not target possession labels.

```bash
python -m pip install -r requirements-training.txt
python tools/calibrated_camera.py --data-dir data --out runs/calibrated-reproduction/profile.json
python tools/train_multimodal.py --data-dir data/external/idsse --previous runs/robust-campaign-v3 --profile runs/calibrated-reproduction/profile.json --out runs/multimodal-reproduction --workers 2
```

The campaign command requires the verified **original v3 local campaign artifacts**: caches, out-of-fold heads, protocol and summary. They are not included in the source archive because the caches contain raw-data-derived observations. Running the old trainer under modified v4 sources is not an equivalent historical artifact and the provenance check will reject it. The archive preserves v1.3 source and weights; reconstructing the historical campaign inputs is a separate prerequisite. The packaged final prediction needs none of these caches or training dependencies.


The data source is Bassek, Rein, Weber and Memmert (2025), [IDSSE/Sportec](https://doi.org/10.6084/m9.figshare.28196177), CC BY 4.0. Seven matches total approximately 2.62 GB. No target reference labels enter the campaign. All seven matches were previously exposed, so the resulting scores are development CV.

See [v4 evidence](docs/MULTIMODAL_CAMPAIGN.md), [independent-data evaluation](docs/INDEPENDENT_DATA.md), and [inactive video research](docs/VIDEO_RECOVERY.md). Video research dependencies are separate in `requirements-video.txt`; inference does not import them.

## Submission status

Team: **Xtension Labs**. The repository remains private at the user's request. Prepared files are not a submitted competition form. The organizer's public-source requirement, emailed submission form and team eligibility remain outstanding. Hidden competition scores, ranking and winning probability are unknown. The repository is [likalight/xtension-labs-possessionflow](https://github.com/likalight/xtension-labs-possessionflow).

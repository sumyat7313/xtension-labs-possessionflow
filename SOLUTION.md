# Xtension Labs — PossessionFlow 1.5.0

Promoted **joint-state** after both the fixed development and fresh-match gates passed.

## Reproduce the prediction

Use Python 3.12 and the three authorized tactical assets described in the README. Production prediction uses no network call, video, GPU or API key. The packaged frozen weights are sufficient; training caches are unnecessary for CSV reproduction.

```bash
python -m pip install -r requirements-solution.txt
python -m unittest discover -s tests -v
python -m possession predict --data-dir data --config configs/default.json --model models/submission.json --out runs/reproduction
python -m possession validate runs/reproduction/predictions.csv
```

`python -m possession doctor --data-dir data` checks the inputs. On Windows, `./run-challenge.ps1 -Action Predict` uses the packaged model; `-PythonPath` selects the interpreter. `MTAH_DATA_DIR` may supply the directory containing `tactical/`.

Each run records its model, configuration, diagnostics, intervals and reproducibility manifest. Package a real validated run with:

```bash
python -m possession bundle --run-dir runs/reproduction --team-name "Xtension Labs" --out dist/xtension-labs
```

Omitting `--model` runs the original heuristic. Incompatible model schemas fail explicitly. `--tracking-only` disables automatic tags. The source archive excludes raw data and video weights.

## Optional research replay

Optional research uses `requirements-training.txt`, which pins CPU PyTorch and SciPy. The v5 campaign requires the verified **original v4 local artifacts**: full feature caches, fold models, protocol, summary and the aggregate calibrated-camera profile. These caches contain raw-data-derived observations and are excluded from the repository. The pinned v1.4 source archive authenticates the historical procedure. Rebuilding those historical inputs is a separate prerequisite; running the old trainer under changed sources is not an equivalent provenance record.

```bash
python -m pip install -r requirements-training.txt
python tools/prepare_joint_cache.py --data-dir data/external/idsse --previous runs/multimodal-campaign-v4 --profile runs/calibrated-camera/profile.json --out runs/joint-cache-reproduction --workers 2
python tools/train_joint_campaign.py --previous runs/multimodal-campaign-v4 --cache runs/joint-cache-reproduction --out runs/joint-campaign-reproduction --workers 2
```

Cache preparation reconstructs the 58 observation-only live/dead features and requires exact agreement with every historical 98-feature control matrix and frozen dead prediction. The campaign compares four candidates in 20 match/condition cells each. Sportec references are used only within their assigned training or evaluation folds; all seven Sportec matches were previously exposed development data. Neither fresh provider enters training.

To reproduce the **already consumed** fresh audit, obtain the pinned provider samples and use the frozen experimental candidate produced above:

```bash
python tools/fresh_benchmark.py download --data-dir runs/fresh-audit-reproduction-data
python tools/fresh_benchmark.py evaluate-pair --data-dir runs/fresh-audit-reproduction-data --model runs/joint-campaign-reproduction/release-candidate-model.json --baseline-model models/baselines/v1.4.json --config runs/joint-campaign-reproduction/release-candidate-config.json --baseline-config configs/baselines/v1.4.json --out runs/fresh-audit-reproduction
```

Creating a new directory does not make these published audit references fresh again. The evaluator freezes models, configuration and source before access, saves all six prediction CSVs before separate scoring, and refuses to reuse a consumed local audit directory. SkillCorner tracking records contain embedded possession fields; the observation adapter whitelists geometry/detection fields and never accesses possession/phase/state for inference. Embedded fields are nevertheless parsed with the JSON records, so the exclusive audit marker is created before final tracking reads.


External training labels are from Bassek, Rein, Weber and Memmert (2025), [IDSSE/Sportec](https://doi.org/10.6084/m9.figshare.28196177), CC BY 4.0: seven matches, approximately 2.62 GB. All seven were exposed before v5. The target's public five-minute reference labels are not used for training or tuning.

Metrica Game 3 supplies sparse pass-origin labels from provider events; the annotation method is unverified, so these are not described as human ground truth. SkillCorner supplies dense H/A labels only inside its automatic phase intervals. Unlabelled gaps remain X and are never interpreted as dead ball. D/X predictions at known H/A points count as errors. These checks do not establish fresh human-labelled dense H/A/D accuracy, possession-spell F1 or stoppage recall. The three audit matches are now exposed and cannot be reused as untouched tuning targets.

See [v5 evidence](docs/JOINT_CAMPAIGN.md) and `models/fresh-audit.json`. The exact v1.4 source and model are preserved under `models/baselines/`. Optional video research remains inactive and has separate dependencies in `requirements-video.txt`.

## Submission status

Team: **Xtension Labs**. The repository remains private under `likalight`, as requested. Prepared files are not a submitted competition form. The organizer's public-source requirement, emailed form and team eligibility remain outstanding. Hidden competition scores, ranking and winning probability are unknown.

# Reproduce Xtension Labs — PossessionFlow 1.5.1

Retained v1.5: experimental combined did not pass both fixed gates (development=False, fresh=False).

Use Python 3.12. Inference requires only requirements-solution.txt; no GPU or external API.

```bash
python -m pip install -r requirements-solution.txt
python -m possession predict --data-dir data --config configs/default.json --model models/submission.json --out runs/reproduction
python -m possession validate runs/reproduction/predictions.csv
python tools/verify_release.py
```

Place authorized BAR-ATH_tactical_atd.xml, BAR-ATH_tactical_atd.txt and BAR-ATH_tactical_smart_tagging.xml under data/tactical/. The files remain local. `python -m possession doctor --data-dir data` validates inputs. `python -m possession demo` creates synthetic demonstration data only. On Windows the packaged run-challenge.ps1 accepts an explicit PythonPath.

```bash
python -m unittest discover -s tests -v
python tools/package_temporal_v6.py --run runs/reproduction --out dist/reproduction --team "Xtension Labs"
```

The v6 packaging wrapper adds new research evidence to the frozen inference source allowlist. Optional video tests require requirements-video.txt; four optional training tests require CPU PyTorch from requirements-training.txt. Core inference does not require either.

## Research reproduction

The fixed campaign requires original v4/v5 caches and fold models from authorized external data. These raw-data-derived files remain local. The archived baseline sources authenticate the historical procedure; changing an old trainer and rerunning it is not equivalent provenance.

```bash
python -m pip install -r requirements-training.txt
python tools/train_temporal_v6.py --previous runs/multimodal-campaign-v4 --previous-v5 runs/joint-campaign-v5 --cache runs/joint-cache-v5 --out runs/temporal-reproduction --workers 2
python tools/fresh_benchmark_v6.py download --data-dir runs/fresh-v6-reproduction-data
python tools/fresh_benchmark_v6.py evaluate-pair --data-dir runs/fresh-v6-reproduction-data --model models/experiments/v6-candidate.json --baseline-model models/baselines/v1.5.json --config configs/default.json --out runs/fresh-v6-reproduction
```

The last command reproduces an already consumed audit. A new directory does not make these sources untouched. Fresh results must never trigger candidate retuning or runner-up searches. See docs/TEMPORAL_CAMPAIGN_V6.md and docs/FRESH_DATA_V6.md for limitations, fixed gates and outcome.

Submission is prepared locally and stored privately; the organizer's form has not been submitted.

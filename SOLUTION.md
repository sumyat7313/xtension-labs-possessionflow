# Reproduce Xtension Labs — PossessionFlow 1.5.2

Retained the original v1.5 joint-state model and configuration. V7 adds completed research and reproducible evidence; phase-only fresh labels cannot validate stoppage or short-possession improvements or authorize replacement.

Use Python 3.12. Production inference needs only requirements-solution.txt, NumPy on CPU, and authorized challenge inputs.

```bash
python -m pip install -r requirements-solution.txt
python -m possession predict --data-dir data --config configs/default.json --model models/submission.json --out runs/reproduction
python -m possession validate runs/reproduction/predictions.csv
python tools/verify_release.py
```

Place authorized BAR-ATH_tactical_atd.xml, BAR-ATH_tactical_atd.txt and BAR-ATH_tactical_smart_tagging.xml in data/tactical/. `python -m possession doctor --data-dir data` checks inputs; `python -m possession demo` uses synthetic demonstration data only. The Windows run-challenge.ps1 wrapper accepts an explicit PythonPath.

```bash
python -m pip install scipy==1.18.1
python -m unittest discover -s tests -v
python tools/package_boundary_v7.py --run runs/reproduction --out dist/reproduction --team "Xtension Labs"
```

The typed-boundary tests require SciPy 1.18.1, installed above. Optional neural training tests require CPU PyTorch from requirements-training.txt; optional video tests use requirements-video.txt (CI uses OpenCV 5.0.0.93). These test dependencies do not activate video or change NumPy-only CPU production inference. The v7 packaging wrapper adds explicit research evidence to the source allowlist and excludes restricted raw data/local runs.

## Research reproduction

Original authorized v4/v5 caches, whole-match fold models and checksum-verified historical archives are required. They are not published. Rebuilding an old trainer with changed code does not reproduce its original provenance.

```bash
python -m pip install -r requirements-training.txt
python tools/train_boundary_campaign_v7.py --previous runs/multimodal-campaign-v4 --previous-v5 runs/joint-campaign-v5 --cache runs/joint-cache-v5 --out runs/boundary-v7-reproduction --workers 2
python tools/contact_restart_v7.py --out runs/contact-v7-reproduction
```

The sealed phase audit used `tools/fresh_benchmark_v7.py evaluate-pair`, the frozen models/experiments/v7-candidate.json, models/baselines/v1.5.json, baseline configuration and two checksum-pinned matches. Its final-reference access marker intentionally prevents rerunning the source directory as an untouched test. A copied directory does not restore freshness. Consult docs/FRESH_DATA_V7.md before any historical replay; never retune or try a runner-up against these consumed references.

The complete fixed protocol, measured comparison and rejected reliability claims are in docs/BOUNDARY_CAMPAIGN_V7.md. Submission is prepared locally; the official organizer form has not been submitted.

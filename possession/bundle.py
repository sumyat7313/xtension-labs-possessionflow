"""Explicit source allowlist prevents restricted data entering the public bundle."""

import json
from pathlib import Path
import re
import zipfile

from games import GAME, GAMES
from .io import sha256, write_json
from .pipeline import ROOT, code_digest, validate_csv


def source_files():
    fixed = [".gitignore", "LICENSE", "games.py", "requirements-solution.txt", "SOLUTION.md",
             "run-challenge.ps1", "docs/verification.json", "docs/COMPETITIVE_AUDIT.md",
             "docs/ALGORITHM_AUDIT.md", "docs/EXPERIMENT_PROTOCOL.md", "docs/EXTERNAL_BENCHMARK.md",
             "tools/audit_observations.py", "tools/external_benchmark.py",
             "tools/train_stoppages.py", "docs/TRAINING_CAMPAIGN.md", "docs/MODEL_REVIEW.md",
             "configs/default.json", "docs/TECHNICAL_REPORT.md", "docs/BUILD_STATUS.md",
             "CHALLENGE.md", "DATA.md", "EVALUATION.md", "RULES.md", "SUBMISSION.md", "FAQ.md", "OVERVIEW.md",
             "evaluation/possession_metrics.py", "evaluation/run_evaluation.py", "evaluation/validate_submission.py",
             "evaluation/gt/BAR-ATH_possession_gt_public.csv"]
    paths = [ROOT / name for name in fixed]
    paths += list((ROOT / "possession").glob("*.py"))
    paths += list((ROOT / "tests").glob("*.py"))
    if (ROOT / "models/selection.json").is_file():
        paths.append(ROOT / "models/selection.json")
    for name in ("tools/train_control.py", "tools/control_diagnostics.py", "tools/verify_release.py",
                 "docs/RESEARCH_V2.md", "docs/REVIEW_V2.md", "docs/CONTROL_CAMPAIGN.md",
                 "models/baselines/v1.1.json", "configs/baselines/v1.1.json", "models/control-selection.json",
                 "tools/train_robust.py", "tools/camera_augmentation.py", "docs/ROBUST_CAMPAIGN.md",
                 "models/baselines/v1.2.json", "configs/baselines/v1.2.json", "models/baselines/v1.2-source.zip",
                 "models/robust-selection.json", "models/baselines/v1.3.json", "configs/baselines/v1.3.json",
                 "models/baselines/v1.3-source.zip", "models/multimodal-selection.json", "models/independent-audit.json",
                 "requirements-training.txt", "requirements-video.txt", "tools/calibrated_camera.py",
                 "tools/train_sequence.py", "tools/train_multimodal.py", "tools/independent_benchmark.py",
                 "tools/recover_video_ball.py", "docs/INDEPENDENT_DATA.md", "docs/VIDEO_RECOVERY.md",
                 "docs/MULTIMODAL_CAMPAIGN.md", "models/baselines/v1.4.json", "configs/baselines/v1.4.json",
                 "models/baselines/v1.4-source.zip", "models/joint-selection.json", "models/fresh-audit.json",
                 "tools/train_joint.py", "tools/train_boundary.py", "tools/train_joint_campaign.py",
                 "tools/prepare_joint_cache.py", "tools/fresh_benchmark.py", "docs/FRESH_DATA.md",
                 "docs/JOINT_CAMPAIGN.md"):
        if (ROOT / name).is_file():
            paths.append(ROOT / name)
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Source bundle incomplete: {missing}")
    return sorted(paths, key=lambda p: p.relative_to(ROOT).as_posix())


def make_bundle(destination, team_name="possessionflow", run_dir=None):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _-]{0,63}", team_name) or team_name != team_name.strip():
        raise ValueError("Team filename must use 1-64 letters, digits, spaces, underscores, or hyphens without edge spaces")
    dest = Path(destination).resolve()
    # Check the run before writing anything that might be mistaken for a submission.
    manifest = None
    model_file = None
    if run_dir:
        run = Path(run_dir).resolve()
        if (run / ".prediction-run.json").exists():
            raise ValueError("Prediction run was interrupted; regenerate it before packaging")
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("synthetic") is not False:
            raise ValueError("Synthetic or unverified demo runs cannot be packaged as a submission")
        prediction = run / "predictions.csv"
        validate_csv(prediction, GAMES[GAME]["n_video_frames"])
        if sha256(prediction) != manifest["output"]["sha256"]:
            raise ValueError("Prediction CSV changed since it was generated")
        if sha256(run / "config.json") != manifest["config_sha256"]:
            raise ValueError("Run configuration changed since inference")
        if manifest["code_sha256"] != code_digest():
            raise ValueError("Inference code changed: regenerate predictions before packaging")
        if "learned_model" in manifest:
            model_file = run / "model.json"
            if sha256(model_file) != manifest["learned_model"]["sha256"]:
                raise ValueError("Trained model changed since inference")
    files = source_files()
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / f"{team_name}_source.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as handle:
        for path in files:
            relative = path.relative_to(ROOT).as_posix()
            info = zipfile.ZipInfo(relative, (2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            handle.writestr(info, path.read_bytes())
        info = zipfile.ZipInfo("README.md", (2026, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        handle.writestr(info, (ROOT / "SOLUTION.md").read_bytes())
        if model_file is not None:
            info = zipfile.ZipInfo("models/submission.json", (2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            handle.writestr(info, model_file.read_bytes())
    (dest / f"{team_name}.md").write_bytes((ROOT / "docs" / "TECHNICAL_REPORT.md").read_bytes())
    if manifest:
        (dest / f"{team_name}.csv").write_bytes(prediction.read_bytes())
        (dest / "config.json").write_bytes((run / "config.json").read_bytes())
        write_json(dest / "manifest.json", manifest)
        if model_file is not None:
            (dest / "model.json").write_bytes(model_file.read_bytes())
    remaining = [
        "Confirm registration, team eligibility, and the registered team name used for filenames",
        "Publish the source as a public repository and record its URL",
        "Upload the exact validated CSV and technical document with the repository URL to the official submission form",
    ]
    if not manifest:
        remaining[:0] = ["Run on actual match assets", "Generate and validate the match CSV"]
    write_json(dest / "bundle-status.json", {
        "source_archive": archive.name, "source_sha256": sha256(archive),
        "technical_document": f"{team_name}.md", "includes_submission_csv": bool(manifest),
        "team_name": team_name,
        "local_deliverables_ready": bool(manifest),
        "submission_ready": False,
        "remaining": remaining,
        "publication": "Source archive is prepared locally; no public repository has been created",
    })
    return archive

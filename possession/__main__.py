"""python -m possession: doctor, predict, demo, validate, evaluate, bundle."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from games import GAME, GAMES, asset_paths
from .config import load_config
from .io import sha256, write_json
from .pipeline import ROOT, predict, validate_csv


def main(argv=None):
    parser = argparse.ArgumentParser(description="PossessionFlow: local, deterministic team-possession inference")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Check for the three required tactical input files")
    doctor.add_argument("--data-dir")
    infer = commands.add_parser("predict", help="Generate full-match predictions and a reproducibility manifest")
    infer.add_argument("--data-dir")
    infer.add_argument("--out", default="runs/possessionflow")
    infer.add_argument("--config")
    infer.add_argument("--model", help="Trained stoppage-model JSON; omit to reproduce the original heuristic")
    infer.add_argument("--tracking-only", action="store_true", help="Explicitly omit Smart Tagging")
    demo = commands.add_parser("demo", help="Run a clearly marked full-size synthetic engineering fixture")
    demo.add_argument("--work-dir", default="runs/synthetic-demo")
    validate = commands.add_parser("validate", help="Apply the official CSV contract")
    validate.add_argument("csv")
    evaluate = commands.add_parser("evaluate", help="One public-sample correctness check; never use for tuning")
    evaluate.add_argument("--run-dir", default="runs/possessionflow")
    bundle = commands.add_parser("bundle", help="Package source and technical document; optionally a real run")
    bundle.add_argument("--out", default="dist/possessionflow")
    bundle.add_argument("--team-name", default="possessionflow")
    bundle.add_argument("--run-dir")
    commands.add_parser("config", help="Print the fixed default configuration")
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            paths = {k: Path(v) for k, v in asset_paths(data_dir=args.data_dir).items()}
            required = {"atd_meta", "atd_raw", "smart_tagging"}
            status = {key: {"file": path.name, "present": path.is_file(), "required": key in required}
                      for key, path in paths.items()}
            print(json.dumps(status, indent=2))
            return 0 if all(paths[key].is_file() for key in required) else 2
        if args.command == "predict":
            predict(args.data_dir, args.out, args.config, args.tracking_only, model_path=args.model)
        elif args.command == "demo":
            from .fixture import create_fixture
            work = Path(args.work_dir)
            data = create_fixture(work / "data")
            predict(data, work / "prediction", allow_synthetic=True)
            print("SYNTHETIC DEMO ONLY. Real-match accuracy is unmeasured.")
        elif args.command == "validate":
            count = validate_csv(args.csv, GAMES[GAME]["n_video_frames"])
            print(f"PASS: {count:,} rows, correct frame ordering and label alphabet")
        elif args.command == "evaluate":
            run = Path(args.run_dir).resolve()
            if (run / ".prediction-run.json").exists():
                raise ValueError("Prediction run was interrupted; regenerate it before evaluation")
            manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
            if manifest.get("synthetic") is not False:
                raise ValueError("Cannot score a synthetic demo against the real match ground truth")
            prediction = run / "predictions.csv"
            if sha256(prediction) != manifest["output"]["sha256"]:
                raise ValueError("CSV changed since inference; regenerate it")
            score_input = run / "public-check" / "input"
            score_input.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(prediction, score_input / "predictions.csv")
            return subprocess.call([sys.executable, str(ROOT / "evaluation" / "run_evaluation.py"),
                                    "--results", str(score_input), "--out", str(run / "public-check"),
                                    "--gt", "public", "--no-detail"])
        elif args.command == "bundle":
            from .bundle import make_bundle
            print(make_bundle(args.out, args.team_name, args.run_dir))
        elif args.command == "config":
            print(json.dumps(load_config().as_dict(), indent=2, sort_keys=True))
        return 0
    except (ValueError, OSError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Synthetic end-to-end checks for learned-model snapshots and reproduction."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from games import GAME, GAMES, asset_paths
from possession.io import sha256
from possession.learned import FEATURE_NAMES, fit_model, save_model
from possession.pipeline import predict


class LearnedPipelineTests(unittest.TestCase):
    def test_model_snapshot_reproduces_csv_and_is_recorded(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = {k: Path(v) for k, v in asset_paths(data_dir=root / "data").items()}
            paths["atd_meta"].parent.mkdir(parents=True)
            paths["atd_meta"].write_text(
                '<EPTS><Teams><Team id="h"><Name>Home</Name></Team>'
                '<Team id="a"><Name>Away</Name></Team></Teams><Players>'
                '<Player teamId="h"/><Player teamId="a"/></Players></EPTS>')
            paths["atd_raw"].write_text("".join(
                f"{i + 1}:0.3,0.5,2;0.7,0.5,2:0.302,0.5,2\n" for i in range(80)))
            rng = np.random.default_rng(7221)
            X = rng.normal(size=(100, len(FEATURE_NAMES)))
            model = fit_model(X, np.where(X[:, 0] > 0, "D", "H"),
                              metadata={"source": "invented test only"})
            model["decoder_mode"] = "separate"
            save_model(root / "model.json", model)
            spec = dict(GAMES[GAME], n_video_frames=80)
            with patch.dict(GAMES, {GAME: spec}), patch("possession.pipeline.half_video_blocks", return_value=[(1, 0, 79)]):
                first, _ = predict(root / "data", root / "first", tracking_only=True,
                                   model_path=root / "model.json", progress=lambda _: None)
                second, _ = predict(root / "data", root / "second", tracking_only=True,
                                    config_path=root / "first/config.json",
                                    model_path=root / "first/model.json", progress=lambda _: None)
            self.assertEqual(first["output"]["sha256"], second["output"]["sha256"])
            self.assertEqual(first["learned_model"]["sha256"], sha256(root / "first/model.json"))
            self.assertEqual(first["learned_model"]["decoder_mode"], "separate")
            self.assertEqual(first["training_data"], {"source": "invented test only"})
            self.assertFalse(first["ground_truth_used_for_inference"])
            self.assertFalse((root / "first/.prediction-run.json").exists())
            self.assertEqual(json.loads((root / "first/model.json").read_text()), model)


if __name__ == "__main__":
    unittest.main()

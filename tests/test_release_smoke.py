"""CPU smoke tests for release portability fixes, without pretrained weights."""

import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Analyze"))


class ReleaseSmokeTests(unittest.TestCase):
    def test_checkpoint_layouts(self):
        source = ROOT / "Policy/hivebench/model/framework/share_tools.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        selected = {"resolve_checkpoint_path", "_checkpoint_run_dir"}
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in selected]
        namespace = {"Path": Path}
        exec(compile(tree, str(source), "exec"), namespace)
        resolve = namespace["resolve_checkpoint_path"]
        find_run = namespace["_checkpoint_run_dir"]
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / "config.yaml").write_text("framework: {}\n", encoding="utf-8")
            (run / "dataset_statistics.json").write_text(json.dumps({}), encoding="utf-8")
            for subdirectory in ("", "checkpoints", "final_model"):
                folder = run / subdirectory
                folder.mkdir(exist_ok=True)
                checkpoint = folder / "pytorch_model.pt"
                checkpoint.touch()
                self.assertEqual(resolve(folder), checkpoint)
                self.assertEqual(find_run(resolve(checkpoint), "config.yaml"), run)
            with self.assertRaises(FileNotFoundError):
                resolve(run / "missing.pt")

    def test_numpy_dataclass_defaults(self):
        source = ROOT / "Bench/Robocasa_tabletop/eval_files/simulation_env.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        from dataclasses import dataclass, field
        tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MultiStepConfig"]
        namespace = {"np": np, "dataclass": dataclass, "field": field}
        exec(compile(tree, str(source), "exec"), namespace)
        first = namespace["MultiStepConfig"]()
        second = namespace["MultiStepConfig"]()
        first.video_delta_indices[0] = 10
        self.assertEqual(second.video_delta_indices[0], 0)

    def test_bfloat16_features(self):
        from tools.multi_frame import extract_pooled_features, process_within_between_var_analysis
        from tools.single_frame import process_single_frame_analysis
        from tools.temporal import process_temporal_analysis

        class Encoder:
            def __call__(self, frames):
                return torch.randn(len(frames), 8, 4, 4, dtype=torch.bfloat16)

        frames = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(4)]
        model = Encoder()
        self.assertEqual(extract_pooled_features(frames, model, 2).dtype, np.float32)
        cfg = OmegaConf.create({"batch_size": 2})
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            result = process_within_between_var_analysis(frames, list(range(4)), model, cfg, output)
            self.assertIn("within_between_var", result)
            process_single_frame_analysis(
                ["avg_token_cos"], frames, model, cfg, output,
                save_frame_outputs=False, save_frame_summaries=False,
            )
            result = process_temporal_analysis(
                ["temporal_smoothness"], frames, model, cfg, output,
            )
            self.assertIn("temporal_smoothness", result)


if __name__ == "__main__":
    unittest.main()

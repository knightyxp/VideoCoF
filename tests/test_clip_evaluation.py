"""Offline regression checks; no checkpoint download or GPU is required."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "metric" / "compute_clip_score.py"
SPEC = importlib.util.spec_from_file_location("clip_evaluation_under_test", str(SCRIPT))
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)
try:
    import numpy as np
except ImportError:
    np = None


class ManifestAndCLI(unittest.TestCase):
    def test_help_does_not_load_torch_or_clip(self):
        code = "import runpy,sys; sys.argv=[%r,'--help']; runpy.run_path(%r,run_name='__main__')" % (str(SCRIPT), str(SCRIPT))
        result = subprocess.run([sys.executable, "-S", "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--clip_model", result.stdout)

    def test_released_jsonl_and_instruction_priority(self):
        row = {"id": "local_style_transfer_033_2", "video": "local_style_transfer/videos/033.mp4", "edit_instruction": "Change only her jacket", "qwen_vl_72b_refined_instruction": "wrong fallback"}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bench.jsonl"
            path.write_text(json.dumps(row) + "\n")
            sample = evaluation.load_and_normalize_samples(str(path))[0]
        self.assertEqual(sample["task_type"], "local_style_transfer")
        self.assertEqual(sample["sample_id"], "033_2")
        self.assertEqual(evaluation.resolve_instruction(sample), row["edit_instruction"])
        sample["instruction"] = "  Explicit instruction  "
        self.assertEqual(evaluation.resolve_instruction(sample), "Explicit instruction")

    def test_explicit_paths_override_filename_pattern(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source.mp4"
            edited = Path(temp) / "edited.mp4"
            source.touch()
            edited.touch()
            sample = {"id": "arbitrary", "original_video_path": str(source), "edited_video_path": "edited.mp4"}
            self.assertEqual(evaluation.construct_paths_for_sample(sample, temp, temp, "unused.mp4"), (str(source), str(edited), False))
            sample["edited_video_path"] = "missing.mp4"
            self.assertIsNone(evaluation.construct_paths_for_sample(sample, temp, temp, "edited.mp4")[1])

    def test_historical_sampler_indices_are_unchanged(self):
        self.assertEqual(evaluation._compute_evenly_spaced_indices(65, 33), list(range(0, 65, 2)))
        self.assertEqual(evaluation._compute_evenly_spaced_indices(None, 3), [0, 1, 2])
        self.assertEqual(evaluation._compute_evenly_spaced_indices(2, 5), [0, 0, 0, 1, 1])

    def test_infinite_imageio_length_preserves_historical_first_frames(self):
        reader = mock.Mock()
        reader.get_length.return_value = float("inf")
        fake_imageio = types.SimpleNamespace(get_reader=lambda path: reader)
        with mock.patch.object(evaluation, "_load_video_dependencies"), mock.patch.object(evaluation, "imageio", fake_imageio):
            self.assertIsNone(evaluation._get_video_length("video.mp4"))
        reader.close.assert_called_once_with()


@unittest.skipIf(np is None, "NumPy is required for the lightweight tensor oracle")
class MetricRegression(unittest.TestCase):
    def test_scaled_logit_mean_adjacent_cosine_and_product(self):
        class Tensor(np.ndarray):
            def __new__(cls, values):
                return np.asarray(values, dtype=float).view(cls)

            device = "cpu"

            def to(self, *args, **kwargs):
                return self

            def unsqueeze(self, axis):
                return np.expand_dims(self, axis).view(Tensor)

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return np.asarray(self)

        @contextlib.contextmanager
        def no_grad():
            yield

        fake_torch = types.SimpleNamespace(
            no_grad=no_grad,
            sqrt=lambda x: Tensor(np.sqrt(x)),
            sum=lambda x, dim=None, keepdim=False: Tensor(np.sum(np.asarray(x), axis=dim, keepdims=keepdim)),
            zeros=lambda n, **kwargs: Tensor(np.zeros(n)),
            tensor=lambda data, **kwargs: Tensor(data),
        )
        fake_clip = types.SimpleNamespace(tokenize=lambda text: Tensor([1]))
        fake_model = mock.Mock()
        fake_model.encode_image.side_effect = lambda tensor: tensor[:, :2]
        fake_model.side_effect = lambda tensor, text: (tensor[:, 2:3], None)
        with mock.patch.object(evaluation, "_ensure_clip_loaded"), mock.patch.object(evaluation, "extract_evenly_spaced_frames_pil", return_value=[[3, 4, 12], [0, 5, 24]]), mock.patch.multiple(evaluation, torch=fake_torch, clip=fake_clip, model=fake_model, preprocess=Tensor, device="cpu"):
            clip_t, clip_f, q_edit = evaluation.compute_clip_temporal_q("unused", "edit instruction", 2, raise_on_error=True)
        # Hand-computed: logits mean 18; normalized [0.6,0.8] dot [0,1] = 0.8.
        self.assertAlmostEqual(clip_t, 18.0)
        self.assertAlmostEqual(clip_f, 0.8)
        self.assertAlmostEqual(q_edit, 14.4)

    def test_report_failure_coverage_and_independent_dino_denominator(self):
        with tempfile.TemporaryDirectory() as temp:
            edited = Path(temp) / "edited.mp4"
            edited.touch()
            source = Path(temp) / "samples.json"
            report = Path(temp) / "report.json"
            rows = [{"id": "obj_swap_%03d" % i, "instruction": "replace object", "edited_video_path": str(edited)} for i in range(4)]
            rows[3]["edited_video_path"] = str(Path(temp) / "missing.mp4")
            source.write_text(json.dumps(rows))
            args = ["--input_json", str(source), "--video_root", temp, "--output_json", str(report), "--dino_model_name", "dinov2_vits14"]
            with mock.patch.object(evaluation, "_load_runtime_dependencies"), mock.patch.object(evaluation, "_get_video_length", return_value=65), mock.patch.object(evaluation, "np", np), mock.patch.object(evaluation, "compute_clip_temporal_q", side_effect=[(10.0, 0.5, 5.0), (20.0, 0.7, 14.0), RuntimeError("synthetic decode failure")]), mock.patch.object(evaluation, "compute_dino_temporal_consistency", side_effect=[0.8, 0.6, 0.7]), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                status = evaluation.main(args)
            data = json.loads(report.read_text())
        self.assertEqual(status, 1)
        self.assertEqual(data["coverage"]["num_input_samples"], 4)
        self.assertEqual(data["coverage"]["num_clip_scored"], 2)
        self.assertEqual(data["coverage"]["num_dino_scored"], 3)
        self.assertEqual(data["coverage"]["num_failed_samples"], 2)
        self.assertAlmostEqual(data["averages"]["clip_t_avg"], 15.0)
        self.assertAlmostEqual(data["averages"]["q_edit_avg"], 9.5)
        self.assertAlmostEqual(data["averages"]["dino_temporal_consistency_avg"], 0.7)
        self.assertEqual({failure["stage"] for failure in data["failures"]}, {"clip", "path"})


if __name__ == "__main__":
    unittest.main()

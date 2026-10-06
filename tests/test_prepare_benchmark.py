import json
from pathlib import Path
import tempfile
import unittest

from metric.prepare_benchmark import prepare_manifest


class PrepareBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        video = self.root / "obj_swap/videos/source.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"test fixture")
        self.rows = [
            {"id": "obj_swap_instance_023_1", "video": "obj_swap/videos/source.mp4", "edit_instruction": "Edit the left person."},
            {"id": "obj_swap_instance_023_2", "video": "obj_swap/videos/source.mp4", "edit_instruction": "Edit the right person."},
        ]
        self.write_rows()

    def write_rows(self):
        (self.root / "videocof_edit.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in self.rows), encoding="utf-8"
        )

    def test_shared_video_keeps_distinct_instance_outputs(self):
        records = prepare_manifest(self.root, self.root / "outputs")
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["sample_id"], "instance_023_1")
        self.assertEqual(records[0]["source_video_path"], records[1]["source_video_path"])
        self.assertNotEqual(records[0]["edited_video_path"], records[1]["edited_video_path"])
        for row, record in zip(self.rows, records):
            expected = f"gen_{record['task_type']}_{record['sample_id']}.mp4"
            self.assertEqual(Path(record["edited_video_path"]).name, expected)
            self.assertEqual(record["instruction"], row["edit_instruction"])

    def test_duplicate_ids_and_output_collisions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "collision"):
            prepare_manifest(self.root, self.root / "outputs", "same.mp4")
        self.rows.append(self.rows[0])
        self.write_rows()
        with self.assertRaisesRegex(ValueError, "duplicate id"):
            prepare_manifest(self.root, self.root / "outputs")

    def test_missing_inputs_and_outputs_are_reported(self):
        with self.assertRaisesRegex(FileNotFoundError, "Missing edited output"):
            prepare_manifest(self.root, self.root / "outputs", check_outputs=True)
        (self.root / self.rows[0]["video"]).unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Missing input"):
            prepare_manifest(self.root, self.root / "outputs")


if __name__ == "__main__":
    unittest.main()

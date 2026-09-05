from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import download_model


class CheckpointStartupTest(unittest.TestCase):
    def checkpoint(self, target):
        for name in download_model.REQUIRED_FILES:
            path = target / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture")

    def run_main(self, target):
        args = ["download_model.py", "--repo-id", "fixture/model", "--local-dir", str(target),
                "--revision", "pinned-fixture-revision", "--ensure-complete"]
        with patch.object(sys, "argv", args):
            download_model.main()

    def test_complete_local_checkpoint_never_calls_network(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(download_model, "snapshot_download") as fetch:
            self.checkpoint(Path(tmp))
            self.run_main(tmp)
            fetch.assert_not_called()

    def test_existing_weights_do_not_hide_missing_processor(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(download_model, "snapshot_download") as fetch:
            target = Path(tmp)
            self.checkpoint(target)
            missing = target / "policy_preprocessor_step_3_normalizer_processor.safetensors"
            missing.unlink()
            fetch.side_effect = lambda **kwargs: missing.write_bytes(b"fixture")
            self.run_main(target)
            self.assertEqual(fetch.call_args.kwargs["revision"], "pinned-fixture-revision")
            self.assertTrue(missing.is_file())

    def test_empty_tokenizer_still_rejected_after_incomplete_download(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(download_model, "snapshot_download") as fetch:
            target = Path(tmp)
            self.checkpoint(target)
            (target / "tokenizer/tokenizer.json").write_bytes(b"")
            with self.assertRaisesRegex(SystemExit, "tokenizer/tokenizer.json"):
                self.run_main(target)
            fetch.assert_called_once()


if __name__ == "__main__":
    unittest.main()

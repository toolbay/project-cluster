import os
import tempfile
import unittest
from pathlib import Path

from lib.constants import DEFAULT_HISTORY_DB
from lib.inferencer import infer_patch
from lib.trainer import train_model


class DefenderIntegrationTest(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("RUN_DEFENDER_INTEGRATION") == "1",
        "set RUN_DEFENDER_INTEGRATION=1 to run",
    )
    def test_sample_patch_top1(self) -> None:
        base_dir = Path(__file__).resolve().parents[1]
        patch_path = base_dir.parent / "0001-Reland-interpreter-Enable-TDZ-elision-by-default.patch"
        model_db = Path(tempfile.mkdtemp()) / "defender_model.db"

        train_model(
            v8_repo="/home/shuni/code/v8/main/v8",
            history_db_path=str(DEFAULT_HISTORY_DB),
            model_db_path=str(model_db),
            reuse_db=True,
            progress=None,
        )

        result = infer_patch(
            model_db_path=str(model_db),
            patch_path=str(patch_path),
            top_k=5,
        )

        self.assertIsNotNone(result["top1"])
        self.assertEqual(
            result["top1"]["flag"],
            "ignition-elide-redundant-tdz-checks",
        )
        self.assertEqual(
            result["top1"]["recommended_cmd"],
            "--no-ignition-elide-redundant-tdz-checks",
        )
        self.assertTrue(result["top1"]["evidence"]["direct_patch_hint"])


if __name__ == "__main__":
    unittest.main()

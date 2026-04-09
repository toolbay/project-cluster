import tempfile
import unittest
from pathlib import Path

from lib.flag_catalog import parse_flag_definitions
from lib.patch_parser import parse_patch
from lib.path_utils import directory_prefixes, normalize_repo_path


class DefenderUnitTest(unittest.TestCase):
    def test_parse_flag_definitions(self) -> None:
        content = """
DEFINE_BOOL(ignition_elide_redundant_tdz_checks, true, "elide")
DEFINE_INT(max_inlined_bytecode_size, 460, "size")
DEFINE_BOOL(id, false, "placeholder")
"""
        with tempfile.NamedTemporaryFile("w", suffix=".h", delete=False) as f:
            f.write(content)
            tmp_path = f.name

        catalog = parse_flag_definitions(tmp_path)
        self.assertIn("ignition-elide-redundant-tdz-checks", catalog)
        self.assertIn("max-inlined-bytecode-size", catalog)
        self.assertNotIn("id", catalog)
        self.assertTrue(catalog["ignition-elide-redundant-tdz-checks"].is_bool)
        self.assertFalse(catalog["max-inlined-bytecode-size"].is_bool)

    def test_parse_patch_sample(self) -> None:
        patch_path = Path(__file__).resolve().parents[2] / (
            "0001-Reland-interpreter-Enable-TDZ-elision-by-default.patch"
        )
        if not patch_path.exists():
            self.skipTest(f"sample patch missing: {patch_path}")
        info = parse_patch(str(patch_path))
        self.assertEqual(len(info.files), 5)
        self.assertIn("src/flags/flag-definitions.h", info.files)
        self.assertIn("ignition-elide-redundant-tdz-checks", info.direct_flag_hints)
        self.assertGreater(info.token_counts.get("tdz", 0), 0)

    def test_path_normalization_and_prefix(self) -> None:
        self.assertEqual(
            normalize_repo_path("a\\src\\flags\\flag-definitions.h"),
            "src/flags/flag-definitions.h",
        )
        self.assertEqual(
            directory_prefixes("src/flags/flag-definitions.h"),
            ["src/flags", "src"],
        )


if __name__ == "__main__":
    unittest.main()

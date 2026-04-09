import argparse
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest import mock

import runme


class _FakeTTYStream:
    def __init__(self) -> None:
        self.buf = StringIO()

    def isatty(self) -> bool:
        return True

    @property
    def encoding(self) -> str:
        return "UTF-8"

    def write(self, data: str) -> int:
        return self.buf.write(data)

    def flush(self) -> None:
        return None

    def getvalue(self) -> str:
        return self.buf.getvalue()


class CliOutputTest(unittest.TestCase):
    def setUp(self) -> None:
        self.result = {
            "input": {
                "patch": "/tmp/sample.patch",
                "model_db": "/tmp/model.db",
            },
            "top_k": [
                {
                    "flag": "flag-a",
                    "recommended_cmd": "--no-flag-a",
                    "score": 10.0,
                    "evidence": {
                        "matched_keywords": ["alpha", "beta"],
                        "matched_clusters": [1],
                        "direct_patch_hint": True,
                    },
                },
                {
                    "flag": "flag-b",
                    "recommended_cmd": None,
                    "score": 8.0,
                    "evidence": {
                        "matched_keywords": ["gamma"],
                        "matched_clusters": [2],
                        "direct_patch_hint": False,
                    },
                },
                {
                    "flag": "flag-c",
                    "recommended_cmd": "--no-flag-c",
                    "score": 6.0,
                    "evidence": {
                        "matched_keywords": [],
                        "matched_clusters": [],
                        "direct_patch_hint": False,
                    },
                },
            ],
            "top1": {
                "flag": "flag-a",
                "recommended_cmd": "--no-flag-a",
                "score": 10.0,
                "evidence": {
                    "matched_keywords": ["alpha", "beta"],
                    "matched_clusters": [1],
                    "direct_patch_hint": True,
                },
            },
        }

    def test_build_top3_summary_rows(self) -> None:
        rows = runme._build_top3_summary_rows(self.result)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0][0], "1")
        self.assertEqual(rows[0][1], "flag-a")
        self.assertEqual(rows[0][2], "--no-flag-a")
        self.assertIn("alpha", rows[0][3])
        self.assertEqual(rows[0][4], "True")

    def test_render_html_report(self) -> None:
        report = runme._render_html_report(self.result)
        self.assertIn("JSFlags Defender Inference Report", report)
        self.assertIn("Top 3 Summary", report)
        self.assertIn("Top-K Details", report)
        self.assertIn("Full JSON", report)
        self.assertIn("flag-a", report)
        self.assertIn("copyJson", report)

    def test_progress_renderer_non_tty_tables_and_stage_names(self) -> None:
        output = StringIO()
        renderer = runme.InferProgressRenderer(stream=output)

        renderer.handle_event(
            {
                "event": "step_start",
                "step": 4,
                "stage_key": "feature_projection",
                "stage_name": "Feature Projection",
                "phase": "mapping",
                "message": "projecting files...",
            }
        )
        renderer.handle_event(
            {
                "event": "step_detail",
                "step": 4,
                "stage_key": "feature_projection",
                "stage_name": "Feature Projection",
                "phase": "mapping",
                "title": "Mapping Table",
                "table": {
                    "headers": ["File", "Resolution", "Cluster"],
                    "rows": [["a.cc", "exact", "C1"]],
                },
            }
        )
        renderer.handle_event(
            {
                "event": "step_detail",
                "step": 4,
                "stage_key": "feature_projection",
                "stage_name": "Feature Projection",
                "phase": "mapping",
                "title": "Feature Space Projection",
                "panel": {
                    "title": "2D Feature Space Projection (41x17)",
                    "lines": ["┌─────┐", "│ ● f1 │", "│  F1  │", "└─────┘"],
                    "legend": {
                        "headers": ["Mark", "Type", "Entity"],
                        "rows": [["●", "patch", "patch-centroid"], ["f1", "file", "a.cc"], ["F1", "flag", "flag-a"]],
                    },
                },
            }
        )
        renderer.handle_event(
            {
                "event": "step_end",
                "step": 4,
                "stage_key": "feature_projection",
                "stage_name": "Feature Projection",
                "phase": "mapping",
                "summary": "feature projection ready",
                "elapsed_ms": 12,
                "metrics": {"mapped": 1, "clusters": 1},
            }
        )
        renderer.close()

        text = output.getvalue()
        self.assertIn("1-4 Feature Projection", text)
        self.assertIn("Step • 1-4 • Feature Projection", text)
        self.assertIn("Status • 1-4 • Feature Projection", text)
        self.assertIn("Mapping Table", text)
        self.assertIn("┌", text)
        self.assertIn("┬", text)
        self.assertIn("│", text)
        self.assertIn("└", text)
        self.assertIn("●", text)
        self.assertIn("f1", text)
        self.assertIn("F1", text)
        self.assertNotIn("06/07", text)
        self.assertNotIn("[runme]", text)

    @mock.patch("runme.time.sleep")
    def test_showtime_tty_uses_sleep_for_padding(self, sleep_mock: mock.Mock) -> None:
        stream = _FakeTTYStream()
        renderer = runme.InferProgressRenderer(stream=stream, showtime=True)
        renderer._showtime_stage_budgets["input_scan"] = 0.05

        renderer.handle_event(
            {
                "event": "step_start",
                "step": 1,
                "stage_key": "input_scan",
                "stage_name": "Input Scan",
                "phase": "thinking",
                "message": "checking input files...",
            }
        )
        renderer.handle_event(
            {
                "event": "step_end",
                "step": 1,
                "stage_key": "input_scan",
                "stage_name": "Input Scan",
                "phase": "thinking",
                "summary": "inputs ready",
                "elapsed_ms": 0,
                "metrics": {"model": "ok"},
            }
        )
        renderer.close()

        self.assertTrue(sleep_mock.called)

    @mock.patch("runme.time.sleep")
    def test_non_tty_showtime_does_not_pad(self, sleep_mock: mock.Mock) -> None:
        stream = StringIO()
        renderer = runme.InferProgressRenderer(stream=stream, showtime=True)

        renderer.handle_event(
            {
                "event": "step_start",
                "step": 1,
                "stage_key": "input_scan",
                "stage_name": "Input Scan",
                "phase": "thinking",
                "message": "checking input files...",
            }
        )
        renderer.handle_event(
            {
                "event": "step_end",
                "step": 1,
                "stage_key": "input_scan",
                "stage_name": "Input Scan",
                "phase": "thinking",
                "summary": "inputs ready",
                "elapsed_ms": 0,
                "metrics": {"model": "ok"},
            }
        )
        renderer.close()

        self.assertFalse(sleep_mock.called)

    @mock.patch("runme.time.sleep")
    def test_default_mode_no_showtime_padding(self, sleep_mock: mock.Mock) -> None:
        stream = _FakeTTYStream()
        renderer = runme.InferProgressRenderer(stream=stream, showtime=False)

        renderer.handle_event(
            {
                "event": "step_start",
                "step": 1,
                "stage_key": "input_scan",
                "stage_name": "Input Scan",
                "phase": "thinking",
                "message": "checking input files...",
            }
        )
        renderer.handle_event(
            {
                "event": "step_end",
                "step": 1,
                "stage_key": "input_scan",
                "stage_name": "Input Scan",
                "phase": "thinking",
                "summary": "inputs ready",
                "elapsed_ms": 0,
                "metrics": {"model": "ok"},
            }
        )
        renderer.close()

        self.assertFalse(sleep_mock.called)

    @mock.patch("runme.infer_patch")
    def test_cmd_infer_default_stdout_summary_only(self, infer_patch_mock: mock.Mock) -> None:
        infer_patch_mock.return_value = self.result
        args = argparse.Namespace(
            model_db="/tmp/model.db",
            patch="/tmp/sample.patch",
            top_k=3,
            out="",
            html_out="",
            showtime=False,
        )

        output = StringIO()
        with redirect_stdout(output):
            rc = runme.cmd_infer(args)

        text = output.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("Summary • Top 3 Candidates", text)
        self.assertIn("flag-a", text)
        self.assertIn("┌", text)
        self.assertNotIn('"top_k":', text)
        self.assertNotIn("06/07", text)


if __name__ == "__main__":
    unittest.main()

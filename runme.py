#!/usr/bin/env python3
import argparse
import html
import json
import os
import shutil
import sys
import threading
from pathlib import Path
from typing import Any, Dict, IO, List, Optional, Sequence

from lib.constants import DEFAULT_HISTORY_DB, DEFAULT_MODEL_DB
from lib.inferencer import infer_patch
from lib.model_store import inspect_model
from lib.trainer import train_model

STAGE_NAME_BY_STEP = {
    1: "Input Scan",
    2: "Model Sync",
    3: "Patch Decode",
    4: "Feature Projection",
    5: "Candidate Scoring",
    6: "Rank Decision",
    7: "Result Packaging",
}


def _supports_unicode(stream: IO[str]) -> bool:
    encoding = getattr(stream, "encoding", None)
    if encoding is None:
        return True
    return "UTF" in encoding.upper()


def _stringify_table_rows(rows: Sequence[Sequence[object]]) -> List[List[str]]:
    return [[str(cell) for cell in row] for row in rows]


def _render_table(
    headers: Sequence[object],
    rows: Sequence[Sequence[object]],
    use_unicode: bool,
) -> List[str]:
    str_headers = [str(x) for x in headers]
    str_rows = _stringify_table_rows(rows)

    if not str_headers:
        return []

    col_count = len(str_headers)
    normalized_rows: List[List[str]] = []
    for row in str_rows:
        cells = list(row[:col_count])
        if len(cells) < col_count:
            cells.extend([""] * (col_count - len(cells)))
        normalized_rows.append(cells)

    widths = [len(str_headers[idx]) for idx in range(col_count)]
    for row in normalized_rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    if use_unicode:
        chars = {
            "top_l": "┌",
            "top_m": "┬",
            "top_r": "┐",
            "mid_l": "├",
            "mid_m": "┼",
            "mid_r": "┤",
            "bot_l": "└",
            "bot_m": "┴",
            "bot_r": "┘",
            "h": "─",
            "v": "│",
        }
    else:
        chars = {
            "top_l": "+",
            "top_m": "+",
            "top_r": "+",
            "mid_l": "+",
            "mid_m": "+",
            "mid_r": "+",
            "bot_l": "+",
            "bot_m": "+",
            "bot_r": "+",
            "h": "-",
            "v": "|",
        }

    def border(left: str, middle: str, right: str) -> str:
        chunks = [chars["h"] * (width + 2) for width in widths]
        return f"{left}{middle.join(chunks)}{right}"

    def row_line(cells: Sequence[str]) -> str:
        padded = [f" {cells[idx].ljust(widths[idx])} " for idx in range(col_count)]
        return f"{chars['v']}{chars['v'].join(padded)}{chars['v']}"

    output = [
        border(chars["top_l"], chars["top_m"], chars["top_r"]),
        row_line(str_headers),
        border(chars["mid_l"], chars["mid_m"], chars["mid_r"]),
    ]
    for row in normalized_rows:
        output.append(row_line(row))
    output.append(border(chars["bot_l"], chars["bot_m"], chars["bot_r"]))
    return output


class InferProgressRenderer:
    _RUNNING_ICON = "..."
    _PHASE_COLORS = {
        "thinking": "96",
        "reading": "94",
        "mapping": "95",
        "deciding": "93",
        "shipping": "92",
    }

    def __init__(self, stream: Optional[IO[str]] = None) -> None:
        self.stream = stream if stream is not None else sys.stdout
        self.is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self.use_color = self.is_tty and os.environ.get("NO_COLOR") is None
        self.use_unicode = _supports_unicode(self.stream)

        self._write_lock = threading.Lock()
        self._state_lock = threading.Lock()

        self._active_step: Optional[Dict[str, object]] = None
        self._live_message = ""
        self._live_progress = ""

    def handle_event(self, event: Dict[str, object]) -> None:
        event_type = str(event.get("event", ""))
        if event_type == "step_start":
            self._on_step_start(event)
        elif event_type == "step_end":
            self._on_step_end(event)
        elif event_type == "step_detail":
            self._on_step_detail(event)
        elif event_type == "step_progress":
            self._on_step_progress(event)

    def close(self) -> None:
        if self.is_tty:
            self._write_tty_line("", newline=False)

    def _stage_name(self, event: Dict[str, object]) -> str:
        if event.get("stage_name"):
            return str(event["stage_name"])
        step = int(event.get("step", 0))
        return STAGE_NAME_BY_STEP.get(step, f"Step {step}")

    def _stage_id(self, event: Dict[str, object]) -> str:
        step = int(event.get("step", 0))
        if step > 0:
            return f"1-{step}"
        return "1-?"

    def _on_step_start(self, event: Dict[str, object]) -> None:
        stage_id = self._stage_id(event)
        stage_name = self._stage_name(event)
        phase = str(event.get("phase", "thinking"))
        message = str(event.get("message", ""))

        separator = self._render_stage_separator(stage_id, stage_name)
        self._write_line("")
        self._write_line(self._color(separator, "95"))
        self._write_line("")

        with self._state_lock:
            self._active_step = {
                "stage_id": stage_id,
                "stage_name": stage_name,
                "phase": phase,
            }
            self._live_message = message
            self._live_progress = ""

        if self.is_tty:
            self._render_live_line()
            return

        line = self._format_running_line(
            stage_id,
            stage_name,
            phase,
            message,
            spinner=self._RUNNING_ICON,
        )
        self._write_line(line)

    def _on_step_end(self, event: Dict[str, object]) -> None:
        stage_id = self._stage_id(event)
        stage_name = self._stage_name(event)
        summary = str(event.get("summary", "done"))
        elapsed_ms = int(event.get("elapsed_ms", 0))
        metrics = event.get("metrics")
        metrics_text = self._format_metrics(metrics if isinstance(metrics, dict) else None)

        done_icon = self._color("OK", "92") if self.is_tty else "OK"
        timing = f"compute={elapsed_ms}ms, total={elapsed_ms}ms"

        status = f"Status • {stage_id} • {stage_name} • {done_icon} • {summary}"
        if metrics_text:
            status += f" • {metrics_text}"
        status += f" • {timing}"

        if self.is_tty:
            self._write_tty_line(status, newline=True)
        else:
            self._write_line(status)

        with self._state_lock:
            self._active_step = None
            self._live_progress = ""

    def _on_step_detail(self, event: Dict[str, object]) -> None:
        stage_id = self._stage_id(event)
        stage_name = self._stage_name(event)
        title = str(event.get("title", "Detail"))
        raw_lines = event.get("lines")
        lines = [str(x) for x in raw_lines] if isinstance(raw_lines, list) else []
        table = event.get("table")
        panel = event.get("panel")

        if not lines and not isinstance(table, dict) and not isinstance(panel, dict):
            return

        should_resume_live = False
        if self.is_tty:
            with self._state_lock:
                should_resume_live = self._active_step is not None
            if should_resume_live:
                self._write_tty_line("", newline=False)

        self._print_detail_block(stage_id, stage_name, title, lines, table, panel)

        if should_resume_live:
            self._render_live_line()

    def _on_step_progress(self, event: Dict[str, object]) -> None:
        stage_id = self._stage_id(event)
        stage_name = self._stage_name(event)
        message = str(event.get("message", ""))

        current = event.get("current")
        total = event.get("total")
        metrics = event.get("metrics")
        metrics_text = self._format_metrics(metrics if isinstance(metrics, dict) else None)

        progress_text = message
        if isinstance(current, int) and isinstance(total, int) and total > 0:
            percent = int((current * 100) / total)
            progress_text += f" [{current}/{total}, {percent}%]"
        if metrics_text:
            progress_text += f" • {metrics_text}"

        if self.is_tty:
            with self._state_lock:
                self._live_progress = progress_text
            self._render_live_line()
            return

        self._write_line(f"Progress • {stage_id} • {stage_name} • {progress_text}")

    def _render_live_line(self) -> None:
        if not self.is_tty:
            return

        with self._state_lock:
            active = dict(self._active_step or {})
            message = self._live_message
            progress = self._live_progress

        if not active:
            self._write_tty_line("", newline=False)
            return

        stage_id = str(active.get("stage_id", "1-?"))
        stage_name = str(active.get("stage_name", ""))
        phase = str(active.get("phase", "thinking"))
        spinner = self._color(self._RUNNING_ICON, "96")
        line = self._format_running_line(stage_id, stage_name, phase, message, spinner)
        if progress:
            line += f" • {progress}"
        self._write_tty_line(line, newline=False)

    def _format_running_line(
        self,
        stage_id: str,
        stage_name: str,
        phase: str,
        message: str,
        spinner: str,
    ) -> str:
        phase_label = self._phase_label(phase)
        return f"Step • {stage_id} • {stage_name} • {spinner} • {phase_label} • {message}"

    def _phase_label(self, phase: str) -> str:
        return self._color(phase, self._PHASE_COLORS.get(phase, "97"))

    def _format_metrics(self, metrics: Optional[Dict[str, object]]) -> str:
        if not metrics:
            return ""
        return ", ".join(f"{key}={value}" for key, value in metrics.items())

    def _print_detail_block(
        self,
        stage_id: str,
        stage_name: str,
        title: str,
        lines: List[str],
        table: object,
        panel: object,
    ) -> None:
        self._write_line(self._color(f"Detail • {stage_id} • {stage_name} • {title}", "90"))

        if isinstance(table, dict):
            headers = table.get("headers") or []
            rows = table.get("rows") or []
            rendered = _render_table(headers, rows, use_unicode=self.use_unicode)
            for line in rendered:
                self._write_line(f"  {line}")

        if lines:
            for line in lines:
                self._write_line(f"  - {line}")

        if isinstance(panel, dict):
            panel_title = str(panel.get("title", "Panel"))
            self._write_line(self._color(f"Panel • {panel_title}", "90"))
            panel_lines = panel.get("lines") or []
            for line in panel_lines:
                self._write_line(f"  {line}")

            legend = panel.get("legend")
            if isinstance(legend, dict):
                headers = legend.get("headers") or []
                rows = legend.get("rows") or []
                rendered = _render_table(headers, rows, use_unicode=self.use_unicode)
                for line in rendered:
                    self._write_line(f"  {line}")

    def _write_tty_line(self, text: str, newline: bool) -> None:
        with self._write_lock:
            suffix = "\n" if newline else ""
            self.stream.write(f"\r\033[2K{text}{suffix}")
            self.stream.flush()

    def _write_line(self, text: str) -> None:
        with self._write_lock:
            self.stream.write(f"{text}\n")
            self.stream.flush()

    def _color(self, text: str, code: str) -> str:
        if not self.use_color:
            return text
        return f"\033[{code}m{text}\033[0m"

    def _render_stage_separator(self, stage_id: str, stage_name: str) -> str:
        label = f" {stage_id} {stage_name} "
        fill = "═" if self.use_unicode else "="
        width = self._terminal_width()

        min_width = len(label) + 6
        if width < min_width:
            width = min_width

        left_len = (width - len(label)) // 2
        right_len = width - len(label) - left_len

        return f"{fill * left_len}{label}{fill * right_len}"

    def _terminal_width(self) -> int:
        if self.is_tty:
            try:
                return max(60, int(shutil.get_terminal_size(fallback=(120, 30)).columns))
            except OSError:
                return 120
        return 120


def _print_progress(message: str) -> None:
    print(message)


def _write_json(path: str, payload: Any) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False),
        encoding="utf-8",
    )


def _write_html(path: str, content: str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")


def _build_top3_summary_rows(result: Dict[str, Any]) -> List[List[str]]:
    items = result.get("top_k") or []
    rows: List[List[str]] = []

    for index, item in enumerate(items[:3], start=1):
        flag = str(item.get("flag", ""))
        cmd = item.get("recommended_cmd")
        cmd_text = str(cmd) if cmd else "null"
        evidence = item.get("evidence") or {}
        keywords = evidence.get("matched_keywords") or []
        keyword_text = ", ".join(str(k) for k in keywords[:5]) if keywords else "-"
        direct_hint = bool(evidence.get("direct_patch_hint", False))
        rows.append([str(index), flag, cmd_text, keyword_text, str(direct_hint)])

    if not rows:
        rows.append(["-", "No candidate flags", "-", "-", "-"])

    return rows


def _print_top3_summary(result: Dict[str, Any]) -> None:
    print("Summary • Top 3 Candidates")
    rows = _build_top3_summary_rows(result)
    table_lines = _render_table(
        headers=["Rank", "Flag", "Recommended Cmd", "Matched Keywords", "Direct Hint"],
        rows=rows,
        use_unicode=_supports_unicode(sys.stdout),
    )
    for line in table_lines:
        print(line)


def _render_html_report(result: Dict[str, Any]) -> str:
    top_items = result.get("top_k") or []
    top3 = top_items[:3]

    summary_rows = []
    for item in top3:
        evidence = item.get("evidence") or {}
        keywords = evidence.get("matched_keywords") or []
        summary_rows.append(
            "<tr>"
            f"<td>{html.escape(str(item.get('flag', '')))}</td>"
            f"<td>{html.escape(str(item.get('recommended_cmd') or 'null'))}</td>"
            f"<td>{html.escape(', '.join(str(k) for k in keywords[:5]) or '-')}</td>"
            f"<td>{html.escape(str(item.get('score', '')))}</td>"
            "</tr>"
        )

    if not summary_rows:
        summary_rows.append(
            "<tr><td colspan='4' style='text-align:center;color:#777;'>No candidates</td></tr>"
        )

    detail_rows = []
    for item in top_items:
        evidence = item.get("evidence") or {}
        detail_rows.append(
            "<tr>"
            f"<td>{html.escape(str(item.get('flag', '')))}</td>"
            f"<td>{html.escape(str(item.get('recommended_cmd') or 'null'))}</td>"
            f"<td>{html.escape(str(item.get('score', '')))}</td>"
            f"<td>{html.escape(', '.join(str(evidence.get('matched_keywords') or [])) or '-')}</td>"
            f"<td>{html.escape(', '.join(str(x) for x in (evidence.get('matched_clusters') or [])) or '-')}</td>"
            "</tr>"
        )

    if not detail_rows:
        detail_rows.append(
            "<tr><td colspan='5' style='text-align:center;color:#777;'>No candidates</td></tr>"
        )

    json_text = html.escape(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=False)
    )

    patch_path = ((result.get("input") or {}).get("patch")) or ""
    model_path = ((result.get("input") or {}).get("model_db")) or ""

    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>JSFlags Defender Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 0; color: #1f2328; background: #f6f8fa; }}
    .wrap {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
    .sticky {{ position: sticky; top: 0; z-index: 10; background: #fff; border-bottom: 1px solid #d0d7de; padding: 12px 20px; }}
    .card {{ background: #fff; border: 1px solid #d0d7de; border-radius: 8px; padding: 16px; margin-bottom: 16px; }}
    h1 {{ margin: 0; font-size: 20px; }}
    h2 {{ margin: 0 0 12px 0; font-size: 16px; }}
    .meta {{ font-size: 13px; color: #57606a; margin-top: 6px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border: 1px solid #d0d7de; padding: 8px; vertical-align: top; text-align: left; }}
    th {{ background: #f6f8fa; }}
    pre {{ margin: 0; background: #0d1117; color: #e6edf3; padding: 12px; border-radius: 6px; overflow: auto; font-size: 12px; line-height: 1.45; }}
    .btn {{ background: #1f6feb; color: #fff; border: 0; border-radius: 6px; padding: 8px 10px; cursor: pointer; font-size: 12px; }}
    .btn:active {{ transform: translateY(1px); }}
  </style>
</head>
<body>
  <div class=\"sticky\">
    <h1>JSFlags Defender Inference Report</h1>
    <div class=\"meta\">Patch: {html.escape(str(patch_path))}</div>
    <div class=\"meta\">Model: {html.escape(str(model_path))}</div>
  </div>

  <div class=\"wrap\">
    <section class=\"card\">
      <h2>Top 3 Summary</h2>
      <table>
        <thead>
          <tr><th>Flag</th><th>Recommended Cmd</th><th>Matched Keywords</th><th>Score</th></tr>
        </thead>
        <tbody>
          {''.join(summary_rows)}
        </tbody>
      </table>
    </section>

    <section class=\"card\">
      <h2>Top-K Details</h2>
      <table>
        <thead>
          <tr><th>Flag</th><th>Recommended Cmd</th><th>Score</th><th>Matched Keywords</th><th>Matched Clusters</th></tr>
        </thead>
        <tbody>
          {''.join(detail_rows)}
        </tbody>
      </table>
    </section>

    <section class=\"card\">
      <h2>Full JSON</h2>
      <div style=\"margin-bottom:8px;\"><button class=\"btn\" onclick=\"copyJson()\">Copy JSON</button></div>
      <pre id=\"full-json\">{json_text}</pre>
    </section>
  </div>

  <script>
    function copyJson() {{
      const text = document.getElementById('full-json').innerText;
      navigator.clipboard.writeText(text);
    }}
  </script>
</body>
</html>
"""


def cmd_train(args: argparse.Namespace) -> int:
    result = train_model(
        v8_repo=args.v8_repo,
        history_db_path=args.history_db,
        model_db_path=args.model_db,
        reuse_db=args.reuse_db,
        progress=_print_progress,
    )
    print(
        json.dumps(
            {
                "model_db": result.model_db,
                "history_db": result.history_db,
                "history_source": result.history_source,
                "commit_count": result.commit_count,
                "file_count": result.file_count,
                "cluster_count": result.cluster_count,
                "flag_count": result.flag_count,
                "weighted_flag_count": result.weighted_flag_count,
                "keyword_weighted_flag_count": result.keyword_weighted_flag_count,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def cmd_infer(args: argparse.Namespace) -> int:
    renderer = InferProgressRenderer()
    try:
        result = infer_patch(
            model_db_path=args.model_db,
            patch_path=args.patch,
            top_k=args.top_k,
            progress=renderer.handle_event,
        )
    finally:
        renderer.close()

    _print_top3_summary(result)

    if args.out:
        _write_json(args.out, result)
        print(f"wrote inference JSON: {Path(args.out).resolve()}")

    if args.html_out:
        html_report = _render_html_report(result)
        _write_html(args.html_out, html_report)
        print(f"wrote inference HTML: {Path(args.html_out).resolve()}")

    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    summary = inspect_model(args.model_db)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="JSFlags runme training and inference CLI",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    train_parser = sub.add_parser("train", help="Train model from V8 history")
    train_parser.add_argument("--v8-repo", required=True, help="Path to V8 repo")
    train_parser.add_argument(
        "--history-db",
        default=str(DEFAULT_HISTORY_DB),
        help="Path to reusable history sqlite db",
    )
    train_parser.add_argument(
        "--model-db",
        default=str(DEFAULT_MODEL_DB),
        help="Path to output model sqlite db",
    )
    train_parser.add_argument(
        "--reuse-db",
        dest="reuse_db",
        action="store_true",
        default=True,
        help="Reuse/normalize existing history db when possible (default)",
    )
    train_parser.add_argument(
        "--no-reuse-db",
        dest="reuse_db",
        action="store_false",
        help="Force rebuild history db from git log",
    )
    train_parser.set_defaults(func=cmd_train)

    infer_parser = sub.add_parser("infer", help="Infer defense flags from patch")
    infer_parser.add_argument(
        "--model-db",
        default=str(DEFAULT_MODEL_DB),
        help="Path to trained model sqlite db",
    )
    infer_parser.add_argument("--patch", required=True, help="Path to patch file")
    infer_parser.add_argument("--top-k", type=int, default=5, help="Top-K flags to output")
    infer_parser.add_argument(
        "--out",
        default="",
        help="Write JSON result to this path",
    )
    infer_parser.add_argument(
        "--html-out",
        default="",
        help="Write readable HTML report to this path",
    )
    infer_parser.set_defaults(func=cmd_infer)

    inspect_parser = sub.add_parser("inspect", help="Inspect model metadata and stats")
    inspect_parser.add_argument(
        "--model-db",
        default=str(DEFAULT_MODEL_DB),
        help="Path to trained model sqlite db",
    )
    inspect_parser.set_defaults(func=cmd_inspect)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

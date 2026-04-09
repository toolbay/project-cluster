from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .constants import (
    CLUSTER_SCORE_WEIGHT,
    DIRECT_HINT_BOOST,
    KEYWORD_SCORE_WEIGHT,
)
from .model_store import load_model
from .patch_parser import parse_patch
from .path_utils import directory_prefixes, normalize_repo_path
from .token_utils import normalize_counter

InferProgressFn = Optional[Callable[[Dict[str, object]], None]]
INFER_TOTAL_STEPS = 7

STAGE_INFO = {
    1: {"key": "input_scan", "name": "Input Scan", "phase": "thinking"},
    2: {"key": "model_sync", "name": "Model Sync", "phase": "thinking"},
    3: {"key": "patch_decode", "name": "Patch Decode", "phase": "reading"},
    4: {"key": "feature_projection", "name": "Feature Projection", "phase": "mapping"},
    5: {"key": "candidate_scoring", "name": "Candidate Scoring", "phase": "deciding"},
    6: {"key": "rank_decision", "name": "Rank Decision", "phase": "deciding"},
    7: {"key": "result_packaging", "name": "Result Packaging", "phase": "shipping"},
}


def _stage_meta(step: int) -> Dict[str, str]:
    meta = STAGE_INFO.get(step)
    if meta:
        return meta
    return {
        "key": f"step_{step}",
        "name": f"Step {step}",
        "phase": "thinking",
    }


def _emit(progress: InferProgressFn, event: Dict[str, object]) -> None:
    if progress is not None:
        progress(event)


def _event_base(step: int) -> Dict[str, object]:
    meta = _stage_meta(step)
    return {
        "step": int(step),
        "total_steps": INFER_TOTAL_STEPS,
        "stage_key": meta["key"],
        "stage_name": meta["name"],
        "phase": meta["phase"],
    }


def _step_start(
    progress: InferProgressFn,
    step: int,
    message: str,
) -> float:
    payload = _event_base(step)
    payload.update(
        {
            "event": "step_start",
            "message": message,
        }
    )
    _emit(progress, payload)
    return time.perf_counter()


def _step_end(
    progress: InferProgressFn,
    step: int,
    summary: str,
    started_at: float,
    metrics: Optional[Dict[str, object]] = None,
) -> None:
    elapsed_ms = int((time.perf_counter() - started_at) * 1000)
    payload = _event_base(step)
    payload.update(
        {
            "event": "step_end",
            "summary": summary,
            "elapsed_ms": elapsed_ms,
            "metrics": metrics or {},
        }
    )
    _emit(progress, payload)


def _step_detail(
    progress: InferProgressFn,
    step: int,
    title: str,
    lines: Optional[List[str]] = None,
    table: Optional[Dict[str, object]] = None,
    panel: Optional[Dict[str, object]] = None,
) -> None:
    payload = _event_base(step)
    payload.update(
        {
            "event": "step_detail",
            "title": title,
        }
    )
    if lines:
        payload["lines"] = lines
    if table:
        payload["table"] = table
    if panel:
        payload["panel"] = panel
    _emit(progress, payload)


def _step_progress(
    progress: InferProgressFn,
    step: int,
    message: str,
    current: Optional[int] = None,
    total: Optional[int] = None,
    metrics: Optional[Dict[str, object]] = None,
) -> None:
    payload = _event_base(step)
    payload.update(
        {
            "event": "step_progress",
            "message": message,
            "current": current,
            "total": total,
            "metrics": metrics or {},
        }
    )
    _emit(progress, payload)


def _resolve_cluster(
    file_path: str,
    file_clusters: Dict[str, int],
    prefix_clusters: Dict[str, Tuple[int, int]],
) -> Tuple[Optional[int], str]:
    normalized = normalize_repo_path(file_path)
    if normalized in file_clusters:
        return file_clusters[normalized], "exact"

    for prefix in directory_prefixes(normalized):
        if prefix in prefix_clusters:
            return prefix_clusters[prefix][0], "prefix"

    return None, "none"


def _compute_cluster_score(
    patch_cluster_weights: Dict[int, float],
    flag_cluster_map: Dict[int, float],
    cluster_priors: Dict[int, float],
) -> Tuple[float, float]:
    if not patch_cluster_weights or not flag_cluster_map:
        return 0.0, 0.0

    flag_total = float(sum(flag_cluster_map.values()))
    if flag_total <= 0:
        return 0.0, 0.0

    positive_priors = [value for value in cluster_priors.values() if value > 0]
    min_prior = min(positive_priors) if positive_priors else 1e-6

    raw_score = 0.0
    for cluster_id, patch_weight in patch_cluster_weights.items():
        flag_prob = flag_cluster_map.get(cluster_id, 0.0) / flag_total
        if flag_prob <= 0:
            continue
        prior = cluster_priors.get(cluster_id, min_prior)
        raw_score += patch_weight * (flag_prob / max(prior, min_prior))

    return math.log1p(raw_score), raw_score


def _compute_keyword_score(
    patch_token_weights: Dict[str, float],
    flag_keyword_map: Dict[str, float],
) -> Tuple[float, list[str]]:
    if not patch_token_weights or not flag_keyword_map:
        return 0.0, []

    keyword_total = float(sum(flag_keyword_map.values()))
    if keyword_total <= 0:
        return 0.0, []

    contributions = []
    overlap = 0.0
    for token, patch_weight in patch_token_weights.items():
        keyword_weight = flag_keyword_map.get(token, 0.0)
        if keyword_weight <= 0:
            continue
        contribution = patch_weight * keyword_weight
        overlap += contribution
        contributions.append((token, contribution))

    contributions.sort(key=lambda item: item[1], reverse=True)
    matched_tokens = [token for token, _ in contributions[:5]]
    return overlap / keyword_total, matched_tokens


def _format_bytes(size: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    value = float(size)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f}{unit}" if unit != "B" else f"{int(value)}B"
        value /= 1024.0
    return f"{int(size)}B"


def _format_mtime(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _to_table(headers: List[str], rows: List[List[object]]) -> Dict[str, object]:
    return {
        "headers": headers,
        "rows": rows,
    }


def _normalized(value: float, max_value: float) -> float:
    if max_value <= 0:
        return 0.0
    return min(1.0, max(0.0, value / max_value))


def _cluster_only_candidates(
    patch_cluster_weights: Dict[int, float],
    flag_cluster_weights: Dict[str, Dict[int, float]],
    cluster_priors: Dict[int, float],
    top_n: int = 5,
) -> List[Dict[str, object]]:
    candidates: List[Dict[str, object]] = []
    if not patch_cluster_weights:
        return candidates

    for flag_name, cluster_map in flag_cluster_weights.items():
        score_log, score_raw = _compute_cluster_score(
            patch_cluster_weights=patch_cluster_weights,
            flag_cluster_map=cluster_map,
            cluster_priors=cluster_priors,
        )
        if score_raw <= 0:
            continue

        total = float(sum(cluster_map.values()))
        primary_cluster = None
        if total > 0:
            primary_cluster, _ = max(cluster_map.items(), key=lambda item: item[1])

        candidates.append(
            {
                "flag": flag_name,
                "score_log": score_log,
                "score_raw": score_raw,
                "primary_cluster": primary_cluster,
            }
        )

    candidates.sort(key=lambda item: item["score_raw"], reverse=True)
    return candidates[:top_n]


def _keyword_only_candidates(
    patch_token_weights: Dict[str, float],
    flag_keyword_weights: Dict[str, Dict[str, float]],
    direct_hints: set[str],
    top_n: int = 5,
) -> List[Dict[str, object]]:
    candidates: List[Dict[str, object]] = []
    if not patch_token_weights and not direct_hints:
        return candidates

    for flag_name, keyword_map in flag_keyword_weights.items():
        keyword_score, _ = _compute_keyword_score(
            patch_token_weights=patch_token_weights,
            flag_keyword_map=keyword_map,
        )
        direct_boost = DIRECT_HINT_BOOST if flag_name in direct_hints else 0.0
        score_raw = keyword_score + direct_boost
        if score_raw <= 0:
            continue
        candidates.append(
            {
                "flag": flag_name,
                "score_log": math.log1p(score_raw),
                "score_raw": score_raw,
                "primary_cluster": None,
            }
        )

    candidates.sort(
        key=lambda item: (float(item["score_raw"]), str(item["flag"])),
        reverse=True,
    )
    return candidates[:top_n]


def _infer_cluster_weights_from_flag_candidates(
    projection_candidates: List[Dict[str, object]],
    flag_cluster_weights: Dict[str, Dict[int, float]],
) -> Dict[int, float]:
    cluster_counter: Counter[int] = Counter()

    for candidate in projection_candidates:
        flag_name = str(candidate.get("flag", ""))
        base_weight = float(candidate.get("score_raw", 0.0))
        if base_weight <= 0.0:
            continue

        cluster_map = flag_cluster_weights.get(flag_name, {})
        if not cluster_map:
            continue

        total = float(sum(cluster_map.values()))
        if total <= 0.0:
            continue

        for cluster_id, weight in cluster_map.items():
            if weight <= 0.0:
                continue
            cluster_counter[int(cluster_id)] += base_weight * (float(weight) / total)

    return normalize_counter(cluster_counter)


def _write_label(grid: List[List[str]], x: int, y: int, label: str) -> bool:
    height = len(grid)
    width = len(grid[0]) if grid else 0
    if width == 0:
        return False

    offsets = [
        (0, 0),
        (1, 0),
        (-1, 0),
        (0, -1),
        (0, 1),
        (1, -1),
        (-1, 1),
        (2, 0),
        (-2, 0),
    ]

    for dx, dy in offsets:
        px = x + dx
        py = y + dy
        if py < 1 or py >= height - 1:
            continue
        if px < 1 or px + len(label) >= width - 1:
            continue
        if all(grid[py][px + i] == " " for i in range(len(label))):
            for i, ch in enumerate(label):
                grid[py][px + i] = ch
            return True

    return False


def _build_base_canvas(width: int = 41, height: int = 17) -> Tuple[List[List[str]], Callable[[float, float], Tuple[int, int]]]:
    grid = [[" " for _ in range(width)] for _ in range(height)]

    grid[0][0] = "┌"
    grid[0][width - 1] = "┐"
    grid[height - 1][0] = "└"
    grid[height - 1][width - 1] = "┘"
    for x in range(1, width - 1):
        grid[0][x] = "─"
        grid[height - 1][x] = "─"
    for y in range(1, height - 1):
        grid[y][0] = "│"
        grid[y][width - 1] = "│"

    x_axis_row = height - 3
    y_axis_col = 3
    for x in range(1, width - 1):
        grid[x_axis_row][x] = "─"
    for y in range(1, height - 1):
        grid[y][y_axis_col] = "│"
    grid[x_axis_row][y_axis_col] = "┼"

    plot_x_min = y_axis_col + 1
    plot_x_max = width - 2
    plot_y_min = 1
    plot_y_max = x_axis_row - 1

    def to_canvas(xv: float, yv: float) -> Tuple[int, int]:
        xv = min(1.0, max(0.0, xv))
        yv = min(1.0, max(0.0, yv))
        px = plot_x_min + int(round(xv * (plot_x_max - plot_x_min)))
        py = plot_y_max - int(round(yv * (plot_y_max - plot_y_min)))
        return px, py

    grid[height - 2][width - 5] = "x"
    grid[1][1] = "y"
    return grid, to_canvas


def _build_flag_relations(
    projection_candidates: List[Dict[str, object]],
    patch_cluster_weights: Dict[int, float],
    cluster_priors: Dict[int, float],
    patch_token_weights: Dict[str, float],
    flag_cluster_weights: Dict[str, Dict[int, float]],
    flag_keyword_weights: Dict[str, Dict[str, float]],
    direct_hints: set[str],
) -> List[Dict[str, object]]:
    relations: List[Dict[str, object]] = []
    for item in projection_candidates:
        flag_name = str(item["flag"])
        cluster_score, cluster_raw = _compute_cluster_score(
            patch_cluster_weights=patch_cluster_weights,
            flag_cluster_map=flag_cluster_weights.get(flag_name, {}),
            cluster_priors=cluster_priors,
        )
        keyword_score, matched_keywords = _compute_keyword_score(
            patch_token_weights=patch_token_weights,
            flag_keyword_map=flag_keyword_weights.get(flag_name, {}),
        )
        direct_boost = DIRECT_HINT_BOOST if flag_name in direct_hints else 0.0
        total_score = (
            CLUSTER_SCORE_WEIGHT * cluster_score
            + KEYWORD_SCORE_WEIGHT * keyword_score
            + direct_boost
        )
        relations.append(
            {
                "flag": flag_name,
                "cluster_raw": float(cluster_raw),
                "keyword_score": float(keyword_score),
                "direct_boost": float(direct_boost),
                "total_score": float(total_score),
                "primary_cluster": item.get("primary_cluster"),
                "matched_keywords": matched_keywords,
            }
        )

    relations.sort(key=lambda row: row["total_score"], reverse=True)
    return relations


def _build_patch_flag_projection_panel(
    flag_relations: List[Dict[str, object]],
) -> Dict[str, object]:
    grid, to_canvas = _build_base_canvas(width=41, height=17)
    legend_rows: List[List[object]] = []

    patch_px, patch_py = to_canvas(0.06, 0.50)
    _write_label(grid, patch_px, patch_py, "●")
    legend_rows.append(
        [
            "●",
            "patch",
            "patch",
            f"({patch_px},{patch_py})",
            "x=0.06, y=0.50",
            "anchor",
        ]
    )

    selected = flag_relations[:5]
    max_cluster = max((float(row["cluster_raw"]) for row in selected), default=0.0)
    max_lexical = max(
        (
            float(row["keyword_score"]) + float(row["direct_boost"])
            for row in selected
        ),
        default=0.0,
    )

    for idx, row in enumerate(selected, start=1):
        cluster_norm = _normalized(float(row["cluster_raw"]), max_cluster) if max_cluster > 0 else 0.05
        lexical_raw = float(row["keyword_score"]) + float(row["direct_boost"])
        if max_lexical > 0:
            lexical_norm = _normalized(lexical_raw, max_lexical)
        else:
            lexical_norm = 1.0 - ((idx - 1) / max(1, len(selected)))

        px, py = to_canvas(max(0.08, cluster_norm), max(0.06, lexical_norm))
        mark = f"F{idx}"
        _write_label(grid, px, py, mark)

        legend_rows.append(
            [
                mark,
                "flag",
                row["flag"],
                f"({px},{py})",
                f"x={max(0.08, cluster_norm):.2f}, y={max(0.06, lexical_norm):.2f}",
                (
                    f"total={float(row['total_score']):.4f}, "
                    f"cluster={float(row['cluster_raw']):.4f}, "
                    f"lex={lexical_raw:.4f}"
                ),
            ]
        )

    return {
        "title": "Patch <> Candidate Flags (41x17)",
        "lines": ["".join(row) for row in grid],
        "legend": _to_table(
            headers=["Mark", "Type", "Entity", "Grid", "Projected", "Meta"],
            rows=legend_rows,
        ),
    }


def _build_cluster_relations(
    patch_cluster_weights: Dict[int, float],
    inferred_cluster_weights: Dict[int, float],
    cluster_priors: Dict[int, float],
    cluster_to_files: Dict[int, List[str]],
) -> List[Dict[str, object]]:
    cluster_ids = sorted(
        set(cluster_priors.keys())
        | set(patch_cluster_weights.keys())
        | set(inferred_cluster_weights.keys())
    )
    positive_priors = [value for value in cluster_priors.values() if value > 0]
    min_prior = min(positive_priors) if positive_priors else 1e-6

    relations: List[Dict[str, object]] = []
    for cluster_id in cluster_ids:
        prior = float(cluster_priors.get(cluster_id, min_prior))
        patch_weight = float(patch_cluster_weights.get(cluster_id, 0.0))
        inferred_weight = float(inferred_cluster_weights.get(cluster_id, 0.0))
        combined_weight = patch_weight if patch_weight > 0.0 else inferred_weight
        lift = combined_weight / max(prior, min_prior)
        file_hits = len(cluster_to_files.get(cluster_id, []))
        relations.append(
            {
                "cluster_id": int(cluster_id),
                "patch_weight": patch_weight,
                "inferred_weight": inferred_weight,
                "combined_weight": combined_weight,
                "prior": prior,
                "lift": lift,
                "file_hits": file_hits,
                "active": combined_weight > 0.0,
            }
        )

    relations.sort(
        key=lambda row: (
            float(row["combined_weight"]),
            float(row["lift"]),
            -int(row["cluster_id"]),
        ),
        reverse=True,
    )
    return relations


def _build_patch_cluster_projection_panel(
    cluster_relations: List[Dict[str, object]],
) -> Dict[str, object]:
    grid, to_canvas = _build_base_canvas(width=41, height=17)
    legend_rows: List[List[object]] = []

    patch_px, patch_py = to_canvas(0.06, 0.50)
    _write_label(grid, patch_px, patch_py, "●")
    legend_rows.append(
        [
            "●",
            "patch",
            "patch",
            f"({patch_px},{patch_py})",
            "x=0.06, y=0.50",
            "anchor",
        ]
    )

    active_rows = [row for row in cluster_relations if bool(row["active"])]
    if active_rows:
        selected = active_rows[:8]
    else:
        selected = sorted(
            cluster_relations,
            key=lambda row: (float(row["prior"]), int(row["cluster_id"])),
        )[:8]

    max_weight = max((float(row["combined_weight"]) for row in selected), default=0.0)
    max_lift = max((float(row["lift"]) for row in selected), default=0.0)
    max_prior = max((float(row["prior"]) for row in selected), default=0.0)

    for idx, row in enumerate(selected, start=1):
        if max_weight > 0:
            x_norm = _normalized(float(row["combined_weight"]), max_weight)
        else:
            x_norm = 0.08

        if max_lift > 0:
            y_norm = _normalized(float(row["lift"]), max_lift)
        elif max_prior > 0:
            y_norm = 1.0 - _normalized(float(row["prior"]), max_prior)
        else:
            y_norm = 0.5

        mark = f"C{idx}"
        px, py = to_canvas(max(0.08, x_norm), max(0.06, y_norm))
        _write_label(grid, px, py, mark)

        legend_rows.append(
            [
                mark,
                "cluster",
                f"cluster-{int(row['cluster_id'])}",
                f"({px},{py})",
                f"x={max(0.08, x_norm):.2f}, y={max(0.06, y_norm):.2f}",
                (
                    f"w={float(row['combined_weight']):.4f}, "
                    f"hard={float(row['patch_weight']):.4f}, "
                    f"infer={float(row['inferred_weight']):.4f}, "
                    f"prior={float(row['prior']):.4f}, "
                    f"lift={float(row['lift']):.2f}, "
                    f"files={int(row['file_hits'])}"
                ),
            ]
        )

    return {
        "title": "Patch <> Clusters (41x17)",
        "lines": ["".join(row) for row in grid],
        "legend": _to_table(
            headers=["Mark", "Type", "Entity", "Grid", "Projected", "Meta"],
            rows=legend_rows,
        ),
    }


def infer_patch(
    model_db_path: str,
    patch_path: str,
    top_k: int,
    progress: InferProgressFn = None,
) -> Dict[str, object]:
    step_start_ts = _step_start(
        progress,
        step=1,
        message="checking input files and metadata...",
    )
    model_path = Path(model_db_path)
    patch_file = Path(patch_path)
    if not model_path.exists():
        raise FileNotFoundError(f"model db not found: {model_path}")
    if not patch_file.exists():
        raise FileNotFoundError(f"patch not found: {patch_file}")

    model_stat = model_path.stat()
    patch_stat = patch_file.stat()
    _step_detail(
        progress,
        step=1,
        title="Input Probe",
        table=_to_table(
            headers=["Field", "Value"],
            rows=[
                ["model_db", str(model_path.resolve())],
                ["patch", str(patch_file.resolve())],
                [
                    "model_file",
                    f"size={_format_bytes(model_stat.st_size)}, mtime={_format_mtime(model_stat.st_mtime)}",
                ],
                [
                    "patch_file",
                    f"size={_format_bytes(patch_stat.st_size)}, mtime={_format_mtime(patch_stat.st_mtime)}",
                ],
            ],
        ),
    )
    _step_end(
        progress,
        step=1,
        summary="inputs ready",
        started_at=step_start_ts,
        metrics={
            "model": "ok",
            "patch": "ok",
            "patch_size": _format_bytes(patch_stat.st_size),
        },
    )

    step_start_ts = _step_start(
        progress,
        step=2,
        message="loading model graph and weight tables...",
    )
    loaded = load_model(str(model_path))
    model_meta = loaded["meta"]
    flag_catalog = loaded["flag_catalog"]
    file_clusters = loaded["file_clusters"]
    prefix_clusters = loaded["prefix_clusters"]
    flag_cluster_weights = loaded["flag_cluster_weights"]
    cluster_priors = loaded.get("cluster_priors", {})
    flag_keyword_weights = loaded.get("flag_keyword_weights", {})

    _step_detail(
        progress,
        step=2,
        title="Model Snapshot",
        table=_to_table(
            headers=["Field", "Value"],
            rows=[
                ["schema_version", model_meta.get("schema_version", "?")],
                ["trained_at", model_meta.get("trained_at", "?")],
                ["v8_head", (model_meta.get("v8_head") or "?")[:12]],
                ["flags", len(flag_catalog)],
                ["weighted_flags", len(flag_cluster_weights)],
                ["clusters", len(set(file_clusters.values()))],
                ["prefix_clusters", len(prefix_clusters)],
            ],
        ),
    )
    _step_end(
        progress,
        step=2,
        summary="model loaded",
        started_at=step_start_ts,
        metrics={
            "flags": len(flag_catalog),
            "clusters": len(set(file_clusters.values())),
            "priors": len(cluster_priors),
        },
    )

    step_start_ts = _step_start(
        progress,
        step=3,
        message="parsing patch structure and lexical hints...",
    )
    known_flags = set(flag_catalog.keys())
    patch_info = parse_patch(str(patch_file), known_flags=known_flags)

    files_rows = [
        [str(idx), file_path, "-"]
        for idx, file_path in enumerate(patch_info.files[:12], start=1)
    ]
    if not files_rows:
        files_rows = [["-", "(no files)", "-"]]

    tokens_rows = [
        [token, str(count)]
        for token, count in sorted(
            patch_info.token_counts.items(), key=lambda item: item[1], reverse=True
        )[:12]
    ]
    if not tokens_rows:
        tokens_rows = [["-", "0"]]

    hints_rows = [[hint] for hint in sorted(patch_info.direct_flag_hints)[:12]]
    if not hints_rows:
        hints_rows = [["none"]]

    _step_detail(
        progress,
        step=3,
        title="Patch Files",
        table=_to_table(
            headers=["#", "File", "Hint"],
            rows=files_rows,
        ),
    )
    _step_detail(
        progress,
        step=3,
        title="Patch Tokens",
        table=_to_table(
            headers=["Token", "Count"],
            rows=tokens_rows,
        ),
    )
    _step_detail(
        progress,
        step=3,
        title="Direct Hint Flags",
        table=_to_table(
            headers=["Flag"],
            rows=hints_rows,
        ),
    )

    _step_end(
        progress,
        step=3,
        summary="patch parsed",
        started_at=step_start_ts,
        metrics={
            "files": len(patch_info.files),
            "tokens": len(patch_info.token_counts),
            "direct_hints": len(patch_info.direct_flag_hints),
        },
    )

    step_start_ts = _step_start(
        progress,
        step=4,
        message="projecting files into cluster feature space...",
    )
    patch_cluster_counter: Counter[int] = Counter()
    cluster_to_files = defaultdict(list)
    file_resolutions = []
    exact_hits = 0
    prefix_hits = 0
    unmapped_hits = 0

    progress_interval = max(1, len(patch_info.files) // 6) if patch_info.files else 1

    for index, file_path in enumerate(patch_info.files, start=1):
        cluster_id, source = _resolve_cluster(file_path, file_clusters, prefix_clusters)
        file_resolutions.append(
            {
                "file_path": file_path,
                "cluster_id": cluster_id,
                "resolution": source,
            }
        )

        if source == "exact":
            exact_hits += 1
        elif source == "prefix":
            prefix_hits += 1
        else:
            unmapped_hits += 1

        if cluster_id is not None:
            patch_cluster_counter[cluster_id] += 1
            cluster_to_files[cluster_id].append(file_path)

        if index == 1 or index == len(patch_info.files) or (index % progress_interval == 0):
            _step_progress(
                progress,
                step=4,
                message=f"mapping file {index}: {file_path}",
                current=index,
                total=len(patch_info.files),
                metrics={
                    "exact": exact_hits,
                    "prefix": prefix_hits,
                    "unmapped": unmapped_hits,
                },
            )

    patch_cluster_weights = normalize_counter(patch_cluster_counter)
    patch_token_weights = normalize_counter(Counter(patch_info.token_counts))

    map_rows = []
    for row in file_resolutions[:12]:
        cluster_id = row["cluster_id"]
        map_rows.append(
            [
                row["file_path"],
                row["resolution"],
                f"C{cluster_id}" if cluster_id is not None else "-",
            ]
        )
    if not map_rows:
        map_rows = [["(no files)", "-", "-"]]

    _step_detail(
        progress,
        step=4,
        title="Mapping Table",
        table=_to_table(
            headers=["File", "Resolution", "Cluster"],
            rows=map_rows,
        ),
    )

    cluster_candidates = _cluster_only_candidates(
        patch_cluster_weights=patch_cluster_weights,
        flag_cluster_weights=flag_cluster_weights,
        cluster_priors=cluster_priors,
        top_n=20,
    )
    projection_source = "cluster"
    projection_candidates = cluster_candidates
    if not projection_candidates:
        projection_candidates = _keyword_only_candidates(
            patch_token_weights=patch_token_weights,
            flag_keyword_weights=flag_keyword_weights,
            direct_hints=patch_info.direct_flag_hints,
            top_n=20,
        )
        projection_source = "keyword"

    seed_rows = []
    for idx, item in enumerate(projection_candidates[:8], start=1):
        cluster_id = item.get("primary_cluster")
        seed_rows.append(
            [
                str(idx),
                str(item["flag"]),
                f"{float(item['score_raw']):.4f}",
                f"C{cluster_id}" if cluster_id is not None else "-",
                projection_source,
            ]
        )
    if not seed_rows:
        seed_rows = [["-", "(none)", "0.0000", "-", projection_source]]

    _step_detail(
        progress,
        step=4,
        title="Projection Seeds",
        table=_to_table(
            headers=["#", "Flag", "BaseScore", "PrimaryCluster", "Source"],
            rows=seed_rows,
        ),
    )

    flag_relations = _build_flag_relations(
        projection_candidates=projection_candidates,
        patch_cluster_weights=patch_cluster_weights,
        cluster_priors=cluster_priors,
        patch_token_weights=patch_token_weights,
        flag_cluster_weights=flag_cluster_weights,
        flag_keyword_weights=flag_keyword_weights,
        direct_hints=patch_info.direct_flag_hints,
    )
    flag_rows = []
    for idx, row in enumerate(flag_relations[:8], start=1):
        flag_rows.append(
            [
                str(idx),
                str(row["flag"]),
                f"{float(row['total_score']):.4f}",
                f"{float(row['cluster_raw']):.4f}",
                f"{float(row['keyword_score']):.4f}",
                f"{float(row['direct_boost']):.2f}",
                ",".join(str(tok) for tok in (row["matched_keywords"] or [])) or "-",
            ]
        )
    if not flag_rows:
        flag_rows = [["-", "(none)", "0.0000", "0.0000", "0.0000", "0.00", "-"]]

    _step_detail(
        progress,
        step=4,
        title="Patch-Flag Relation Matrix",
        table=_to_table(
            headers=["#", "Flag", "Total", "Cluster", "Keyword", "Direct", "Keywords"],
            rows=flag_rows,
        ),
    )

    flag_projection_panel = _build_patch_flag_projection_panel(flag_relations=flag_relations)
    _step_detail(
        progress,
        step=4,
        title="Patch-Flag Projection",
        panel=flag_projection_panel,
    )

    inferred_cluster_weights = _infer_cluster_weights_from_flag_candidates(
        projection_candidates=projection_candidates,
        flag_cluster_weights=flag_cluster_weights,
    )
    cluster_relations = _build_cluster_relations(
        patch_cluster_weights=patch_cluster_weights,
        inferred_cluster_weights=inferred_cluster_weights,
        cluster_priors=cluster_priors,
        cluster_to_files=cluster_to_files,
    )
    cluster_rows = []
    for row in cluster_relations[:16]:
        cluster_rows.append(
            [
                f"C{int(row['cluster_id'])}",
                f"{float(row['patch_weight']):.4f}",
                f"{float(row['inferred_weight']):.4f}",
                f"{float(row['combined_weight']):.4f}",
                f"{float(row['prior']):.4f}",
                f"{float(row['lift']):.2f}",
                str(int(row["file_hits"])),
                str(bool(row["active"])),
            ]
        )
    if not cluster_rows:
        cluster_rows = [["-", "0.0000", "0.0000", "0.0000", "0.0000", "0.00", "0", "False"]]

    _step_detail(
        progress,
        step=4,
        title="Patch-Cluster Relation Matrix",
        table=_to_table(
            headers=[
                "Cluster",
                "HardW",
                "InferW",
                "CombinedW",
                "Prior",
                "Lift",
                "FileHits",
                "Active",
            ],
            rows=cluster_rows,
        ),
    )

    cluster_projection_panel = _build_patch_cluster_projection_panel(
        cluster_relations=cluster_relations
    )
    _step_detail(
        progress,
        step=4,
        title="Patch-Cluster Projection",
        panel=cluster_projection_panel,
    )

    _step_end(
        progress,
        step=4,
        summary="feature projection ready",
        started_at=step_start_ts,
        metrics={
            "mapped": exact_hits + prefix_hits,
            "exact": exact_hits,
            "prefix": prefix_hits,
            "unmapped": unmapped_hits,
            "clusters": len(patch_cluster_weights),
            "proj_flags": len(projection_candidates),
            "active_clusters": len([row for row in cluster_relations if row["active"]]),
            "inferred_clusters": len(inferred_cluster_weights),
        },
    )

    step_start_ts = _step_start(
        progress,
        step=5,
        message="scoring candidate flags across cluster/keyword/direct-hint signals...",
    )
    scored = []
    leader_flag = "-"
    leader_score = -1.0
    leader_switches = 0

    total_flags = len(flag_catalog)
    progress_interval = max(1, total_flags // 18)

    for index, (flag_name, info) in enumerate(flag_catalog.items(), start=1):
        cluster_map = flag_cluster_weights.get(flag_name, {})
        keyword_map = flag_keyword_weights.get(flag_name, {})

        cluster_score, cluster_score_raw = _compute_cluster_score(
            patch_cluster_weights=patch_cluster_weights,
            flag_cluster_map=cluster_map,
            cluster_priors=cluster_priors,
        )
        keyword_score, matched_keywords = _compute_keyword_score(
            patch_token_weights=patch_token_weights,
            flag_keyword_map=keyword_map,
        )

        direct_boost = DIRECT_HINT_BOOST if flag_name in patch_info.direct_flag_hints else 0.0

        total_score = (
            CLUSTER_SCORE_WEIGHT * cluster_score
            + KEYWORD_SCORE_WEIGHT * keyword_score
            + direct_boost
        )

        matched_clusters = sorted(
            cluster_id
            for cluster_id in patch_cluster_counter
            if cluster_id in cluster_map and cluster_map[cluster_id] > 0.0
        )

        if total_score > leader_score:
            leader_score = total_score
            leader_flag = flag_name
            leader_switches += 1
            if leader_switches <= 6:
                _step_progress(
                    progress,
                    step=5,
                    message=f"leader switch -> {leader_flag}",
                    current=index,
                    total=total_flags,
                    metrics={"score": round(leader_score, 4)},
                )

        if total_score <= 0.0 and not matched_clusters:
            if index % progress_interval == 0 or index == total_flags:
                _step_progress(
                    progress,
                    step=5,
                    message="scanning candidates",
                    current=index,
                    total=total_flags,
                    metrics={"leader": leader_flag, "scored": len(scored)},
                )
            continue

        recommended_cmd = f"--no-{flag_name}" if info.is_bool else None

        scored.append(
            {
                "flag": flag_name,
                "recommended_cmd": recommended_cmd,
                "score": round(total_score, 6),
                "evidence": {
                    "matched_clusters": matched_clusters,
                    "matched_keywords": matched_keywords,
                    "direct_patch_hint": flag_name in patch_info.direct_flag_hints,
                    "cluster_score": round(cluster_score, 6),
                    "cluster_score_raw": round(cluster_score_raw, 6),
                    "keyword_score": round(keyword_score, 6),
                    "direct_hint_boost": round(direct_boost, 6),
                },
            }
        )

        if index % progress_interval == 0 or index == total_flags:
            _step_progress(
                progress,
                step=5,
                message="scoring in progress",
                current=index,
                total=total_flags,
                metrics={"leader": leader_flag, "scored": len(scored)},
            )

    preview = sorted(scored, key=lambda item: item["score"], reverse=True)[:5]
    radar_rows = []
    for idx, item in enumerate(preview, start=1):
        radar_rows.append(
            [
                str(idx),
                item["flag"],
                f"{item['score']:.4f}",
                f"{item['evidence']['cluster_score']:.4f}",
                f"{item['evidence']['keyword_score']:.4f}",
                f"{item['evidence']['direct_hint_boost']:.1f}",
                ",".join(str(x) for x in item["evidence"]["matched_clusters"]) or "-",
            ]
        )
    if not radar_rows:
        radar_rows = [["-", "no scored candidates", "-", "-", "-", "-", "-"]]

    _step_detail(
        progress,
        step=5,
        title="Scoring Radar",
        table=_to_table(
            headers=["#", "Flag", "Total", "Cluster", "Keyword", "Direct", "Clusters"],
            rows=radar_rows,
        ),
    )

    _step_end(
        progress,
        step=5,
        summary="scoring done",
        started_at=step_start_ts,
        metrics={
            "scored_flags": len(scored),
            "direct_hints": len(patch_info.direct_flag_hints),
            "leader_switches": leader_switches,
        },
    )

    step_start_ts = _step_start(
        progress,
        step=6,
        message="sorting and applying tie-break decisions...",
    )
    _step_progress(
        progress,
        step=6,
        message="phase 1: sort by total score",
        current=1,
        total=3,
    )

    scored.sort(
        key=lambda item: (
            item["score"],
            item["evidence"]["direct_hint_boost"],
            item["evidence"]["cluster_score"],
            item["evidence"]["keyword_score"],
            item["flag"],
        ),
        reverse=True,
    )

    _step_progress(
        progress,
        step=6,
        message="phase 2: apply tie-breakers",
        current=2,
        total=3,
    )

    top_k_items = scored[: max(1, top_k)]
    top1 = top_k_items[0] if top_k_items else None

    _step_progress(
        progress,
        step=6,
        message="phase 3: freeze final top-k",
        current=3,
        total=3,
        metrics={"top1": top1["flag"] if top1 else "-"},
    )

    decision_rows = []
    for idx, item in enumerate(top_k_items[:8], start=1):
        decision_rows.append(
            [
                str(idx),
                item["flag"],
                f"{item['score']:.4f}",
                ",".join(str(x) for x in item["evidence"]["matched_clusters"]) or "-",
                ",".join(str(x) for x in item["evidence"]["matched_keywords"]) or "-",
                str(item["evidence"]["direct_patch_hint"]),
            ]
        )
    if not decision_rows:
        decision_rows = [["-", "no candidates", "-", "-", "-", "-"]]

    _step_detail(
        progress,
        step=6,
        title="Decision Matrix",
        table=_to_table(
            headers=["#", "Flag", "Score", "MatchedClusters", "MatchedKeywords", "DirectHint"],
            rows=decision_rows,
        ),
    )

    _step_end(
        progress,
        step=6,
        summary="ranking done",
        started_at=step_start_ts,
        metrics={
            "top_k": len(top_k_items),
            "top1": top1["flag"] if top1 else "-",
        },
    )

    step_start_ts = _step_start(
        progress,
        step=7,
        message="packing result payload and evidence blocks...",
    )
    patch_clusters = [
        {
            "cluster_id": int(cluster_id),
            "weight": round(weight, 6),
            "source_files": sorted(set(cluster_to_files[cluster_id])),
        }
        for cluster_id, weight in sorted(
            patch_cluster_weights.items(), key=lambda item: item[1], reverse=True
        )
    ]

    result = {
        "input": {
            "model_db": str(model_path.resolve()),
            "patch": str(patch_file.resolve()),
            "top_k": int(top_k),
        },
        "model_meta": model_meta,
        "patch_files": patch_info.files,
        "patch_tokens": sorted(
            patch_info.token_counts.items(), key=lambda item: item[1], reverse=True
        )[:30],
        "patch_file_resolutions": file_resolutions,
        "patch_clusters": patch_clusters,
        "top_k": top_k_items,
        "top1": top1,
    }

    _step_detail(
        progress,
        step=7,
        title="Final Payload",
        table=_to_table(
            headers=["Field", "Value"],
            rows=[
                ["top1", top1["flag"] if top1 else "-"],
                ["top_k_count", len(top_k_items)],
                ["patch_clusters", len(patch_clusters)],
                ["json_keys", ", ".join(result.keys())],
            ],
        ),
    )

    _step_end(
        progress,
        step=7,
        summary="result ready",
        started_at=step_start_ts,
        metrics={
            "candidates": len(top_k_items),
            "patch_clusters": len(patch_clusters),
        },
    )

    return result

from __future__ import annotations

import re
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Callable, Dict, Optional, Set, Tuple

import networkx as nx

from .constants import (
    FLAG_DEFINITIONS_REL_PATH,
    FLAG_NAME_TOKEN_WEIGHT,
    KEYWORD_TOP_K,
    MODEL_SCHEMA_VERSION,
    RANDOM_SEED,
)
from .flag_catalog import parse_flag_definitions
from .history_db import ensure_history_db, iter_history_commits
from .model_store import write_model
from .path_utils import directory_prefixes, normalize_repo_path, uniq_preserve_order
from .token_utils import STOPWORDS, canonicalize_token, extract_token_counter, normalize_counter

ProgressFn = Optional[Callable[[str], None]]

DASH_FLAG_RE = re.compile(r"--([a-z0-9][a-z0-9_-]*)")
DEFINE_RE = re.compile(r"DEFINE_[A-Z0-9_]+\(\s*([a-zA-Z0-9_]+)\s*,")
TAG_RE = re.compile(r"\[([a-z0-9_/,\-]+)\]")
SEPARATOR_TOKEN_RE = re.compile(r"\b([a-z0-9]+(?:[_-][a-z0-9]+)+)\b")


@dataclass
class TrainResult:
    model_db: str
    history_db: str
    history_source: str
    commit_count: int
    file_count: int
    cluster_count: int
    flag_count: int
    weighted_flag_count: int
    keyword_weighted_flag_count: int



def _log(progress: ProgressFn, message: str) -> None:
    if progress:
        progress(message)



def _extract_flags_from_text(text: str, known_flags: Set[str]) -> Set[str]:
    if not text:
        return set()

    found: Set[str] = set()
    lower = text.lower()

    for match in DASH_FLAG_RE.finditer(lower):
        candidate = canonicalize_token(match.group(1))
        if candidate in known_flags:
            found.add(candidate)

    # Common in V8 commit subjects, e.g. "[maglev] ...", "[interpreter,wasm] ...".
    for match in TAG_RE.finditer(lower):
        raw_group = match.group(1)
        for token in re.split(r"[,/]", raw_group):
            candidate = canonicalize_token(token.strip())
            if candidate in known_flags:
                found.add(candidate)

    # Only accept identifier-like tokens containing separators to avoid
    # broad generic words (e.g. "log", "future") being over-counted.
    for match in SEPARATOR_TOKEN_RE.finditer(lower):
        candidate = canonicalize_token(match.group(1))
        if candidate in known_flags:
            found.add(candidate)

    return found



def _extract_define_hints_for_commit(
    v8_repo: str,
    commit_hash: str,
    known_flags: Set[str],
    cache: Dict[str, Set[str]],
) -> Set[str]:
    if commit_hash in cache:
        return cache[commit_hash]

    cmd = [
        "git",
        "-C",
        v8_repo,
        "show",
        "--no-color",
        "--pretty=format:",
        "--unified=0",
        commit_hash,
        "--",
        FLAG_DEFINITIONS_REL_PATH,
    ]
    output = subprocess.check_output(cmd, text=True, errors="ignore")

    hints: Set[str] = set()
    for line in output.splitlines():
        if not line.startswith(("+", "-")):
            continue
        if line.startswith(("+++", "---")):
            continue

        payload = line[1:]
        match = DEFINE_RE.search(payload)
        if match:
            candidate = canonicalize_token(match.group(1))
            if candidate in known_flags:
                hints.add(candidate)

        for flag_match in DASH_FLAG_RE.finditer(payload.lower()):
            candidate = canonicalize_token(flag_match.group(1))
            if candidate in known_flags:
                hints.add(candidate)

    cache[commit_hash] = hints
    return hints



def _extract_commit_token_counter(commit_msg: str, files: list[str]) -> Counter[str]:
    token_counter = extract_token_counter(commit_msg)
    for file_path in files:
        token_counter.update(extract_token_counter(file_path))
    return token_counter



def _build_file_graph(
    history_db_path: str,
    progress: ProgressFn,
    max_pair_files: int = 64,
) -> Tuple[nx.Graph, int, int]:
    graph = nx.Graph()
    commit_count = 0
    skipped_large = 0

    for _commit_hash, _commit_msg, files in iter_history_commits(history_db_path):
        uniq_files = uniq_preserve_order(normalize_repo_path(path) for path in files)
        uniq_files = [file_path for file_path in uniq_files if file_path]
        if not uniq_files:
            continue

        commit_count += 1
        graph.add_nodes_from(uniq_files)

        if len(uniq_files) <= 1:
            continue

        if len(uniq_files) > max_pair_files:
            skipped_large += 1
            continue

        for left, right in combinations(uniq_files, 2):
            data = graph.get_edge_data(left, right)
            if data is None:
                graph.add_edge(left, right, weight=1.0)
            else:
                data["weight"] += 1.0

        if commit_count % 10000 == 0:
            _log(
                progress,
                (
                    f"graph progress commits={commit_count} "
                    f"nodes={graph.number_of_nodes()} edges={graph.number_of_edges()}"
                ),
            )

    return graph, commit_count, skipped_large



def _cluster_files(graph: nx.Graph, progress: ProgressFn) -> Dict[str, int]:
    if graph.number_of_nodes() == 0:
        return {}

    _log(
        progress,
        (
            f"running community detection "
            f"nodes={graph.number_of_nodes()} edges={graph.number_of_edges()}"
        ),
    )
    communities = nx.algorithms.community.asyn_lpa_communities(
        graph,
        weight="weight",
        seed=RANDOM_SEED,
    )

    file_clusters: Dict[str, int] = {}
    for cluster_id, members in enumerate(communities):
        for file_path in members:
            file_clusters[file_path] = cluster_id

    _log(progress, f"clustered files into {len(set(file_clusters.values()))} communities")
    return file_clusters



def _build_prefix_clusters(file_clusters: Dict[str, int]) -> Dict[str, Tuple[int, int]]:
    prefix_cluster_counter: Dict[str, Counter[int]] = defaultdict(Counter)

    for file_path, cluster_id in file_clusters.items():
        for prefix in directory_prefixes(file_path):
            prefix_cluster_counter[prefix][cluster_id] += 1

    prefix_clusters: Dict[str, Tuple[int, int]] = {}
    for prefix, cluster_counter in prefix_cluster_counter.items():
        cluster_id, support = sorted(
            cluster_counter.items(), key=lambda item: (-item[1], item[0])
        )[0]
        prefix_clusters[prefix] = (int(cluster_id), int(support))

    return prefix_clusters



def _build_training_signals(
    v8_repo: str,
    history_db_path: str,
    file_clusters: Dict[str, int],
    known_flags: Set[str],
    progress: ProgressFn,
) -> Tuple[Dict[str, Dict[int, float]], Dict[int, float], Dict[str, Dict[str, float]]]:
    define_diff_cache: Dict[str, Set[str]] = {}

    flag_cluster_weights: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
    flag_keyword_weights: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    cluster_priors_counter: Counter[int] = Counter()

    commit_index = 0
    for commit_hash, commit_msg, files in iter_history_commits(history_db_path):
        commit_index += 1
        uniq_files = uniq_preserve_order(normalize_repo_path(path) for path in files)
        uniq_files = [file_path for file_path in uniq_files if file_path]

        clusters = [file_clusters[file_path] for file_path in uniq_files if file_path in file_clusters]
        if not clusters:
            continue

        cluster_counter = Counter(clusters)
        cluster_priors_counter.update(cluster_counter)

        hinted_flags = _extract_flags_from_text(commit_msg, known_flags)
        if FLAG_DEFINITIONS_REL_PATH in uniq_files:
            hinted_flags |= _extract_define_hints_for_commit(
                v8_repo=v8_repo,
                commit_hash=commit_hash,
                known_flags=known_flags,
                cache=define_diff_cache,
            )

        if not hinted_flags:
            continue

        cluster_weights = normalize_counter(cluster_counter)
        token_weights = normalize_counter(_extract_commit_token_counter(commit_msg, uniq_files))

        for flag_name in hinted_flags:
            for cluster_id, weight in cluster_weights.items():
                flag_cluster_weights[flag_name][cluster_id] += weight
            for token, weight in token_weights.items():
                flag_keyword_weights[flag_name][token] += weight

        if commit_index % 20000 == 0:
            _log(
                progress,
                (
                    f"signal progress commits={commit_index} "
                    f"weighted_flags={len(flag_cluster_weights)}"
                ),
            )

    cluster_priors = normalize_counter(cluster_priors_counter)

    # Add weak lexical priors from flag names themselves.
    for flag_name in known_flags:
        candidates = {flag_name}
        candidates.update(part for part in flag_name.split("-") if len(part) >= 3)
        for token in candidates:
            if token in STOPWORDS:
                continue
            flag_keyword_weights[flag_name][token] += FLAG_NAME_TOKEN_WEIGHT

    compact_keywords: Dict[str, Dict[str, float]] = {}
    for flag_name, token_map in flag_keyword_weights.items():
        top_pairs = sorted(token_map.items(), key=lambda item: item[1], reverse=True)[:KEYWORD_TOP_K]
        compact_keywords[flag_name] = {token: float(weight) for token, weight in top_pairs}

    return (
        {flag_name: dict(cluster_map) for flag_name, cluster_map in flag_cluster_weights.items()},
        {int(cluster_id): float(prior) for cluster_id, prior in cluster_priors.items()},
        compact_keywords,
    )



def train_model(
    v8_repo: str,
    history_db_path: str,
    model_db_path: str,
    reuse_db: bool,
    progress: ProgressFn = print,
) -> TrainResult:
    start_ts = time.time()
    _log(progress, "starting training pipeline")

    history_info = ensure_history_db(
        v8_repo=v8_repo,
        history_db_path=history_db_path,
        reuse_db=reuse_db,
        progress=progress,
    )

    flag_def_path = Path(v8_repo) / FLAG_DEFINITIONS_REL_PATH
    flag_catalog = parse_flag_definitions(str(flag_def_path))
    known_flags = set(flag_catalog.keys())
    _log(progress, f"loaded flag catalog: {len(flag_catalog)} flags")

    graph, graph_commit_count, skipped_large = _build_file_graph(
        history_db_path=history_db_path,
        progress=progress,
    )
    _log(
        progress,
        (
            f"graph ready commits={graph_commit_count} skipped_large_commits={skipped_large} "
            f"nodes={graph.number_of_nodes()} edges={graph.number_of_edges()}"
        ),
    )

    file_clusters = _cluster_files(graph=graph, progress=progress)
    prefix_clusters = _build_prefix_clusters(file_clusters)

    flag_cluster_weights, cluster_priors, flag_keyword_weights = _build_training_signals(
        v8_repo=v8_repo,
        history_db_path=history_db_path,
        file_clusters=file_clusters,
        known_flags=known_flags,
        progress=progress,
    )

    elapsed = time.time() - start_ts
    meta = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": f"{elapsed:.3f}",
        "v8_repo": str(Path(v8_repo).resolve()),
        "v8_head": subprocess.check_output(
            ["git", "-C", v8_repo, "rev-parse", "HEAD"], text=True
        ).strip(),
        "history_db": str(Path(history_db_path).resolve()),
        "history_source": history_info["history_source"],
        "commit_count": history_info["commit_count"],
        "file_count": history_info["file_count"],
        "graph_nodes": str(graph.number_of_nodes()),
        "graph_edges": str(graph.number_of_edges()),
        "cluster_count": str(len(set(file_clusters.values()))),
        "cluster_prior_count": str(len(cluster_priors)),
        "flag_count": str(len(flag_catalog)),
        "weighted_flag_count": str(len(flag_cluster_weights)),
        "keyword_weighted_flag_count": str(len(flag_keyword_weights)),
    }

    write_model(
        model_db_path=model_db_path,
        meta=meta,
        flag_catalog=flag_catalog,
        file_clusters=file_clusters,
        prefix_clusters=prefix_clusters,
        flag_cluster_weights=flag_cluster_weights,
        cluster_priors=cluster_priors,
        flag_keyword_weights=flag_keyword_weights,
    )

    _log(progress, f"model saved: {model_db_path}")

    return TrainResult(
        model_db=str(Path(model_db_path).resolve()),
        history_db=str(Path(history_db_path).resolve()),
        history_source=history_info["history_source"],
        commit_count=int(history_info["commit_count"]),
        file_count=int(history_info["file_count"]),
        cluster_count=len(set(file_clusters.values())),
        flag_count=len(flag_catalog),
        weighted_flag_count=len(flag_cluster_weights),
        keyword_weighted_flag_count=len(flag_keyword_weights),
    )

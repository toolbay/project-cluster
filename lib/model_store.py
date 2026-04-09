from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

from .flag_catalog import FlagInfo



def write_model(
    model_db_path: str,
    meta: Dict[str, str],
    flag_catalog: Dict[str, FlagInfo],
    file_clusters: Dict[str, int],
    prefix_clusters: Dict[str, Tuple[int, int]],
    flag_cluster_weights: Dict[str, Dict[int, float]],
    cluster_priors: Dict[int, float],
    flag_keyword_weights: Dict[str, Dict[str, float]],
) -> None:
    path = Path(model_db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path)
    try:
        conn.execute("DROP TABLE IF EXISTS model_meta")
        conn.execute("DROP TABLE IF EXISTS flag_catalog")
        conn.execute("DROP TABLE IF EXISTS file_clusters")
        conn.execute("DROP TABLE IF EXISTS prefix_clusters")
        conn.execute("DROP TABLE IF EXISTS flag_cluster_weights")
        conn.execute("DROP TABLE IF EXISTS cluster_priors")
        conn.execute("DROP TABLE IF EXISTS flag_keyword_weights")

        conn.execute(
            """
            CREATE TABLE model_meta (
                key TEXT PRIMARY KEY,
                value TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE flag_catalog (
                flag_name TEXT PRIMARY KEY,
                source_name TEXT NOT NULL,
                macro_type TEXT NOT NULL,
                default_raw TEXT,
                is_bool INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE file_clusters (
                file_path TEXT PRIMARY KEY,
                cluster_id INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE prefix_clusters (
                prefix_path TEXT PRIMARY KEY,
                cluster_id INTEGER NOT NULL,
                support INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE flag_cluster_weights (
                flag_name TEXT NOT NULL,
                cluster_id INTEGER NOT NULL,
                weight REAL NOT NULL,
                PRIMARY KEY(flag_name, cluster_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE cluster_priors (
                cluster_id INTEGER PRIMARY KEY,
                prior REAL NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE flag_keyword_weights (
                flag_name TEXT NOT NULL,
                keyword TEXT NOT NULL,
                weight REAL NOT NULL,
                PRIMARY KEY(flag_name, keyword)
            )
            """
        )

        conn.executemany(
            "INSERT INTO model_meta(key, value) VALUES (?, ?)",
            sorted(meta.items()),
        )

        conn.executemany(
            (
                "INSERT INTO flag_catalog(flag_name, source_name, macro_type, "
                "default_raw, is_bool) VALUES (?, ?, ?, ?, ?)"
            ),
            [
                (
                    name,
                    info.source_name,
                    info.macro_type,
                    info.default_raw,
                    1 if info.is_bool else 0,
                )
                for name, info in sorted(flag_catalog.items())
            ],
        )

        conn.executemany(
            "INSERT INTO file_clusters(file_path, cluster_id) VALUES (?, ?)",
            sorted(file_clusters.items()),
        )

        conn.executemany(
            "INSERT INTO prefix_clusters(prefix_path, cluster_id, support) VALUES (?, ?, ?)",
            [
                (prefix, cluster_id, support)
                for prefix, (cluster_id, support) in sorted(prefix_clusters.items())
            ],
        )

        cluster_rows: List[Tuple[str, int, float]] = []
        for flag_name, cluster_map in flag_cluster_weights.items():
            for cluster_id, weight in cluster_map.items():
                if weight <= 0:
                    continue
                cluster_rows.append((flag_name, int(cluster_id), float(weight)))

        conn.executemany(
            "INSERT INTO flag_cluster_weights(flag_name, cluster_id, weight) VALUES (?, ?, ?)",
            cluster_rows,
        )

        conn.executemany(
            "INSERT INTO cluster_priors(cluster_id, prior) VALUES (?, ?)",
            [(int(cluster_id), float(prior)) for cluster_id, prior in sorted(cluster_priors.items())],
        )

        keyword_rows: List[Tuple[str, str, float]] = []
        for flag_name, keyword_map in flag_keyword_weights.items():
            for keyword, weight in keyword_map.items():
                if weight <= 0:
                    continue
                keyword_rows.append((flag_name, keyword, float(weight)))

        conn.executemany(
            "INSERT INTO flag_keyword_weights(flag_name, keyword, weight) VALUES (?, ?, ?)",
            keyword_rows,
        )

        conn.execute(
            "CREATE INDEX idx_flag_cluster_weights_flag ON flag_cluster_weights(flag_name)"
        )
        conn.execute(
            "CREATE INDEX idx_flag_cluster_weights_cluster ON flag_cluster_weights(cluster_id)"
        )
        conn.execute("CREATE INDEX idx_file_clusters_cluster ON file_clusters(cluster_id)")
        conn.execute("CREATE INDEX idx_prefix_clusters_cluster ON prefix_clusters(cluster_id)")
        conn.execute(
            "CREATE INDEX idx_flag_keyword_weights_flag ON flag_keyword_weights(flag_name)"
        )

        conn.commit()
    finally:
        conn.close()



def load_model(model_db_path: str) -> Dict[str, object]:
    conn = sqlite3.connect(model_db_path)
    try:
        meta = dict(conn.execute("SELECT key, value FROM model_meta"))

        flag_catalog: Dict[str, FlagInfo] = {}
        for row in conn.execute(
            "SELECT flag_name, source_name, macro_type, default_raw, is_bool FROM flag_catalog"
        ):
            flag_name, source_name, macro_type, default_raw, is_bool = row
            flag_catalog[flag_name] = FlagInfo(
                canonical_name=flag_name,
                source_name=source_name,
                macro_type=macro_type,
                default_raw=default_raw,
                is_bool=bool(is_bool),
            )

        file_clusters = dict(conn.execute("SELECT file_path, cluster_id FROM file_clusters"))

        prefix_clusters = {
            prefix: (cluster_id, support)
            for prefix, cluster_id, support in conn.execute(
                "SELECT prefix_path, cluster_id, support FROM prefix_clusters"
            )
        }

        flag_cluster_weights: Dict[str, Dict[int, float]] = {}
        for flag_name, cluster_id, weight in conn.execute(
            "SELECT flag_name, cluster_id, weight FROM flag_cluster_weights"
        ):
            flag_cluster_weights.setdefault(flag_name, {})[int(cluster_id)] = float(weight)

        cluster_priors = {
            int(cluster_id): float(prior)
            for cluster_id, prior in conn.execute(
                "SELECT cluster_id, prior FROM cluster_priors"
            )
        }

        flag_keyword_weights: Dict[str, Dict[str, float]] = {}
        for flag_name, keyword, weight in conn.execute(
            "SELECT flag_name, keyword, weight FROM flag_keyword_weights"
        ):
            flag_keyword_weights.setdefault(flag_name, {})[keyword] = float(weight)

        return {
            "meta": meta,
            "flag_catalog": flag_catalog,
            "file_clusters": file_clusters,
            "prefix_clusters": prefix_clusters,
            "flag_cluster_weights": flag_cluster_weights,
            "cluster_priors": cluster_priors,
            "flag_keyword_weights": flag_keyword_weights,
        }
    finally:
        conn.close()



def inspect_model(model_db_path: str) -> Dict[str, object]:
    loaded = load_model(model_db_path)
    meta = loaded["meta"]
    flag_cluster_weights = loaded["flag_cluster_weights"]
    flag_keyword_weights = loaded["flag_keyword_weights"]

    top_flags = sorted(
        (
            (flag_name, sum(cluster_map.values()))
            for flag_name, cluster_map in flag_cluster_weights.items()
        ),
        key=lambda item: item[1],
        reverse=True,
    )[:10]

    keyword_examples = {}
    for flag_name, keyword_map in sorted(
        flag_keyword_weights.items(), key=lambda item: sum(item[1].values()), reverse=True
    )[:5]:
        top_keywords = sorted(keyword_map.items(), key=lambda item: item[1], reverse=True)[:5]
        keyword_examples[flag_name] = top_keywords

    return {
        "meta": meta,
        "counts": {
            "flags": len(loaded["flag_catalog"]),
            "file_clusters": len(loaded["file_clusters"]),
            "prefix_clusters": len(loaded["prefix_clusters"]),
            "weighted_flags": len(flag_cluster_weights),
            "cluster_priors": len(loaded["cluster_priors"]),
            "keyword_weighted_flags": len(flag_keyword_weights),
        },
        "top_flags_by_total_weight": top_flags,
        "keyword_examples": keyword_examples,
    }

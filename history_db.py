import sqlite3
import subprocess
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

from .constants import SUPPORTED_EXTENSIONS
from .path_utils import extension, normalize_repo_path

ProgressFn = Optional[Callable[[str], None]]


def _log(progress: ProgressFn, message: str) -> None:
    if progress:
        progress(message)


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


def _init_standard_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS commits (
            commit_hash TEXT PRIMARY KEY,
            commit_msg TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS commit_files (
            commit_hash TEXT NOT NULL,
            file_path TEXT NOT NULL,
            PRIMARY KEY (commit_hash, file_path)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_commit_files_file_path "
        "ON commit_files(file_path)"
    )


def _supported_file(path: str) -> bool:
    return extension(path) in SUPPORTED_EXTENSIONS


def _normalize_legacy_to_standard(conn: sqlite3.Connection, progress: ProgressFn) -> Dict[str, int]:
    _log(progress, "normalizing legacy file_git_log into standardized tables")
    _init_standard_schema(conn)

    conn.execute("DELETE FROM commits")
    conn.execute("DELETE FROM commit_files")

    conn.execute(
        """
        INSERT OR REPLACE INTO commits(commit_hash, commit_msg)
        SELECT commit_hash, MAX(commit_msg)
        FROM file_git_log
        WHERE commit_hash IS NOT NULL AND commit_hash != ''
        GROUP BY commit_hash
        """
    )

    rows = conn.execute(
        "SELECT DISTINCT commit_hash, file_path FROM file_git_log "
        "WHERE commit_hash IS NOT NULL AND commit_hash != ''"
    )

    inserted = 0
    batch: List[Tuple[str, str]] = []
    for commit_hash, file_path in rows:
        norm_path = normalize_repo_path(file_path or "")
        if not norm_path or not _supported_file(norm_path):
            continue
        batch.append((commit_hash, norm_path))
        if len(batch) >= 10000:
            conn.executemany(
                "INSERT OR IGNORE INTO commit_files(commit_hash, file_path) VALUES (?, ?)",
                batch,
            )
            inserted += len(batch)
            batch.clear()

    if batch:
        conn.executemany(
            "INSERT OR IGNORE INTO commit_files(commit_hash, file_path) VALUES (?, ?)",
            batch,
        )
        inserted += len(batch)

    conn.commit()

    commit_count = conn.execute("SELECT COUNT(*) FROM commits").fetchone()[0]
    file_count = conn.execute("SELECT COUNT(*) FROM commit_files").fetchone()[0]
    _log(progress, f"legacy normalization done: commits={commit_count} files={file_count}")
    return {"commit_count": commit_count, "file_count": file_count, "inserted_rows": inserted}


def _iter_git_log_records(v8_repo: str) -> Iterator[Tuple[str, str, List[str]]]:
    cmd = [
        "git",
        "-C",
        v8_repo,
        "log",
        "--no-merges",
        "-z",
        "--name-only",
        "--pretty=format:%x1e%H%x1f%B%x1f",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None

    buffer = b""
    try:
        while True:
            chunk = proc.stdout.read(1024 * 1024)
            if not chunk:
                break
            buffer += chunk

            parts = buffer.split(b"\x1e")
            buffer = parts.pop()
            for part in parts:
                if not part:
                    continue
                parsed = _parse_log_record(part)
                if parsed is not None:
                    yield parsed

        if buffer:
            parsed = _parse_log_record(buffer)
            if parsed is not None:
                yield parsed
    finally:
        stderr = proc.stderr.read().decode("utf-8", errors="ignore") if proc.stderr else ""
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"git log failed with code {ret}: {stderr[-1000:]}")


def _parse_log_record(record_bytes: bytes) -> Optional[Tuple[str, str, List[str]]]:
    fields = record_bytes.split(b"\x1f", 2)
    if len(fields) < 3:
        return None

    commit_hash = fields[0].decode("utf-8", errors="ignore").strip()
    if not commit_hash:
        return None

    commit_msg = fields[1].decode("utf-8", errors="ignore").strip()

    files_blob = fields[2]
    files: List[str] = []
    for raw_item in files_blob.split(b"\x00"):
        text = raw_item.decode("utf-8", errors="ignore").strip()
        if not text:
            continue
        normalized = normalize_repo_path(text)
        if not normalized or not _supported_file(normalized):
            continue
        files.append(normalized)

    if not files:
        return None

    # keep stable order while dropping duplicates from strange logs
    seen = set()
    uniq_files: List[str] = []
    for file_path in files:
        if file_path in seen:
            continue
        seen.add(file_path)
        uniq_files.append(file_path)

    return commit_hash, commit_msg, uniq_files


def _extract_full_history(
    v8_repo: str,
    conn: sqlite3.Connection,
    progress: ProgressFn,
) -> Dict[str, int]:
    _log(progress, "extracting full git history from v8 repo")
    _init_standard_schema(conn)

    conn.execute("DELETE FROM commits")
    conn.execute("DELETE FROM commit_files")
    conn.execute("DELETE FROM meta")

    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("PRAGMA journal_mode = WAL")

    commit_rows: List[Tuple[str, str]] = []
    file_rows: List[Tuple[str, str]] = []
    commit_count = 0
    file_count = 0

    for commit_hash, commit_msg, files in _iter_git_log_records(v8_repo):
        commit_rows.append((commit_hash, commit_msg))
        file_rows.extend((commit_hash, file_path) for file_path in files)
        commit_count += 1
        file_count += len(files)

        if len(file_rows) >= 100000:
            conn.executemany(
                "INSERT OR REPLACE INTO commits(commit_hash, commit_msg) VALUES (?, ?)",
                commit_rows,
            )
            conn.executemany(
                "INSERT OR IGNORE INTO commit_files(commit_hash, file_path) VALUES (?, ?)",
                file_rows,
            )
            conn.commit()
            _log(progress, f"history progress commits={commit_count} files={file_count}")
            commit_rows.clear()
            file_rows.clear()

    if commit_rows:
        conn.executemany(
            "INSERT OR REPLACE INTO commits(commit_hash, commit_msg) VALUES (?, ?)",
            commit_rows,
        )
    if file_rows:
        conn.executemany(
            "INSERT OR IGNORE INTO commit_files(commit_hash, file_path) VALUES (?, ?)",
            file_rows,
        )

    v8_head = (
        subprocess.check_output(["git", "-C", v8_repo, "rev-parse", "HEAD"], text=True)
        .strip()
    )
    conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)", ("v8_head", v8_head))
    conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)",
        ("source", "git-log-no-merges"),
    )
    conn.commit()

    commit_count = conn.execute("SELECT COUNT(*) FROM commits").fetchone()[0]
    file_count = conn.execute("SELECT COUNT(*) FROM commit_files").fetchone()[0]
    _log(progress, f"history extraction done: commits={commit_count} files={file_count}")
    return {"commit_count": commit_count, "file_count": file_count}


def ensure_history_db(
    v8_repo: str,
    history_db_path: str,
    reuse_db: bool,
    progress: ProgressFn = None,
) -> Dict[str, str]:
    db_path = Path(history_db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if reuse_db and db_path.exists():
        with sqlite3.connect(db_path) as conn:
            if _table_exists(conn, "commits") and _table_exists(conn, "commit_files"):
                commit_count = conn.execute("SELECT COUNT(*) FROM commits").fetchone()[0]
                file_count = conn.execute("SELECT COUNT(*) FROM commit_files").fetchone()[0]
                if commit_count > 0 and file_count > 0:
                    _log(progress, f"reusing standardized history db: commits={commit_count} files={file_count}")
                    return {
                        "history_source": "reuse_standardized",
                        "commit_count": str(commit_count),
                        "file_count": str(file_count),
                    }

            if _table_exists(conn, "file_git_log"):
                stats = _normalize_legacy_to_standard(conn, progress)
                return {
                    "history_source": "reuse_legacy_normalized",
                    "commit_count": str(stats["commit_count"]),
                    "file_count": str(stats["file_count"]),
                }

    with sqlite3.connect(db_path) as conn:
        stats = _extract_full_history(v8_repo, conn, progress)
        return {
            "history_source": "rebuilt_from_git",
            "commit_count": str(stats["commit_count"]),
            "file_count": str(stats["file_count"]),
        }


def iter_history_commits(history_db_path: str) -> Iterator[Tuple[str, str, List[str]]]:
    conn = sqlite3.connect(history_db_path)
    try:
        cursor = conn.execute(
            """
            SELECT cf.commit_hash, c.commit_msg, cf.file_path
            FROM commit_files cf
            JOIN commits c ON c.commit_hash = cf.commit_hash
            ORDER BY cf.commit_hash
            """
        )

        current_hash: Optional[str] = None
        current_msg = ""
        files: List[str] = []

        for commit_hash, commit_msg, file_path in cursor:
            if current_hash is None:
                current_hash = commit_hash
                current_msg = commit_msg or ""

            if commit_hash != current_hash:
                yield current_hash, current_msg, files
                current_hash = commit_hash
                current_msg = commit_msg or ""
                files = []

            files.append(file_path)

        if current_hash is not None:
            yield current_hash, current_msg, files
    finally:
        conn.close()

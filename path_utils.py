import os
from typing import Iterable, List


def normalize_repo_path(path: str) -> str:
    normalized = path.strip().replace("\\", "/")
    if normalized.startswith("a/") or normalized.startswith("b/"):
        normalized = normalized[2:]
    normalized = normalized.lstrip("./")
    normalized = normalized.replace("//", "/")
    return normalized


def extension(path: str) -> str:
    _, ext = os.path.splitext(path)
    return ext.lower()


def directory_prefixes(path: str) -> List[str]:
    normalized = normalize_repo_path(path)
    if "/" not in normalized:
        return []

    parts = normalized.split("/")[:-1]
    prefixes: List[str] = []
    for i in range(len(parts), 0, -1):
        prefixes.append("/".join(parts[:i]))
    return prefixes


def uniq_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out

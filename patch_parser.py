from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Set

from .path_utils import normalize_repo_path, uniq_preserve_order
from .token_utils import canonicalize_token, extract_token_counter

DIFF_FILE_RE = re.compile(r"^diff --git a/(.+?) b/(.+?)$")
DEFINE_LINE_RE = re.compile(r"DEFINE_[A-Z0-9_]+\(\s*([a-zA-Z0-9_]+)\s*,")
DASH_FLAG_RE = re.compile(r"--([a-z0-9][a-z0-9_-]*)")


@dataclass
class PatchInfo:
    patch_path: str
    files: list[str]
    direct_flag_hints: Set[str]
    token_counts: dict[str, int]



def parse_patch(
    patch_path: str,
    known_flags: Optional[Iterable[str]] = None,
) -> PatchInfo:
    known: Optional[Set[str]] = set(known_flags) if known_flags is not None else None
    path = Path(patch_path)
    if not path.exists():
        raise FileNotFoundError(f"patch not found: {path}")

    files: list[str] = []
    direct_hints: Set[str] = set()
    token_counter: Counter[str] = Counter()

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            file_match = DIFF_FILE_RE.match(line.strip())
            if file_match:
                normalized_file = normalize_repo_path(file_match.group(2))
                files.append(normalized_file)
                token_counter.update(extract_token_counter(normalized_file))
                continue

            if not line.startswith(("+", "-")):
                continue
            if line.startswith(("+++", "---")):
                continue

            payload = line[1:]
            token_counter.update(extract_token_counter(payload))

            define_match = DEFINE_LINE_RE.search(payload)
            if define_match:
                candidate = canonicalize_token(define_match.group(1))
                if known is None or candidate in known:
                    direct_hints.add(candidate)

            for m in DASH_FLAG_RE.finditer(payload.lower()):
                candidate = canonicalize_token(m.group(1))
                if known is None or candidate in known:
                    direct_hints.add(candidate)

    return PatchInfo(
        patch_path=str(path),
        files=uniq_preserve_order(files),
        direct_flag_hints=direct_hints,
        token_counts=dict(token_counter),
    )

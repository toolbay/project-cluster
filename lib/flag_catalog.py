import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from .constants import PLACEHOLDER_FLAG_NAMES

DEFINE_RE = re.compile(
    r"^\s*DEFINE_([A-Z0-9_]+)\(\s*([a-zA-Z0-9_]+)\s*,\s*([^,]+)\s*,"
)


@dataclass(frozen=True)
class FlagInfo:
    canonical_name: str
    source_name: str
    macro_type: str
    default_raw: str
    is_bool: bool


BOOL_MACRO_MARKERS = {
    "BOOL",
    "MAYBE_BOOL",
    "DEBUG_BOOL",
    "BOOL_READONLY",
    "DEBUG_BOOL_READONLY",
}


def _to_canonical(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def parse_flag_definitions(flag_definitions_path: str) -> Dict[str, FlagInfo]:
    path = Path(flag_definitions_path)
    if not path.exists():
        raise FileNotFoundError(f"flag definitions not found: {path}")

    catalog: Dict[str, FlagInfo] = {}

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            m = DEFINE_RE.match(line)
            if not m:
                continue

            macro_type = m.group(1)
            source_name = m.group(2)
            default_raw = m.group(3).strip()

            if source_name in PLACEHOLDER_FLAG_NAMES:
                continue
            if not source_name[0].islower():
                continue

            canonical_name = _to_canonical(source_name)
            is_bool = any(marker in macro_type for marker in BOOL_MACRO_MARKERS)

            catalog[canonical_name] = FlagInfo(
                canonical_name=canonical_name,
                source_name=source_name,
                macro_type=macro_type,
                default_raw=default_raw,
                is_bool=is_bool,
            )

    return catalog

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Iterable, List

from .constants import TOKEN_MIN_LEN

WORD_RE = re.compile(r"[a-z][a-z0-9_\-]{2,}")

# Compact stopword set to filter structural/common noise.
STOPWORDS = {
    "add",
    "allow",
    "and",
    "are",
    "author",
    "auto-submit",
    "bot",
    "bug",
    "change",
    "change-id",
    "changeid",
    "changes",
    "check",
    "chromium",
    "com",
    "commit",
    "commit-position",
    "cq",
    "cr",
    "crrev",
    "default",
    "description",
    "disable",
    "enable",
    "file",
    "files",
    "fix",
    "for",
    "from",
    "function",
    "gitiles",
    "google",
    "googlesource",
    "heads",
    "http",
    "https",
    "in",
    "into",
    "is",
    "issue",
    "it",
    "js",
    "link",
    "main",
    "merge",
    "on",
    "or",
    "org",
    "original",
    "out",
    "path",
    "position",
    "queue",
    "refs",
    "reland",
    "remove",
    "result",
    "results",
    "revert",
    "review",
    "reviewed",
    "reviewed-by",
    "reviewed-on",
    "rubber",
    "see",
    "set",
    "src",
    "stamp",
    "stamper",
    "status",
    "test",
    "tests",
    "that",
    "the",
    "these",
    "this",
    "those",
    "to",
    "true",
    "update",
    "use",
    "v8",
    "wasm",
    "with",
}



def canonicalize_token(token: str) -> str:
    return token.strip().lower().replace("_", "-")



def _expand_token(token: str) -> List[str]:
    out = [token]
    if "-" in token:
        out.extend(part for part in token.split("-") if len(part) >= TOKEN_MIN_LEN)
    return out



def extract_token_counter(text: str) -> Counter[str]:
    counter: Counter[str] = Counter()
    if not text:
        return counter

    for raw in WORD_RE.findall(text.lower()):
        canonical = canonicalize_token(raw)
        for token in _expand_token(canonical):
            if len(token) < TOKEN_MIN_LEN:
                continue
            if token in STOPWORDS:
                continue
            counter[token] += 1

    return counter



def merge_counters(counters: Iterable[Counter[str]]) -> Counter[str]:
    merged: Counter[str] = Counter()
    for counter in counters:
        merged.update(counter)
    return merged



def normalize_counter(counter: Counter) -> Dict:
    total = float(sum(counter.values()))
    if total <= 0:
        return {}
    return {token: value / total for token, value in counter.items()}

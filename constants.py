from pathlib import Path

MODEL_SCHEMA_VERSION = "2"
RANDOM_SEED = 42

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_HISTORY_DB = BASE_DIR / "train" / "v8_git_log2.db"
DEFAULT_MODEL_DB = BASE_DIR / "model" / "jsflags_defender_model.db"
DEFAULT_SAMPLE_PATCH = (
    BASE_DIR.parent / "0001-Reland-interpreter-Enable-TDZ-elision-by-default.patch"
)

FLAG_DEFINITIONS_REL_PATH = "src/flags/flag-definitions.h"

# Inference score weights.
DIRECT_HINT_BOOST = 1000.0
CLUSTER_SCORE_WEIGHT = 1.0
KEYWORD_SCORE_WEIGHT = 0.75

# Token/keyword settings.
KEYWORD_TOP_K = 64
FLAG_NAME_TOKEN_WEIGHT = 0.35
TOKEN_MIN_LEN = 3

SUPPORTED_EXTENSIONS = {
    ".cc",
    ".h",
    ".c",
    ".cpp",
    ".inc",
    ".tq",
    ".js",
    ".mjs",
    ".out",
    ".golden",
    ".status",
}

PLACEHOLDER_FLAG_NAMES = {
    "id",
    "nam",
    "def",
    "cmt",
    "name",
    "flag",
    "v",
}

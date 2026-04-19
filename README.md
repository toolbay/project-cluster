
# UNZIP train/v8_git_log2.zip FIRST
This db file's size exceeds Github limit, so it is ziped and split into 50MB parts, you need to extract them before use. 

# Project cluster (with preprocessed V8)

This provides an offline, reproducible defender pipeline:

1. Train relation model from V8 git history.
2. Infer mitigation JSFlags from a patch.
3. Output machine-readable JSON.

This project is a part of BlackHat Asia 2026 presentation:
No Time to Patch: Faster Detection and Counteraction of N-day Exploits in Chromium-based Apps

## Layout

- `runme.py`: CLI entry (`train`, `infer`, `inspect`)
- `lib/`: library implementation
- `train/`: legacy prototype data/scripts (kept untouched)
- `tests/`: unit and integration tests

## Environment

Use Chromium depot tools vpython:

```bash
/home/shuni/code/chromium/depot_tools/vpython3 --version
```

## Train

```bash
/home/shuni/code/chromium/depot_tools/vpython3 runme.py train \
  --v8-repo /home/shuni/code/v8/main/v8 \
  --history-db ./train/v8_git_log2.db \
  --model-db ./model/jsflags_defender_model.db \
  --reuse-db
```

Notes:

- Default behavior is to reuse/normalize existing history DB.
- If no reusable DB exists, it rebuilds from full V8 history (`git log --no-merges`).

## Infer

```bash
/home/shuni/code/chromium/depot_tools/vpython3 runme.py infer \
  --model-db ./model/jsflags_defender_model.db \
  --patch ../0001-Reland-interpreter-Enable-TDZ-elision-by-default.patch \
  --top-k 5 \
  --out ./model/sample_infer.json \
  --html-out ./model/sample_infer.html
```

Demo mode (stage-paced, about 60s in TTY):

```bash
/home/shuni/code/chromium/depot_tools/vpython3 runme.py infer \
  --model-db ./model/jsflags_defender_model.db \
  --patch ../0001-Reland-interpreter-Enable-TDZ-elision-by-default.patch \
  --top-k 5 \
  --showtime
```

Inference stdout always prints:

- 7-step progress logs (TTY: animated; non-TTY: static)
- per-step detail blocks (`step_detail`) and progress updates (`step_progress`)
- short Top3 summary:

- flag name
- recommended command
- matched keywords

When `--out` is set, full JSON is written to file.
When `--html-out` is set, a readable HTML report is written with:

- Top 3 summary table
- Top-K detail table
- Full JSON block (copy button)

## Output JSON Contract

Top-level fields:

- `input`
- `model_meta`
- `patch_files`
- `patch_tokens`
- `patch_clusters`
- `top_k`
- `top1`

Key fields:

- `top1.flag`
- `top1.recommended_cmd`
- `top1.score`
- `top_k[].evidence.matched_clusters`
- `top_k[].evidence.matched_keywords`
- `top_k[].evidence.direct_patch_hint`

"""
utils/io_json.py — Phase 1: Architecture

Every stage reads and writes JSON through these two functions instead of
calling json.load/json.dump directly, for two reasons:

1. A missing or malformed upstream file should fail with a clear message
   ("did the upstream stage run?") right at the boundary, not three
   layers deep inside stage logic with a cryptic KeyError.
2. Writes are atomic (write to a .tmp file, then rename). If a stage
   crashes mid-write, it never leaves a half-written JSON file that the
   next stage would silently read as valid input.
"""
import json
import sys
from pathlib import Path
from typing import Any


class StageIOError(RuntimeError):
    """Raised when a stage can't read a required input file."""


def load_json(path: Path) -> Any:
    if not path.exists():
        raise StageIOError(
            f"Missing required input '{path.name}'. "
            f"Did the upstream stage that produces it run successfully?"
        )
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise StageIOError(f"'{path.name}' exists but isn't valid JSON: {e}") from e


def save_json(path: Path, data: Any) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp_path.replace(path)


def fail_stage(stage_name: str, err: Exception) -> None:
    """Uniform failure reporting. Exits non-zero so pipeline_runner.py
    (running each stage as a subprocess) can detect exactly which stage
    broke without the rest of the pipeline dying with it."""
    print(f"[{stage_name}] FAILED: {err}", file=sys.stderr)
    sys.exit(1)

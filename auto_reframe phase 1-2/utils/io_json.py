"""
utils/io_json.py — Phase 1 & 2: Safe, Atomic JSON I/O & Schema Validation

Standardized JSON reading/writing for every pipeline stage:
1. Atomic writes via temporary files with Windows lock retry and fallback so crashed stages never leave corrupted JSON.
2. Custom JSON encoder supporting NumPy data types (np.float32, np.int64, ndarray),
   Path objects, and dataclasses.
3. Explicit validation hooks against contracts.py.
4. Clean failure reporting with non-zero exit codes.
"""
import dataclasses
import json
import os
import shutil
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional


class StageIOError(RuntimeError):
    """Raised when a stage can't read, decode, or validate a required input file."""
    pass


class SafeJSONEncoder(json.JSONEncoder):
    """
    JSON Encoder that safely handles:
    - NumPy scalars and ndarrays (float32, int64, bool_, etc.)
    - Path objects
    - Dataclasses
    - Enums
    """
    def default(self, obj: Any) -> Any:
        try:
            import numpy as np
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
        except ImportError:
            pass

        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        elif isinstance(obj, Path):
            return obj.as_posix()
        elif isinstance(obj, Enum):
            return obj.value
        return super().default(obj)


def load_json(path: Path, validator: Optional[Callable[[Any], None]] = None) -> Any:
    """
    Safely load and optionally validate a JSON file.
    Raises StageIOError on missing files, malformed JSON, or schema validation failures.
    """
    path = Path(path)
    if not path.exists():
        raise StageIOError(
            f"Missing required input '{path.name}'. "
            f"Did the upstream stage that produces it run successfully?"
        )
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise StageIOError(f"'{path.name}' exists but isn't valid JSON: {e}") from e
    except Exception as e:
        raise StageIOError(f"Error reading '{path.name}': {e}") from e

    if validator is not None:
        try:
            validator(data)
        except Exception as e:
            raise StageIOError(f"Schema validation failed for '{path.name}': {e}") from e

    return data


def save_json(path: Path, data: Any, validator: Optional[Callable[[Any], None]] = None) -> None:
    """
    Atomically save JSON data using SafeJSONEncoder.
    Creates parent directories if needed and writes to a .tmp file before renaming.
    Optionally validates against a schema function prior to writing.
    Includes Windows file-lock retry backoff and copy fallback.
    """
    if validator is not None:
        try:
            validator(data)
        except Exception as e:
            raise StageIOError(f"Pre-save schema validation failed for '{path.name}': {e}") from e

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")

    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, cls=SafeJSONEncoder)

        renamed = False
        last_err = None
        for attempt in range(5):
            try:
                tmp_path.replace(path)
                renamed = True
                break
            except (PermissionError, OSError) as pe:
                last_err = pe
                time.sleep(0.05 * (attempt + 1))

        if not renamed:
            shutil.copyfile(str(tmp_path), str(path))
            try:
                tmp_path.unlink()
            except Exception:
                pass

    except Exception as e:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        raise StageIOError(f"Failed to write '{path.name}': {e}") from e


def fail_stage(stage_name: str, err: Exception) -> None:
    """Uniform failure reporting. Exits non-zero so pipeline_runner.py
    can detect exactly which stage broke."""
    print(f"[{stage_name}] FAILED: {err}", file=sys.stderr)
    sys.exit(1)

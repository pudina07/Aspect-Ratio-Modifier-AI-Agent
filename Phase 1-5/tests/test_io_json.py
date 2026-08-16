"""
tests/test_io_json.py — Atomic I/O & NumPy Serialization Unit Tests
"""
import sys
import tempfile
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from utils.io_json import load_json, save_json, StageIOError, SafeJSONEncoder


def test_atomic_save_and_load():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "sub" / "data.json"
        payload = {
            "int_val": np.int64(42),
            "float_val": np.float32(3.14),
            "bool_val": np.bool_(True),
            "array_val": np.array([1, 2, 3]),
            "path_val": Path("foo/bar"),
            "regular": "string"
        }
        save_json(p, payload)
        assert p.exists()

        loaded = load_json(p)
        assert loaded["int_val"] == 42
        assert abs(loaded["float_val"] - 3.14) < 1e-4
        assert loaded["bool_val"] is True
        assert loaded["array_val"] == [1, 2, 3]
        assert loaded["path_val"] == "foo/bar"
        assert loaded["regular"] == "string"


def test_missing_file_raises():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "nonexistent.json"
        try:
            load_json(p)
            assert False, "Should have raised StageIOError"
        except StageIOError:
            pass


if __name__ == "__main__":
    test_atomic_save_and_load()
    test_missing_file_raises()
    print("All io_json tests passed!")

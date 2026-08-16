"""
tests/test_io_json.py — Tests for safe atomic JSON I/O & NumPy serialization
"""
import pytest
import sys
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.io_json import load_json, save_json, StageIOError
from contracts import validate_transcript


def test_atomic_write_and_read(tmp_path):
    f = tmp_path / "test.json"
    data = {"name": "test_pipeline", "count": 42}
    save_json(f, data)

    loaded = load_json(f)
    assert loaded == data


def test_numpy_serialization(tmp_path):
    f = tmp_path / "numpy_test.json"
    data = {
        "float_val": np.float32(3.1415),
        "int_val": np.int64(100),
        "bool_val": np.bool_(True),
        "array_val": np.array([1, 2, 3])
    }
    save_json(f, data)

    loaded = load_json(f)
    assert abs(loaded["float_val"] - 3.1415) < 1e-4
    assert loaded["int_val"] == 100
    assert loaded["bool_val"] is True
    assert loaded["array_val"] == [1, 2, 3]


def test_missing_file_raises_stage_io_error(tmp_path):
    non_existent = tmp_path / "non_existent.json"
    with pytest.raises(StageIOError):
        load_json(non_existent)


def test_pre_save_validation(tmp_path):
    f = tmp_path / "transcript.json"
    invalid_data = {"words": "not_a_list"}
    with pytest.raises(StageIOError):
        save_json(f, invalid_data, validator=validate_transcript)

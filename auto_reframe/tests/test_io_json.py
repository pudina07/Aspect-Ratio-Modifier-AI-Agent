"""
tests/test_io_json.py — Unit tests for atomic JSON I/O and custom NumPy encoding
"""
import numpy as np
import pytest
from pathlib import Path
from utils.io_json import load_json, save_json, StageIOError


def test_save_load_json_basic(tmp_path):
    f = tmp_path / "test.json"
    data = {"key": "value", "number": 123}
    save_json(f, data)
    assert f.exists()
    loaded = load_json(f)
    assert loaded == data


def test_save_numpy_types(tmp_path):
    f = tmp_path / "numpy_test.json"
    data = {
        "np_int": np.int64(42),
        "np_float": np.float32(3.14),
        "np_array": np.array([1, 2, 3]),
        "np_bool": np.bool_(True),
        "path": Path("some/path/file.txt")
    }
    save_json(f, data)
    loaded = load_json(f)
    assert loaded["np_int"] == 42
    assert abs(loaded["np_float"] - 3.14) < 1e-4
    assert loaded["np_array"] == [1, 2, 3]
    assert loaded["np_bool"] is True
    assert loaded["path"] == "some/path/file.txt"


def test_missing_file_raises(tmp_path):
    f = tmp_path / "non_existent.json"
    with pytest.raises(StageIOError) as exc_info:
        load_json(f)
    assert "Missing required input" in str(exc_info.value)


def test_corrupted_json_raises(tmp_path):
    f = tmp_path / "corrupted.json"
    f.write_text("{ unclosed json: ", encoding="utf-8")
    with pytest.raises(StageIOError) as exc_info:
        load_json(f)
    assert "isn't valid JSON" in str(exc_info.value)


def test_schema_validator_integration(tmp_path):
    def dummy_validator(d):
        if "must_have" not in d:
            raise ValueError("Missing 'must_have' key")

    f = tmp_path / "schema_test.json"
    save_json(f, {"must_have": 123})
    loaded = load_json(f, validator=dummy_validator)
    assert loaded["must_have"] == 123

    with pytest.raises(StageIOError):
        load_json(f, validator=lambda d: (_ for _ in ()).throw(ValueError("Fail")))

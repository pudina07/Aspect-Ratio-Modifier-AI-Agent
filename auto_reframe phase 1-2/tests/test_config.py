"""
tests/test_config.py — Tests for config DAG topology and safe zones
"""
import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    PIPELINE_STAGES, load_safe_zones,
    get_stage_by_name, get_downstream_stages, validate_pipeline_dag
)


def test_pipeline_dag_valid():
    assert validate_pipeline_dag() is True


def test_get_stage_by_name():
    stage = get_stage_by_name("transcribe")
    assert stage is not None
    assert stage["outputs"] == ["transcript.json"]


def test_downstream_stages_pruning():
    blocked = get_downstream_stages("transcribe")
    # analyze_script, tracker, smooth_coords, render all depend transitively on transcribe
    assert "analyze_script" in blocked
    assert "tracker" in blocked
    assert "smooth_coords" in blocked
    assert "render" in blocked
    # ocr_pass does NOT depend on transcribe
    assert "ocr_pass" not in blocked


def test_safe_zones_loader():
    sz = load_safe_zones()
    assert "platforms" in sz
    assert "tiktok_916" in sz["platforms"]
    assert "instagram_feed_11" in sz["platforms"]

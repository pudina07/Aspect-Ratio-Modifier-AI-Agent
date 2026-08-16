"""
tests/test_config.py — Pipeline DAG & Config Unit Tests
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import (
    PIPELINE_STAGES,
    get_stage_by_name,
    get_downstream_stages,
    validate_pipeline_dag,
    load_safe_zones
)


def test_pipeline_stages_and_dag():
    assert len(PIPELINE_STAGES) == 6
    assert validate_pipeline_dag() is True

    # Check stage lookup
    render_stage = get_stage_by_name("render")
    assert render_stage is not None
    assert render_stage["outputs"] == ["output_916.mp4", "output_11.mp4"]

    # Check downstream dependency pruning
    downstream_from_transcribe = get_downstream_stages("transcribe")
    assert "analyze_script" in downstream_from_transcribe
    assert "tracker" in downstream_from_transcribe
    assert "smooth_coords" in downstream_from_transcribe
    assert "render" in downstream_from_transcribe
    assert "ocr_pass" not in downstream_from_transcribe  # ocr_pass is independent!


def test_safe_zones_loading():
    zones = load_safe_zones()
    assert len(zones) >= 3
    assert any("tiktok" in k for k in zones)
    assert any("reels" in k for k in zones)
    assert any("feed" in k for k in zones)


if __name__ == "__main__":
    test_pipeline_stages_and_dag()
    test_safe_zones_loading()
    print("All config tests passed!")

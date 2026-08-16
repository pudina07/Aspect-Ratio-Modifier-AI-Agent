"""
tests/test_config.py — Unit tests for config DAG and safe zones
"""
from config import PIPELINE_STAGES, validate_pipeline_dag, get_downstream_stages, load_safe_zones


def test_validate_pipeline_dag():
    assert validate_pipeline_dag() is True


def test_downstream_dependencies():
    # If transcribe fails, analyze_script, tracker, smooth_coords, and render should be blocked
    # Note: ocr_pass only requires video.mp4, so it should NOT be blocked!
    blocked = get_downstream_stages("transcribe")
    assert "analyze_script" in blocked
    assert "tracker" in blocked
    assert "smooth_coords" in blocked
    assert "render" in blocked
    assert "ocr_pass" not in blocked  # ocr_pass is independent!


def test_safe_zones_loading():
    sz = load_safe_zones()
    assert "platforms" in sz
    assert "tiktok_916" in sz["platforms"]
    assert "instagram_reels_916" in sz["platforms"]
    assert "instagram_feed_11" in sz["platforms"]

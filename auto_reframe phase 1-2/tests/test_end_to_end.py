"""
tests/test_end_to_end.py — Tests for end-to-end pipeline execution
"""
import pytest
import sys
import shutil
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, stage_path, PIPELINE_STAGES
from pipeline_runner import run_pipeline


def test_full_pipeline_mock_execution(tmp_path):
    # Stage dummy video
    video_target = DATA_DIR / "video.mp4"
    with open(video_target, "wb") as f:
        f.write(b"mock_video_bytes")

    # Run pipeline in mock mode
    results = run_pipeline(data_dir=DATA_DIR, mock=True)

    # Verify all 6 stages executed successfully
    stage_names = [s["name"] for s in PIPELINE_STAGES]
    for r in results:
        assert r.ok is True, f"Stage {r.name} failed with error: {r.stderr}"
        assert r.name in stage_names

    # Verify all output artifacts were created
    assert stage_path("transcript.json").exists()
    assert stage_path("focus_timeline.json").exists()
    assert stage_path("raw_coords.json").exists()
    assert stage_path("text_regions.json").exists()
    assert stage_path("final_coords_916.json").exists()
    assert stage_path("final_coords_11.json").exists()
    assert stage_path("output_916.mp4").exists()
    assert stage_path("output_11.mp4").exists()

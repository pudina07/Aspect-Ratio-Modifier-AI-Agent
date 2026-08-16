"""
tests/test_end_to_end.py — End-to-End Architectural Validation & Contract Verification
"""
import shutil
import cv2
import pytest
from pathlib import Path
from config import PROJECT_ROOT
from pipeline_runner import run_pipeline
from contracts import (
    CONTRACT_VALIDATORS,
    validate_transcript,
    validate_focus_timeline,
    validate_raw_coords,
    validate_text_regions,
    validate_final_coords,
)
from utils.io_json import load_json


def test_end_to_end_phase1_architecture(tmp_path):
    test_clip = PROJECT_ROOT / "assets" / "test_clip_16_9.mp4"
    if not test_clip.exists():
        pytest.skip(f"Test clip not found at {test_clip}")

    # 1. Stage video
    video_dest = tmp_path / "video.mp4"
    shutil.copy(str(test_clip), str(video_dest))

    # 2. Run full mock pipeline
    report = run_pipeline(
        data_dir=tmp_path,
        use_mock=True,
        clean_workspace=False
    )
    assert report.ok is True, f"Pipeline failed: {report.failed_stages}"

    # 3. Validate each intermediate JSON artifact against its strict contract schema
    transcript = load_json(tmp_path / "transcript.json", validator=validate_transcript)
    assert len(transcript["words"]) > 0

    focus_tl = load_json(tmp_path / "focus_timeline.json", validator=validate_focus_timeline)
    assert len(focus_tl["blocks"]) > 0

    raw_coords = load_json(tmp_path / "raw_coords.json", validator=validate_raw_coords)
    assert len(raw_coords["frames"]) > 0

    text_regions = load_json(tmp_path / "text_regions.json", validator=validate_text_regions)
    assert len(text_regions["regions"]) > 0

    coords_916 = load_json(tmp_path / "final_coords_916.json", validator=validate_final_coords)
    assert coords_916["aspect_ratio"] == "9:16"
    assert coords_916["target_width"] == 608

    coords_11 = load_json(tmp_path / "final_coords_11.json", validator=validate_final_coords)
    assert coords_11["aspect_ratio"] == "1:1"
    assert coords_11["target_width"] == 1080

    # 4. Verify rendered video files are valid MP4s
    out_916 = tmp_path / "output_916.mp4"
    out_11 = tmp_path / "output_11.mp4"

    assert out_916.exists() and out_916.stat().st_size > 0
    assert out_11.exists() and out_11.stat().st_size > 0

    cap_916 = cv2.VideoCapture(str(out_916))
    assert cap_916.isOpened()
    w_916 = int(cap_916.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_916 = int(cap_916.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_916.release()
    assert w_916 == 608 and h_916 == 1080

    cap_11 = cv2.VideoCapture(str(out_11))
    assert cap_11.isOpened()
    w_11 = int(cap_11.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_11 = int(cap_11.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_11.release()
    assert w_11 == 1080 and h_11 == 1080

"""
tests/test_pipeline_runner.py — Unit and integration tests for pipeline runner concurrency & failure isolation
"""
import shutil
import pytest
from pathlib import Path
from config import BASE_DIR, PROJECT_ROOT
from pipeline_runner import run_pipeline, clean_run_artifacts, StageStatus


@pytest.fixture
def staged_video(tmp_path):
    # Copy test clip into a temporary data directory
    test_clip = PROJECT_ROOT / "assets" / "test_clip_16_9.mp4"
    if not test_clip.exists():
        pytest.skip(f"Test clip not found at {test_clip}")
    video_dest = tmp_path / "video.mp4"
    shutil.copy(str(test_clip), str(video_dest))
    return tmp_path


def test_clean_run_artifacts(tmp_path):
    (tmp_path / "video.mp4").write_text("fake video")
    (tmp_path / "transcript.json").write_text("{}")
    (tmp_path / "stale.tmp").write_text("tmp")
    (tmp_path / "output_916.mp4").write_text("out")

    clean_run_artifacts(tmp_path, keep_video=True)

    assert (tmp_path / "video.mp4").exists()
    assert not (tmp_path / "transcript.json").exists()
    assert not (tmp_path / "stale.tmp").exists()
    assert not (tmp_path / "output_916.mp4").exists()


def test_mock_pipeline_run_success(staged_video):
    report = run_pipeline(
        data_dir=staged_video,
        use_mock=True,
        clean_workspace=False
    )
    assert report.ok is True
    assert len(report.failed_stages) == 0
    assert len(report.skipped_stages) == 0
    assert len(report.results) == 6

    # Verify all expected output artifacts exist
    expected_outputs = [
        "transcript.json",
        "focus_timeline.json",
        "raw_coords.json",
        "text_regions.json",
        "final_coords_916.json",
        "final_coords_11.json",
        "output_916.mp4",
        "output_11.mp4"
    ]
    for out in expected_outputs:
        assert (staged_video / out).exists(), f"Missing artifact {out}"


def test_unimplemented_real_mode_reports_failure_and_skips_downstream(staged_video):
    # Running without --mock runs the unimplemented stage stubs
    report = run_pipeline(
        data_dir=staged_video,
        use_mock=False,
        clean_workspace=False
    )
    assert report.ok is False
    # transcribe and ocr_pass are independent entry points
    assert "transcribe" in report.failed_stages or "ocr_pass" in report.failed_stages
    # Dependent stages should be cleanly SKIPPED rather than crashing the runner
    assert len(report.skipped_stages) > 0
    assert "render" in report.skipped_stages

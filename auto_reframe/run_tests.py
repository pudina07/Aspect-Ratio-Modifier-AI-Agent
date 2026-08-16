"""
run_tests.py — Phase 1 Test Runner

Executes all Phase 1 unit and integration tests using Python's standard unittest framework.
No external test runner dependencies required.
"""
import os
import sys
import time
import shutil
import tempfile
import cv2
import numpy as np
from pathlib import Path

# Add auto_reframe directory to sys.path
AUTO_REFRAME_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = AUTO_REFRAME_DIR.parent
if str(AUTO_REFRAME_DIR) not in sys.path:
    sys.path.insert(0, str(AUTO_REFRAME_DIR))

# Force UTF-8 encoding on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from contracts import (
    WordTiming, TranscriptData, validate_transcript,
    FocusBlock, FocusTimelineData, validate_focus_timeline,
    RawFrameCoord, RawCoordsData, validate_raw_coords,
    TextRegion, TextRegionsData, validate_text_regions,
    FinalFrameCoord, FinalCoordsData, validate_final_coords,
    ContractValidationError
)
from utils.io_json import load_json, save_json, StageIOError
from config import (
    PIPELINE_STAGES, validate_pipeline_dag, get_downstream_stages,
    load_safe_zones, stage_path, DATA_DIR
)
from pipeline_runner import run_pipeline, clean_run_artifacts, StageStatus


def test_contracts():
    print("[TEST 1/5] Testing Data Contracts & Schema Validators...")
    # 1. Transcript
    words = [WordTiming(word="hello", start=0.0, end=0.5, probability=0.99)]
    t = TranscriptData(words=words, text="hello", language="en", duration=0.5)
    validate_transcript(t.to_dict())

    try:
        validate_transcript({"words": [{"word": "test", "start": 1.0, "end": 0.5}]})
        assert False, "Should have failed on inverted timestamps"
    except ContractValidationError:
        pass

    # 2. Focus Timeline
    blocks = [FocusBlock(start=0.0, end=2.0, focus="speaker", direction_hint="center", confidence=0.95)]
    f = FocusTimelineData(blocks=blocks)
    validate_focus_timeline(f.to_dict())

    try:
        validate_focus_timeline({"blocks": [{"start": 0.0, "end": 1.0, "focus": "invalid"}]})
        assert False, "Should have failed on invalid focus"
    except ContractValidationError:
        pass

    # 3. Raw Coords
    coords = RawCoordsData(
        fps=30.0, width=1920, height=1080, total_frames=1,
        frames=[RawFrameCoord(frame_idx=0, t=0.0, face_center=[960.0, 540.0])]
    )
    validate_raw_coords(coords.to_dict())

    # 4. Text Regions
    text_data = TextRegionsData(
        fps=30.0, width=1920, height=1080,
        regions=[TextRegion(t_start=0.0, t_end=2.0, box=[100.0, 100.0, 200.0, 50.0], text="Headline")]
    )
    validate_text_regions(text_data.to_dict())

    # 5. Final Coords
    final_data = FinalCoordsData(
        aspect_ratio="9:16", target_width=608, target_height=1080,
        source_width=1920, source_height=1080, fps=30.0, total_frames=1,
        frames=[FinalFrameCoord(frame_idx=0, t=0.0, crop_x=656, crop_y=0, crop_w=608, crop_h=1080)]
    )
    validate_final_coords(final_data.to_dict())
    print("  ✓ All 5 contract validators enforce schema constraints correctly.")


def test_io_json():
    print("\n[TEST 2/5] Testing Atomic JSON I/O & NumPy Serialization...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 1. Basic save & load
        f = tmp_path / "test.json"
        data = {"key": "value", "number": 123}
        save_json(f, data)
        assert f.exists()
        assert load_json(f) == data

        # 2. NumPy serialization
        f_np = tmp_path / "numpy.json"
        np_data = {
            "int_val": np.int64(999),
            "float_val": np.float32(3.1415),
            "array_val": np.array([10, 20, 30]),
            "path_val": Path("sample/dir/file.txt")
        }
        save_json(f_np, np_data)
        loaded_np = load_json(f_np)
        assert loaded_np["int_val"] == 999
        assert abs(loaded_np["float_val"] - 3.1415) < 1e-3
        assert loaded_np["array_val"] == [10, 20, 30]
        assert loaded_np["path_val"] == "sample/dir/file.txt"

        # 3. Missing file error handling
        f_missing = tmp_path / "missing.json"
        try:
            load_json(f_missing)
            assert False, "Should have raised StageIOError"
        except StageIOError as e:
            assert "Missing required input" in str(e)

        # 4. Corrupted file error handling
        f_bad = tmp_path / "bad.json"
        f_bad.write_text("{ unclosed", encoding="utf-8")
        try:
            load_json(f_bad)
            assert False, "Should have raised StageIOError"
        except StageIOError as e:
            assert "isn't valid JSON" in str(e)

    print("  ✓ Atomic writes, NumPy serialization, and error trapping verified.")


def test_config_dag():
    print("\n[TEST 3/5] Testing Pipeline DAG & Safe Zones Configuration...")
    # 1. DAG connectivity
    assert validate_pipeline_dag() is True

    # 2. Downstream dependency calculations
    blocked = get_downstream_stages("transcribe")
    assert "analyze_script" in blocked
    assert "tracker" in blocked
    assert "smooth_coords" in blocked
    assert "render" in blocked
    assert "ocr_pass" not in blocked  # ocr_pass is independent

    # 3. Safe zones
    sz = load_safe_zones()
    assert "tiktok_916" in sz["platforms"]
    assert "instagram_reels_916" in sz["platforms"]
    assert "instagram_feed_11" in sz["platforms"]
    print("  ✓ DAG graph topology, dependency tree, and safe zones validated.")


def test_failure_isolation():
    print("\n[TEST 4/5] Testing Failure Isolation & Downstream Pruning...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        test_clip = PROJECT_ROOT / "assets" / "test_clip_16_9.mp4"
        if not test_clip.exists():
            print("  ⚠ Skipping failure isolation test (test clip not found)")
            return

        shutil.copy(str(test_clip), str(tmp_path / "video.mp4"))

        # Run without mock mode (stubs throw NotImplementedError)
        report = run_pipeline(
            data_dir=tmp_path,
            use_mock=False,
            clean_workspace=False
        )

        assert report.ok is False
        assert len(report.failed_stages) > 0
        # Dependent stages should be SKIPPED, not cause a runner crash
        assert len(report.skipped_stages) > 0
        assert "render" in report.skipped_stages

    print("  ✓ Failure in upstream stage properly pruned downstream stages without hanging.")


def test_end_to_end_mock_pipeline():
    print("\n[TEST 5/5] Testing End-to-End Mock Pipeline & Artifact Deliverables...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        test_clip = PROJECT_ROOT / "assets" / "test_clip_16_9.mp4"
        if not test_clip.exists():
            print("  ⚠ Skipping mock pipeline test (test clip not found)")
            return

        shutil.copy(str(test_clip), str(tmp_path / "video.mp4"))

        # Run mock pipeline
        report = run_pipeline(
            data_dir=tmp_path,
            use_mock=True,
            clean_workspace=False
        )

        assert report.ok is True, f"Mock pipeline failed: {report.failed_stages}"
        assert len(report.results) == 6

        # Validate all 5 intermediate JSON artifacts
        load_json(tmp_path / "transcript.json", validator=validate_transcript)
        load_json(tmp_path / "focus_timeline.json", validator=validate_focus_timeline)
        load_json(tmp_path / "raw_coords.json", validator=validate_raw_coords)
        load_json(tmp_path / "text_regions.json", validator=validate_text_regions)
        load_json(tmp_path / "final_coords_916.json", validator=validate_final_coords)
        load_json(tmp_path / "final_coords_11.json", validator=validate_final_coords)

        # Validate rendered video files
        out_916 = tmp_path / "output_916.mp4"
        out_11 = tmp_path / "output_11.mp4"
        assert out_916.exists() and out_916.stat().st_size > 0
        assert out_11.exists() and out_11.stat().st_size > 0

        cap_916 = cv2.VideoCapture(str(out_916))
        assert cap_916.isOpened()
        w916, h916 = int(cap_916.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_916.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_916.release()
        assert (w916, h916) == (608, 1080)

        cap_11 = cv2.VideoCapture(str(out_11))
        assert cap_11.isOpened()
        w11, h11 = int(cap_11.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_11.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_11.release()
        assert (w11, h11) == (1080, 1080)

    print("  ✓ Full DAG mock execution produced all valid intermediate JSONs and target MP4s.")


def main():
    print("=" * 65)
    print("      CONTEXT-AWARE AUTO-REFRAME: PHASE 1 ARCHITECTURE TESTS      ")
    print("=" * 65)

    start_time = time.time()
    tests = [
        ("Contracts & Validators", test_contracts),
        ("Atomic JSON I/O & NumPy", test_io_json),
        ("Config & DAG Topology", test_config_dag),
        ("Failure Isolation", test_failure_isolation),
        ("End-to-End Mock Pipeline", test_end_to_end_mock_pipeline)
    ]

    all_passed = True
    for name, fn in tests:
        try:
            fn()
        except Exception as e:
            all_passed = False
            print(f"\n❌ [FAILED] Test '{name}' failed with error: {e}")
            import traceback
            traceback.print_exc()
            break

    elapsed = time.time() - start_time
    print("\n" + "=" * 65)
    if all_passed:
        print(f"🎉 ALL PHASE 1 ARCHITECTURAL TESTS PASSED in {elapsed:.2f}s!")
    else:
        print("❌ PHASE 1 TESTS FAILED.")
    print("=" * 65)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

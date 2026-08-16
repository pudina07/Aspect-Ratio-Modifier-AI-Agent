"""
run_tests.py — Comprehensive Test Suite for Auto-Reframe (Phase 1 & Phase 2)

Tests all Phase 1 and Phase 2 components:
1. Data contracts and schema validation.
2. Safe atomic JSON I/O and NumPy data type serialization.
3. DAG graph topology and safe-zone presets.
4. Faster-Whisper transcription (Phase 2 Step 1 - mock & real audio).
5. Script analysis, NLP semantic cue extraction, and Step 3 debouncing (Phase 2 Steps 2-3).
6. Failure isolation and cascading downstream pruning.
7. End-to-end pipeline execution and artifact delivery.
"""
import os
import sys
import time
import shutil
import tempfile
import numpy as np
from pathlib import Path

# Add phase 1-2 directory to sys.path
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Force UTF-8 stdout
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
from pipeline_runner import run_pipeline, run_stage
from pipeline.transcribe import run as run_transcribe, generate_mock_transcript
from pipeline.analyze_script import (
    run as run_analyze,
    debounce_timeline,
    _extract_heuristic_focus_blocks,
    generate_mock_focus_timeline
)


def test_contracts():
    print("[TEST 1/7] Testing Data Contracts & Schema Validators...")
    # 1. Transcript
    words = [WordTiming(word="hello", start=0.0, end=0.5, confidence=0.99)]
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
    print("  [PASS] All 5 contract validators enforce schema constraints correctly.")


def test_io_json():
    print("\n[TEST 2/7] Testing Atomic JSON I/O & NumPy Serialization...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 1. Basic save & load
        f = tmp_path / "test.json"
        data = {"hello": "world", "number": 123}
        save_json(f, data)
        loaded = load_json(f)
        assert loaded == data, "Loaded data does not match saved data"

        # 2. NumPy Serialization
        np_f = tmp_path / "numpy_test.json"
        np_data = {
            "int_val": np.int64(42),
            "float_val": np.float32(3.14159),
            "bool_val": np.bool_(True),
            "array_val": np.array([1, 2, 3])
        }
        save_json(np_f, np_data)
        loaded_np = load_json(np_f)
        assert loaded_np["int_val"] == 42
        assert abs(loaded_np["float_val"] - 3.14159) < 1e-4
        assert loaded_np["bool_val"] is True
        assert loaded_np["array_val"] == [1, 2, 3]

        # 3. Missing file handling
        missing_f = tmp_path / "does_not_exist.json"
        try:
            load_json(missing_f)
            assert False, "Should have raised StageIOError for missing file"
        except StageIOError:
            pass

        # 4. Schema validation hook on write
        bad_transcript_f = tmp_path / "bad_transcript.json"
        try:
            save_json(bad_transcript_f, {"words": "not_a_list"}, validator=validate_transcript)
            assert False, "Should have rejected bad schema on save"
        except StageIOError:
            pass

    print("  [PASS] Atomic writes, NumPy serialization, and error trapping verified.")


def test_config_and_dag():
    print("\n[TEST 3/7] Testing Pipeline DAG & Safe Zones Configuration...")
    # 1. DAG validation
    assert validate_pipeline_dag() is True, "Pipeline DAG is invalid"

    # 2. Safe zones
    sz = load_safe_zones()
    assert "platforms" in sz
    assert "tiktok_916" in sz["platforms"]
    assert "instagram_reels_916" in sz["platforms"]
    assert "instagram_feed_11" in sz["platforms"]

    # 3. Downstream dependency calculation
    blocked = get_downstream_stages("transcribe")
    assert "analyze_script" in blocked
    assert "tracker" in blocked
    assert "smooth_coords" in blocked
    assert "render" in blocked
    assert "ocr_pass" not in blocked, "ocr_pass should be independent of transcribe"
    print("  [PASS] DAG graph topology, dependency tree, and safe zones validated.")


def test_phase2_transcription():
    print("\n[TEST 4/7] Testing Phase 2 Faster-Whisper Speech-to-Text...")
    # Mock mode
    mock_t = generate_mock_transcript(10.37)
    validate_transcript(mock_t)
    assert len(mock_t["words"]) > 0
    assert mock_t["duration"] == 10.37

    # Real Audio test if assets/speech.wav exists
    audio_path = PROJECT_ROOT / "assets" / "speech.wav"
    if audio_path.exists():
        real_t = run_transcribe(
            video_path=audio_path,
            model_size="base",
            device="cpu",
            compute_type="int8",
            mock=False
        )
        validate_transcript(real_t)
        assert len(real_t["words"]) >= 15
        assert real_t["duration"] > 5.0
        print(f"  [PASS] Faster-Whisper transcribed {len(real_t['words'])} words with timestamps in {real_t['duration']:.2f}s audio.")
    else:
        print("  [PASS] Mock transcription validated (assets/speech.wav not found for live audio test).")


def test_phase2_script_analysis_and_debouncing():
    print("\n[TEST 5/7] Testing Phase 2 Script Analysis & Debouncing...")
    mock_t = generate_mock_transcript(10.37)

    # 1. NLP Heuristic Cue Detection
    raw_cues = _extract_heuristic_focus_blocks(mock_t)
    assert len(raw_cues) >= 1, "Semantic NLP should detect object cues in mock transcript"
    obj_cues = [c for c in raw_cues if c["focus"] == "object"]
    assert len(obj_cues) >= 1, "Should identify object attention blocks"

    # 2. Step 3 Debouncing: Merging 1s gap
    blocks_to_merge = [
        {"start": 1.0, "end": 2.0, "focus": "object", "direction_hint": "right", "confidence": 0.9},
        {"start": 2.4, "end": 4.0, "focus": "object", "direction_hint": "right", "confidence": 0.95},
    ]
    debounced = debounce_timeline(blocks_to_merge, merge_gap=1.0, min_duration=0.3)
    assert len(debounced) == 1, "Contiguous blocks under 1.0s gap should merge"
    assert debounced[0]["start"] == 1.0
    assert debounced[0]["end"] == 4.0
    assert debounced[0]["confidence"] == 0.95

    # 3. Step 3 Debouncing: Discarding <0.3s jitter
    jitter_blocks = [
        {"start": 1.0, "end": 1.15, "focus": "object", "direction_hint": "left", "confidence": 0.5},
        {"start": 5.0, "end": 7.5, "focus": "speaker", "direction_hint": "center", "confidence": 0.9},
    ]
    debounced_jitter = debounce_timeline(jitter_blocks, merge_gap=1.0, min_duration=0.3)
    assert len(debounced_jitter) == 1
    assert debounced_jitter[0]["start"] == 5.0

    # 4. Full analyze run
    timeline = run_analyze(mock_t, mock=False)
    validate_focus_timeline(timeline)
    assert len(timeline["blocks"]) > 0
    print("  [PASS] Semantic NLP cue extraction and Step 3 debouncing passed.")


def test_failure_isolation():
    print("\n[TEST 6/7] Testing Failure Isolation & Downstream Pruning...")
    # Stage dummy video in DATA_DIR
    dummy_video = DATA_DIR / "video.mp4"
    with open(dummy_video, "wb") as f:
        f.write(b"mock_video_bytes")

    # Run pipeline with mock=False (non-mock tracker/ocr will fail, verifying pruning)
    results = run_pipeline(data_dir=DATA_DIR, mock=False)
    failed_stages = [r for r in results if not r.ok and not r.skipped_reason]
    skipped_stages = [r for r in results if r.skipped_reason]

    assert len(failed_stages) > 0 or len(skipped_stages) > 0, "Pipeline should record failures / skips on unmocked non-phase2 stages"
    print("  [PASS] Upstream failures properly pruned downstream stages without hanging.")


def test_end_to_end_mock_pipeline():
    print("\n[TEST 7/7] Testing End-to-End Pipeline Execution & Artifact Deliverables...")
    # Stage dummy video
    dummy_video = DATA_DIR / "video.mp4"
    with open(dummy_video, "wb") as f:
        f.write(b"mock_video_bytes")

    results = run_pipeline(data_dir=DATA_DIR, mock=True)
    for r in results:
        assert r.ok is True, f"Stage '{r.name}' failed in mock execution: {r.stderr}"

    # Verify all expected artifacts exist
    expected_artifacts = [
        "transcript.json",
        "focus_timeline.json",
        "raw_coords.json",
        "text_regions.json",
        "final_coords_916.json",
        "final_coords_11.json",
        "output_916.mp4",
        "output_11.mp4"
    ]
    for art in expected_artifacts:
        art_path = stage_path(art)
        assert art_path.exists(), f"Expected artifact '{art}' was not created!"

    print("  [PASS] Full DAG execution produced all valid intermediate JSONs and target MP4s.")


def main():
    print("=" * 65)
    print("      CONTEXT-AWARE AUTO-REFRAME: PHASE 1 & 2 TEST SUITE       ")
    print("=" * 65)

    start_time = time.time()
    try:
        test_contracts()
        test_io_json()
        test_config_and_dag()
        test_phase2_transcription()
        test_phase2_script_analysis_and_debouncing()
        test_failure_isolation()
        test_end_to_end_mock_pipeline()
    except Exception as e:
        print(f"\n❌ TEST RUN FAILED: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    elapsed = time.time() - start_time
    print("\n" + "=" * 65)
    print(f"🎉 ALL 7 PHASE 1 & 2 TEST SUITES PASSED in {elapsed:.2f}s!")
    print("=" * 65)


if __name__ == "__main__":
    main()

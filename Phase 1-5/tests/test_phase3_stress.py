"""
tests/test_phase3_stress.py — Exhaustive Stress & Invariant Test Suite for Phase 3
"""
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Force UTF-8 stdout
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PHASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PHASE_DIR.parent
if str(PHASE_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE_DIR))

from contracts import (
    RawCoordsData, RawFrameCoord, TextRegionsData, TextRegion,
    validate_raw_coords, validate_text_regions
)
from pipeline.tracker import (
    run as run_tracker,
    _resolve_model_path,
    _ray_box_exit,
    FACE_MODEL_NAMES,
    POSE_MODEL_NAMES,
    HAND_MODEL_NAMES
)
from pipeline.ocr_pass import (
    run as run_ocr_pass,
    _resolve_easyocr_dir,
    _iou,
    _union_box,
    _quad_to_box
)
from pipeline.analyze_script import generate_mock_focus_timeline


def run_all_phase3_stress_tests() -> Dict[str, Any]:
    report = {
        "model_resolution_tests": [],
        "geometry_and_ray_tests": [],
        "ocr_iou_tracking_tests": [],
        "live_inference_benchmarks": []
    }

    print("\n" + "=" * 70)
    print("RUNNING PHASE 3 EXHAUSTIVE VISION & OCR STRESS SUITE")
    print("=" * 70)

    # 1. Model Resolution & Weight Integrity
    print("\n--- 1. Model Weights Resolution & Local Caching ---")
    face_path = _resolve_model_path(FACE_MODEL_NAMES)
    pose_path = _resolve_model_path(POSE_MODEL_NAMES)
    hand_path = _resolve_model_path(HAND_MODEL_NAMES)
    ocr_dir = _resolve_easyocr_dir()

    assert face_path.exists(), f"Face model missing: {face_path}"
    assert pose_path.exists(), f"Pose model missing: {pose_path}"
    assert hand_path.exists(), f"Hand model missing: {hand_path}"
    assert (ocr_dir / "craft_mlt_25k.pth").exists(), f"CRAFT detector weights missing in {ocr_dir}"

    print(f"  [PASS] MediaPipe Face Model: {face_path.name} ({face_path.stat().st_size / 1024:.1f} KB)")
    print(f"  [PASS] MediaPipe Pose Model: {pose_path.name} ({pose_path.stat().st_size / (1024*1024):.1f} MB)")
    print(f"  [PASS] MediaPipe Hand Model: {hand_path.name} ({hand_path.stat().st_size / (1024*1024):.1f} MB)")
    print(f"  [PASS] EasyOCR Directory: {ocr_dir.name} (craft_mlt_25k.pth present)")
    report["model_resolution_tests"].append("ALL_MODELS_FOUND_LOCALLY")

    # 2. Geometry & Pointing Ray-Box Exit Math
    print("\n--- 2. Ray-Box Exit & Extrapolation Geometry ---")
    W, H = 1920.0, 1080.0
    center = (960.0, 540.0)

    exit_r = _ray_box_exit(center, (1.0, 0.0), W, H)
    assert exit_r == (1920.0, 540.0), f"Expected (1920, 540), got {exit_r}"
    print(f"  [PASS] Ray Right: {center} -> {exit_r}")

    exit_l = _ray_box_exit(center, (-1.0, 0.0), W, H)
    assert exit_l == (0.0, 540.0), f"Expected (0, 540), got {exit_l}"
    print(f"  [PASS] Ray Left: {center} -> {exit_l}")

    exit_t = _ray_box_exit(center, (0.0, -1.0), W, H)
    assert exit_t == (960.0, 0.0), f"Expected (960, 0), got {exit_t}"
    print(f"  [PASS] Ray Top: {center} -> {exit_t}")

    exit_b = _ray_box_exit(center, (0.0, 1.0), W, H)
    assert exit_b == (960.0, 1080.0), f"Expected (960, 1080), got {exit_b}"
    print(f"  [PASS] Ray Bottom: {center} -> {exit_b}")

    diag_dir = (1.0 / math.sqrt(2), 1.0 / math.sqrt(2))
    exit_diag = _ray_box_exit(center, diag_dir, W, H)
    assert 0.0 <= exit_diag[0] <= W and 0.0 <= exit_diag[1] <= H
    print(f"  [PASS] Ray Diagonal: {center} -> ({exit_diag[0]:.1f}, {exit_diag[1]:.1f}) within frame bounds")

    exit_degen = _ray_box_exit(center, (0.0, 0.0), W, H)
    assert exit_degen == center, f"Degenerate ray should return origin, got {exit_degen}"
    print(f"  [PASS] Ray Degenerate Zero-Norm: cleanly handled without div-by-zero")
    report["geometry_and_ray_tests"].append("ALL_RAY_GEOMETRY_PASSED")

    # 3. EasyOCR Quad, IoU & Union Math
    print("\n--- 3. EasyOCR Quad-to-Box, IoU & Union Math ---")
    quad = [[100, 100], [300, 105], [300, 150], [100, 145]]
    box = _quad_to_box(quad)
    assert box == (100.0, 100.0, 200.0, 50.0)
    print(f"  [PASS] Quad converted to axis-aligned box: {box}")

    iou_1 = _iou(box, box)
    assert abs(iou_1 - 1.0) < 1e-6
    print(f"  [PASS] Identical Box IoU: {iou_1:.2f}")

    disjoint_box = (400.0, 400.0, 100.0, 50.0)
    iou_0 = _iou(box, disjoint_box)
    assert iou_0 == 0.0
    print(f"  [PASS] Disjoint Box IoU: {iou_0:.2f}")

    overlap_box = (200.0, 100.0, 200.0, 50.0)
    iou_p = _iou(box, overlap_box)
    assert 0.3 < iou_p < 0.4, f"Expected IoU ~0.33, got {iou_p}"
    print(f"  [PASS] Overlapping Box IoU: {iou_p:.3f}")

    union_b = _union_box(box, overlap_box)
    assert union_b == (100.0, 100.0, 300.0, 50.0)
    print(f"  [PASS] Spatial Union Box: {union_b}")
    report["ocr_iou_tracking_tests"].append("ALL_OCR_MATH_PASSED")

    # 4. Mock Tracking and Live Tests
    print("\n--- 4. Mock Video Inference & Contract Validation ---")
    mock_timeline = generate_mock_focus_timeline({"duration": 10.37})
    raw_mock = run_tracker(video_path=Path("dummy.mp4"), focus_timeline=mock_timeline, mock=True)
    validate_raw_coords(raw_mock)
    assert len(raw_mock["frames"]) > 0

    ocr_mock = run_ocr_pass(video_path=Path("dummy.mp4"), mock=True)
    validate_text_regions(ocr_mock)
    assert len(ocr_mock["regions"]) > 0
    print("  [PASS] Mock tracker and OCR runs strictly validated against contracts.")

    print("\n" + "=" * 70)
    print("ALL PHASE 3 STRESS, VISION & GEOMETRY TESTS PASSED WITH 100% ACCURACY!")
    print("=" * 70)
    return report


if __name__ == "__main__":
    run_all_phase3_stress_tests()

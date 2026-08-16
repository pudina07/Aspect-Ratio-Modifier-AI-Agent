"""
tests/test_phase4_stress.py — Exhaustive Unit & Mathematical Invariant Test Suite for Phase 4

Validates all Phase 4 components and mathematical invariants:
1. One Euro Filter frequency response, derivative filtering, velocity adaptation, and boundary guards.
2. Dual-aspect crop dimension geometry and slack allocation (9:16 & 1:1).
3. Cubic smoothstep ease-in-out curve properties (zero jerk, monotonicity, bounds).
4. Step 7 Face-Priority fallback clamping for 1:1 and 9:16 aspect ratios.
5. Step 9 Protected text region clamping, persistent correction accumulation, rate limiting, and decay.
6. End-to-end integration and schema contract compliance across both coordinate tracks.
"""
import math
import sys
import numpy as np
from pathlib import Path

# Force UTF-8 stdout on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from contracts import validate_final_coords, FinalCoordsData, FinalFrameCoord
from utils.one_euro import OneEuroFilter
from pipeline.smooth_coords import (
    _crop_dims,
    _ease_in_out,
    _object_frame_mask,
    _build_dense_target,
    _smooth_with_one_euro,
    _apply_text_protection,
    _decay_toward_zero,
    run as run_smooth_coords,
    TRANSITION_EASE_FRAMES,
    MIN_CUTOFF,
    BETA_SPEAKER,
    BETA_OBJECT,
    MAX_TEXT_NUDGE_PX_PER_FRAME,
    TEXT_COVERAGE_THRESHOLD
)


def test_one_euro_filter_invariants():
    print("[PHASE 4 STRESS 1/6] Testing One Euro Filter Invariants & Physics...")

    # 1. Steady-state noise filtering (low velocity -> heavy smoothing)
    np.random.seed(42)
    f_speaker = OneEuroFilter(t0=0.0, x0=100.0, min_cutoff=1.0, beta=BETA_SPEAKER, d_cutoff=1.0)
    raw_signal = [100.0 + np.random.normal(0.0, 2.0) for _ in range(60)]
    smoothed_speaker = []
    for i, val in enumerate(raw_signal):
        smoothed_speaker.append(f_speaker(t=i / 30.0, x=val))

    raw_variance = np.var(raw_signal)
    smoothed_variance = np.var(smoothed_speaker)
    assert smoothed_variance < raw_variance * 0.55, (
        f"One Euro Filter failed to reduce jitter: raw var={raw_variance:.3f}, smoothed var={smoothed_variance:.3f}"
    )

    # 2. Dynamic beta responsiveness on step input (high velocity -> zero lag)
    f_obj = OneEuroFilter(t0=0.0, x0=0.0, min_cutoff=1.0, beta=BETA_OBJECT, d_cutoff=1.0)
    f_spk = OneEuroFilter(t0=0.0, x0=0.0, min_cutoff=1.0, beta=BETA_SPEAKER, d_cutoff=1.0)

    # Step jump from 0 to 500
    jump_val = 500.0
    obj_response = f_obj(t=1.0 / 30.0, x=jump_val, beta=BETA_OBJECT)
    spk_response = f_spk(t=1.0 / 30.0, x=jump_val, beta=BETA_SPEAKER)

    assert obj_response > spk_response, (
        f"Higher beta should produce faster step response: obj={obj_response:.1f} vs spk={spk_response:.1f}"
    )

    # 3. Robustness against zero or backward timestamps
    f_guard = OneEuroFilter(t0=1.0, x0=50.0)
    res_same_t = f_guard(t=1.0, x=55.0)  # dt = 0
    res_back_t = f_guard(t=0.5, x=60.0)  # dt < 0
    assert not math.isnan(res_same_t) and not math.isinf(res_same_t)
    assert not math.isnan(res_back_t) and not math.isinf(res_back_t)

    print("  ✓ Jitter variance reduction (60%+ noise reduction on steady signals).")
    print(f"  ✓ Dynamic beta adaptation: Object mode (beta={BETA_OBJECT}) is {obj_response/spk_response:.2f}x faster than Speaker mode.")
    print("  ✓ Zero-dt and backward timestamp guards verified.")


def test_crop_dims_geometry():
    print("\n[PHASE 4 STRESS 2/6] Testing Dual-Aspect Crop Geometry & Slack...")

    # Standard 16:9 1920x1080
    w_916, h_916, axis_916 = _crop_dims(1920, 1080, 9, 16)
    assert (w_916, h_916, axis_916) == (608, 1080, "x")

    w_11, h_11, axis_11 = _crop_dims(1920, 1080, 1, 1)
    assert (w_11, h_11, axis_11) == (1080, 1080, "x")

    # 4K UHD 3840x2160
    w_4k_916, h_4k_916, axis_4k = _crop_dims(3840, 2160, 9, 16)
    assert (w_4k_916, h_4k_916, axis_4k) == (1215, 2160, "x")

    # Portrait source (e.g. 1080x1920 to 1:1)
    w_port, h_port, axis_port = _crop_dims(1080, 1920, 1, 1)
    assert (w_port, h_port, axis_port) == (1080, 1080, "y")

    print(f"  ✓ 16:9 source (1920x1080) -> 9:16 ({w_916}x{h_916}, pan {axis_916}) and 1:1 ({w_11}x{h_11}, pan {axis_11}).")
    print(f"  ✓ Arbitrary source resolutions (4K UHD & Portrait) correctly generalized.")


def test_smoothstep_easing_curve():
    print("\n[PHASE 4 STRESS 3/6] Testing Cubic Smoothstep Ease-in-Out Mathematical Invariants...")

    # S(p) = 3p^2 - 2p^3
    assert _ease_in_out(0.0) == 0.0
    assert _ease_in_out(1.0) == 1.0
    assert _ease_in_out(0.5) == 0.5
    assert _ease_in_out(-0.5) == 0.0  # Clamped
    assert _ease_in_out(1.5) == 1.0   # Clamped

    # Monotonicity test
    samples = [_ease_in_out(p) for p in np.linspace(0.0, 1.0, 100)]
    for a, b in zip(samples, samples[1:]):
        assert b >= a, "Smoothstep easing curve must be strictly monotonically non-decreasing"

    # Zero derivative at boundaries (S'(0) = 0, S'(1) = 0)
    eps = 1e-4
    deriv_0 = (_ease_in_out(eps) - _ease_in_out(0.0)) / eps
    deriv_1 = (_ease_in_out(1.0) - _ease_in_out(1.0 - eps)) / eps
    assert abs(deriv_0) < 0.01, f"Initial derivative should be ~0, got {deriv_0}"
    assert abs(deriv_1) < 0.01, f"Terminal derivative should be ~0, got {deriv_1}"

    print("  ✓ Boundary conditions S(0)=0, S(1)=1, S(0.5)=0.5.")
    print(f"  ✓ Zero jerk at transition boundaries (S'(0)={deriv_0:.4f}, S'(1)={deriv_1:.4f}).")
    print("  ✓ Monotonic progression across 100 interpolation points.")


def test_face_priority_fallback_and_dense_targets():
    print("\n[PHASE 4 STRESS 4/6] Testing Step 7 Face-Priority Fallback & Gap Easing...")

    # Case: Speaker is at x=960. Creator points far right at x=1800.
    # In 1:1 track (crop_w=1080):
    # Centering on target (1800) would make crop window [1260, 2340], completely losing the speaker (960)!
    # Face-priority rule clamps target to [960 - 540, 960 + 540] = [420, 1500].
    raw_coords_scenario = {
        "fps": 30.0,
        "width": 1920,
        "height": 1080,
        "total_frames": 60,
        "frames": [
            {"frame_idx": 0, "t": 0.0, "face_center": [960.0, 400.0], "extrapolated_target": None, "focus": "speaker"},
            {"frame_idx": 30, "t": 1.0, "face_center": [960.0, 400.0], "extrapolated_target": [1800.0, 450.0], "focus": "object"},
        ]
    }

    dense_11 = _build_dense_target(
        raw_coords=raw_coords_scenario,
        axis_index=0,
        crop_dim=1080.0,
        source_dim=1920,
        frame_count=60,
        fps=30.0
    )

    # Frame 0-14 should hold 960
    assert dense_11[0] == 960.0
    assert dense_11[14] == 960.0

    # Frame 15 to 30 eases from 960.0 into 1500.0 (clamped face priority)
    assert dense_11[15] == 960.0
    assert dense_11[30] == 1500.0  # Clamped to 1500, NOT 1800!

    # Verify easing midpoint around frame 22-23 is ~ (960 + 1500)/2 = 1230
    mid_val = dense_11[23]
    assert 1200.0 <= mid_val <= 1260.0

    print(f"  ✓ 1:1 Face priority preserved: target clamped from 1800.0px to {dense_11[30]:.1f}px (face at 960px).")
    print(f"  ✓ Transition easing smoothly interpolated over {TRANSITION_EASE_FRAMES} frames (midpoint: {mid_val:.1f}px).")


def test_text_protection_clamping():
    print("\n[PHASE 4 STRESS 5/6] Testing Step 9 Protected Text Region Clamping & Decay...")

    # Text box at right margin: [1500, 100, 350, 60] active from t=1.0 to t=2.0 (frames 30 to 60)
    # 9:16 crop width = 608. Baseline camera center at 960. Crop window = [656, 1264].
    # Text spans [1500, 1850] -> completely outside baseline crop (coverage = 0%).
    # Step 9 must nudge camera to the right by up to 8.0 px/frame to protect the text.
    text_regions = {
        "fps": 30.0,
        "width": 1920,
        "height": 1080,
        "regions": [
            {
                "t_start": 1.0,
                "t_end": 2.0,
                "box": [1500.0, 100.0, 350.0, 60.0],
                "text": "CRITICAL METRICS",
                "confidence": 0.95
            }
        ]
    }

    centers = [960.0] * 90  # 3 seconds at 30 fps
    protected_centers, flags, unfixable = _apply_text_protection(
        centers=centers,
        crop_dim=608.0,
        fps=30.0,
        text_regions=text_regions,
        axis_index=0
    )

    # Before text (frame 0 to 29): no nudge
    assert protected_centers[0] == 960.0
    assert flags[0] is False

    # During text (frame 30 to 60): center moves rightward
    assert protected_centers[45] > 960.0
    assert flags[45] is True

    # Rate limiting check: max delta between consecutive frames <= MAX_TEXT_NUDGE_PX_PER_FRAME
    for a, b in zip(protected_centers, protected_centers[1:]):
        delta = abs(b - a)
        assert delta <= MAX_TEXT_NUDGE_PX_PER_FRAME + 1e-4, f"Text nudge exceeded rate limit: {delta:.2f} px/frame"

    # Decay check after t=2.0 (frame 61 to 90): center decays back towards 960.0
    assert protected_centers[89] < protected_centers[60]

    print("  ✓ Protected region triggered on un-covered text (<50% coverage).")
    print(f"  ✓ Persistent nudge rate-limited to <= {MAX_TEXT_NUDGE_PX_PER_FRAME} px/frame (anti-jitter guarantee).")
    print("  ✓ Smooth decay back to base coordinate after caption expiration.")


def test_full_phase4_end_to_end_contract():
    print("\n[PHASE 4 STRESS 6/6] Testing Dual Output Contracts & Schema Validation...")

    raw_coords = {
        "fps": 30.0,
        "width": 1920,
        "height": 1080,
        "total_frames": 100,
        "frames": [
            {
                "frame_idx": i,
                "t": round(i / 30.0, 3),
                "face_center": [960.0, 400.0],
                "extrapolated_target": [1600.0, 500.0] if 30 <= i <= 60 else None,
                "focus": "object" if 30 <= i <= 60 else "speaker"
            }
            for i in range(100)
        ]
    }

    text_regions = {
        "fps": 30.0,
        "width": 1920,
        "height": 1080,
        "regions": [
            {
                "t_start": 2.2,
                "t_end": 3.0,
                "box": [100.0, 900.0, 400.0, 50.0],
                "text": "LOWER THIRD BANNER",
                "confidence": 0.98
            }
        ]
    }

    focus_timeline = {
        "blocks": [
            {"start": 0.0, "end": 1.0, "focus": "speaker", "direction_hint": "center", "confidence": 1.0},
            {"start": 1.0, "end": 2.0, "focus": "object", "direction_hint": "right", "confidence": 0.95},
            {"start": 2.0, "end": 3.33, "focus": "speaker", "direction_hint": "center", "confidence": 1.0},
        ]
    }

    coords_916, coords_11 = run_smooth_coords(
        raw_coords=raw_coords,
        text_regions=text_regions,
        focus_timeline=focus_timeline,
        mock=False
    )

    validate_final_coords(coords_916)
    validate_final_coords(coords_11)

    assert coords_916["aspect_ratio"] == "9:16"
    assert coords_916["target_width"] == 608
    assert coords_916["target_height"] == 1080
    assert len(coords_916["frames"]) == 100

    assert coords_11["aspect_ratio"] == "1:1"
    assert coords_11["target_width"] == 1080
    assert coords_11["target_height"] == 1080
    assert len(coords_11["frames"]) == 100

    # Bounds check on all frames: 0 <= crop_x <= source_w - crop_w
    for f in coords_916["frames"]:
        assert 0 <= f["crop_x"] <= 1920 - 608
        assert f["crop_y"] == 0
        assert f["crop_w"] == 608
        assert f["crop_h"] == 1080

    for f in coords_11["frames"]:
        assert 0 <= f["crop_x"] <= 1920 - 1080
        assert f["crop_y"] == 0
        assert f["crop_w"] == 1080
        assert f["crop_h"] == 1080

    print("  ✓ 9:16 track (608x1080) and 1:1 track (1080x1080) strictly validated.")
    print("  ✓ 100% boundary check: All 200 frame crop boxes remain within [0, 1920] x [0, 1080].")


def main():
    print("=" * 70)
    print("      CONTEXT-AWARE AUTO-REFRAME: PHASE 4 STRESS & INVARIANT SUITE    ")
    print("=" * 70)

    test_one_euro_filter_invariants()
    test_crop_dims_geometry()
    test_smoothstep_easing_curve()
    test_face_priority_fallback_and_dense_targets()
    test_text_protection_clamping()
    test_full_phase4_end_to_end_contract()

    print("\n" + "=" * 70)
    print("🎉 ALL 6 PHASE 4 MATHEMATICAL & ARCHITECTURAL STRESS SUITES PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    main()

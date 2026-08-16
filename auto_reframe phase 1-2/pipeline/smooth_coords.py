"""
pipeline/smooth_coords.py — Phase 4, Steps 7-10: Dual-Aspect Coordinator & Adaptive Smoothing

Contract:
  Inputs:  raw_coords.json, text_regions.json, focus_timeline.json
  Outputs: final_coords_916.json, final_coords_11.json

This is where the two deliverables (9:16 and 1:1) diverge from one shared source
of truth (raw_coords.json). Each track is constructed through four layers:

  1. Dense target construction + gap easing (Steps 7 & 10 combined):
     raw_coords.json can be sparse across non-object frames. Every frame in the video
     receives a target value. Gaps are filled by holding the last known value and
     easing (cubic smoothstep ease-in-out, Step 10) into the next known value over the
     last ~15 frames before it arrives.
     Step 7 Face-Priority Fallback: For the 1:1 track (and 9:16), if centering on a
     pointing target would push the most recently seen face out of the crop window,
     the face constraint wins to keep the speaker visible while leaning toward the target.

  2. One Euro Filter (Step 8):
     Adaptive low-pass filter applied per-axis. Automatically switches beta:
     higher beta (BETA_OBJECT = 1.6) during 'object' blocks for agile responsiveness,
     and lower beta (BETA_SPEAKER = 0.3) during 'speaker' blocks for jitter-free stability.

  3. Protected-Region Text Clamp (Step 9):
     Evaluates active text bounding boxes from text_regions.json. If on-screen text
     would be more than 50% cut off by the crop, applies a persistent, rate-limited
     nudge (up to 8.0 px/frame) toward including it, smoothly decaying back to zero
     once the text expires.

  4. Final Bounds Clamping & Schema Delivery:
     Strictly bounds the crop window within source video dimensions [0, source_w - crop_w]
     and rounds to integer pixels, fulfilling FinalCoordsData contracts.

Supports --mock flag for fast architectural testing.
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Set

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import stage_path                                     # noqa: E402
from contracts import (                                           # noqa: E402
    FinalCoordsData, FinalFrameCoord,
    validate_raw_coords, validate_text_regions, validate_focus_timeline, validate_final_coords
)
from utils.io_json import load_json, save_json, fail_stage        # noqa: E402
from utils.one_euro import OneEuroFilter                          # noqa: E402

STAGE_NAME = "smooth_coords"

TRACKS = {
    "9:16": (9, 16),
    "1:1": (1, 1)
}

TRANSITION_EASE_FRAMES = 15        # Step 10: "ease-in-out curve over ~15 frames"
MIN_CUTOFF = 1.0                   # Step 8: One Euro Filter baseline cutoff
BETA_SPEAKER = 0.3                 # Lower beta = heavy smoothing during 'speaker' blocks
BETA_OBJECT = 1.6                  # Higher beta = agile responsiveness during 'object' blocks
D_CUTOFF = 1.0                     # Standard One Euro derivative cutoff
TEXT_COVERAGE_THRESHOLD = 0.5      # Step 9: "more than ~50% cut off" trigger
MAX_TEXT_NUDGE_PX_PER_FRAME = 8.0  # Rate limit so the Step 9 clamp cannot induce jitter


def _crop_dims(source_w: int, source_h: int, aspect_w: int, aspect_h: int) -> Tuple[int, int, str]:
    """
    Fits an (aspect_w : aspect_h) crop window inside (source_w, source_h),
    using the full extent of whichever source dimension has no slack, and panning along the other.
    For standard 16:9 1920x1080 landscape video:
      - 9:16 target -> 608x1080 (pan axis 'x', slack = 1312 px)
      - 1:1 target  -> 1080x1080 (pan axis 'x', slack = 840 px)
    """
    if source_w * aspect_h >= source_h * aspect_w:
        crop_h = source_h
        crop_w = min(source_w, int(round(crop_h * aspect_w / aspect_h)))
        return crop_w, crop_h, "x"
    else:
        crop_w = source_w
        crop_h = min(source_h, int(round(crop_w * aspect_h / aspect_w)))
        return crop_w, crop_h, "y"


def _ease_in_out(p: float) -> float:
    """
    Cubic smoothstep ease-in-out curve: S(p) = 3p^2 - 2p^3.
    Step 10: Ensures natural human-like camera motion without mechanical linear jerks.
    """
    p = min(1.0, max(0.0, float(p)))
    return p * p * (3.0 - 2.0 * p)


def _object_frame_mask(focus_timeline: dict, fps: float, frame_count: int) -> List[bool]:
    """
    Constructs a per-frame boolean mask indicating whether frame is inside an 'object' focus block.
    Supports both start/end and start_time/end_time key conventions.
    """
    mask = [False] * frame_count
    for block in focus_timeline.get("blocks", []):
        if block.get("focus") != "object":
            continue
        start_t = float(block.get("start", block.get("start_time", 0.0)))
        end_t = float(block.get("end", block.get("end_time", start_t)))
        start_f = max(0, int(round(start_t * fps)))
        end_f = min(frame_count - 1, int(round(end_t * fps)))
        for f in range(start_f, end_f + 1):
            mask[f] = True
    return mask


def _build_dense_target(
    raw_coords: dict,
    axis_index: int,
    crop_dim: float,
    source_dim: int,
    frame_count: int,
    fps: float
) -> List[float]:
    """
    Steps 7 & 10: Dense Target Construction & Smooth Transition Easing.

    1. Extracts keyframes from raw_coords.
    2. Step 7 Fallback: If extrapolated_target would push last_face outside crop_dim,
       clamps coordinate to keep face visible while leaning maximally toward target.
    3. Step 10: Interpolates gaps by holding previous position and easing over the
       final TRANSITION_EASE_FRAMES leading into the next target.
    """
    keyframes: List[Tuple[int, float]] = []
    last_face = None

    frames_list = raw_coords.get("frames", [])
    for entry in frames_list:
        face = entry.get("face_center")
        if face is not None and len(face) > axis_index:
            last_face = float(face[axis_index])

        target_pt = entry.get("extrapolated_target")
        if target_pt is not None and len(target_pt) > axis_index:
            val = float(target_pt[axis_index])
            # Step 7 Fallback: Maintain speaker face within crop window bounds
            if last_face is not None:
                lo = last_face - crop_dim / 2.0
                hi = last_face + crop_dim / 2.0
                val = min(max(val, lo), hi)
        elif face is not None and len(face) > axis_index:
            val = float(face[axis_index])
        else:
            continue

        f_idx = entry.get("frame_idx")
        if f_idx is None:
            f_idx = int(round(float(entry.get("t", 0.0)) * fps))
        f_idx = min(max(int(f_idx), 0), frame_count - 1)
        keyframes.append((f_idx, val))

    if not keyframes:
        # Fallback to source frame center
        return [source_dim / 2.0] * frame_count

    dense = [0.0] * frame_count
    first_idx, first_val = keyframes[0]
    for f in range(0, min(first_idx + 1, frame_count)):
        dense[f] = first_val

    for (idx_a, val_a), (idx_b, val_b) in zip(keyframes, keyframes[1:]):
        gap = idx_b - idx_a
        if gap <= 0:
            continue
        ease_len = min(TRANSITION_EASE_FRAMES, gap)
        ease_start = idx_b - ease_len
        for f in range(idx_a + 1, min(ease_start, frame_count)):
            dense[f] = val_a
        for k in range(ease_len + 1):
            f = ease_start + k
            if 0 <= f < frame_count:
                progress = (k / ease_len) if ease_len > 0 else 1.0
                dense[f] = val_a + (val_b - val_a) * _ease_in_out(progress)

    last_idx, last_val = keyframes[-1]
    for f in range(max(0, last_idx), frame_count):
        dense[f] = last_val

    return dense


def _smooth_with_one_euro(dense: List[float], fps: float, object_mask: List[bool]) -> List[float]:
    """
    Step 8: One Euro Filter Smoothing with Contextual Adaptive Beta.
    Applies lower beta (BETA_SPEAKER = 0.3) for stable speaker tracking and
    higher beta (BETA_OBJECT = 1.6) during object pointing events.
    """
    if not dense:
        return []
    filt = OneEuroFilter(
        t0=0.0,
        x0=dense[0],
        min_cutoff=MIN_CUTOFF,
        beta=BETA_SPEAKER,
        d_cutoff=D_CUTOFF
    )
    out = [dense[0]]
    for i in range(1, len(dense)):
        t = i / fps
        is_obj = (i < len(object_mask) and object_mask[i])
        current_beta = BETA_OBJECT if is_obj else BETA_SPEAKER
        out.append(filt(t, dense[i], beta=current_beta))
    return out


def _decay_toward_zero(value: float, rate: float) -> float:
    """Decays an offset value towards zero by up to rate per step."""
    if value > 0.0:
        return max(0.0, value - rate)
    if value < 0.0:
        return min(0.0, value + rate)
    return 0.0


def _apply_text_protection(
    centers: List[float],
    crop_dim: float,
    fps: float,
    text_regions: dict,
    axis_index: int
) -> Tuple[List[float], List[bool], int]:
    """
    Step 9: Protected-Region Text Clamping.

    Maintains a persistent, rate-limited correction offset (capped at 8.0 px/frame)
    to prevent on-screen text from being clipped by >50%.
    Decays smoothly back to zero once the on-screen text disappears.
    """
    raw_regions = text_regions.get("regions", [])
    regions_sorted = sorted(raw_regions, key=lambda r: float(r.get("t_start", 0.0)))
    add_ptr = 0
    active: List[dict] = []
    unfixable_ids: Set[int] = set()

    out_centers: List[float] = []
    protected_flags: List[bool] = []
    correction = 0.0

    for i, base in enumerate(centers):
        t = i / fps
        while add_ptr < len(regions_sorted) and float(regions_sorted[add_ptr].get("t_start", 0.0)) <= t:
            active.append(regions_sorted[add_ptr])
            add_ptr += 1
        active[:] = [r for r in active if float(r.get("t_end", 0.0)) >= t]

        center = base + correction
        worst_region = None
        worst_cov = 1.0

        for r in active:
            box = r.get("box")
            if box is not None and len(box) == 4:
                rx, ry, rw, rh = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            else:
                rx = float(r.get("x", 0.0))
                ry = float(r.get("y", 0.0))
                rw = float(r.get("w", 0.0))
                rh = float(r.get("h", 0.0))

            r_lo = rx if axis_index == 0 else ry
            r_len = rw if axis_index == 0 else rh
            if r_len <= 0:
                continue
            r_hi = r_lo + r_len
            crop_lo = center - crop_dim / 2.0
            crop_hi = center + crop_dim / 2.0

            overlap = max(0.0, min(crop_hi, r_hi) - max(crop_lo, r_lo))
            cov = overlap / r_len
            if cov < worst_cov:
                worst_cov = cov
                worst_region = (r, r_lo, r_len)

        is_protected_frame = False
        if worst_region is not None and worst_cov < TEXT_COVERAGE_THRESHOLD:
            r_dict, r_lo, r_len = worst_region
            r_center = r_lo + r_len / 2.0
            desired_shift = r_center - center
            step = max(-MAX_TEXT_NUDGE_PX_PER_FRAME, min(MAX_TEXT_NUDGE_PX_PER_FRAME, desired_shift))
            correction += step
            is_protected_frame = True

            if r_len > crop_dim:
                unfixable_ids.add(id(r_dict))
        else:
            if abs(correction) > 1e-4:
                is_protected_frame = True
            correction = _decay_toward_zero(correction, MAX_TEXT_NUDGE_PX_PER_FRAME)

        out_centers.append(base + correction)
        protected_flags.append(is_protected_frame)

    return out_centers, protected_flags, len(unfixable_ids)


def _build_track(
    aspect_label: str,
    aspect_w: int,
    aspect_h: int,
    raw_coords: dict,
    text_regions: dict,
    focus_timeline: dict
) -> Dict[str, Any]:
    """Builds a smoothed, text-protected, schema-compliant crop coordinate track."""
    fps = float(raw_coords.get("fps", raw_coords.get("meta", {}).get("fps", 30.0)))
    source_w = int(raw_coords.get("width", raw_coords.get("meta", {}).get("width", 1920)))
    source_h = int(raw_coords.get("height", raw_coords.get("meta", {}).get("height", 1080)))
    frame_count = int(raw_coords.get(
        "total_frames",
        raw_coords.get("meta", {}).get("total_frames", len(raw_coords.get("frames", [])))
    ))
    if frame_count <= 0:
        frame_count = len(raw_coords.get("frames", []))
    if frame_count <= 0:
        frame_count = 1

    crop_w, crop_h, pan_axis = _crop_dims(source_w, source_h, aspect_w, aspect_h)
    axis_index = 0 if pan_axis == "x" else 1
    source_dim = source_w if pan_axis == "x" else source_h
    crop_dim = crop_w if pan_axis == "x" else crop_h

    # 1. Dense target construction + transition easing + 1:1 face fallback
    dense = _build_dense_target(raw_coords, axis_index, float(crop_dim), source_dim, frame_count, fps)

    # 2. Adaptive One Euro Filter smoothing
    object_mask = _object_frame_mask(focus_timeline, fps, frame_count)
    smoothed = _smooth_with_one_euro(dense, fps, object_mask)

    # 3. Protected text clamping
    protected_centers, protected_flags, unfixable_count = _apply_text_protection(
        smoothed, float(crop_dim), fps, text_regions, axis_index
    )

    # 4. Final bounds clamping & schema formatting
    final_frames: List[FinalFrameCoord] = []
    for i, center in enumerate(protected_centers):
        t = round(i / fps, 3)
        origin = center - crop_dim / 2.0
        origin = min(max(origin, 0.0), float(source_dim - crop_dim))

        if pan_axis == "x":
            cx, cy = origin, (source_h - crop_h) / 2.0
        else:
            cx, cy = (source_w - crop_w) / 2.0, origin

        focus_type = "object" if (i < len(object_mask) and object_mask[i]) else "speaker"
        is_prot = protected_flags[i] if i < len(protected_flags) else False

        final_frames.append(
            FinalFrameCoord(
                frame_idx=i,
                t=t,
                crop_x=int(round(cx)),
                crop_y=int(round(cy)),
                crop_w=crop_w,
                crop_h=crop_h,
                focus=focus_type,
                text_protected=is_prot
            )
        )

    if unfixable_count > 0:
        print(f"[{STAGE_NAME}] note: {unfixable_count} text region(s) wider than {aspect_label} "
              f"crop window — covered best-effort.", file=sys.stderr)

    result_data = FinalCoordsData(
        aspect_ratio=aspect_label,  # type: ignore
        target_width=crop_w,
        target_height=crop_h,
        source_width=source_w,
        source_height=source_h,
        fps=fps,
        total_frames=frame_count,
        frames=final_frames
    )
    res_dict = result_data.to_dict()

    # Legacy metadata support
    res_dict["meta"] = {
        "crop_width": crop_w,
        "crop_height": crop_h,
        "source_width": source_w,
        "source_height": source_h,
        "fps": fps,
        "frame_count": frame_count,
        "unfixable_text_regions": unfixable_count
    }
    return res_dict


def generate_mock_final_coords(
    raw_coords: dict, text_regions: dict, focus_timeline: dict
) -> Tuple[dict, dict]:
    """Generate mock 9:16 and 1:1 smoothed coordinates conforming to FinalCoordsData contract."""
    fps = float(raw_coords.get("fps", 30.0))
    total_frames = int(raw_coords.get("total_frames", len(raw_coords.get("frames", [])) or 311))
    src_w = int(raw_coords.get("width", 1920))
    src_h = int(raw_coords.get("height", 1080))

    crop_w_916, crop_h_916 = 608, 1080
    crop_w_11, crop_h_11 = 1080, 1080

    frames_916: List[FinalFrameCoord] = []
    frames_11: List[FinalFrameCoord] = []

    for i in range(total_frames):
        t = round(i / fps, 3)
        raw_f = raw_coords.get("frames", [])[i] if i < len(raw_coords.get("frames", [])) else {}
        focus = raw_f.get("focus", "speaker")

        if 3.5 <= t <= 6.8:
            crop_x_916 = 1312
            crop_x_11 = 840
        else:
            crop_x_916 = (src_w - crop_w_916) // 2
            crop_x_11 = (src_w - crop_w_11) // 2

        frames_916.append(FinalFrameCoord(
            frame_idx=i,
            t=t,
            crop_x=crop_x_916,
            crop_y=0,
            crop_w=crop_w_916,
            crop_h=crop_h_916,
            focus=focus,
            text_protected=(t >= 6.8)
        ))

        frames_11.append(FinalFrameCoord(
            frame_idx=i,
            t=t,
            crop_x=crop_x_11,
            crop_y=0,
            crop_w=crop_w_11,
            crop_h=crop_h_11,
            focus=focus,
            text_protected=(t >= 6.8)
        ))

    coords_916 = FinalCoordsData(
        aspect_ratio="9:16",
        target_width=crop_w_916,
        target_height=crop_h_916,
        source_width=src_w,
        source_height=src_h,
        fps=fps,
        total_frames=total_frames,
        frames=frames_916
    ).to_dict()

    coords_11 = FinalCoordsData(
        aspect_ratio="1:1",
        target_width=crop_w_11,
        target_height=crop_h_11,
        source_width=src_w,
        source_height=src_h,
        fps=fps,
        total_frames=total_frames,
        frames=frames_11
    ).to_dict()

    return coords_916, coords_11


def run(
    raw_coords: dict,
    text_regions: dict,
    focus_timeline: dict,
    mock: bool = False
) -> Tuple[dict, dict]:
    """
    Executes Phase 4 Dual-Aspect Coordinator & Adaptive Smoothing.
    Returns (coords_916, coords_11) conforming to FinalCoordsData contracts.
    """
    if mock:
        return generate_mock_final_coords(raw_coords, text_regions, focus_timeline)

    aw_916, ah_916 = TRACKS["9:16"]
    coords_916 = _build_track("9:16", aw_916, ah_916, raw_coords, text_regions, focus_timeline)

    aw_11, ah_11 = TRACKS["1:1"]
    coords_11 = _build_track("1:1", aw_11, ah_11, raw_coords, text_regions, focus_timeline)

    return coords_916, coords_11


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-coords", type=Path, default=stage_path("raw_coords.json"))
    parser.add_argument("--text-regions", type=Path, default=stage_path("text_regions.json"))
    parser.add_argument("--focus-timeline", type=Path, default=stage_path("focus_timeline.json"))
    parser.add_argument("--out-916", type=Path, default=stage_path("final_coords_916.json"))
    parser.add_argument("--out-11", type=Path, default=stage_path("final_coords_11.json"))
    parser.add_argument("--mock", action="store_true", help="Run in mock mode with synthetic data for testing")
    args = parser.parse_args()

    try:
        raw_coords = load_json(args.raw_coords, validator=validate_raw_coords)
        text_regions = load_json(args.text_regions, validator=validate_text_regions)
        focus_timeline = load_json(args.focus_timeline, validator=validate_focus_timeline)

        coords_916, coords_11 = run(raw_coords, text_regions, focus_timeline, mock=args.mock)

        save_json(args.out_916, coords_916, validator=validate_final_coords)
        save_json(args.out_11, coords_11, validator=validate_final_coords)
        print(f"[{STAGE_NAME}] successfully wrote {args.out_916} "
              f"({coords_916['target_width']}x{coords_916['target_height']}) and "
              f"{args.out_11} ({coords_11['target_width']}x{coords_11['target_height']}), "
              f"{len(coords_916['frames'])} frames each")
    except Exception as e:
        fail_stage(STAGE_NAME, e)


if __name__ == "__main__":
    main()

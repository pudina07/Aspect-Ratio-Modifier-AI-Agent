"""
pipeline/tracker.py — Phase 3, Steps 4 & 5: MediaPipe Face, Pose & Hand Pointing Vector Tracker

Contract:
  Inputs:  video.mp4, focus_timeline.json
  Output:  raw_coords.json

Step 4: Runs baseline FaceDetector (MediaPipe Tasks, BlazeFace short-range) every 5th frame
        to provide a reliable talking-head crop center across the entire clip.

Step 5: During 'object' blocks (flagged in focus_timeline.json), runs PoseLandmarker (wrists 15/16)
        and HandLandmarker (index fingertip 8). Calculates the 2D ray vector from wrist -> fingertip
        and extrapolates target coordinate 35-40% toward frame boundaries.
        Extrapolating beyond fingertip resolves the classic wrist-only undershoot problem.

Supports --mock flag for fast architectural testing.
"""
import argparse
import math
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import stage_path, MODELS_DIR, PROJECT_ROOT                     # noqa: E402
from contracts import (                                                     # noqa: E402
    RawCoordsData, RawFrameCoord,
    validate_focus_timeline, validate_raw_coords
)
from utils.io_json import load_json, save_json, fail_stage                  # noqa: E402

STAGE_NAME = "tracker"

FACE_SAMPLE_RATE = 5             # Step 4: baseline face track every 5th frame
POSE_WRIST_LANDMARKS = (15, 16)  # 15=left wrist, 16=right wrist in BlazePose topology
HAND_WRIST_LANDMARK = 0          # Hand root wrist landmark
HAND_FINGERTIP_LANDMARK = 8      # Index fingertip
EXTRAPOLATION_FRACTION = 0.375   # Midpoint of plan's 35-40% range
MIN_WRIST_VISIBILITY = 0.25      # Visibility threshold for pose landmarks

FACE_MODEL_NAMES = ["blaze_face_short_range.tflite", "face_detector.tflite"]
POSE_MODEL_NAMES = ["pose_landmarker_full.task", "pose_landmarker_lite.task"]
HAND_MODEL_NAMES = ["hand_landmarker.task"]


def _resolve_model_path(candidates: List[str]) -> Path:
    """Finds existing model weights file across candidate locations."""
    search_dirs = [
        MODELS_DIR / "mediapipe",
        PROJECT_ROOT / "models" / "mediapipe",
        MODELS_DIR,
        PROJECT_ROOT / "models",
    ]
    for filename in candidates:
        for sdir in search_dirs:
            p = sdir / filename
            if p.exists():
                return p

    raise FileNotFoundError(
        f"Could not find any of model weights {candidates} in {search_dirs}. "
        f"Run 'python download_models.py' to download required weights."
    )


def _build_detectors(delegate: str = "CPU"):
    """Constructs MediaPipe Tasks detectors."""
    import mediapipe as mp

    base_options_cls = mp.tasks.BaseOptions
    mp_delegate = (
        base_options_cls.Delegate.GPU if delegate.upper() == "GPU"
        else base_options_cls.Delegate.CPU
    )

    face_model_p = _resolve_model_path(FACE_MODEL_NAMES)
    pose_model_p = _resolve_model_path(POSE_MODEL_NAMES)
    hand_model_p = _resolve_model_path(HAND_MODEL_NAMES)

    print(f"[{STAGE_NAME}] Initializing MediaPipe Tasks (Face: {face_model_p.name}, Pose: {pose_model_p.name}, Hand: {hand_model_p.name}, Delegate: {delegate})...")

    face_detector = mp.tasks.vision.FaceDetector.create_from_options(
        mp.tasks.vision.FaceDetectorOptions(
            base_options=base_options_cls(model_asset_path=str(face_model_p), delegate=mp_delegate),
            min_detection_confidence=0.45,
        )
    )
    pose_landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(
        mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_options_cls(model_asset_path=str(pose_model_p), delegate=mp_delegate),
            num_poses=1,
            min_pose_detection_confidence=0.45,
        )
    )
    hand_landmarker = mp.tasks.vision.HandLandmarker.create_from_options(
        mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options_cls(model_asset_path=str(hand_model_p), delegate=mp_delegate),
            num_hands=2,
            min_hand_detection_confidence=0.45,
        )
    )
    return mp, face_detector, pose_landmarker, hand_landmarker


def _extract_object_frame_ranges(focus_timeline: dict, fps: float) -> List[Tuple[int, int]]:
    """Extracts frame index ranges where focus == 'object', handling start/end & start_time/end_time."""
    ranges = []
    blocks = focus_timeline.get("blocks", [])
    for b in blocks:
        if b.get("focus") == "object":
            s_t = float(b.get("start", b.get("start_time", 0.0)))
            e_t = float(b.get("end", b.get("end_time", s_t + 1.0)))
            s_f = max(0, int(round(s_t * fps)))
            e_f = max(s_f, int(round(e_t * fps)))
            ranges.append((s_f, e_f))
    return sorted(ranges, key=lambda r: r[0])


def _is_in_object_range(frame_idx: int, ranges: List[Tuple[int, int]], pointer: int) -> Tuple[bool, int]:
    """Efficient 2-pointer range membership check for sequential video iteration."""
    while pointer < len(ranges) and ranges[pointer][1] < frame_idx:
        pointer += 1
    in_range = (pointer < len(ranges) and ranges[pointer][0] <= frame_idx <= ranges[pointer][1])
    return in_range, pointer


def _get_largest_face_box_and_center(detection_result, width: int, height: int) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    """Finds largest detected face bounding box and center coordinate."""
    if not detection_result.detections:
        return None, None

    best = max(
        detection_result.detections,
        key=lambda d: d.bounding_box.width * d.bounding_box.height
    )
    bb = best.bounding_box
    cx = float(bb.origin_x + bb.width / 2.0)
    cy = float(bb.origin_y + bb.height / 2.0)
    box = [float(bb.origin_x), float(bb.origin_y), float(bb.width), float(bb.height)]
    center = [round(cx, 1), round(cy, 1)]
    return box, center


def _ray_box_exit(origin: Tuple[float, float], direction: Tuple[float, float], width: float, height: float) -> Tuple[float, float]:
    """Calculates ray intersection point with frame boundaries [0, width] x [0, height]."""
    ox, oy = origin
    dx, dy = direction
    t_candidates = []
    if dx > 1e-6:
        t_candidates.append((width - ox) / dx)
    elif dx < -1e-6:
        t_candidates.append((0.0 - ox) / dx)

    if dy > 1e-6:
        t_candidates.append((height - oy) / dy)
    elif dy < -1e-6:
        t_candidates.append((0.0 - oy) / dy)

    pos_t = [t for t in t_candidates if t > 0]
    if not pos_t:
        return origin
    min_t = min(pos_t)
    return (ox + dx * min_t, oy + dy * min_t)


def _compute_pointing_target(pose_result, hand_result, width: int, height: int) -> Tuple[Optional[List[float]], Optional[List[float]], Optional[List[float]]]:
    """
    Computes wrist, index fingertip, and 35-40% extrapolated target point.
    """
    if not pose_result.pose_landmarks or not hand_result.hand_landmarks:
        return None, None, None

    pose = pose_result.pose_landmarks[0]
    pose_wrists: Dict[int, Tuple[float, float]] = {}

    for idx in POSE_WRIST_LANDMARKS:
        lm = pose[idx]
        vis = getattr(lm, "visibility", 1.0)
        if vis is None or vis >= MIN_WRIST_VISIBILITY:
            pose_wrists[idx] = (lm.x * width, lm.y * height)

    if not pose_wrists:
        return None, None, None

    best_pair = None
    best_ext = -1.0

    for hand in hand_result.hand_landmarks:
        hand_w = (hand[HAND_WRIST_LANDMARK].x * width, hand[HAND_WRIST_LANDMARK].y * height)
        tip = (hand[HAND_FINGERTIP_LANDMARK].x * width, hand[HAND_FINGERTIP_LANDMARK].y * height)

        # Match hand to nearest pose wrist
        matched_idx = min(pose_wrists.keys(), key=lambda i: math.dist(pose_wrists[i], hand_w))
        wrist_coord = pose_wrists[matched_idx]
        ext = math.dist(wrist_coord, tip)

        if ext > best_ext:
            best_ext = ext
            best_pair = (wrist_coord, tip)

    if best_pair is None or best_ext < 5.0:
        return None, None, None

    wrist_pt, tip_pt = best_pair
    dx = tip_pt[0] - wrist_pt[0]
    dy = tip_pt[1] - wrist_pt[1]
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return None, None, None

    dir_vec = (dx / norm, dy / norm)
    exit_pt = _ray_box_exit(tip_pt, dir_vec, float(width), float(height))

    target_x = tip_pt[0] + EXTRAPOLATION_FRACTION * (exit_pt[0] - tip_pt[0])
    target_y = tip_pt[1] + EXTRAPOLATION_FRACTION * (exit_pt[1] - tip_pt[1])

    target_clamped = [
        round(max(0.0, min(float(width), target_x)), 1),
        round(max(0.0, min(float(height), target_y)), 1)
    ]

    return (
        [round(wrist_pt[0], 1), round(wrist_pt[1], 1)],
        [round(tip_pt[0], 1), round(tip_pt[1], 1)],
        target_clamped
    )


def generate_mock_raw_coords(video_path: Path, focus_timeline: dict) -> dict:
    """Generate mock tracking coordinates conforming to RawCoordsData contract."""
    fps = 30.0
    total_frames = 311
    width = 1920
    height = 1080

    frames = []
    for i in range(total_frames):
        t = i / fps
        face_center = [960.0, 400.0]
        face_box = [885.0, 295.0, 150.0, 210.0]
        wrist = None
        fingertip = None
        target = None
        focus = "speaker"

        for b in focus_timeline.get("blocks", []):
            s_t = b.get("start", b.get("start_time", 0.0))
            e_t = b.get("end", b.get("end_time", 0.0))
            if b.get("focus") == "object" and s_t <= t <= e_t:
                focus = "object"
                wrist = [1540.0, 520.0]
                fingertip = [1590.0, 510.0]
                target = [1720.0, 480.0]
                break

        frames.append(RawFrameCoord(
            frame_idx=i,
            t=t,
            face_center=face_center,
            face_box=face_box,
            wrist=wrist,
            fingertip=fingertip,
            extrapolated_target=target,
            focus=focus
        ))

    data = RawCoordsData(
        fps=fps,
        width=width,
        height=height,
        total_frames=total_frames,
        frames=frames
    )
    res = data.to_dict()
    res["meta"] = {
        "width": width,
        "height": height,
        "fps": fps,
        "total_frames": total_frames
    }
    return res


def run(
    video_path: Path,
    focus_timeline: dict,
    delegate: str = "CPU",
    face_sample_rate: int = FACE_SAMPLE_RATE,
    mock: bool = False
) -> dict:
    """
    Executes MediaPipe face tracking and pose/hand pointing vector extrapolation.
    """
    if mock:
        return generate_mock_raw_coords(video_path, focus_timeline)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found at: {video_path}")

    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV failed to open video at {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if fps <= 0:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    object_ranges = _extract_object_frame_ranges(focus_timeline, fps)
    range_pointer = 0

    mp, face_detector, pose_landmarker, hand_landmarker = _build_detectors(delegate)

    frames_out: List[RawFrameCoord] = []
    frame_idx = 0

    last_face_center: Optional[List[float]] = [width / 2.0, height * 0.38]
    last_face_box: Optional[List[float]] = [width * 0.45, height * 0.25, width * 0.1, height * 0.2]

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            t = round(frame_idx / fps, 3)
            needs_face = (frame_idx % face_sample_rate == 0)
            in_object_block, range_pointer = _is_in_object_range(frame_idx, object_ranges, range_pointer)

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            face_center = last_face_center
            face_box = last_face_box

            if needs_face:
                face_result = face_detector.detect(mp_image)
                f_box, f_center = _get_largest_face_box_and_center(face_result, width, height)
                if f_center is not None:
                    face_center = f_center
                    face_box = f_box
                    last_face_center = f_center
                    last_face_box = f_box

            wrist = None
            fingertip = None
            target = None
            focus = "object" if in_object_block else "speaker"

            if in_object_block:
                pose_result = pose_landmarker.detect(mp_image)
                hand_result = hand_landmarker.detect(mp_image)
                w_pt, f_pt, t_pt = _compute_pointing_target(pose_result, hand_result, width, height)
                wrist = w_pt
                fingertip = f_pt
                target = t_pt

            frames_out.append(
                RawFrameCoord(
                    frame_idx=frame_idx,
                    t=t,
                    face_center=face_center,
                    face_box=face_box,
                    wrist=wrist,
                    fingertip=fingertip,
                    extrapolated_target=target,
                    focus=focus
                )
            )

            frame_idx += 1
    finally:
        cap.release()
        face_detector.close()
        pose_landmarker.close()
        hand_landmarker.close()

    actual_total = len(frames_out)
    data = RawCoordsData(
        fps=fps,
        width=width,
        height=height,
        total_frames=actual_total,
        frames=frames_out
    )
    res_dict = data.to_dict()
    res_dict["meta"] = {
        "width": width,
        "height": height,
        "fps": fps,
        "total_frames": actual_total,
        "object_blocks": len(object_ranges)
    }
    return res_dict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=stage_path("video.mp4"))
    parser.add_argument("--focus-timeline", type=Path, default=stage_path("focus_timeline.json"))
    parser.add_argument("--out", type=Path, default=stage_path("raw_coords.json"))
    parser.add_argument("--delegate", default="CPU", choices=["CPU", "GPU"])
    parser.add_argument("--face-sample-rate", type=int, default=FACE_SAMPLE_RATE)
    parser.add_argument("--mock", action="store_true", help="Generate mock tracking coords for testing")
    args = parser.parse_args()

    try:
        focus_timeline = load_json(args.focus_timeline, validator=validate_focus_timeline)
        raw_coords = run(
            video_path=args.video,
            focus_timeline=focus_timeline,
            delegate=args.delegate,
            face_sample_rate=args.face_sample_rate,
            mock=args.mock
        )
        save_json(args.out, raw_coords, validator=validate_raw_coords)
        n_point = sum(1 for f in raw_coords.get("frames", []) if f.get("extrapolated_target") is not None)
        print(f"[{STAGE_NAME}] successfully wrote {args.out} ({len(raw_coords.get('frames', []))} frames, {n_point} pointing vectors)")
    except Exception as e:
        fail_stage(STAGE_NAME, e)


if __name__ == "__main__":
    main()

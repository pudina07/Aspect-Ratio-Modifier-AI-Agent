"""
pipeline/tracker.py — Phase 3, Steps 4-5

Contract:  video.mp4 + focus_timeline.json  ->  raw_coords.json

Step 4: baseline FaceDetector (MediaPipe Tasks, short-range BlazeFace)
every 5th frame — always runs, regardless of what focus_timeline says,
so there's a fallback crop center for any stretch of video the LLM
didn't flag as an 'object' moment.

Step 5: during 'object' blocks from focus_timeline.json, run
PoseLandmarker (wrist, landmark 15/16) + HandLandmarker (index
fingertip, landmark 8) on every frame in the block, then extrapolate a
target point ~35-40% of the way from the fingertip toward the frame
edge along the wrist->fingertip vector. This is the fix over a
wrist-only approach: the wrist alone tends to undershoot because it's
still near the body, not at the object actually being pointed at.

focus_timeline.json is a dependency (not just video.mp4) specifically
so this stage knows *when* to spend the extra compute on pose+hand
instead of running it on every single frame — see config.py's
PIPELINE_STAGES for how that dependency is wired.

Uses the modern `mediapipe.tasks` Tasks API throughout (mp.tasks.vision),
not the legacy `mp.solutions` API. Model weights
(pose_landmarker_full.task, hand_landmarker.task,
blaze_face_short_range.tflite) are expected in MODELS_DIR, downloaded
once during Phase 0 setup — this stage does not download them itself,
matching the plan's "no multi-hundred-MB download mid-pipeline" rule.

Runs on CPU by default. The plan's own benchmarking note applies here:
the GPU delegate has shown inconsistent speedups over CPU on recent
MediaPipe Python builds, so don't flip --delegate to GPU without timing
both on your machine first.
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import stage_path, MODELS_DIR                    # noqa: E402
from utils.io_json import load_json, save_json, fail_stage   # noqa: E402

STAGE_NAME = "tracker"

FACE_SAMPLE_RATE = 5             # Step 4: baseline face track every 5th frame
POSE_WRIST_LANDMARKS = (15, 16)  # BlazePose 33-point topology: 15=left wrist, 16=right wrist
HAND_WRIST_LANDMARK = 0          # hand's own wrist landmark — used only to match a hand to an arm
HAND_FINGERTIP_LANDMARK = 8      # index fingertip
EXTRAPOLATION_FRACTION = 0.375   # midpoint of the plan's 35-40% range
MIN_WRIST_VISIBILITY = 0.3       # ignore a pose wrist MediaPipe itself isn't confident about

FACE_MODEL = "blaze_face_short_range.tflite"
POSE_MODEL = "pose_landmarker_full.task"
HAND_MODEL = "hand_landmarker.task"


def _model_path(filename: str) -> Path:
    path = MODELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Missing model weight '{filename}' in {MODELS_DIR}. "
            f"Download it during Phase 0 setup before running tracker.py."
        )
    return path


def _build_detectors(delegate: str):
    """Lazy import + construct all three MediaPipe Tasks detectors.
    Imported lazily (like faster_whisper/openai in the Phase 2 stages) so
    --help and every other stage keep working without mediapipe installed."""
    import mediapipe as mp

    base_options_cls = mp.tasks.BaseOptions
    mp_delegate = (base_options_cls.Delegate.GPU if delegate.upper() == "GPU"
                   else base_options_cls.Delegate.CPU)

    face_detector = mp.tasks.vision.FaceDetector.create_from_options(
        mp.tasks.vision.FaceDetectorOptions(
            base_options=base_options_cls(model_asset_path=str(_model_path(FACE_MODEL)),
                                           delegate=mp_delegate),
            min_detection_confidence=0.5,
        )
    )
    pose_landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(
        mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_options_cls(model_asset_path=str(_model_path(POSE_MODEL)),
                                           delegate=mp_delegate),
            num_poses=1,
            min_pose_detection_confidence=0.5,
        )
    )
    hand_landmarker = mp.tasks.vision.HandLandmarker.create_from_options(
        mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options_cls(model_asset_path=str(_model_path(HAND_MODEL)),
                                           delegate=mp_delegate),
            num_hands=2,
            min_hand_detection_confidence=0.5,
        )
    )
    return mp, face_detector, pose_landmarker, hand_landmarker


def _object_frame_ranges(focus_timeline: dict, fps: float) -> list[tuple[int, int]]:
    """Convert focus_timeline's 'object' blocks (in seconds) into inclusive
    [start_frame, end_frame] ranges, sorted by start. Sorted + non-decreasing
    frame_idx during the main loop is what lets _advance_and_check use a
    single forward-moving pointer instead of re-scanning every block on
    every frame."""
    ranges = []
    for block in focus_timeline.get("blocks", []):
        if block.get("focus") != "object":
            continue
        start_f = max(0, int(round(block["start_time"] * fps)))
        end_f = max(start_f, int(round(block["end_time"] * fps)))
        ranges.append((start_f, end_f))
    return sorted(ranges)


def _advance_and_check(frame_idx: int, ranges: list[tuple[int, int]], pointer: int) -> tuple[bool, int]:
    """Two-pointer membership test: is frame_idx inside any range? Assumes
    frame_idx is non-decreasing across calls, which holds here since the
    video is scanned sequentially frame by frame."""
    while pointer < len(ranges) and ranges[pointer][1] < frame_idx:
        pointer += 1
    in_range = pointer < len(ranges) and ranges[pointer][0] <= frame_idx <= ranges[pointer][1]
    return in_range, pointer


def _largest_face_center(detection_result, width: int, height: int):
    """Pick the largest detected face by bounding-box area — the speaker
    talking to camera, not someone passing behind them — and return its
    pixel-space center. None if nothing was detected this frame."""
    if not detection_result.detections:
        return None
    best = max(detection_result.detections,
               key=lambda d: d.bounding_box.width * d.bounding_box.height)
    bb = best.bounding_box
    cx = bb.origin_x + bb.width / 2
    cy = bb.origin_y + bb.height / 2
    return [round(cx, 1), round(cy, 1)]


def _ray_box_exit(origin, direction, width: int, height: int):
    """Where does the ray from `origin` along unit `direction` exit the
    [0,width] x [0,height] frame? Standard slab clipping against the two
    axes; falls back to `origin` itself if direction is degenerate (a hand
    detected essentially on top of the wrist, no real gesture to read)."""
    ox, oy = origin
    dx, dy = direction
    t_candidates = []
    if dx > 1e-9:
        t_candidates.append((width - ox) / dx)
    elif dx < -1e-9:
        t_candidates.append((0 - ox) / dx)
    if dy > 1e-9:
        t_candidates.append((height - oy) / dy)
    elif dy < -1e-9:
        t_candidates.append((0 - oy) / dy)
    positive = [t for t in t_candidates if t > 0]
    if not positive:
        return list(origin)
    t = min(positive)
    return [ox + dx * t, oy + dy * t]


def _best_pointing_pair(pose_result, hand_result, width: int, height: int):
    """Match each detected hand to the nearer pose wrist (15 left / 16
    right) by pixel distance between the hand's own wrist landmark (0) and
    each pose wrist, then keep whichever matched pair has the greatest
    wrist->fingertip extension. A fully extended arm is the clearest signal
    of an intentional point, as opposed to e.g. a resting hand caught
    mid-frame during a 'speaker' moment that happened to land inside a
    debounced 'object' block's edges.

    Returns (wrist_px, fingertip_px, extrapolated_target_px), each a
    [x, y] list, or (None, None, None) if no usable pair was found.
    """
    if not pose_result.pose_landmarks or not hand_result.hand_landmarks:
        return None, None, None

    pose = pose_result.pose_landmarks[0]
    pose_wrists_px = {}
    for idx in POSE_WRIST_LANDMARKS:
        lm = pose[idx]
        visibility = getattr(lm, "visibility", None)
        if visibility is None or visibility > MIN_WRIST_VISIBILITY:
            pose_wrists_px[idx] = (lm.x * width, lm.y * height)
    if not pose_wrists_px:
        return None, None, None

    best_pair = None
    best_extension = -1.0
    for hand in hand_result.hand_landmarks:
        hand_wrist_px = (hand[HAND_WRIST_LANDMARK].x * width, hand[HAND_WRIST_LANDMARK].y * height)
        fingertip_px = (hand[HAND_FINGERTIP_LANDMARK].x * width, hand[HAND_FINGERTIP_LANDMARK].y * height)

        matched_idx = min(pose_wrists_px, key=lambda i: math.dist(pose_wrists_px[i], hand_wrist_px))
        wrist_px = pose_wrists_px[matched_idx]
        extension = math.dist(wrist_px, fingertip_px)
        if extension > best_extension:
            best_extension = extension
            best_pair = (wrist_px, fingertip_px)

    if best_pair is None or best_extension < 1e-6:
        return None, None, None

    wrist_px, fingertip_px = best_pair
    dx, dy = fingertip_px[0] - wrist_px[0], fingertip_px[1] - wrist_px[1]
    norm = math.hypot(dx, dy)
    direction = (dx / norm, dy / norm)

    edge_point = _ray_box_exit(fingertip_px, direction, width, height)
    target = [
        fingertip_px[0] + EXTRAPOLATION_FRACTION * (edge_point[0] - fingertip_px[0]),
        fingertip_px[1] + EXTRAPOLATION_FRACTION * (edge_point[1] - fingertip_px[1]),
    ]
    target = [
        round(min(max(target[0], 0.0), width), 1),
        round(min(max(target[1], 0.0), height), 1),
    ]
    return (
        [round(wrist_px[0], 1), round(wrist_px[1], 1)],
        [round(fingertip_px[0], 1), round(fingertip_px[1], 1)],
        target,
    )


def run(video_path: Path, focus_timeline: dict, delegate: str = "CPU",
        face_sample_rate: int = FACE_SAMPLE_RATE) -> dict:
    """
    Returns:
        {"meta": {"width": 1920, "height": 1080, "fps": 30.0, "frame_count": 900},
         "frames": [
            {"t": 4.13, "face_center": [960.0, 540.0],
             "wrist": [820.3, 610.1], "fingertip": [750.2, 590.4],
             "extrapolated_target": [412.7, 560.0]},
            ...
        ]}

    Each entry only carries the fields actually computed on that frame:
    face_center is non-null only on face-sample frames (every Nth), and
    wrist/fingertip/extrapolated_target are non-null only on frames inside
    an 'object' block where a pointing pair was actually found. Frames
    where nothing was computed are omitted entirely — smooth_coords.py
    (Phase 4) interpolates/holds across the gaps, same as it already has
    to for the sparse face samples.
    """
    import cv2  # lazy import, same reasoning as mediapipe above

    if not video_path.exists():
        raise FileNotFoundError(f"No video at {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    object_ranges = _object_frame_ranges(focus_timeline, fps)
    range_pointer = 0

    mp, face_detector, pose_landmarker, hand_landmarker = _build_detectors(delegate)

    frames_out = []
    frame_idx = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            needs_face = (frame_idx % face_sample_rate == 0)
            in_object_block, range_pointer = _advance_and_check(frame_idx, object_ranges, range_pointer)

            if needs_face or in_object_block:
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                entry = {
                    "t": round(frame_idx / fps, 3),
                    "face_center": None,
                    "wrist": None,
                    "fingertip": None,
                    "extrapolated_target": None,
                }

                if needs_face:
                    face_result = face_detector.detect(mp_image)
                    entry["face_center"] = _largest_face_center(face_result, width, height)

                if in_object_block:
                    pose_result = pose_landmarker.detect(mp_image)
                    hand_result = hand_landmarker.detect(mp_image)
                    wrist, fingertip, target = _best_pointing_pair(pose_result, hand_result, width, height)
                    entry["wrist"] = wrist
                    entry["fingertip"] = fingertip
                    entry["extrapolated_target"] = target

                if any(v is not None for k, v in entry.items() if k != "t"):
                    frames_out.append(entry)

            frame_idx += 1
    finally:
        cap.release()
        face_detector.close()
        pose_landmarker.close()
        hand_landmarker.close()

    return {
        "meta": {"width": width, "height": height, "fps": fps, "frame_count": frame_count},
        "frames": frames_out,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=stage_path("video.mp4"))
    parser.add_argument("--focus-timeline", type=Path, default=stage_path("focus_timeline.json"))
    parser.add_argument("--out", type=Path, default=stage_path("raw_coords.json"))
    parser.add_argument("--delegate", default="CPU", choices=["CPU", "GPU"],
                         help="Benchmark both before trusting GPU — see module docstring.")
    parser.add_argument("--face-sample-rate", type=int, default=FACE_SAMPLE_RATE,
                         help="Run the baseline FaceDetector every Nth frame (Phase 3 Step 4).")
    args = parser.parse_args()

    try:
        focus_timeline = load_json(args.focus_timeline)
        raw_coords = run(args.video, focus_timeline, args.delegate, args.face_sample_rate)
        save_json(args.out, raw_coords)
        n_face = sum(1 for f in raw_coords["frames"] if f["face_center"] is not None)
        n_point = sum(1 for f in raw_coords["frames"] if f["extrapolated_target"] is not None)
        print(f"[{STAGE_NAME}] wrote {args.out} "
              f"({len(raw_coords['frames'])} sampled frames: "
              f"{n_face} face, {n_point} pointing-vector)")
    except Exception as e:
        fail_stage(STAGE_NAME, e)


if __name__ == "__main__":
    main()

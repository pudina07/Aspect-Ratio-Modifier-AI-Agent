"""
pipeline/tracker.py — Phase 1 & 3: Vision Tracking Stub & Contracts

Contract:
  Input:  video.mp4, focus_timeline.json
  Output: raw_coords.json

Filled in during Phase 3, Steps 4-5:
  - Step 4: baseline FaceDetector every 5th frame, always runs, fallback crop center.
  - Step 5: during 'object' blocks (from focus_timeline.json), run PoseLandmarker
    (wrist, landmark 15/16) + HandLandmarker (index fingertip, landmark 8),
    extrapolate 35-40% past fingertip along wrist->fingertip vector.

Supports --mock flag to generate valid schema-compliant mock coords for testing Phase 1/2.
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import stage_path                                     # noqa: E402
from contracts import (                                           # noqa: E402
    RawCoordsData, RawFrameCoord,
    validate_focus_timeline, validate_raw_coords
)
from utils.io_json import load_json, save_json, fail_stage        # noqa: E402

STAGE_NAME = "tracker"


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

        # Check if t falls inside any object block from timeline
        for b in focus_timeline.get("blocks", []):
            if b.get("focus") == "object" and b.get("start", 0) <= t <= b.get("end", 0):
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
    return data.to_dict()


def run(video_path: Path, focus_timeline: dict, mock: bool = False) -> dict:
    if mock:
        return generate_mock_raw_coords(video_path, focus_timeline)
    raise NotImplementedError(
        "Phase 3, Step 4-5: Face detection & pose/hand pointing vector extrapolation. (Use --mock for testing)"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=stage_path("video.mp4"))
    parser.add_argument("--focus-timeline", type=Path, default=stage_path("focus_timeline.json"))
    parser.add_argument("--out", type=Path, default=stage_path("raw_coords.json"))
    parser.add_argument("--mock", action="store_true", help="Run in mock mode with synthetic data for testing")
    args = parser.parse_args()

    try:
        focus_timeline = load_json(args.focus_timeline, validator=validate_focus_timeline)
        raw_coords = run(args.video, focus_timeline, mock=args.mock)
        save_json(args.out, raw_coords, validator=validate_raw_coords)
        print(f"[{STAGE_NAME}] wrote {args.out}")
    except Exception as e:
        fail_stage(STAGE_NAME, e)


if __name__ == "__main__":
    main()

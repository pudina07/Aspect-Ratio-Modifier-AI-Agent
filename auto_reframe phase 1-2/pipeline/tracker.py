"""
pipeline/tracker.py

Contract:  video.mp4 + focus_timeline.json  ->  raw_coords.json

Filled in during Phase 3, Steps 4-5:
  - Step 4: baseline FaceDetector every 5th frame, always runs, gives a
    fallback crop center for any stretch the LLM didn't flag.
  - Step 5: during 'object' blocks (from focus_timeline.json), run
    PoseLandmarker (wrist, landmark 15/16) + HandLandmarker (index
    fingertip, landmark 8), extrapolate ~35-40% past the fingertip along
    the wrist->fingertip vector to estimate the pointed-at target.
focus_timeline.json is a dependency (not just video.mp4) specifically so
this stage knows *when* to spend the extra compute on pose+hand instead
of running it on every frame.
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import stage_path                      # noqa: E402
from utils.io_json import load_json, save_json, fail_stage  # noqa: E402

STAGE_NAME = "tracker"


def run(video_path: Path, focus_timeline: dict) -> dict:
    """
    Returns something shaped like:
        {"frames": [
            {"t": 4.13, "face_center": [x, y],
             "wrist": [x, y], "fingertip": [x, y],
             "extrapolated_target": [x, y]},
            ...
        ]}
    """
    raise NotImplementedError(
        "Phase 3, Step 4: baseline face track every 5th frame. "
        "Phase 3, Step 5: pose+hand pointing vector during 'object' blocks."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=stage_path("video.mp4"))
    parser.add_argument("--focus-timeline", type=Path, default=stage_path("focus_timeline.json"))
    parser.add_argument("--out", type=Path, default=stage_path("raw_coords.json"))
    args = parser.parse_args()

    try:
        focus_timeline = load_json(args.focus_timeline)
        raw_coords = run(args.video, focus_timeline)
        save_json(args.out, raw_coords)
        print(f"[{STAGE_NAME}] wrote {args.out}")
    except Exception as e:
        fail_stage(STAGE_NAME, e)


if __name__ == "__main__":
    main()

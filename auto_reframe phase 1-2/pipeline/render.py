"""
pipeline/render.py — Phase 1 & 5: Video Rendering Stub & Contracts

Contract:
  Inputs:  video.mp4, final_coords_916.json, final_coords_11.json
  Outputs: output_916.mp4, output_11.mp4

Filled in during Phase 5, Step 11:
  - For each frame, build blurred full-bleed background at target aspect ratio.
  - Crop region per final_coords track, overlay centered.
  - Render with cv2.VideoWriter, then mux original audio via FFmpeg (-c:v copy -c:a aac).

Supports --mock flag to generate valid test MP4 deliverables for testing.
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import stage_path                             # noqa: E402
from contracts import validate_final_coords                # noqa: E402
from utils.io_json import load_json, fail_stage            # noqa: E402

STAGE_NAME = "render"


def generate_mock_rendered_videos(
    video_path: Path, coords_916: dict, coords_11: dict, out_916: Path, out_11: Path
) -> None:
    """Generate lightweight valid test video files for testing."""
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 30.0
    if fps <= 0:
        fps = 30.0
    cap.release()

    total_frames = 30
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Render 9:16 mock video (608x1080)
    out_916.parent.mkdir(parents=True, exist_ok=True)
    writer_916 = cv2.VideoWriter(str(out_916), fourcc, fps, (608, 1080))
    for i in range(total_frames):
        frame = np.zeros((1080, 608, 3), dtype=np.uint8)
        frame[:, :] = (40, 20, 20)
        cv2.putText(frame, "MOCK 9:16 REFRAME", (50, 540), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 200), 2)
        cv2.putText(frame, f"Frame {i}/{total_frames}", (50, 600), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        writer_916.write(frame)
    writer_916.release()

    # Render 1:1 mock video (1080x1080)
    out_11.parent.mkdir(parents=True, exist_ok=True)
    writer_11 = cv2.VideoWriter(str(out_11), fourcc, fps, (1080, 1080))
    for i in range(total_frames):
        frame = np.zeros((1080, 1080, 3), dtype=np.uint8)
        frame[:, :] = (20, 40, 20)
        cv2.putText(frame, "MOCK 1:1 REFRAME", (200, 540), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 2)
        cv2.putText(frame, f"Frame {i}/{total_frames}", (200, 600), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        writer_11.write(frame)
    writer_11.release()


def run(
    video_path: Path,
    coords_916: dict,
    coords_11: dict,
    out_916: Path,
    out_11: Path,
    mock: bool = False,
) -> None:
    if mock:
        generate_mock_rendered_videos(video_path, coords_916, coords_11, out_916, out_11)
        return
    raise NotImplementedError(
        "Phase 5, Step 11: per-frame crop + blurred-background composite via OpenCV & FFmpeg audio mux. (Use --mock for testing)"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=stage_path("video.mp4"))
    parser.add_argument("--coords-916", type=Path, default=stage_path("final_coords_916.json"))
    parser.add_argument("--coords-11", type=Path, default=stage_path("final_coords_11.json"))
    parser.add_argument("--out-916", type=Path, default=stage_path("output_916.mp4"))
    parser.add_argument("--out-11", type=Path, default=stage_path("output_11.mp4"))
    parser.add_argument("--mock", action="store_true", help="Run in mock mode with synthetic data for testing")
    args = parser.parse_args()

    try:
        coords_916 = load_json(args.coords_916, validator=validate_final_coords)
        coords_11 = load_json(args.coords_11, validator=validate_final_coords)
        run(args.video, coords_916, coords_11, args.out_916, args.out_11, mock=args.mock)
        print(f"[{STAGE_NAME}] wrote {args.out_916} and {args.out_11}")
    except Exception as e:
        fail_stage(STAGE_NAME, e)


if __name__ == "__main__":
    main()

"""
pipeline/render.py

Contract:  video.mp4 + final_coords_916.json + final_coords_11.json
           ->  output_916.mp4 + output_11.mp4

Filled in during Phase 5, Step 11: for each frame, build a blurred
full-bleed background at the target aspect ratio, crop per the coord
track, overlay it centered, optionally draw the safe-zone rectangle for
QA. Written with cv2.VideoWriter (OpenCV as primary path, not fallback),
then mux the original audio back in with
`ffmpeg -c:v copy -c:a aac`.

The only stage whose outputs aren't JSON — kept in the same try/except/
fail_stage pattern as every other stage anyway, so a render crash is
reported the same way the rest of the pipeline reports failures.
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import stage_path            # noqa: E402
from utils.io_json import load_json, fail_stage  # noqa: E402

STAGE_NAME = "render"


def run(video_path: Path, coords_916: dict, coords_11: dict,
        out_916: Path, out_11: Path) -> None:
    raise NotImplementedError(
        "Phase 5, Step 11: per-frame crop + blurred-background composite "
        "via OpenCV, then FFmpeg audio mux."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=stage_path("video.mp4"))
    parser.add_argument("--coords-916", type=Path, default=stage_path("final_coords_916.json"))
    parser.add_argument("--coords-11", type=Path, default=stage_path("final_coords_11.json"))
    parser.add_argument("--out-916", type=Path, default=stage_path("output_916.mp4"))
    parser.add_argument("--out-11", type=Path, default=stage_path("output_11.mp4"))
    args = parser.parse_args()

    try:
        coords_916 = load_json(args.coords_916)
        coords_11 = load_json(args.coords_11)
        run(args.video, coords_916, coords_11, args.out_916, args.out_11)
        print(f"[{STAGE_NAME}] wrote {args.out_916} and {args.out_11}")
    except Exception as e:
        fail_stage(STAGE_NAME, e)


if __name__ == "__main__":
    main()

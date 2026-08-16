"""
pipeline/ocr_pass.py

Contract:  video.mp4  ->  text_regions.json

Filled in during Phase 3, Step 6: EasyOCR over every 8th frame, tag each
frame range with any detected text bounding box as a protected_region
(x, y, w, h) so smooth_coords.py knows not to crop it out.

Note: this stage only needs video.mp4 — it does NOT depend on
transcript.json or focus_timeline.json. That's deliberate: it means
pipeline_runner.py can run this concurrently with the
transcribe -> analyze_script -> tracker chain instead of waiting behind it.
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import stage_path               # noqa: E402
from utils.io_json import save_json, fail_stage  # noqa: E402

STAGE_NAME = "ocr_pass"


def run(video_path: Path) -> dict:
    """
    Returns something shaped like:
        {"regions": [
            {"t_start": 2.0, "t_end": 5.5, "x": 40, "y": 800, "w": 600, "h": 90},
            ...
        ]}
    """
    raise NotImplementedError(
        "Phase 3, Step 6: EasyOCR every 8th frame -> protected_region boxes."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=stage_path("video.mp4"))
    parser.add_argument("--out", type=Path, default=stage_path("text_regions.json"))
    args = parser.parse_args()

    try:
        text_regions = run(args.video)
        save_json(args.out, text_regions)
        print(f"[{STAGE_NAME}] wrote {args.out}")
    except Exception as e:
        fail_stage(STAGE_NAME, e)


if __name__ == "__main__":
    main()

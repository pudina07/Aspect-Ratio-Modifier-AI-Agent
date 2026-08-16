"""
pipeline/smooth_coords.py — Phase 1: Architecture & Contract Stub

Contract:
  Inputs:  raw_coords.json, text_regions.json, focus_timeline.json
  Outputs: final_coords_916.json, final_coords_11.json

Filled in during Phase 4, Steps 7-10:
  - Step 7:  Dual-aspect crop windowing (9:16 = 608x1080, 1:1 = 1080x1080 with face priority).
  - Step 8:  One Euro Filter per-axis (lower beta for speaker, higher beta for transitions).
  - Step 9:  Protected-region clamp against text_regions.json.
  - Step 10: Eased transitions over ~15 frames.

Supports --mock flag to generate valid schema-compliant mock coords for testing Phase 1.
"""
import argparse
import sys
from pathlib import Path
from typing import Tuple

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import stage_path                                     # noqa: E402
from contracts import (                                           # noqa: E402
    FinalCoordsData, FinalFrameCoord,
    validate_raw_coords, validate_text_regions, validate_focus_timeline, validate_final_coords
)
from utils.io_json import load_json, save_json, fail_stage        # noqa: E402

STAGE_NAME = "smooth_coords"


def generate_mock_final_coords(
    raw_coords: dict, text_regions: dict, focus_timeline: dict
) -> Tuple[dict, dict]:
    """Generate mock 9:16 and 1:1 smoothed coordinates conforming to FinalCoordsData contract."""
    fps = raw_coords.get("fps", 30.0)
    total_frames = raw_coords.get("total_frames", 311)
    src_w = raw_coords.get("width", 1920)
    src_h = raw_coords.get("height", 1080)

    # 9:16 Crop: 608x1080
    crop_w_916 = 608
    crop_h_916 = 1080

    # 1:1 Crop: 1080x1080
    crop_w_11 = 1080
    crop_h_11 = 1080

    frames_916 = []
    frames_11 = []

    for i in range(total_frames):
        t = i / fps
        raw_f = raw_coords.get("frames", [])[i] if i < len(raw_coords.get("frames", [])) else {}
        focus = raw_f.get("focus", "speaker")

        # In segment 2 (3.5s - 6.8s), crop shifts right towards the pointed chart
        if 3.5 <= t <= 6.8:
            crop_x_916 = 1312  # Clamped to right edge (1920 - 608)
            crop_x_11 = 840   # Centered on chart/presenter right area
        else:
            crop_x_916 = (src_w - crop_w_916) // 2  # 656 (Center presenter)
            crop_x_11 = (src_w - crop_w_11) // 2    # 420 (Center presenter)

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


def run(raw_coords: dict, text_regions: dict, focus_timeline: dict, mock: bool = False) -> Tuple[dict, dict]:
    if mock:
        return generate_mock_final_coords(raw_coords, text_regions, focus_timeline)
    raise NotImplementedError(
        "Phase 4, Steps 7-10: One Euro smoothing, dual-aspect crop selection & text clamping. (Use --mock for architecture testing)"
    )


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
        print(f"[{STAGE_NAME}] wrote {args.out_916} and {args.out_11}")
    except Exception as e:
        fail_stage(STAGE_NAME, e)


if __name__ == "__main__":
    main()

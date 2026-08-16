"""
pipeline/smooth_coords.py

Contract:  raw_coords.json + text_regions.json + focus_timeline.json
           ->  final_coords_916.json + final_coords_11.json

This is where the two deliverables (9:16 and 1:1) diverge from one
shared source of truth. Filled in during Phase 4, Steps 7-10:
  - Step 7:  compute the 9:16 and 1:1 crop windows separately (1:1 has a
             face-priority fallback rule when object + face can't both fit).
  - Step 8:  One Euro Filter per-axis on both tracks (lower beta on
             'speaker' blocks, higher beta on 'object' transitions).
  - Step 9:  protected-region clamp against text_regions.json.
  - Step 10: ease-in-out transition interpolation over ~15 frames.
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import stage_path                      # noqa: E402
from utils.io_json import load_json, save_json, fail_stage  # noqa: E402

STAGE_NAME = "smooth_coords"


def run(raw_coords: dict, text_regions: dict, focus_timeline: dict) -> tuple[dict, dict]:
    """
    Returns (coords_916, coords_11), each shaped like:
        {"frames": [{"t": 4.13, "crop_x": 210, "crop_y": 0}, ...]}
    """
    raise NotImplementedError(
        "Phase 4, Steps 7-10: dual-aspect crop computation, One Euro "
        "smoothing, protected-region clamp, eased transitions."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-coords", type=Path, default=stage_path("raw_coords.json"))
    parser.add_argument("--text-regions", type=Path, default=stage_path("text_regions.json"))
    parser.add_argument("--focus-timeline", type=Path, default=stage_path("focus_timeline.json"))
    parser.add_argument("--out-916", type=Path, default=stage_path("final_coords_916.json"))
    parser.add_argument("--out-11", type=Path, default=stage_path("final_coords_11.json"))
    args = parser.parse_args()

    try:
        raw_coords = load_json(args.raw_coords)
        text_regions = load_json(args.text_regions)
        focus_timeline = load_json(args.focus_timeline)

        coords_916, coords_11 = run(raw_coords, text_regions, focus_timeline)

        save_json(args.out_916, coords_916)
        save_json(args.out_11, coords_11)
        print(f"[{STAGE_NAME}] wrote {args.out_916} and {args.out_11}")
    except Exception as e:
        fail_stage(STAGE_NAME, e)


if __name__ == "__main__":
    main()

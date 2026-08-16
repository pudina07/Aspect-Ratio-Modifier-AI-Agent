"""
pipeline/ocr_pass.py — Phase 1 & 3: OCR Text Regions Stub & Contracts

Contract:
  Input:  video.mp4
  Output: text_regions.json

Filled in during Phase 3, Step 6: EasyOCR sampled every 8th frame, detects
text bounding boxes and tags time ranges with protected_region (x, y, w, h).

Deliberately independent of transcript.json / focus_timeline.json so it
runs concurrently with the transcript branch in pipeline_runner.py.

Supports --mock flag to generate valid schema-compliant mock text regions for testing.
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import stage_path                               # noqa: E402
from contracts import TextRegionsData, TextRegion, validate_text_regions  # noqa: E402
from utils.io_json import save_json, fail_stage             # noqa: E402

STAGE_NAME = "ocr_pass"


def generate_mock_text_regions(video_path: Path) -> dict:
    """Generate mock text regions conforming to TextRegionsData contract."""
    regions = [
        TextRegion(
            t_start=6.8,
            t_end=10.37,
            box=[1380.0, 70.0, 500.0, 60.0],
            text="CRITICAL: 95% RETENTION",
            confidence=0.97
        ),
        TextRegion(
            t_start=6.8,
            t_end=10.37,
            box=[40.0, 960.0, 540.0, 50.0],
            text="KEY TAKEAWAY: SUBSCRIBE NOW",
            confidence=0.96
        )
    ]
    data = TextRegionsData(
        fps=30.0,
        width=1920,
        height=1080,
        regions=regions
    )
    return data.to_dict()


def run(video_path: Path, mock: bool = False) -> dict:
    if mock:
        return generate_mock_text_regions(video_path)
    raise NotImplementedError(
        "Phase 3, Step 6: EasyOCR every 8th frame -> protected text regions. (Use --mock for testing)"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=stage_path("video.mp4"))
    parser.add_argument("--out", type=Path, default=stage_path("text_regions.json"))
    parser.add_argument("--mock", action="store_true", help="Run in mock mode with synthetic data for testing")
    args = parser.parse_args()

    try:
        text_regions = run(args.video, mock=args.mock)
        save_json(args.out, text_regions, validator=validate_text_regions)
        print(f"[{STAGE_NAME}] wrote {args.out}")
    except Exception as e:
        fail_stage(STAGE_NAME, e)


if __name__ == "__main__":
    main()

"""
app.py — Phase 1 & 2: Pipeline CLI Driver

Stages input video into data/, orchestrates the pipeline execution,
and verifies intermediate artifacts and outputs.
"""
import argparse
import shutil
import sys
from pathlib import Path

from config import stage_path, DATA_DIR
from pipeline_runner import run_pipeline


def main(video_path: Path, mock: bool = False) -> None:
    dest = stage_path("video.mp4")
    if not video_path.exists():
        print(f"Error: Input video not found at {video_path}")
        sys.exit(1)

    shutil.copy(video_path, dest)
    print(f"Input video staged at {dest}\n")

    results = run_pipeline(mock=mock)
    failed = [r for r in results if not r.ok]

    if failed:
        print(f"\nPipeline stopped — '{failed[0].name}' failed:")
        if failed[0].stderr:
            print(failed[0].stderr)
        elif failed[0].skipped_reason:
            print(failed[0].skipped_reason)
        sys.exit(1)

    print("\n==========================================")
    print("?? Pipeline Execution Complete:")
    print(f"  Transcript:     {stage_path('transcript.json')}")
    print(f"  Focus Timeline: {stage_path('focus_timeline.json')}")
    print(f"  9:16 Video:     {stage_path('output_916.mp4')}")
    print(f"  1:1 Video:      {stage_path('output_11.mp4')}")
    print("==========================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Path to input video MP4 file")
    parser.add_argument("--mock", action="store_true", help="Run stages with mock data for testing")
    args = parser.parse_args()

    main(args.video, mock=args.mock)

"""
app.py — Phase 1: Architecture

This is the architectural skeleton, NOT the Phase 6 Streamlit UI. Its
only job right now is to prove config.py's stage graph and
pipeline_runner.py's scheduler wire together end to end, via a plain
CLI: stage a video into data/, run the pipeline, report where the
outputs landed.

Phase 6 replaces main()'s body with real Streamlit widgets (file
uploader, platform multi-select, st.spinner per stage, st.video
side-by-side) — run_pipeline() itself doesn't change.
"""
import shutil
import sys

from config import stage_path
from pipeline_runner import run_pipeline


def main(video_path: str) -> None:
    dest = stage_path("video.mp4")
    shutil.copy(video_path, dest)
    print(f"Input video staged at {dest}\n")

    results = run_pipeline()
    failed = [r for r in results if not r.ok]

    if failed:
        print(f"\nPipeline stopped — '{failed[0].name}' failed:\n{failed[0].stderr}")
        sys.exit(1)

    print("\nPipeline complete:")
    print(f"  9:16 -> {stage_path('output_916.mp4')}")
    print(f"  1:1  -> {stage_path('output_11.mp4')}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python app.py <path-to-video.mp4>")
        sys.exit(1)
    main(sys.argv[1])

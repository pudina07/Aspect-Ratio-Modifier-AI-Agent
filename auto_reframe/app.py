"""
app.py — Phase 1: Architecture & CLI Orchestration Harness

Proves the architecture, DAG scheduling, and data contract flow end-to-end.
Stages a video into data/, executes the pipeline runner, validates outputs,
and reports execution metrics.

Phase 6 replaces this CLI interface with a Streamlit web UI.
"""
import argparse
import shutil
import sys
from pathlib import Path

# Force UTF-8 encoding on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from config import stage_path, DATA_DIR, BASE_DIR
from pipeline_runner import run_pipeline, clean_run_artifacts


def main():
    parser = argparse.ArgumentParser(description="Context-Aware Auto-Reframe (Phase 1 CLI Harness)")
    parser.add_argument("video", type=Path, help="Path to input 16:9 MP4 video")
    parser.add_argument("--mock", action="store_true", help="Run with mock data to test DAG and contract architecture")
    parser.add_argument("--no-clean", action="store_true", help="Do not clean data directory before starting")
    args = parser.parse_args()

    if not args.video.exists():
        print(f"Error: Input video '{args.video}' does not exist.", file=sys.stderr)
        sys.exit(1)

    dest = stage_path("video.mp4")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not args.no_clean:
        clean_run_artifacts(DATA_DIR, keep_video=False)

    shutil.copy(str(args.video), str(dest))
    print(f"============================================================")
    print(f"  CONTEXT-AWARE AUTO-REFRAME: PIPELINE EXECUTION")
    print(f"============================================================")
    print(f"Input video staged: {dest}")
    print(f"Mode: {'MOCK (Architecture Test)' if args.mock else 'PRODUCTION (ML Inference)'}\n")

    report = run_pipeline(
        data_dir=DATA_DIR,
        use_mock=args.mock,
        clean_workspace=False,  # Already cleaned above
    )

    print("\n" + report.summary())

    if not report.ok:
        sys.exit(1)

    out_916 = stage_path("output_916.mp4")
    out_11 = stage_path("output_11.mp4")

    print("\nDeliverables generated successfully:")
    print(f"  9:16 Video -> {out_916} ({'Exists' if out_916.exists() else 'Missing'})")
    print(f"  1:1  Video -> {out_11} ({'Exists' if out_11.exists() else 'Missing'})")
    print(f"============================================================\n")


if __name__ == "__main__":
    main()

"""
app.py — Phases 1 to 5: Context-Aware Auto-Reframe CLI Harness

Orchestrates the entire pipeline end-to-end:
1. Stages video into data/ directory.
2. Runs DAG pipeline governor (speech STT -> NLP timeline -> vision tracker & OCR -> smoothing -> rendering).
3. Produces final deliverables: output_916.mp4 (1080x1920) and output_11.mp4 (1080x1080).
4. Emits structured execution metrics and deliverable validation.
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
    parser = argparse.ArgumentParser(description="Context-Aware Auto-Reframe (Phases 1-5 CLI Harness)")
    parser.add_argument("video", type=Path, help="Path to input 16:9 MP4 video")
    parser.add_argument("--mock", action="store_true", help="Run with mock data for fast architectural testing")
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
    print("=" * 60)
    print("  CONTEXT-AWARE AUTO-REFRAME: PIPELINE EXECUTION (PHASES 1-5)")
    print("=" * 60)
    print(f"Input video staged : {dest}")
    print(f"Execution Mode     : {'MOCK (Architecture Test)' if args.mock else 'PRODUCTION (ML Inference)'}\n")

    report = run_pipeline(
        data_dir=DATA_DIR,
        mock=args.mock,
        clean_workspace=False
    )

    print("\n" + report.summary())

    if not report.ok:
        sys.exit(1)

    out_916 = stage_path("output_916.mp4")
    out_11 = stage_path("output_11.mp4")

    print("\nDeliverables generated successfully:")
    print(f"  9:16 Video (1080x1920) -> {out_916} ({'Ready' if out_916.exists() else 'Missing'})")
    print(f"  1:1  Video (1080x1080) -> {out_11} ({'Ready' if out_11.exists() else 'Missing'})")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

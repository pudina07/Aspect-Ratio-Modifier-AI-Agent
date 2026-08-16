"""
pipeline/transcribe.py — Phase 1: Architecture & Contract Stub

Contract:
  Input:  video.mp4 (or extracted audio)
  Output: transcript.json

Filled in during Phase 2, Step 1: run faster-whisper (distil-large-v3 / base) locally,
extract audio with FFmpeg first, and produce word-level timestamps.

Supports --mock flag to generate valid schema-compliant mock transcript for testing Phase 1.
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import stage_path                               # noqa: E402
from contracts import TranscriptData, WordTiming, validate_transcript  # noqa: E402
from utils.io_json import save_json, fail_stage             # noqa: E402

STAGE_NAME = "transcribe"


def generate_mock_transcript(video_path: Path) -> dict:
    """Generate mock transcript conforming to TranscriptData contract."""
    words = [
        WordTiming("Hello", 0.0, 0.4, 0.98),
        WordTiming("and", 0.4, 0.6, 0.99),
        WordTiming("welcome", 0.6, 1.1, 0.97),
        WordTiming("to", 1.1, 1.3, 0.99),
        WordTiming("the", 1.3, 1.5, 0.99),
        WordTiming("auto-reframe", 1.5, 2.2, 0.95),
        WordTiming("demonstration.", 2.2, 2.9, 0.96),
        WordTiming("Look", 3.6, 4.0, 0.99),
        WordTiming("at", 4.0, 4.2, 0.99),
        WordTiming("this", 4.2, 4.5, 0.98),
        WordTiming("chart", 4.5, 4.9, 0.97),
        WordTiming("on", 4.9, 5.1, 0.99),
        WordTiming("the", 5.1, 5.3, 0.99),
        WordTiming("right", 5.3, 5.6, 0.98),
        WordTiming("side", 5.6, 5.8, 0.99),
        WordTiming("over", 5.8, 6.0, 0.98),
        WordTiming("here.", 6.0, 6.3, 0.99),
        WordTiming("Notice", 6.9, 7.3, 0.98),
        WordTiming("the", 7.3, 7.5, 0.99),
        WordTiming("key", 7.5, 7.8, 0.97),
        WordTiming("metric", 7.8, 8.2, 0.96),
        WordTiming("in", 8.2, 8.4, 0.99),
        WordTiming("the", 8.4, 8.6, 0.99),
        WordTiming("corner", 8.6, 9.0, 0.98),
        WordTiming("of", 9.0, 9.2, 0.99),
        WordTiming("your", 9.2, 9.4, 0.98),
        WordTiming("screen.", 9.4, 9.8, 0.99),
    ]
    data = TranscriptData(
        words=words,
        text="Hello and welcome to the auto-reframe demonstration. Look at this chart on the right side over here. Notice the key metric in the corner of your screen.",
        language="en",
        duration=10.37
    )
    return data.to_dict()


def run(video_path: Path, mock: bool = False) -> dict:
    if mock:
        return generate_mock_transcript(video_path)
    raise NotImplementedError(
        "Phase 2, Step 1: call faster-whisper here and return word-level timestamps. (Use --mock for architecture testing)"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=stage_path("video.mp4"))
    parser.add_argument("--out", type=Path, default=stage_path("transcript.json"))
    parser.add_argument("--mock", action="store_true", help="Run in mock mode with synthetic data for testing")
    args = parser.parse_args()

    try:
        transcript = run(args.video, mock=args.mock)
        save_json(args.out, transcript, validator=validate_transcript)
        print(f"[{STAGE_NAME}] wrote {args.out}")
    except Exception as e:
        fail_stage(STAGE_NAME, e)


if __name__ == "__main__":
    main()

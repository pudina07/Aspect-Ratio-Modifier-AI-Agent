"""
pipeline/analyze_script.py — Phase 1: Architecture & Contract Stub

Contract:
  Input:  transcript.json
  Output: focus_timeline.json

Filled in during Phase 2, Steps 2-3: send transcript to LLM (structured JSON mode)
to identify timestamps where speaker points/references an object, then debounce
focus blocks under 1s apart.

Supports --mock flag to generate valid schema-compliant mock timeline for testing Phase 1.
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import stage_path                                     # noqa: E402
from contracts import (                                           # noqa: E402
    FocusTimelineData, FocusBlock,
    validate_transcript, validate_focus_timeline
)
from utils.io_json import load_json, save_json, fail_stage        # noqa: E402

STAGE_NAME = "analyze_script"


def generate_mock_focus_timeline(transcript: dict) -> dict:
    """Generate mock focus timeline conforming to FocusTimelineData contract."""
    blocks = [
        FocusBlock(start=0.0, end=3.5, focus="speaker", direction_hint="center", confidence=0.98),
        FocusBlock(start=3.5, end=6.8, focus="object", direction_hint="right", confidence=0.95),
        FocusBlock(start=6.8, end=10.37, focus="speaker", direction_hint="center", confidence=0.92),
    ]
    data = FocusTimelineData(
        blocks=blocks,
        metadata={"total_duration": transcript.get("duration", 10.37), "model": "mock-llm-v1"}
    )
    return data.to_dict()


def run(transcript: dict, mock: bool = False) -> dict:
    if mock:
        return generate_mock_focus_timeline(transcript)
    raise NotImplementedError(
        "Phase 2, Step 2-3: LLM pointing/reference classification & debouncing. (Use --mock for architecture testing)"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript", type=Path, default=stage_path("transcript.json"))
    parser.add_argument("--out", type=Path, default=stage_path("focus_timeline.json"))
    parser.add_argument("--mock", action="store_true", help="Run in mock mode with synthetic data for testing")
    args = parser.parse_args()

    try:
        transcript = load_json(args.transcript, validator=validate_transcript)
        focus_timeline = run(transcript, mock=args.mock)
        save_json(args.out, focus_timeline, validator=validate_focus_timeline)
        print(f"[{STAGE_NAME}] wrote {args.out}")
    except Exception as e:
        fail_stage(STAGE_NAME, e)


if __name__ == "__main__":
    main()

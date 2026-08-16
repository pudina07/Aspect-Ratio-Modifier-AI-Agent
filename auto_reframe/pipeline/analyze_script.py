"""
pipeline/analyze_script.py — Phase 2, Steps 2 & 3: LLM & Semantic Focus Timeline Analysis

Contract:
  Input:  transcript.json
  Output: focus_timeline.json

Step 2: Identifies moments where the speaker verbally references or points at an object
        (e.g., "look at this", "here is the chart", "notice the metric on the right").
        - Supports live OpenAI LLM execution when OPENAI_API_KEY is available.
        - Provides intelligent semantic NLP heuristic fallback when offline / no API key.
        - Supports --mock flag for architectural test execution.

Step 3: Debounces focus blocks — merges contiguous blocks within 1.0s gap sharing the same
        direction and focus, and discards transient jitter blocks shorter than 0.3s.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import stage_path                                                      # noqa: E402
from contracts import (                                                            # noqa: E402
    FocusTimelineData, FocusBlock, normalize_focus_block,
    validate_transcript, validate_focus_timeline
)
from utils.io_json import load_json, save_json, fail_stage                         # noqa: E402

STAGE_NAME = "analyze_script"
DEFAULT_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = (
    "You analyze a video transcript with word-level timestamps to identify moments "
    "where the speaker verbally references or points at something (e.g. \"look at this\", "
    "\"here's the chart\", \"notice the metric on the right\").\n"
    "For each moment, return an object with:\n"
    "- start: float (start time in seconds)\n"
    "- end: float (end time in seconds)\n"
    "- focus: \"object\" (if drawing attention to an object/graphic/text) or \"speaker\"\n"
    "- direction_hint: \"left\", \"right\", \"center\", or \"unknown\"\n"
    "- confidence: float between 0.0 and 1.0\n\n"
    "Respond ONLY with a JSON object in this format:\n"
    "{\"blocks\": [{\"start\": 3.5, \"end\": 6.2, \"focus\": \"object\", \"direction_hint\": \"right\", \"confidence\": 0.95}]}"
)


def _transcript_to_compact_text(transcript: dict) -> str:
    """Format word timestamps into concise lines for LLM prompt."""
    lines = []
    words = transcript.get("words", [])
    for w in words:
        w_text = w.get("word", "").strip()
        s = w.get("start", 0.0)
        e = w.get("end", 0.0)
        lines.append(f"[{s:.2f}-{e:.2f}] {w_text}")
    return "\n".join(lines)


def _call_openai_llm(transcript_text: str, model_name: str = DEFAULT_MODEL) -> List[Dict[str, Any]]:
    """Invokes OpenAI Chat API with strict JSON mode."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment.")

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=model_name,
        response_format={"type": "json_object"},
        temperature=0.0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": transcript_text},
        ],
    )
    raw_content = response.choices[0].message.content or "{}"
    parsed = json.loads(raw_content)
    raw_blocks = parsed.get("blocks", [])
    return [normalize_focus_block(b) for b in raw_blocks]


def _extract_heuristic_focus_blocks(transcript: dict) -> List[Dict[str, Any]]:
    """
    Intelligent NLP semantic cue extractor for offline / keyless execution.
    Detects reference phrases and correlates them directly with word timestamp boundaries.
    """
    words = transcript.get("words", [])
    if not words:
        return []

    cues = [
        (re.compile(r"\b(look\s+at|look\s+here|see\s+this|notice\s+the|here\s+is|check\s+out|watch\s+this)\b", re.IGNORECASE), "object", 0.95),
        (re.compile(r"\b(chart|graph|metric|screen|slide|table|diagram|figure|corner)\b", re.IGNORECASE), "object", 0.90),
        (re.compile(r"\b(on\s+the\s+right|to\s+the\s+right|right\s+side)\b", re.IGNORECASE), "right", 0.95),
        (re.compile(r"\b(on\s+the\s+left|to\s+the\s+left|left\s+side)\b", re.IGNORECASE), "left", 0.95),
    ]

    total_duration = transcript.get("duration", 0.0)
    if not total_duration and words:
        total_duration = words[-1].get("end", 10.0)

    extracted_blocks: List[Dict[str, Any]] = []
    num_words = len(words)

    i = 0
    while i < num_words:
        window_end = min(i + 8, num_words)
        window_words = words[i:window_end]
        window_text = " ".join([w.get("word", "").strip() for w in window_words])

        is_object = False
        direction = "center"
        confidence = 0.85

        for pattern, kind, conf in cues:
            if pattern.search(window_text):
                if kind == "object":
                    is_object = True
                    confidence = max(confidence, conf)
                elif kind in ("right", "left"):
                    direction = kind
                    is_object = True
                    confidence = max(confidence, conf)

        if is_object:
            start_t = window_words[0].get("start", 0.0)
            end_t = window_words[-1].get("end", start_t + 2.0)
            start_t = max(0.0, start_t - 0.2)
            end_t = min(total_duration, end_t + 0.8)

            extracted_blocks.append({
                "start": round(start_t, 3),
                "end": round(end_t, 3),
                "focus": "object",
                "direction_hint": direction,
                "confidence": round(confidence, 3)
            })
            i = window_end
        else:
            i += 1

    return extracted_blocks


def debounce_timeline(blocks: List[Dict[str, Any]], merge_gap: float = 1.0, min_duration: float = 0.3) -> List[Dict[str, Any]]:
    """
    Step 3 Debouncing:
    - Merges blocks within merge_gap (default 1.0s) that share focus & direction_hint.
    - Discards blocks shorter than min_duration (default 0.3s).
    """
    if not blocks:
        return []

    normalized = [normalize_focus_block(b) for b in blocks if b.get("end", b.get("end_time", 0)) > b.get("start", b.get("start_time", 0))]
    if not normalized:
        return []

    ordered = sorted(normalized, key=lambda b: b["start"])
    merged: List[Dict[str, Any]] = [ordered[0]]

    for block in ordered[1:]:
        prev = merged[-1]
        same_kind = (block["focus"] == prev["focus"] and block["direction_hint"] == prev["direction_hint"])
        gap = block["start"] - prev["end"]

        if same_kind and gap <= merge_gap:
            prev["end"] = max(prev["end"], block["end"])
            prev["confidence"] = max(prev["confidence"], block["confidence"])
        else:
            merged.append(block)

    debounced = [b for b in merged if (b["end"] - b["start"]) >= min_duration]
    return debounced


def generate_mock_focus_timeline(transcript: dict) -> dict:
    """Generate deterministic mock focus timeline for testing."""
    duration = transcript.get("duration", 10.37)
    blocks = [
        FocusBlock(start=0.0, end=3.5, focus="speaker", direction_hint="center", confidence=0.98),
        FocusBlock(start=3.5, end=6.8, focus="object", direction_hint="right", confidence=0.95),
        FocusBlock(start=6.8, end=round(duration, 3), focus="speaker", direction_hint="center", confidence=0.92),
    ]
    data = FocusTimelineData(
        blocks=blocks,
        metadata={"total_duration": duration, "mode": "mock"}
    )
    return data.to_dict()


def run(transcript: dict, model_name: str = DEFAULT_MODEL, mock: bool = False) -> dict:
    if mock:
        return generate_mock_focus_timeline(transcript)

    words = transcript.get("words", [])
    if not words:
        return FocusTimelineData(blocks=[], metadata={"mode": "empty"}).to_dict()

    raw_blocks: List[Dict[str, Any]] = []
    mode_used = "nlp_heuristic"

    if os.environ.get("OPENAI_API_KEY"):
        try:
            print(f"[{STAGE_NAME}] Calling OpenAI LLM ({model_name}) with structured JSON output...")
            transcript_text = _transcript_to_compact_text(transcript)
            raw_blocks = _call_openai_llm(transcript_text, model_name)
            mode_used = f"openai_{model_name}"
            print(f"[{STAGE_NAME}] LLM detected {len(raw_blocks)} raw focus blocks.")
        except Exception as e:
            print(f"[{STAGE_NAME}] LLM call failed ({e}). Falling back to local semantic NLP heuristic...")
            raw_blocks = _extract_heuristic_focus_blocks(transcript)
    else:
        print(f"[{STAGE_NAME}] OPENAI_API_KEY not set. Running high-precision local semantic NLP cue analyzer...")
        raw_blocks = _extract_heuristic_focus_blocks(transcript)

    debounced_blocks = debounce_timeline(raw_blocks)
    print(f"[{STAGE_NAME}] Debounced into {len(debounced_blocks)} stable focus blocks.")

    block_objs = [
        FocusBlock(
            start=b["start"],
            end=b["end"],
            focus=b["focus"],
            direction_hint=b["direction_hint"],
            confidence=b["confidence"]
        )
        for b in debounced_blocks
    ]

    result = FocusTimelineData(
        blocks=block_objs,
        metadata={
            "mode": mode_used,
            "raw_block_count": len(raw_blocks),
            "debounced_block_count": len(debounced_blocks),
            "total_duration": transcript.get("duration", 0.0)
        }
    )
    return result.to_dict()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript", type=Path, default=stage_path("transcript.json"))
    parser.add_argument("--out", type=Path, default=stage_path("focus_timeline.json"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--mock", action="store_true", help="Generate mock focus timeline for testing")
    args = parser.parse_args()

    try:
        transcript = load_json(args.transcript, validator=validate_transcript)
        focus_timeline = run(transcript, model_name=args.model, mock=args.mock)
        save_json(args.out, focus_timeline, validator=validate_focus_timeline)
        count = len(focus_timeline.get("blocks", []))
        print(f"[{STAGE_NAME}] successfully wrote {args.out} ({count} focus blocks)")
    except Exception as e:
        fail_stage(STAGE_NAME, e)


if __name__ == "__main__":
    main()

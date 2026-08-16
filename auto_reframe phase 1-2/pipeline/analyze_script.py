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
    Detects reference phrases and correlates them directly with exact word timestamp boundaries.
    """
    words = transcript.get("words", [])
    if not words:
        return []

    # Direct deictic cues indicating verbal pointing or attention redirection
    cues = [
        (re.compile(r"\b(look\s+at|look\s+here|see\s+this|see\s+right\s+here|notice\s+the|notice\s+this|here\s+is|check\s+out|watch\s+this|pay\s+close\s+attention|pointing\s+to)\b", re.IGNORECASE), "object", 0.95),
    ]
    directions = [
        (re.compile(r"\b(on\s+the\s+right|to\s+the\s+right|right\s+side|right\s+here|on\s+my\s+right)\b", re.IGNORECASE), "right"),
        (re.compile(r"\b(on\s+the\s+left|to\s+the\s+left|left\s+side|left\s+here|on\s+my\s+left)\b", re.IGNORECASE), "left"),
    ]

    total_duration = float(transcript.get("duration", 0.0))
    if total_duration <= 0 and words:
        total_duration = float(words[-1].get("end", 10.0))

    # Map character offsets in reconstructed transcript to word indices
    char_to_word: List[Tuple[int, int, int, dict]] = []
    accum = 0
    for idx, w in enumerate(words):
        w_str = w.get("word", "").strip()
        s_char = accum
        e_char = accum + len(w_str)
        char_to_word.append((s_char, e_char, idx, w))
        accum = e_char + 1

    full_reconstructed = " ".join(w.get("word", "").strip() for w in words)
    extracted_blocks: List[Dict[str, Any]] = []

    for cue_pat, kind, conf in cues:
        for m in cue_pat.finditer(full_reconstructed):
            s_char, e_char = m.start(), m.end()
            matching_words = [cw for cw in char_to_word if cw[1] >= s_char and cw[0] <= e_char]
            if matching_words:
                first_w_idx = matching_words[0][2]
                end_w_idx = min(len(words) - 1, first_w_idx + 8)

                # Scan phrase window for directional modifier
                sub_phrase = " ".join(w.get("word", "") for w in words[first_w_idx:end_w_idx + 1])
                dir_hint = "center"
                for dir_pat, dir_name in directions:
                    if dir_pat.search(sub_phrase):
                        dir_hint = dir_name
                        break

                start_t = max(0.0, float(words[first_w_idx].get("start", 0.0)) - 0.15)
                end_t = min(total_duration, float(words[end_w_idx].get("end", start_t + 2.5)) + 0.3)

                extracted_blocks.append({
                    "start": round(start_t, 3),
                    "end": round(end_t, 3),
                    "focus": "object",
                    "direction_hint": dir_hint,
                    "confidence": round(conf, 3)
                })

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

    # Discard transient flickers shorter than min_duration
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
    """
    Executes script analysis and timeline generation with debouncing.
    """
    if mock:
        return generate_mock_focus_timeline(transcript)

    words = transcript.get("words", [])
    if not words:
        return FocusTimelineData(blocks=[], metadata={"mode": "empty"}).to_dict()

    raw_blocks: List[Dict[str, Any]] = []
    mode_used = "nlp_heuristic"

    # Attempt OpenAI LLM if API key is present
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

    # Step 3: Debounce raw blocks
    debounced_blocks = debounce_timeline(raw_blocks)
    print(f"[{STAGE_NAME}] Debounced into {len(debounced_blocks)} stable focus blocks.")

    # Convert to FocusBlock dataclasses
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

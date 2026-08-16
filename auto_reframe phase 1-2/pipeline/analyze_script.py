"""
pipeline/analyze_script.py — Phase 2, Steps 2-3

Contract:  transcript.json  ->  focus_timeline.json

Step 2: send the transcript to an LLM (JSON mode enforced) to find
verbal pointing/reference moments ("look at this", "here's the chart").
Step 3: debounce the raw result — merge blocks under 1s apart that share
a direction, drop blocks shorter than 0.3s — so a creator saying
"here — no wait, here" doesn't produce two whip-pans in one second.

Requires OPENAI_API_KEY in the environment (the plan's reference
implementation uses gpt-4o-mini; swap MODEL_NAME below for any
JSON-mode-capable chat model on an OpenAI-compatible endpoint).
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import stage_path                               # noqa: E402
from utils.io_json import load_json, save_json, fail_stage  # noqa: E402

STAGE_NAME = "analyze_script"
MODEL_NAME = "gpt-4o-mini"

SYSTEM_PROMPT = (
    "You analyze a video transcript with word-level timestamps to find "
    "moments where the speaker verbally references or points at "
    "something, e.g. \"look at this\", \"here's the chart\", \"notice this\". "
    "For each moment, output an object with: start_time (seconds), "
    "end_time (seconds), focus (\"speaker\" or \"object\"), direction_hint "
    "(\"left\", \"right\", \"center\", or \"unknown\"), and confidence (0-1). "
    "focus=\"object\" means the speaker is drawing attention to something "
    "other than themselves, in-frame or off-camera; focus=\"speaker\" is "
    "everything else and does not need to be listed explicitly — only "
    "list moments worth flagging. "
    "Respond with ONLY a JSON object of the shape "
    "{\"blocks\": [{\"start_time\": .., \"end_time\": .., \"focus\": .., "
    "\"direction_hint\": .., \"confidence\": ..}, ...]}. No prose, no markdown fences."
)


def _transcript_to_prompt(transcript: dict) -> str:
    """Collapse word-level timestamps into compact timestamped lines —
    enough resolution for the LLM to anchor start/end times, without the
    token overhead of the full JSON structure."""
    lines = [f"[{w['start']:.2f}-{w['end']:.2f}] {w['word']}" for w in transcript["words"]]
    return "\n".join(lines)


def _call_llm(transcript_text: str, model: str = MODEL_NAME) -> list[dict]:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it before running this stage "
            "(see Phase 0 setup)."
        )
    from openai import OpenAI  # imported lazily, same reasoning as faster_whisper above

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": transcript_text},
        ],
    )
    parsed = json.loads(response.choices[0].message.content)
    return parsed.get("blocks", [])


def _debounce(blocks: list[dict], merge_gap: float = 1.0, min_duration: float = 0.3) -> list[dict]:
    """Step 3. Blocks must share both focus and direction_hint to merge —
    a "here" (object/left) immediately followed by a "here" (object/left)
    half a second later is one gesture, not two; a block that flips focus
    or direction in that same window is a real second event and stays
    separate."""
    if not blocks:
        return []

    ordered = sorted(
        (dict(b) for b in blocks if b.get("end_time", 0) > b.get("start_time", 0)),
        key=lambda b: b["start_time"],
    )
    if not ordered:
        return []

    merged = [ordered[0]]
    for block in ordered[1:]:
        prev = merged[-1]
        same_kind = (block.get("focus") == prev.get("focus")
                     and block.get("direction_hint") == prev.get("direction_hint"))
        gap = block["start_time"] - prev["end_time"]
        if same_kind and gap < merge_gap:
            prev["end_time"] = max(prev["end_time"], block["end_time"])
            prev["confidence"] = max(prev.get("confidence", 0), block.get("confidence", 0))
        else:
            merged.append(block)

    return [b for b in merged if (b["end_time"] - b["start_time"]) >= min_duration]


def run(transcript: dict, model: str = MODEL_NAME) -> dict:
    """
    Returns:
        {"blocks": [
            {"start_time": 4.1, "end_time": 6.3, "focus": "object",
             "direction_hint": "left", "confidence": 0.87},
            ...
        ]}
    """
    if not transcript.get("words"):
        return {"blocks": []}

    transcript_text = _transcript_to_prompt(transcript)
    raw_blocks = _call_llm(transcript_text, model)
    debounced = _debounce(raw_blocks)
    return {"blocks": debounced}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript", type=Path, default=stage_path("transcript.json"))
    parser.add_argument("--out", type=Path, default=stage_path("focus_timeline.json"))
    parser.add_argument("--model", default=MODEL_NAME)
    args = parser.parse_args()

    try:
        transcript = load_json(args.transcript)
        focus_timeline = run(transcript, args.model)
        save_json(args.out, focus_timeline)
        print(f"[{STAGE_NAME}] wrote {args.out} ({len(focus_timeline['blocks'])} blocks)")
    except Exception as e:
        fail_stage(STAGE_NAME, e)


if __name__ == "__main__":
    main()

"""
tests/test_phase2_analyze.py — Unit and integration tests for analyze_script.py (Phase 2 Steps 2 & 3)
"""
import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from pipeline.analyze_script import (
    run as run_analyze,
    debounce_timeline,
    _extract_heuristic_focus_blocks,
    generate_mock_focus_timeline
)
from pipeline.transcribe import generate_mock_transcript
from contracts import validate_focus_timeline


def test_mock_analyze():
    transcript = generate_mock_transcript(10.37)
    timeline = generate_mock_focus_timeline(transcript)
    validate_focus_timeline(timeline)
    assert len(timeline["blocks"]) == 3
    assert timeline["blocks"][1]["focus"] == "object"
    assert timeline["blocks"][1]["direction_hint"] == "right"


def test_heuristic_nlp_extraction():
    transcript = generate_mock_transcript(10.37)
    raw_blocks = _extract_heuristic_focus_blocks(transcript)
    assert len(raw_blocks) > 0
    # Should detect the object cue ("look at this chart on the right")
    obj_blocks = [b for b in raw_blocks if b["focus"] == "object"]
    assert len(obj_blocks) >= 1
    # Direction hint should be extracted as right or center
    assert any(b["direction_hint"] in ("right", "center") for b in obj_blocks)


def test_step3_debouncing_merge():
    # Two adjacent blocks with same focus and direction within 0.5s gap should merge
    blocks = [
        {"start": 1.0, "end": 2.0, "focus": "object", "direction_hint": "right", "confidence": 0.9},
        {"start": 2.4, "end": 4.0, "focus": "object", "direction_hint": "right", "confidence": 0.95},
    ]
    debounced = debounce_timeline(blocks, merge_gap=1.0, min_duration=0.3)
    assert len(debounced) == 1
    assert debounced[0]["start"] == 1.0
    assert debounced[0]["end"] == 4.0
    assert debounced[0]["confidence"] == 0.95


def test_step3_debouncing_discard_short():
    # A tiny glitch block < 0.3s should be discarded
    blocks = [
        {"start": 1.0, "end": 1.15, "focus": "object", "direction_hint": "left", "confidence": 0.5},
        {"start": 5.0, "end": 7.5, "focus": "speaker", "direction_hint": "center", "confidence": 0.9},
    ]
    debounced = debounce_timeline(blocks, merge_gap=1.0, min_duration=0.3)
    assert len(debounced) == 1
    assert debounced[0]["start"] == 5.0
    assert debounced[0]["end"] == 7.5


def test_run_analyze_end_to_end():
    transcript = generate_mock_transcript(10.37)
    timeline = run_analyze(transcript, mock=False)
    validate_focus_timeline(timeline)
    assert len(timeline["blocks"]) > 0

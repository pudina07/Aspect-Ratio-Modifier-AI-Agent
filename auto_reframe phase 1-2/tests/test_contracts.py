"""
tests/test_contracts.py — Validation test suite for data contracts
"""
import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from contracts import (
    TranscriptData, WordTiming, FocusTimelineData, FocusBlock,
    RawCoordsData, RawFrameCoord, TextRegionsData, TextRegion,
    FinalCoordsData, FinalFrameCoord,
    validate_transcript, validate_focus_timeline, validate_raw_coords,
    validate_text_regions, validate_final_coords,
    ContractValidationError
)


def test_transcript_contract_valid():
    t = TranscriptData(
        words=[WordTiming("Hello", 0.0, 0.5, 0.99), WordTiming("world", 0.5, 1.0, 0.98)],
        text="Hello world",
        duration=1.0
    ).to_dict()
    validate_transcript(t)


def test_transcript_contract_invalid():
    bad_t = {"words": [{"word": "Test", "start": 5.0, "end": 2.0}]}
    with pytest.raises(ContractValidationError):
        validate_transcript(bad_t)


def test_focus_timeline_contract_valid():
    f = FocusTimelineData(
        blocks=[
            FocusBlock(0.0, 3.5, "speaker", "center", 0.99),
            FocusBlock(3.5, 6.0, "object", "right", 0.95)
        ]
    ).to_dict()
    validate_focus_timeline(f)


def test_focus_timeline_contract_invalid():
    bad_f = {"blocks": [{"start": 0.0, "end": 1.0, "focus": "invalid_focus", "direction_hint": "center"}]}
    with pytest.raises(ContractValidationError):
        validate_focus_timeline(bad_f)


def test_raw_coords_contract_valid():
    r = RawCoordsData(
        fps=30.0,
        width=1920,
        height=1080,
        total_frames=1,
        frames=[RawFrameCoord(0, 0.0, face_center=[960.0, 400.0])]
    ).to_dict()
    validate_raw_coords(r)


def test_text_regions_contract_valid():
    tr = TextRegionsData(
        fps=30.0,
        width=1920,
        height=1080,
        regions=[TextRegion(0.0, 2.0, [100.0, 100.0, 200.0, 50.0], "Title", 0.95)]
    ).to_dict()
    validate_text_regions(tr)


def test_final_coords_contract_valid():
    fc = FinalCoordsData(
        aspect_ratio="9:16",
        target_width=608,
        target_height=1080,
        source_width=1920,
        source_height=1080,
        fps=30.0,
        total_frames=1,
        frames=[FinalFrameCoord(0, 0.0, 656, 0, 608, 1080)]
    ).to_dict()
    validate_final_coords(fc)

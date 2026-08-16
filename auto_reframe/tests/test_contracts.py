"""
tests/test_contracts.py — Unit tests for data contracts & schema validators
"""
import pytest
from contracts import (
    WordTiming, TranscriptData, validate_transcript,
    FocusBlock, FocusTimelineData, validate_focus_timeline,
    RawFrameCoord, RawCoordsData, validate_raw_coords,
    TextRegion, TextRegionsData, validate_text_regions,
    FinalFrameCoord, FinalCoordsData, validate_final_coords,
    ContractValidationError
)


def test_transcript_contract_valid():
    words = [WordTiming(word="hello", start=0.0, end=0.5, probability=0.99)]
    t = TranscriptData(words=words, text="hello", language="en", duration=0.5)
    d = t.to_dict()
    validate_transcript(d)


def test_transcript_contract_invalid():
    with pytest.raises(ContractValidationError):
        validate_transcript({"words": [{"word": "test", "start": 1.0, "end": 0.5}]})  # End before start

    with pytest.raises(ContractValidationError):
        validate_transcript({"text": "no words list"})


def test_focus_timeline_contract_valid():
    blocks = [FocusBlock(start=0.0, end=2.0, focus="speaker", direction_hint="center", confidence=0.95)]
    f = FocusTimelineData(blocks=blocks)
    validate_focus_timeline(f.to_dict())


def test_focus_timeline_contract_invalid():
    with pytest.raises(ContractValidationError):
        validate_focus_timeline({"blocks": [{"start": 0.0, "end": 1.0, "focus": "invalid_focus", "direction_hint": "center", "confidence": 1.0}]})


def test_raw_coords_contract_valid():
    coords = RawCoordsData(
        fps=30.0,
        width=1920,
        height=1080,
        total_frames=1,
        frames=[RawFrameCoord(frame_idx=0, t=0.0, face_center=[960.0, 540.0])]
    )
    validate_raw_coords(coords.to_dict())


def test_text_regions_contract_valid():
    data = TextRegionsData(
        fps=30.0,
        width=1920,
        height=1080,
        regions=[TextRegion(t_start=0.0, t_end=2.0, box=[100.0, 100.0, 200.0, 50.0], text="Headline")]
    )
    validate_text_regions(data.to_dict())


def test_final_coords_contract_valid():
    data = FinalCoordsData(
        aspect_ratio="9:16",
        target_width=608,
        target_height=1080,
        source_width=1920,
        source_height=1080,
        fps=30.0,
        total_frames=1,
        frames=[FinalFrameCoord(frame_idx=0, t=0.0, crop_x=656, crop_y=0, crop_w=608, crop_h=1080)]
    )
    validate_final_coords(data.to_dict())

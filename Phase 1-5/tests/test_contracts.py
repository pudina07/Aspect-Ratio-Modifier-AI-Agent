"""
tests/test_contracts.py — Schema & Contract Unit Tests
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from contracts import (
    WordTiming, TranscriptData, validate_transcript,
    FocusBlock, FocusTimelineData, validate_focus_timeline, normalize_focus_block,
    RawFrameCoord, RawCoordsData, validate_raw_coords,
    TextRegion, TextRegionsData, validate_text_regions,
    FinalFrameCoord, FinalCoordsData, validate_final_coords,
    ContractValidationError
)


def test_transcript_contract():
    valid = {
        "text": "Hello world",
        "words": [
            {"word": "Hello", "start": 0.0, "end": 0.5, "confidence": 0.95},
            {"word": "world", "start": 0.5, "end": 1.0, "confidence": 0.98}
        ]
    }
    validate_transcript(valid)

    # Inverted timestamps
    try:
        validate_transcript({
            "words": [{"word": "bad", "start": 1.0, "end": 0.5}]
        })
        assert False, "Should have raised ContractValidationError"
    except ContractValidationError:
        pass


def test_focus_timeline_contract():
    valid = {
        "blocks": [
            {"start": 0.0, "end": 3.0, "focus": "speaker", "direction_hint": "center", "confidence": 0.95},
            {"start": 3.0, "end": 5.5, "focus": "object", "direction_hint": "right", "confidence": 0.9}
        ]
    }
    validate_focus_timeline(valid)

    # Missing focus
    try:
        validate_focus_timeline({"blocks": [{"start": 0.0, "end": 1.0}]})
        assert False, "Should have raised ContractValidationError"
    except ContractValidationError:
        pass


def test_raw_coords_contract():
    valid = {
        "fps": 30.0,
        "width": 1920,
        "height": 1080,
        "total_frames": 2,
        "frames": [
            {"frame_idx": 0, "t": 0.0, "face_center": [960.0, 540.0], "focus": "speaker"},
            {"frame_idx": 1, "t": 0.033, "face_center": [960.0, 540.0], "focus": "speaker"}
        ]
    }
    validate_raw_coords(valid)


def test_text_regions_contract():
    valid = {
        "fps": 30.0,
        "width": 1920,
        "height": 1080,
        "regions": [
            {"t_start": 1.0, "t_end": 3.0, "box": [100.0, 200.0, 300.0, 50.0], "text": "TEST", "confidence": 0.99}
        ]
    }
    validate_text_regions(valid)


def test_final_coords_contract():
    valid = {
        "aspect_ratio": "9:16",
        "target_width": 608,
        "target_height": 1080,
        "fps": 30.0,
        "frames": [
            {"frame_idx": 0, "t": 0.0, "crop_x": 656, "crop_y": 0, "crop_w": 608, "crop_h": 1080}
        ]
    }
    validate_final_coords(valid)


def main():
    test_transcript_contract()
    test_focus_timeline_contract()
    test_raw_coords_contract()
    test_text_regions_contract()
    test_final_coords_contract()
    print("  ✓ All contracts & schema validations passed!")


if __name__ == "__main__":
    main()

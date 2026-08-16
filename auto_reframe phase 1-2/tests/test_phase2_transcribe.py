"""
tests/test_phase2_transcribe.py — Unit and integration tests for transcribe.py (Phase 2 Step 1)
"""
import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from pipeline.transcribe import run as run_transcribe, generate_mock_transcript
from contracts import validate_transcript, TranscriptData


def test_mock_transcribe():
    t = generate_mock_transcript(10.37)
    validate_transcript(t)
    assert len(t["words"]) > 0
    assert t["duration"] == 10.37
    assert "Hello" in t["text"]


def test_real_audio_transcribe():
    # Test on local assets/speech.wav if available
    base_dir = Path(__file__).resolve().parent.parent.parent
    audio_path = base_dir / "assets" / "speech.wav"
    if not audio_path.exists():
        pytest.skip(f"Audio file {audio_path} not found")

    result = run_transcribe(
        video_path=audio_path,
        model_size="base",
        device="cpu",
        compute_type="int8",
        mock=False
    )
    validate_transcript(result)
    assert len(result["words"]) >= 10
    assert result["duration"] > 5.0
    # Verify word timestamps are monotonically increasing
    words = result["words"]
    for i in range(len(words) - 1):
        assert words[i]["start"] <= words[i]["end"]
        assert words[i]["start"] <= words[i+1]["start"] + 0.5

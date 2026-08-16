"""
pipeline/transcribe.py — Phase 2, Step 1: Faster-Whisper Speech-to-Text

Contract:
  Input:  video.mp4 (or audio file)
  Output: transcript.json

Runs faster-whisper locally on 16kHz mono audio extracted via FFmpeg,
producing precise word-level timestamps conforming to TranscriptData contract.

Supports --mock flag to generate valid schema-compliant transcript for testing.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import stage_path, MODELS_DIR, PROJECT_ROOT               # noqa: E402
from contracts import TranscriptData, WordTiming, validate_transcript  # noqa: E402
from utils.io_json import save_json, fail_stage                        # noqa: E402

STAGE_NAME = "transcribe"


def _get_ffmpeg_binary() -> str:
    """Finds available FFmpeg binary via parent ffmpeg_utils, local tools, imageio_ffmpeg, or PATH."""
    try:
        sys.path.append(str(PROJECT_ROOT))
        import ffmpeg_utils
        return ffmpeg_utils.get_ffmpeg_exe()
    except Exception:
        pass

    local_candidates = [
        PROJECT_ROOT / "tools" / "ffmpeg.exe",
        PROJECT_ROOT / "tools" / "ffmpeg",
    ]
    for cand in local_candidates:
        if cand.exists():
            return str(cand)

    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass

    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        return sys_ffmpeg

    raise FileNotFoundError("FFmpeg binary not found on system. Ensure imageio-ffmpeg or ffmpeg is installed.")


def _extract_audio(video_path: Path) -> Path:
    """Extracts audio to 16kHz mono PCM WAV for Faster-Whisper."""
    audio_path = video_path.with_suffix(".wav")
    ffmpeg_exe = _get_ffmpeg_binary()
    cmd = [
        ffmpeg_exe, "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        str(audio_path)
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg audio extraction failed:\n{proc.stderr}")
    return audio_path


def _resolve_whisper_model_path(model_name: str) -> str:
    """Checks for locally cached Whisper model folder before falling back to model name."""
    candidates = [
        MODELS_DIR / "whisper" / model_name,
        MODELS_DIR / model_name,
        PROJECT_ROOT / "models" / "whisper" / model_name,
        PROJECT_ROOT / "models" / model_name,
    ]
    for c in candidates:
        if c.exists() and (c / "model.bin").exists():
            return str(c)

    # Fallback to local base model if large model not found
    fallback_base = [
        MODELS_DIR / "whisper" / "base",
        PROJECT_ROOT / "models" / "whisper" / "base",
    ]
    for c in fallback_base:
        if c.exists() and (c / "model.bin").exists():
            print(f"[{STAGE_NAME}] Model '{model_name}' not cached locally, falling back to cached 'base' model at {c}")
            return str(c)

    return model_name


def generate_mock_transcript(duration: float = 10.37) -> dict:
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
    full_text = " ".join([w.word for w in words])
    data = TranscriptData(
        words=words,
        text=full_text,
        language="en",
        duration=duration
    )
    return data.to_dict()


def run(video_path: Path, model_size: str = "distil-large-v3",
        device: str = "auto", compute_type: str = "auto", mock: bool = False) -> dict:
    """
    Executes Faster-Whisper transcription on the video file.
    Returns:
        {"language": "en", "duration": 10.37, "text": "...",
         "words": [{"word": "look", "start": 1.02, "end": 1.20, "confidence": 0.98}, ...]}
    """
    if mock:
        return generate_mock_transcript()

    if not video_path.exists():
        raise FileNotFoundError(f"Source video/audio file not found at: {video_path}")

    from faster_whisper import WhisperModel

    # Extract audio if video format
    is_wav = video_path.suffix.lower() == ".wav"
    audio_path = video_path if is_wav else _extract_audio(video_path)

    model_target = _resolve_whisper_model_path(model_size)
    print(f"[{STAGE_NAME}] Loading Faster-Whisper model from '{model_target}' (device={device}, compute_type={compute_type})...")

    # On CPU with int8 quantization
    comp_type = compute_type
    if comp_type == "auto":
        comp_type = "int8" if device in ("auto", "cpu") else "float16"

    dev = "cpu" if device == "auto" else device

    model = WhisperModel(
        model_target,
        device=dev,
        compute_type=comp_type,
        download_root=str(MODELS_DIR / "whisper")
    )

    segments, info = model.transcribe(str(audio_path), word_timestamps=True)

    words_list: List[WordTiming] = []
    full_text_parts: List[str] = []

    for segment in segments:
        full_text_parts.append(segment.text.strip())
        if segment.words:
            for w in segment.words:
                cleaned = w.word.strip()
                if cleaned:
                    words_list.append(
                        WordTiming(
                            word=cleaned,
                            start=w.start,
                            end=w.end,
                            confidence=w.probability
                        )
                    )

    # Fallback duration calculation
    dur = info.duration if info.duration > 0 else (words_list[-1].end if words_list else 0.0)
    full_text = " ".join(full_text_parts)

    data = TranscriptData(
        words=words_list,
        text=full_text,
        language=info.language or "en",
        duration=dur
    )
    return data.to_dict()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=stage_path("video.mp4"))
    parser.add_argument("--out", type=Path, default=stage_path("transcript.json"))
    parser.add_argument("--model-size", default="distil-large-v3",
                        help="e.g. distil-large-v3, base, small, large-v3")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--compute-type", default="auto")
    parser.add_argument("--mock", action="store_true", help="Generate mock transcript without running model")
    args = parser.parse_args()

    try:
        transcript = run(
            video_path=args.video,
            model_size=args.model_size,
            device=args.device,
            compute_type=args.compute_type,
            mock=args.mock
        )
        save_json(args.out, transcript, validator=validate_transcript)
        word_count = len(transcript.get("words", []))
        duration = transcript.get("duration", 0.0)
        print(f"[{STAGE_NAME}] successfully wrote {args.out} ({word_count} words, {duration:.2f}s)")
    except Exception as e:
        fail_stage(STAGE_NAME, e)


if __name__ == "__main__":
    main()

"""
pipeline/transcribe.py — Phase 2, Step 1

Contract:  video.mp4  ->  transcript.json

Runs faster-whisper locally (CTranslate2 backend) on audio extracted
from the source video via FFmpeg, and writes word-level timestamps.

Model size defaults to distil-large-v3 — the plan's speed-safe demo
default. Pass --model-size large-v3 if your time/hardware budget allows
the accuracy bump; either way, do the actual model download during
Phase 0 (see the plan's setup checklist), not the first time this
script runs mid-demo.
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import stage_path, MODELS_DIR          # noqa: E402
from utils.io_json import save_json, fail_stage    # noqa: E402

STAGE_NAME = "transcribe"


def _extract_audio(video_path: Path) -> Path:
    """FFmpeg -> 16kHz mono PCM wav, the format faster-whisper expects.
    Extracting explicitly (rather than letting faster-whisper decode the
    video itself) keeps this stage's dependency surface to 'ffmpeg is on
    PATH', which Phase 0 already checks for."""
    audio_path = video_path.with_suffix(".wav")
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        str(audio_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed:\n{proc.stderr}")
    return audio_path


def run(video_path: Path, model_size: str = "distil-large-v3",
        device: str = "auto", compute_type: str = "auto") -> dict:
    """
    Returns:
        {"language": "en", "duration": 42.3,
         "words": [{"word": "look", "start": 1.02, "end": 1.20,
                     "confidence": 0.98}, ...]}
    """
    from faster_whisper import WhisperModel  # imported lazily so every
    # other stage's --help / argparse still works without this installed

    if not video_path.exists():
        raise FileNotFoundError(f"No video at {video_path}")

    audio_path = _extract_audio(video_path)
    model = WhisperModel(
        model_size, device=device, compute_type=compute_type,
        download_root=str(MODELS_DIR),
    )
    segments, info = model.transcribe(str(audio_path), word_timestamps=True)

    words = []
    for segment in segments:
        for word in segment.words:
            words.append({
                "word": word.word.strip(),
                "start": round(word.start, 3),
                "end": round(word.end, 3),
                "confidence": round(word.probability, 3) if word.probability is not None else None,
            })

    return {"language": info.language, "duration": info.duration, "words": words}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=stage_path("video.mp4"))
    parser.add_argument("--out", type=Path, default=stage_path("transcript.json"))
    parser.add_argument("--model-size", default="distil-large-v3",
                         help="e.g. distil-large-v3 (fast) or large-v3 (most accurate)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--compute-type", default="auto",
                         help="e.g. int8, float16, int8_float16 — see CTranslate2 docs")
    args = parser.parse_args()

    try:
        transcript = run(args.video, args.model_size, args.device, args.compute_type)
        save_json(args.out, transcript)
        print(f"[{STAGE_NAME}] wrote {args.out} "
              f"({len(transcript['words'])} words, {transcript['duration']:.1f}s)")
    except Exception as e:
        fail_stage(STAGE_NAME, e)


if __name__ == "__main__":
    main()

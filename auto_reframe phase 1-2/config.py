"""
config.py — Phase 1: Architecture

Single source of truth for:
  1. where files live (DATA_DIR holds every intermediate artifact for a run)
  2. the stage/JSON contract graph from the plan's Phase 1 diagram

Nothing here does real work. The point of this file is that no other
script hardcodes a filename or a "what depends on what" assumption —
they all read it from PIPELINE_STAGES. That's what lets pipeline_runner.py
figure out which stages can run in parallel (see its docstring) and lets
app.py stay dumb about pipeline internals.
"""
from pathlib import Path

# --- Directory layout ----------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"       # per-run working dir: video.mp4 in, everything else out
MODELS_DIR = BASE_DIR / "models"   # local model weights (whisper / mediapipe / easyocr)

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


def stage_path(filename: str) -> Path:
    """Resolve an artifact filename to its path inside the run's data dir.
    Every stage script should get its default in/out paths through this,
    not by string-concatenating paths themselves."""
    return DATA_DIR / filename


# --- Pipeline stage / dependency graph ------------------------------------
# "inputs"/"outputs" are filenames inside DATA_DIR. This is a DAG, not a
# strict line: ocr_pass only needs video.mp4, so it has no dependency on
# the transcribe -> analyze_script -> tracker chain and can run alongside
# it. pipeline_runner.py schedules stages off this graph rather than
# assuming top-to-bottom order.
PIPELINE_STAGES = [
    {
        "name": "transcribe",
        "script": "pipeline/transcribe.py",
        "inputs": ["video.mp4"],
        "outputs": ["transcript.json"],
    },
    {
        "name": "analyze_script",
        "script": "pipeline/analyze_script.py",
        "inputs": ["transcript.json"],
        "outputs": ["focus_timeline.json"],
    },
    {
        "name": "tracker",
        "script": "pipeline/tracker.py",
        # needs focus_timeline so it knows which frames are "object" blocks
        # (i.e. worth running pose+hand on) vs. plain speaker frames
        "inputs": ["video.mp4", "focus_timeline.json"],
        "outputs": ["raw_coords.json"],
    },
    {
        "name": "ocr_pass",
        "script": "pipeline/ocr_pass.py",
        # deliberately independent of the transcript/focus branch
        "inputs": ["video.mp4"],
        "outputs": ["text_regions.json"],
    },
    {
        "name": "smooth_coords",
        "script": "pipeline/smooth_coords.py",
        "inputs": ["raw_coords.json", "text_regions.json", "focus_timeline.json"],
        "outputs": ["final_coords_916.json", "final_coords_11.json"],
    },
    {
        "name": "render",
        "script": "pipeline/render.py",
        "inputs": ["video.mp4", "final_coords_916.json", "final_coords_11.json"],
        "outputs": ["output_916.mp4", "output_11.mp4"],
    },
]

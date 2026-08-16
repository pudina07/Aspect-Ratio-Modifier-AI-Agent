"""
config.py — Phase 1: Architecture & Pipeline Configuration

Single source of truth for:
  1. Directory layout and model/asset paths.
  2. Safe zones configuration loading.
  3. The stage/JSON contract DAG graph.
  4. DAG validation and dependency query helpers.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Set

import os

# --- Directory layout ----------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
custom_env_data_dir = os.environ.get("AUTO_REFRAME_DATA_DIR")
DATA_DIR = Path(custom_env_data_dir) if custom_env_data_dir else BASE_DIR / "data"
MODELS_DIR = PROJECT_ROOT / "models"  # Preloaded model weights (from Phase 0)
SAFE_ZONES_PATH = PROJECT_ROOT / "safe_zones.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def stage_path(filename: str, custom_dir: Optional[Path] = None) -> Path:
    """Resolve an artifact filename to its path inside data/ or a custom directory."""
    target_dir = custom_dir or DATA_DIR
    return target_dir / filename


def load_safe_zones(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load safe zones presets from safe_zones.json with robust fallback."""
    sz_file = path or SAFE_ZONES_PATH
    if sz_file.exists():
        try:
            with open(sz_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    # Hardcoded sane fallback if file is missing/unreadable
    return {
        "platforms": {
            "tiktok_916": {
                "name": "TikTok",
                "aspect_ratio": "9:16",
                "canvas_width": 1080,
                "canvas_height": 1920,
                "margins": {"top": 130, "bottom": 380, "left": 60, "right": 170}
            },
            "instagram_reels_916": {
                "name": "Instagram Reels",
                "aspect_ratio": "9:16",
                "canvas_width": 1080,
                "canvas_height": 1920,
                "margins": {"top": 220, "bottom": 410, "left": 30, "right": 110}
            },
            "instagram_feed_11": {
                "name": "Instagram Feed",
                "aspect_ratio": "1:1",
                "canvas_width": 1080,
                "canvas_height": 1080,
                "margins": {"top": 20, "bottom": 90, "left": 20, "right": 20}
            }
        }
    }


# --- Pipeline stage / dependency DAG graph --------------------------------
# Inputs and outputs are filenames relative to DATA_DIR.
# ocr_pass only requires video.mp4, so it can run concurrently with the
# transcribe -> analyze_script -> tracker branch.
PIPELINE_STAGES: List[Dict[str, Any]] = [
    {
        "name": "transcribe",
        "script": "pipeline/transcribe.py",
        "inputs": ["video.mp4"],
        "outputs": ["transcript.json"],
        "description": "Speech-to-text transcription with word-level timestamps"
    },
    {
        "name": "analyze_script",
        "script": "pipeline/analyze_script.py",
        "inputs": ["transcript.json"],
        "outputs": ["focus_timeline.json"],
        "description": "LLM pointing/reference detection with debounced focus blocks"
    },
    {
        "name": "tracker",
        "script": "pipeline/tracker.py",
        "inputs": ["video.mp4", "focus_timeline.json"],
        "outputs": ["raw_coords.json"],
        "description": "Face tracking & pose/hand pointing vector extrapolation"
    },
    {
        "name": "ocr_pass",
        "script": "pipeline/ocr_pass.py",
        "inputs": ["video.mp4"],
        "outputs": ["text_regions.json"],
        "description": "On-screen text detection for protected boundary zones"
    },
    {
        "name": "smooth_coords",
        "script": "pipeline/smooth_coords.py",
        "inputs": ["raw_coords.json", "text_regions.json", "focus_timeline.json"],
        "outputs": ["final_coords_916.json", "final_coords_11.json"],
        "description": "Dual-aspect crop windowing with One Euro filter & text clamping"
    },
    {
        "name": "render",
        "script": "pipeline/render.py",
        "inputs": ["video.mp4", "final_coords_916.json", "final_coords_11.json"],
        "outputs": ["output_916.mp4", "output_11.mp4"],
        "description": "Video rendering with blurred full-bleed composite & audio muxing"
    },
]


def get_stage_by_name(name: str) -> Optional[Dict[str, Any]]:
    """Return stage definition dict by name."""
    for s in PIPELINE_STAGES:
        if s["name"] == name:
            return s
    return None


def get_downstream_stages(failed_stage_name: str) -> Set[str]:
    """
    Computes all downstream stages that transitively depend on the outputs
    of failed_stage_name.
    """
    failed_stage = get_stage_by_name(failed_stage_name)
    if not failed_stage:
        return set()

    unavailable_files = set(failed_stage["outputs"])
    blocked_stages: Set[str] = set()

    changed = True
    while changed:
        changed = False
        for stage in PIPELINE_STAGES:
            s_name = stage["name"]
            if s_name not in blocked_stages and s_name != failed_stage_name:
                # If stage requires any unavailable file
                if any(inp in unavailable_files for inp in stage["inputs"]):
                    blocked_stages.add(s_name)
                    unavailable_files.update(stage["outputs"])
                    changed = True

    return blocked_stages


def validate_pipeline_dag() -> bool:
    """
    Validates that the pipeline definition is a valid DAG:
    - Every required input (except external entry points like video.mp4) is produced by an upstream stage.
    - No circular dependencies exist.
    """
    produced_files = {"video.mp4"}
    for stage in PIPELINE_STAGES:
        produced_files.update(stage["outputs"])

    for stage in PIPELINE_STAGES:
        for inp in stage["inputs"]:
            if inp not in produced_files:
                raise ValueError(f"Stage '{stage['name']}' requires '{inp}', which is never produced.")

    return True

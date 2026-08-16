"""
contracts.py — Phase 1: Architecture & Data Contracts

Defines the exact schemas and data structures for every intermediate JSON artifact
produced and consumed by the pipeline stages.

Graph Contract:
  video.mp4
    -> transcribe.py     -> transcript.json
    -> analyze_script.py -> focus_timeline.json
    -> tracker.py        -> raw_coords.json
    -> ocr_pass.py       -> text_regions.json
    -> smooth_coords.py  -> final_coords_916.json, final_coords_11.json
    -> render.py         -> output_916.mp4, output_11.mp4
"""
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Literal, Dict, Any


class ContractValidationError(ValueError):
    """Raised when an artifact fails schema or constraint validation."""
    pass


# -------------------------------------------------------------------------
# 1. Transcript Contract: transcript.json
# -------------------------------------------------------------------------

@dataclass
class WordTiming:
    word: str
    start: float
    end: float
    probability: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {"word": str(self.word), "start": float(self.start), "end": float(self.end)}
        if self.probability is not None:
            d["probability"] = float(self.probability)
        return d


@dataclass
class TranscriptData:
    words: List[WordTiming]
    text: str = ""
    language: str = "en"
    duration: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "language": self.language,
            "duration": float(self.duration),
            "words": [w.to_dict() if isinstance(w, WordTiming) else w for w in self.words]
        }


def validate_transcript(data: Any) -> None:
    if not isinstance(data, dict):
        raise ContractValidationError("Transcript root must be a JSON object (dict)")
    if "words" not in data or not isinstance(data["words"], list):
        raise ContractValidationError("Transcript must contain a 'words' list")
    for i, w in enumerate(data["words"]):
        if not isinstance(w, dict):
            raise ContractValidationError(f"Word at index {i} must be a dict")
        if "word" not in w or "start" not in w or "end" not in w:
            raise ContractValidationError(f"Word at index {i} missing required keys ('word', 'start', 'end')")
        if w["start"] < 0 or w["end"] < w["start"]:
            raise ContractValidationError(f"Invalid timestamp range at word {i}: {w['start']} -> {w['end']}")


# -------------------------------------------------------------------------
# 2. Focus Timeline Contract: focus_timeline.json
# -------------------------------------------------------------------------

@dataclass
class FocusBlock:
    start: float
    end: float
    focus: Literal["speaker", "object"] = "speaker"
    direction_hint: Literal["left", "right", "center", "unknown"] = "center"
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": round(float(self.start), 3),
            "end": round(float(self.end), 3),
            "focus": self.focus,
            "direction_hint": self.direction_hint,
            "confidence": round(float(self.confidence), 3),
        }


@dataclass
class FocusTimelineData:
    blocks: List[FocusBlock]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "blocks": [b.to_dict() if isinstance(b, FocusBlock) else b for b in self.blocks],
            "metadata": self.metadata
        }


def validate_focus_timeline(data: Any) -> None:
    if not isinstance(data, dict):
        raise ContractValidationError("Focus timeline root must be a JSON object (dict)")
    if "blocks" not in data or not isinstance(data["blocks"], list):
        raise ContractValidationError("Focus timeline must contain a 'blocks' list")
    for i, b in enumerate(data["blocks"]):
        if not isinstance(b, dict):
            raise ContractValidationError(f"Focus block at index {i} must be a dict")
        for key in ("start", "end", "focus", "direction_hint", "confidence"):
            if key not in b:
                raise ContractValidationError(f"Focus block at index {i} missing key: '{key}'")
        if b["focus"] not in ("speaker", "object"):
            raise ContractValidationError(f"Invalid focus value at block {i}: {b['focus']}")
        if b["direction_hint"] not in ("left", "right", "center", "unknown"):
            raise ContractValidationError(f"Invalid direction_hint at block {i}: {b['direction_hint']}")
        if b["start"] < 0 or b["end"] < b["start"]:
            raise ContractValidationError(f"Invalid timestamp range at block {i}: {b['start']} -> {b['end']}")


# -------------------------------------------------------------------------
# 3. Raw Coords Contract: raw_coords.json
# -------------------------------------------------------------------------

@dataclass
class RawFrameCoord:
    frame_idx: int
    t: float
    face_center: Optional[List[float]] = None          # [x, y]
    face_box: Optional[List[float]] = None             # [x, y, w, h]
    wrist: Optional[List[float]] = None                # [x, y]
    fingertip: Optional[List[float]] = None            # [x, y]
    extrapolated_target: Optional[List[float]] = None  # [x, y]
    focus: str = "speaker"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_idx": int(self.frame_idx),
            "t": round(float(self.t), 3),
            "face_center": [round(float(v), 1) for v in self.face_center] if self.face_center else None,
            "face_box": [round(float(v), 1) for v in self.face_box] if self.face_box else None,
            "wrist": [round(float(v), 1) for v in self.wrist] if self.wrist else None,
            "fingertip": [round(float(v), 1) for v in self.fingertip] if self.fingertip else None,
            "extrapolated_target": [round(float(v), 1) for v in self.extrapolated_target] if self.extrapolated_target else None,
            "focus": self.focus,
        }


@dataclass
class RawCoordsData:
    fps: float
    width: int
    height: int
    total_frames: int
    frames: List[RawFrameCoord]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fps": float(self.fps),
            "width": int(self.width),
            "height": int(self.height),
            "total_frames": int(self.total_frames),
            "frames": [f.to_dict() if isinstance(f, RawFrameCoord) else f for f in self.frames]
        }


def validate_raw_coords(data: Any) -> None:
    if not isinstance(data, dict):
        raise ContractValidationError("Raw coords root must be a JSON object (dict)")
    for key in ("fps", "width", "height", "total_frames", "frames"):
        if key not in data:
            raise ContractValidationError(f"Raw coords missing required root key '{key}'")
    if not isinstance(data["frames"], list):
        raise ContractValidationError("Raw coords 'frames' must be a list")
    if len(data["frames"]) > 0:
        f0 = data["frames"][0]
        if "frame_idx" not in f0 or "t" not in f0:
            raise ContractValidationError("Frame coords must include 'frame_idx' and 't'")


# -------------------------------------------------------------------------
# 4. Text Regions Contract: text_regions.json
# -------------------------------------------------------------------------

@dataclass
class TextRegion:
    t_start: float
    t_end: float
    box: List[float]             # [x, y, w, h] in source video pixel coordinates
    text: str = ""
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "t_start": round(float(self.t_start), 3),
            "t_end": round(float(self.t_end), 3),
            "box": [round(float(v), 1) for v in self.box],
            "text": str(self.text),
            "confidence": round(float(self.confidence), 3),
        }


@dataclass
class TextRegionsData:
    fps: float
    width: int
    height: int
    regions: List[TextRegion]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fps": float(self.fps),
            "width": int(self.width),
            "height": int(self.height),
            "regions": [r.to_dict() if isinstance(r, TextRegion) else r for r in self.regions]
        }


def validate_text_regions(data: Any) -> None:
    if not isinstance(data, dict):
        raise ContractValidationError("Text regions root must be a JSON object (dict)")
    if "regions" not in data or not isinstance(data["regions"], list):
        raise ContractValidationError("Text regions must contain a 'regions' list")
    for i, r in enumerate(data["regions"]):
        if not isinstance(r, dict):
            raise ContractValidationError(f"Text region at index {i} must be a dict")
        for key in ("t_start", "t_end", "box"):
            if key not in r:
                raise ContractValidationError(f"Text region at index {i} missing key '{key}'")
        if not isinstance(r["box"], list) or len(r["box"]) != 4:
            raise ContractValidationError(f"Text region box at index {i} must be [x, y, w, h]")


# -------------------------------------------------------------------------
# 5. Final Coords Contract: final_coords_916.json & final_coords_11.json
# -------------------------------------------------------------------------

@dataclass
class FinalFrameCoord:
    frame_idx: int
    t: float
    crop_x: int
    crop_y: int
    crop_w: int
    crop_h: int
    focus: str = "speaker"
    text_protected: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_idx": int(self.frame_idx),
            "t": round(float(self.t), 3),
            "crop_x": int(self.crop_x),
            "crop_y": int(self.crop_y),
            "crop_w": int(self.crop_w),
            "crop_h": int(self.crop_h),
            "focus": str(self.focus),
            "text_protected": bool(self.text_protected)
        }


@dataclass
class FinalCoordsData:
    aspect_ratio: Literal["9:16", "1:1"]
    target_width: int
    target_height: int
    source_width: int
    source_height: int
    fps: float
    total_frames: int
    frames: List[FinalFrameCoord]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "aspect_ratio": self.aspect_ratio,
            "target_width": int(self.target_width),
            "target_height": int(self.target_height),
            "source_width": int(self.source_width),
            "source_height": int(self.source_height),
            "fps": float(self.fps),
            "total_frames": int(self.total_frames),
            "frames": [f.to_dict() if isinstance(f, FinalFrameCoord) else f for f in self.frames]
        }


def validate_final_coords(data: Any) -> None:
    if not isinstance(data, dict):
        raise ContractValidationError("Final coords root must be a JSON object (dict)")
    for key in ("aspect_ratio", "target_width", "target_height", "fps", "frames"):
        if key not in data:
            raise ContractValidationError(f"Final coords missing required key '{key}'")
    if not isinstance(data["frames"], list):
        raise ContractValidationError("Final coords 'frames' must be a list")
    if len(data["frames"]) > 0:
        f0 = data["frames"][0]
        for fkey in ("frame_idx", "t", "crop_x", "crop_y", "crop_w", "crop_h"):
            if fkey not in f0:
                raise ContractValidationError(f"Final coords frame item missing key '{fkey}'")


# Contract registry mapping output artifact filenames to validators
CONTRACT_VALIDATORS = {
    "transcript.json": validate_transcript,
    "focus_timeline.json": validate_focus_timeline,
    "raw_coords.json": validate_raw_coords,
    "text_regions.json": validate_text_regions,
    "final_coords_916.json": validate_final_coords,
    "final_coords_11.json": validate_final_coords,
}

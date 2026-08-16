"""
contracts.py — Architecture & Data Contracts (Phases 1 to 5)

Defines exact schemas, dataclasses, and validators for every intermediate
JSON artifact produced and consumed across the pipeline stages:

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
# 1. Transcript Contract: transcript.json (Phase 2, Step 1)
# -------------------------------------------------------------------------

@dataclass
class WordTiming:
    word: str
    start: float
    end: float
    confidence: Optional[float] = None
    probability: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        conf = self.confidence if self.confidence is not None else self.probability
        d: Dict[str, Any] = {
            "word": str(self.word),
            "start": round(float(self.start), 3),
            "end": round(float(self.end), 3)
        }
        if conf is not None:
            d["confidence"] = round(float(conf), 3)
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
            "duration": round(float(self.duration), 3),
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
# 2. Focus Timeline Contract: focus_timeline.json (Phase 2, Steps 2-3)
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


def normalize_focus_block(b: dict) -> dict:
    """Normalizes focus block keys to handle both start/end and start_time/end_time."""
    start = b.get("start", b.get("start_time", 0.0))
    end = b.get("end", b.get("end_time", start))
    focus = b.get("focus", "speaker")
    direction = b.get("direction_hint", b.get("direction", "center"))
    confidence = b.get("confidence", 1.0)
    return {
        "start": round(float(start), 3),
        "end": round(float(end), 3),
        "focus": str(focus),
        "direction_hint": str(direction),
        "confidence": round(float(confidence), 3)
    }


def validate_focus_timeline(data: Any) -> None:
    if not isinstance(data, dict):
        raise ContractValidationError("Focus timeline root must be a JSON object (dict)")
    if "blocks" not in data or not isinstance(data["blocks"], list):
        raise ContractValidationError("Focus timeline must contain a 'blocks' list")
    for i, b in enumerate(data["blocks"]):
        if not isinstance(b, dict):
            raise ContractValidationError(f"Focus block at index {i} must be a dict")
        has_start = "start" in b or "start_time" in b
        has_end = "end" in b or "end_time" in b
        if not has_start or not has_end or "focus" not in b or "direction_hint" not in b:
            raise ContractValidationError(f"Focus block at index {i} missing required keys (start, end, focus, direction_hint)")
        start_val = b.get("start", b.get("start_time"))
        end_val = b.get("end", b.get("end_time"))
        if b["focus"] not in ("speaker", "object"):
            raise ContractValidationError(f"Invalid focus value at block {i}: {b['focus']}")
        if b["direction_hint"] not in ("left", "right", "center", "unknown"):
            raise ContractValidationError(f"Invalid direction_hint at block {i}: {b['direction_hint']}")
        if start_val < 0 or end_val < start_val:
            raise ContractValidationError(f"Invalid timestamp range at block {i}: {start_val} -> {end_val}")


# -------------------------------------------------------------------------
# 3. Raw Coords Contract: raw_coords.json (Phase 3, Steps 4-5)
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
    has_root_keys = all(k in data for k in ("fps", "width", "height", "frames"))
    has_meta_keys = "meta" in data and "frames" in data
    if not (has_root_keys or has_meta_keys):
        raise ContractValidationError("Raw coords missing required dimensions/fps metadata and frames list")
    if not isinstance(data["frames"], list):
        raise ContractValidationError("Raw coords 'frames' must be a list")
    if len(data["frames"]) > 0:
        f0 = data["frames"][0]
        if not isinstance(f0, dict):
            raise ContractValidationError("Raw coords frame item must be a dict")
        if "t" not in f0:
            raise ContractValidationError("Frame coords must include timestamp 't'")


# -------------------------------------------------------------------------
# 4. Text Regions Contract: text_regions.json (Phase 3, Step 6)
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
        for key in ("t_start", "t_end"):
            if key not in r:
                raise ContractValidationError(f"Text region at index {i} missing key '{key}'")
        has_box = "box" in r and isinstance(r["box"], list) and len(r["box"]) == 4
        has_xywh = all(k in r for k in ("x", "y", "w", "h"))
        if not (has_box or has_xywh):
            raise ContractValidationError(f"Text region box at index {i} must specify bounding box")


# -------------------------------------------------------------------------
# 5. Final Coords Contract: final_coords_916.json & final_coords_11.json (Phase 4)
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
    aspect_ratio: Literal["9:16", "1:1", "916", "11"]
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
    has_root_dims = "target_width" in data and "target_height" in data
    has_meta_dims = "meta" in data and "crop_width" in data["meta"] and "crop_height" in data["meta"]
    if not (has_root_dims or has_meta_dims):
        raise ContractValidationError("Final coords missing required crop target dimensions")
    if "frames" not in data or not isinstance(data["frames"], list):
        raise ContractValidationError("Final coords must contain a 'frames' list")
    if len(data["frames"]) > 0:
        f0 = data["frames"][0]
        if not isinstance(f0, dict):
            raise ContractValidationError("Final coords frame item must be a dict")
        for fkey in ("t", "crop_x", "crop_y"):
            if fkey not in f0:
                raise ContractValidationError(f"Final coords frame item missing key '{fkey}'")


CONTRACT_VALIDATORS = {
    "transcript.json": validate_transcript,
    "focus_timeline.json": validate_focus_timeline,
    "raw_coords.json": validate_raw_coords,
    "text_regions.json": validate_text_regions,
    "final_coords_916.json": validate_final_coords,
    "final_coords_11.json": validate_final_coords,
}

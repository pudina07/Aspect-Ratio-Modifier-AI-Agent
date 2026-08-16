"""
pipeline/ocr_pass.py — Phase 3, Step 6: EasyOCR Protected Text Regions

Contract:
  Input:  video.mp4
  Output: text_regions.json

Runs EasyOCR (English detector + recognizer) over every 8th frame,
axis-aligns detected text bounding boxes, and links boxes across consecutive
sampled frames by IoU overlap. This ensures on-screen captions, lower thirds,
and graphics spanning multiple seconds become consolidated protected_regions [t_start, t_end]
so smooth_coords.py (Phase 4) can clamp crop windows to prevent text truncation.

Deliberately independent of transcript.json / focus_timeline.json so it runs
concurrently with the speech/transcription branch in pipeline_runner.py.

Supports --mock flag for fast architectural testing.
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import stage_path, MODELS_DIR, PROJECT_ROOT                     # noqa: E402
from contracts import TextRegionsData, TextRegion, validate_text_regions     # noqa: E402
from utils.io_json import save_json, fail_stage                              # noqa: E402

STAGE_NAME = "ocr_pass"

SAMPLE_RATE = 8               # Step 6: sample every 8th frame
MIN_CONFIDENCE = 0.35         # drop low-confidence OCR noise
IOU_LINK_THRESHOLD = 0.20     # overlap required to track continuous on-screen text


def _resolve_easyocr_dir() -> Path:
    """Resolves local EasyOCR models directory."""
    candidates = [
        MODELS_DIR / "easyocr",
        PROJECT_ROOT / "models" / "easyocr",
        MODELS_DIR,
        PROJECT_ROOT / "models"
    ]
    for c in candidates:
        if c.exists() and (c / "craft_mlt_25k.pth").exists():
            return c
    target = MODELS_DIR / "easyocr"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _quad_to_box(quad) -> Tuple[float, float, float, float]:
    """Convert 4-corner quad to axis-aligned bounding box (x, y, w, h)."""
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    x, y = min(xs), min(ys)
    return float(x), float(y), float(max(xs) - x), float(max(ys) - y)


def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    """Compute Intersection over Union between two bounding boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = (aw * ah) + (bw * bh) - inter
    return inter / union if union > 0 else 0.0


def _union_box(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """Expand bounding box to encompass union of both detections."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = min(ax, bx), min(ay, by)
    x2, y2 = max(ax + aw, bx + bw), max(ay + ah, by + bh)
    return float(x1), float(y1), float(x2 - x1), float(y2 - y1)


def generate_mock_text_regions(video_path: Path) -> dict:
    """Generate mock text regions conforming to TextRegionsData contract."""
    regions = [
        TextRegion(
            t_start=6.8,
            t_end=10.37,
            box=[1380.0, 70.0, 500.0, 60.0],
            text="CRITICAL: 95% RETENTION",
            confidence=0.97
        ),
        TextRegion(
            t_start=6.8,
            t_end=10.37,
            box=[40.0, 960.0, 540.0, 50.0],
            text="KEY TAKEAWAY: SUBSCRIBE NOW",
            confidence=0.96
        )
    ]
    data = TextRegionsData(
        fps=30.0,
        width=1920,
        height=1080,
        regions=regions
    )
    res = data.to_dict()
    for r in res.get("regions", []):
        b = r.get("box", [0, 0, 0, 0])
        r["x"] = b[0]
        r["y"] = b[1]
        r["w"] = b[2]
        r["h"] = b[3]
    return res


def run(
    video_path: Path,
    sample_rate: int = SAMPLE_RATE,
    min_confidence: float = MIN_CONFIDENCE,
    gpu: bool = False,
    mock: bool = False
) -> dict:
    """
    Executes EasyOCR text extraction and temporal IoU tracking across sampled video frames.
    """
    if mock:
        return generate_mock_text_regions(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    import cv2
    import easyocr

    ocr_dir = _resolve_easyocr_dir()
    print(f"[{STAGE_NAME}] Initializing EasyOCR Reader from '{ocr_dir}' (gpu={gpu})...")

    reader = easyocr.Reader(
        ["en"],
        gpu=gpu,
        model_storage_directory=str(ocr_dir),
        download_enabled=False,
        verbose=False
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV failed to open video at {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if fps <= 0:
        fps = 30.0

    finished_regions: List[Dict[str, Any]] = []
    active: List[Dict[str, Any]] = []

    frame_idx = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            if frame_idx % sample_rate == 0:
                t = round(frame_idx / fps, 3)
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                detections = reader.readtext(rgb)

                boxes_with_meta = [
                    (_quad_to_box(quad), text, float(conf))
                    for quad, text, conf in detections
                    if conf >= min_confidence
                ]

                matched = set()
                for box, text, conf in boxes_with_meta:
                    best_i = None
                    best_iou = 0.0
                    for i, region in enumerate(active):
                        if i in matched:
                            continue
                        iou = _iou(region["box"], box)
                        if iou > best_iou:
                            best_i, best_iou = i, iou

                    if best_i is not None and best_iou >= IOU_LINK_THRESHOLD:
                        active[best_i]["box"] = _union_box(active[best_i]["box"], box)
                        active[best_i]["t_end"] = t
                        active[best_i]["conf"] = max(active[best_i]["conf"], conf)
                        if text and len(text) > len(active[best_i]["text"]):
                            active[best_i]["text"] = text
                        matched.add(best_i)
                    else:
                        active.append({
                            "box": box,
                            "t_start": t,
                            "t_end": t,
                            "text": text,
                            "conf": conf
                        })
                        matched.add(len(active) - 1)

                still_active = []
                for i, region in enumerate(active):
                    if i in matched:
                        still_active.append(region)
                    else:
                        finished_regions.append(region)
                active = still_active

            frame_idx += 1
    finally:
        cap.release()

    finished_regions.extend(active)

    region_objs: List[TextRegion] = []
    for r in finished_regions:
        bx = r["box"]
        region_objs.append(
            TextRegion(
                t_start=r["t_start"],
                t_end=r["t_end"],
                box=[round(bx[0], 1), round(bx[1], 1), round(bx[2], 1), round(bx[3], 1)],
                text=r.get("text", ""),
                confidence=round(r.get("conf", 1.0), 3)
            )
        )

    region_objs.sort(key=lambda r: r.t_start)

    data = TextRegionsData(
        fps=fps,
        width=width,
        height=height,
        regions=region_objs
    )
    result_dict = data.to_dict()
    result_dict["meta"] = {
        "width": width,
        "height": height,
        "fps": fps,
        "total_regions": len(region_objs)
    }
    for r in result_dict.get("regions", []):
        b = r.get("box", [0, 0, 0, 0])
        r["x"] = b[0]
        r["y"] = b[1]
        r["w"] = b[2]
        r["h"] = b[3]

    return result_dict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=stage_path("video.mp4"))
    parser.add_argument("--out", type=Path, default=stage_path("text_regions.json"))
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE,
                        help="Run EasyOCR every Nth frame (Phase 3 Step 6)")
    parser.add_argument("--min-confidence", type=float, default=MIN_CONFIDENCE)
    parser.add_argument("--gpu", action="store_true", help="Enable GPU delegate for EasyOCR")
    parser.add_argument("--mock", action="store_true", help="Generate mock text regions for testing")
    args = parser.parse_args()

    try:
        text_regions = run(
            video_path=args.video,
            sample_rate=args.sample_rate,
            min_confidence=args.min_confidence,
            gpu=args.gpu,
            mock=args.mock
        )
        save_json(args.out, text_regions, validator=validate_text_regions)
        region_count = len(text_regions.get("regions", []))
        print(f"[{STAGE_NAME}] successfully wrote {args.out} ({region_count} protected text regions)")
    except Exception as e:
        fail_stage(STAGE_NAME, e)


if __name__ == "__main__":
    main()

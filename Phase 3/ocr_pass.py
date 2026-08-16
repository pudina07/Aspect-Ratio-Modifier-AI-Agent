"""
pipeline/ocr_pass.py — Phase 3, Step 6

Contract:  video.mp4  ->  text_regions.json

Runs EasyOCR (English detector+recognizer) over every 8th frame,
axis-aligns each detected text box, and links boxes across consecutive
sampled frames by IoU overlap — so a caption that's on screen for
several seconds becomes one protected_region spanning
[t_start, t_end] instead of a new, disconnected box every sample.
smooth_coords.py (Phase 4) uses these regions to nudge the crop so
on-screen text doesn't get cut off.

This stage only needs video.mp4 — it does NOT depend on transcript.json
or focus_timeline.json. That's deliberate: it lets pipeline_runner.py
run this concurrently with the transcribe -> analyze_script -> tracker
chain instead of waiting behind it (see config.py's PIPELINE_STAGES).

EasyOCR's English weights are expected pre-downloaded into
MODELS_DIR/easyocr during Phase 0 setup. This stage runs with
download_enabled=False so a missing model fails fast with a clear
message instead of silently attempting (and stalling on) a
multi-hundred-MB download mid-pipeline.
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import stage_path, MODELS_DIR           # noqa: E402
from utils.io_json import save_json, fail_stage     # noqa: E402

STAGE_NAME = "ocr_pass"

SAMPLE_RATE = 8               # Step 6: sample every 8th frame
MIN_CONFIDENCE = 0.4          # drop low-confidence OCR noise before it becomes a protected zone
IOU_LINK_THRESHOLD = 0.2      # overlap required to treat a box as "same text, next sample"
EASYOCR_MODEL_DIR = MODELS_DIR / "easyocr"


def _check_models_present():
    if not EASYOCR_MODEL_DIR.exists() or not any(EASYOCR_MODEL_DIR.iterdir()):
        raise FileNotFoundError(
            f"No EasyOCR weights found in {EASYOCR_MODEL_DIR}. "
            f"Download the English detector+recognizer weights during "
            f"Phase 0 setup before running ocr_pass.py."
        )


def _quad_to_box(quad) -> tuple[float, float, float, float]:
    """EasyOCR returns each detection's box as 4 [x, y] corner points
    (not necessarily axis-aligned — text can be slightly rotated/skewed).
    Collapse to the axis-aligned (x, y, w, h) box smooth_coords.py needs."""
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    x, y = min(xs), min(ys)
    return x, y, max(xs) - x, max(ys) - y


def _iou(a: tuple, b: tuple) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _union_box(a: tuple, b: tuple) -> tuple:
    """Expand rather than replace on a match — a caption box that drifts a
    few px between OCR samples (or gets re-detected slightly differently)
    should still be fully covered by the protected region, not just its
    most recent snapshot."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = min(ax, bx), min(ay, by)
    x2, y2 = max(ax + aw, bx + bw), max(ay + ah, by + bh)
    return x1, y1, x2 - x1, y2 - y1


def run(video_path: Path, sample_rate: int = SAMPLE_RATE,
        min_confidence: float = MIN_CONFIDENCE, gpu: bool = False) -> dict:
    """
    Returns:
        {"meta": {"width": 1920, "height": 1080, "fps": 30.0},
         "regions": [
            {"t_start": 2.0, "t_end": 5.5, "x": 40, "y": 800, "w": 600, "h": 90},
            ...
        ]}
    """
    import cv2
    import easyocr

    if not video_path.exists():
        raise FileNotFoundError(f"No video at {video_path}")
    _check_models_present()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    reader = easyocr.Reader(
        ["en"], gpu=gpu,
        model_storage_directory=str(EASYOCR_MODEL_DIR),
        download_enabled=False,
    )

    finished_regions = []
    active = []  # each: {"box": (x, y, w, h), "t_start": t, "t_end": t}

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

                boxes = [
                    _quad_to_box(quad) for quad, _text, conf in detections
                    if conf >= min_confidence
                ]

                matched = set()
                for box in boxes:
                    best_i, best_iou = None, 0.0
                    for i, region in enumerate(active):
                        if i in matched:
                            continue
                        iou = _iou(region["box"], box)
                        if iou > best_iou:
                            best_i, best_iou = i, iou

                    if best_i is not None and best_iou >= IOU_LINK_THRESHOLD:
                        active[best_i]["box"] = _union_box(active[best_i]["box"], box)
                        active[best_i]["t_end"] = t
                        matched.add(best_i)
                    else:
                        active.append({"box": box, "t_start": t, "t_end": t})
                        matched.add(len(active) - 1)

                # Anything not matched this sample has stopped appearing on screen.
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

    finished_regions.extend(active)  # close out whatever was still on screen at video end

    regions = [
        {
            "t_start": r["t_start"], "t_end": r["t_end"],
            "x": round(r["box"][0], 1), "y": round(r["box"][1], 1),
            "w": round(r["box"][2], 1), "h": round(r["box"][3], 1),
        }
        for r in finished_regions
    ]
    regions.sort(key=lambda r: r["t_start"])

    return {"meta": {"width": width, "height": height, "fps": fps}, "regions": regions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=stage_path("video.mp4"))
    parser.add_argument("--out", type=Path, default=stage_path("text_regions.json"))
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE,
                         help="Run EasyOCR every Nth frame (Phase 3 Step 6).")
    parser.add_argument("--min-confidence", type=float, default=MIN_CONFIDENCE)
    parser.add_argument("--gpu", action="store_true", help="Use GPU for EasyOCR if available.")
    args = parser.parse_args()

    try:
        text_regions = run(args.video, args.sample_rate, args.min_confidence, args.gpu)
        save_json(args.out, text_regions)
        print(f"[{STAGE_NAME}] wrote {args.out} ({len(text_regions['regions'])} protected regions)")
    except Exception as e:
        fail_stage(STAGE_NAME, e)


if __name__ == "__main__":
    main()

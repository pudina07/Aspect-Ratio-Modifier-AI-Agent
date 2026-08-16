"""
pipeline/render.py — Phase 5, Step 11: Platform-Aware Video Rendering & Compositing

Contract:
  Inputs:  video.mp4, final_coords_916.json, final_coords_11.json
  Outputs: output_916.mp4, output_11.mp4

For each source frame:
  1. Builds a blurred, scaled full-bleed backdrop at the target platform canvas size
     (1080x1920 for 9:16 and 1080x1080 for 1:1).
  2. Crops the sharp foreground region per final_coords_*.json (crop_x, crop_y, crop_w, crop_h)
     and scales it to fill the canvas.
  3. Composites the foreground centered over the backdrop.
  4. Writes frame-by-frame via cv2.VideoWriter in a single decode pass across video.mp4.
  5. Muxes original audio track back in via FFmpeg (-c:v copy -c:a aac).

Safe-Zone Presets:
  Configured in safe_zones.json. Pass --qa-overlay <preset_name> (e.g. tiktok_9x16, reels_9x16, feed_1x1)
  to render an additional QA preview with safe-zone margin guides drawn in.

Supports --mock flag for fast architectural test execution.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import stage_path, SAFE_ZONES_PATH, PROJECT_ROOT, load_safe_zones  # noqa: E402
from contracts import validate_final_coords                                    # noqa: E402
from utils.io_json import load_json, fail_stage                                # noqa: E402

STAGE_NAME = "render"

OUTPUT_CANVAS = {
    "9:16": (1080, 1920),
    "916": (1080, 1920),
    "1:1": (1080, 1080),
    "11": (1080, 1080)
}

GAUSSIAN_BLUR_KERNEL = 61   # Must be odd; obscures backdrop detail
OVERLAY_ALPHA = 0.35        # QA safe-zone margin guide opacity


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


def _make_background(frame_bgr, canvas_w: int, canvas_h: int):
    """Full-bleed blurred backdrop: scale entire source frame to cover canvas and blur."""
    import cv2

    src_h, src_w = frame_bgr.shape[:2]
    scale = max(canvas_w / src_w, canvas_h / src_h)
    scaled_w = max(canvas_w, int(round(src_w * scale)))
    scaled_h = max(canvas_h, int(round(src_h * scale)))
    scaled = cv2.resize(frame_bgr, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

    x0 = max(0, (scaled_w - canvas_w) // 2)
    y0 = max(0, (scaled_h - canvas_h) // 2)
    cropped = scaled[y0:y0 + canvas_h, x0:x0 + canvas_w]

    # Ensure blur kernel is odd and <= dimensions
    k = GAUSSIAN_BLUR_KERNEL
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(cropped, (k, k), 0)


def _make_foreground(frame_bgr, crop_x: int, crop_y: int, crop_w: int, crop_h: int,
                      canvas_w: int, canvas_h: int):
    """The sharp crop, scaled to fill canvas dimensions."""
    import cv2

    src_h, src_w = frame_bgr.shape[:2]
    w = min(max(int(crop_w), 1), src_w)
    h = min(max(int(crop_h), 1), src_h)
    x0 = min(max(int(crop_x), 0), max(0, src_w - w))
    y0 = min(max(int(crop_y), 0), max(0, src_h - h))

    crop = frame_bgr[y0:y0 + h, x0:x0 + w]
    return cv2.resize(crop, (canvas_w, canvas_h), interpolation=cv2.INTER_LINEAR)


def _composite(background, foreground):
    """Center foreground over background."""
    canvas = background.copy()
    fg_h, fg_w = foreground.shape[:2]
    bg_h, bg_w = background.shape[:2]
    x0 = max(0, (bg_w - fg_w) // 2)
    y0 = max(0, (bg_h - fg_h) // 2)
    canvas[y0:y0 + fg_h, x0:x0 + fg_w] = foreground[:min(fg_h, bg_h - y0), :min(fg_w, bg_w - x0)]
    return canvas


def _draw_safe_zone(canvas, preset: dict):
    """Draws semi-transparent safe-zone margin rectangle for QA preview."""
    import cv2

    h, w = canvas.shape[:2]

    # Support multiple preset schema shapes
    margins = preset.get("margins", {})
    top = preset.get("top_clear_px", margins.get("top", 0))
    bottom = preset.get("bottom_clear_px", margins.get("bottom"))
    if bottom is None:
        pct = preset.get("bottom_clear_pct", margins.get("bottom_pct", 0.0))
        bottom = int(round(pct * h))
    left = preset.get("left_clear_px", margins.get("left", 0))
    right = preset.get("right_clear_px", margins.get("right", 0))

    overlay = canvas.copy()
    color = (0, 0, 255)  # BGR red

    if top > 0:
        cv2.rectangle(overlay, (0, 0), (w, int(top)), color, -1)
    if bottom > 0:
        cv2.rectangle(overlay, (0, h - int(bottom)), (w, h), color, -1)
    if left > 0:
        cv2.rectangle(overlay, (0, 0), (int(left), h), color, -1)
    if right > 0:
        cv2.rectangle(overlay, (w - int(right), 0), (w, h), color, -1)

    return cv2.addWeighted(overlay, OVERLAY_ALPHA, canvas, 1.0 - OVERLAY_ALPHA, 0)


def _new_writer(out_path: Path, canvas_w: int, canvas_h: int, fps: float):
    """Creates a temporary VideoWriter for video stream only."""
    import cv2

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.stem + f"_temp_video_{canvas_w}x{canvas_h}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp_path), fourcc, fps, (canvas_w, canvas_h))
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter failed to open for {tmp_path}")
    return writer, tmp_path


def _mux_audio(video_only_path: Path, source_video: Path, final_path: Path) -> None:
    """Muxes audio from source_video into video_only_path via FFmpeg."""
    ffmpeg_exe = _get_ffmpeg_binary()
    final_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg_exe, "-y",
        "-i", str(video_only_path),
        "-i", str(source_video),
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "copy", "-c:a", "aac",
        "-shortest",
        str(final_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        # Fallback if mux failed: copy video_only to final_path
        print(f"[{STAGE_NAME}] Warning: FFmpeg audio mux returned non-zero code ({proc.stderr.strip()}). Staging video stream directly.", file=sys.stderr)
        shutil.copyfile(str(video_only_path), str(final_path))


def _extract_track_params(coords: dict) -> Tuple[int, int, float, int, List[dict]]:
    """Extracts width, height, fps, total_frames, and frames list with schema normalization."""
    meta = coords.get("meta", {})
    crop_w = int(coords.get("target_width", meta.get("crop_width", 608)))
    crop_h = int(coords.get("target_height", meta.get("crop_height", 1080)))
    fps = float(coords.get("fps", meta.get("fps", 30.0)))
    frames = coords.get("frames", [])
    total_frames = int(coords.get("total_frames", meta.get("frame_count", len(frames))))
    if total_frames <= 0:
        total_frames = len(frames)
    return crop_w, crop_h, fps, total_frames, frames


def _render_all(video_path: Path, jobs: List[dict], fps: float, expected_frames: int) -> None:
    """Single decode pass over video.mp4 feeding all render jobs."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video at {video_path}")

    actual_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    if actual_fps <= 0:
        actual_fps = fps

    for job in jobs:
        job["writer"], job["tmp_path"] = _new_writer(
            job["out_path"], job["canvas_w"], job["canvas_h"], actual_fps
        )

    frame_idx = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            for job in jobs:
                frames_list = job["frames"]
                if frame_idx < len(frames_list):
                    frame_info = frames_list[frame_idx]
                elif frames_list:
                    frame_info = frames_list[-1]
                else:
                    frame_info = {"crop_x": 0, "crop_y": 0}

                crop_x = frame_info.get("crop_x", 0)
                crop_y = frame_info.get("crop_y", 0)
                crop_w = frame_info.get("crop_w", job["crop_w"])
                crop_h = frame_info.get("crop_h", job["crop_h"])

                background = _make_background(frame_bgr, job["canvas_w"], job["canvas_h"])
                foreground = _make_foreground(
                    frame_bgr, crop_x, crop_y,
                    crop_w, crop_h, job["canvas_w"], job["canvas_h"]
                )
                canvas = _composite(background, foreground)

                if job.get("qa_preset") is not None:
                    canvas = _draw_safe_zone(canvas, job["qa_preset"])

                job["writer"].write(canvas)

            frame_idx += 1
    finally:
        cap.release()
        for job in jobs:
            if "writer" in job and job["writer"] is not None:
                job["writer"].release()

    for job in jobs:
        _mux_audio(job["tmp_path"], video_path, job["out_path"])
        if job["tmp_path"].exists():
            try:
                job["tmp_path"].unlink()
            except Exception:
                pass


def generate_mock_rendered_videos(
    video_path: Path, coords_916: dict, coords_11: dict, out_916: Path, out_11: Path,
    qa_overlay: Optional[str] = None
) -> None:
    """Generate lightweight valid test video deliverables for testing."""
    import cv2
    import numpy as np

    fps = float(coords_916.get("fps", 30.0))
    total_frames = 30
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # 1. 9:16 Output (1080x1920)
    out_916.parent.mkdir(parents=True, exist_ok=True)
    writer_916 = cv2.VideoWriter(str(out_916), fourcc, fps, (1080, 1920))
    for i in range(total_frames):
        frame = np.zeros((1920, 1080, 3), dtype=np.uint8)
        frame[:, :] = (35, 20, 20)
        cv2.putText(frame, "CONTEXT-AWARE AUTO-REFRAME (9:16)", (80, 900), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 200), 2)
        cv2.putText(frame, f"Delivery 1080x1920 | Frame {i+1}/{total_frames}", (80, 980), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        writer_916.write(frame)
    writer_916.release()

    # 2. 1:1 Output (1080x1080)
    out_11.parent.mkdir(parents=True, exist_ok=True)
    writer_11 = cv2.VideoWriter(str(out_11), fourcc, fps, (1080, 1080))
    for i in range(total_frames):
        frame = np.zeros((1080, 1080, 3), dtype=np.uint8)
        frame[:, :] = (20, 35, 20)
        cv2.putText(frame, "CONTEXT-AWARE AUTO-REFRAME (1:1)", (100, 500), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 2)
        cv2.putText(frame, f"Delivery 1080x1080 | Frame {i+1}/{total_frames}", (100, 580), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        writer_11.write(frame)
    writer_11.release()

    # 3. Optional QA overlay
    if qa_overlay:
        qa_out = out_916.with_name(f"{out_916.stem}_qa_{qa_overlay}{out_916.suffix}")
        writer_qa = cv2.VideoWriter(str(qa_out), fourcc, fps, (1080, 1920))
        for i in range(total_frames):
            frame = np.zeros((1920, 1080, 3), dtype=np.uint8)
            frame[:, :] = (35, 20, 20)
            cv2.rectangle(frame, (0, 0), (1080, 130), (0, 0, 255), -1)
            cv2.rectangle(frame, (0, 1920 - 380), (1080, 1920), (0, 0, 255), -1)
            cv2.putText(frame, f"QA OVERLAY: {qa_overlay}", (80, 960), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
            writer_qa.write(frame)
        writer_qa.release()


def run(
    video_path: Path,
    coords_916: dict,
    coords_11: dict,
    out_916: Path,
    out_11: Path,
    qa_overlay: Optional[str] = None,
    mock: bool = False
) -> None:
    """
    Executes Phase 5 Platform-Aware Video Rendering & Compositing.
    """
    if mock:
        generate_mock_rendered_videos(video_path, coords_916, coords_11, out_916, out_11, qa_overlay)
        return

    if not video_path.exists():
        raise FileNotFoundError(f"Source video not found at: {video_path}")

    crop_w_916, crop_h_916, fps, count_916, frames_916 = _extract_track_params(coords_916)
    crop_w_11, crop_h_11, _, count_11, frames_11 = _extract_track_params(coords_11)

    canvas_w_916, canvas_h_916 = OUTPUT_CANVAS["9:16"]
    canvas_w_11, canvas_h_11 = OUTPUT_CANVAS["1:1"]

    jobs = [
        {
            "name": "9:16 Output",
            "frames": frames_916,
            "crop_w": crop_w_916,
            "crop_h": crop_h_916,
            "canvas_w": canvas_w_916,
            "canvas_h": canvas_h_916,
            "out_path": out_916,
            "qa_preset": None
        },
        {
            "name": "1:1 Output",
            "frames": frames_11,
            "crop_w": crop_w_11,
            "crop_h": crop_h_11,
            "canvas_w": canvas_w_11,
            "canvas_h": canvas_h_11,
            "out_path": out_11,
            "qa_preset": None
        }
    ]

    if qa_overlay:
        safe_zones = load_safe_zones()
        if qa_overlay not in safe_zones:
            # Check normalized keys
            alt_key = qa_overlay.replace(":", "").replace("_", "")
            found_k = None
            for k in safe_zones:
                if k.replace(":", "").replace("_", "") == alt_key:
                    found_k = k
                    break
            if found_k:
                preset = safe_zones[found_k]
            else:
                raise KeyError(f"Preset '{qa_overlay}' not found in safe_zones.json. Available: {list(safe_zones.keys())}")
        else:
            preset = safe_zones[qa_overlay]

        asp = preset.get("aspect", preset.get("aspect_ratio", "916"))
        base_job = jobs[0] if "9" in str(asp) else jobs[1]
        qa_out = base_job["out_path"].with_name(
            f"{base_job['out_path'].stem}_qa_{qa_overlay}{base_job['out_path'].suffix}"
        )
        qa_job = {
            "name": f"QA Overlay ({qa_overlay})",
            "frames": base_job["frames"],
            "crop_w": base_job["crop_w"],
            "crop_h": base_job["crop_h"],
            "canvas_w": base_job["canvas_w"],
            "canvas_h": base_job["canvas_h"],
            "out_path": qa_out,
            "qa_preset": preset
        }
        jobs.append(qa_job)

    print(f"[{STAGE_NAME}] Rendering {len(jobs)} output streams (9:16 at {canvas_w_916}x{canvas_h_916}, 1:1 at {canvas_w_11}x{canvas_h_11})...")
    _render_all(video_path, jobs, fps, max(count_916, count_11))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=stage_path("video.mp4"))
    parser.add_argument("--coords-916", type=Path, default=stage_path("final_coords_916.json"))
    parser.add_argument("--coords-11", type=Path, default=stage_path("final_coords_11.json"))
    parser.add_argument("--out-916", type=Path, default=stage_path("output_916.mp4"))
    parser.add_argument("--out-11", type=Path, default=stage_path("output_11.mp4"))
    parser.add_argument("--qa-overlay", default=None,
                        help="Render QA preview with safe-zone guides (e.g. tiktok_9x16, reels_9x16, feed_1x1)")
    parser.add_argument("--mock", action="store_true", help="Generate mock rendered videos for testing")
    args = parser.parse_args()

    try:
        coords_916 = load_json(args.coords_916, validator=validate_final_coords)
        coords_11 = load_json(args.coords_11, validator=validate_final_coords)
        run(
            video_path=args.video,
            coords_916=coords_916,
            coords_11=coords_11,
            out_916=args.out_916,
            out_11=args.out_11,
            qa_overlay=args.qa_overlay,
            mock=args.mock
        )
        print(f"[{STAGE_NAME}] successfully rendered {args.out_916} and {args.out_11}")
    except Exception as e:
        fail_stage(STAGE_NAME, e)


if __name__ == "__main__":
    main()

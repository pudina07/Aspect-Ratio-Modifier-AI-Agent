"""
End-to-End Phase 0 Verification Script for Context-Aware Auto-Reframe.
Verifies all models, dependencies, FFmpeg, safe-zones config, and the 3-segment test clip.
"""
import os
import sys
import json
import time
from pathlib import Path

# Force UTF-8 encoding for Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent

def test_dependencies():
    print("\n[1/6] Testing Python Dependencies...")
    import cv2
    import mediapipe as mp
    import easyocr
    import faster_whisper
    import streamlit
    import torch
    import imageio_ffmpeg

    print(f"  ✓ OpenCV: {cv2.__version__}")
    print(f"  ✓ MediaPipe: {mp.__version__}")
    print(f"  ✓ EasyOCR: {easyocr.__version__}")
    print(f"  ✓ Faster-Whisper: {faster_whisper.__version__}")
    print(f"  ✓ Streamlit: {streamlit.__version__}")
    print(f"  ✓ PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})")
    print(f"  ✓ imageio-ffmpeg: {imageio_ffmpeg.__version__}")
    return True

def test_models_exist():
    print("\n[2/6] Checking Preloaded Model Weights...")
    expected_files = [
        ("MediaPipe Pose Full", BASE_DIR / "models" / "mediapipe" / "pose_landmarker_full.task", 5 * 1024 * 1024),
        ("MediaPipe Pose Lite", BASE_DIR / "models" / "mediapipe" / "pose_landmarker_lite.task", 3 * 1024 * 1024),
        ("MediaPipe Hand", BASE_DIR / "models" / "mediapipe" / "hand_landmarker.task", 5 * 1024 * 1024),
        ("MediaPipe Face Detector", BASE_DIR / "models" / "mediapipe" / "blaze_face_short_range.tflite", 100 * 1024),
        ("EasyOCR Craft MLT", BASE_DIR / "models" / "easyocr" / "craft_mlt_25k.pth", 50 * 1024 * 1024),
        ("EasyOCR English G2", BASE_DIR / "models" / "easyocr" / "english_g2.pth", 10 * 1024 * 1024),
        ("Whisper Base Config", BASE_DIR / "models" / "whisper" / "base" / "config.json", 100),
        ("Whisper distil-large-v3 Config", BASE_DIR / "models" / "whisper" / "distil-large-v3" / "config.json", 100)
    ]
    
    for name, path, min_size in expected_files:
        if not path.exists():
            print(f"  ✗ MISSING: {name} at {path}")
            return False
        size = path.stat().st_size
        if size < min_size:
            print(f"  ✗ CORRUPTED / TOO SMALL: {name} ({size} bytes)")
            return False
        print(f"  ✓ {name}: {size / (1024*1024):.2f} MB")
    return True

def test_ffmpeg():
    print("\n[3/6] Testing FFmpeg Binary and Muxing Support...")
    from ffmpeg_utils import get_ffmpeg_exe, run_ffmpeg_command
    exe = get_ffmpeg_exe()
    print(f"  ✓ Resolved FFmpeg: {exe}")
    res = run_ffmpeg_command(["-version"])
    version_line = res.stdout.splitlines()[0] if res.stdout else "Unknown"
    print(f"  ✓ Version: {version_line}")
    return True

def test_safe_zones():
    print("\n[4/6] Validating Safe Zones Configuration...")
    sz_path = BASE_DIR / "safe_zones.json"
    if not sz_path.exists():
        print(f"  ✗ MISSING: {sz_path}")
        return False
    with open(sz_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    platforms = data.get("platforms", {})
    required = ["tiktok_916", "instagram_reels_916", "instagram_feed_11"]
    for p in required:
        if p not in platforms:
            print(f"  ✗ MISSING platform preset: {p}")
            return False
        cfg = platforms[p]
        m = cfg["margins"]
        print(f"  ✓ Platform '{cfg['name']}' ({cfg['aspect_ratio']}): Canvas {cfg['canvas_width']}x{cfg['canvas_height']}, Margins T={m['top']} B={m['bottom']} L={m['left']} R={m['right']}")
    return True

def test_test_clip():
    print("\n[5/6] Validating Deliberate Test Video Clip...")
    clip_path = BASE_DIR / "assets" / "test_clip_16_9.mp4"
    if not clip_path.exists():
        print(f"  ✗ MISSING: {clip_path}")
        return False
    
    import cv2
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        print(f"  ✗ Failed to open video: {clip_path}")
        return False
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frames / fps if fps > 0 else 0
    cap.release()
    
    print(f"  ✓ Resolution: {width}x{height} (Aspect: {width/height:.3f})")
    print(f"  ✓ FPS: {fps}, Total Frames: {frames}, Duration: {duration:.2f}s")
    
    if width != 1920 or height != 1080:
        print(f"  ✗ Expected 1920x1080, got {width}x{height}")
        return False
    return True

def test_model_inference():
    print("\n[6/6] Smoketesting Model Inferences on Test Clip Frames...")
    import cv2
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python import BaseOptions
    import mediapipe as mp
    import easyocr
    from faster_whisper import WhisperModel

    clip_path = BASE_DIR / "assets" / "test_clip_16_9.mp4"
    cap = cv2.VideoCapture(str(clip_path))

    # Frame 30 (Segment 1 - Face check)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
    ret, frame_face = cap.read()
    assert ret, "Failed to read frame 30"
    
    # Test MediaPipe FaceDetector
    face_model_path = BASE_DIR / "models" / "mediapipe" / "blaze_face_short_range.tflite"
    face_opts = vision.FaceDetectorOptions(base_options=BaseOptions(model_asset_path=str(face_model_path)))
    face_detector = vision.FaceDetector.create_from_options(face_opts)
    rgb_face = cv2.cvtColor(frame_face, cv2.COLOR_BGR2RGB)
    mp_img_face = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_face)
    face_res = face_detector.detect(mp_img_face)
    print(f"  ✓ Face Detector: Detected {len(face_res.detections)} face(s) on frame 30")

    # Frame 150 (Segment 2 - Pointing gesture check)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 150)
    ret, frame_point = cap.read()
    assert ret, "Failed to read frame 150"

    # Test MediaPipe PoseLandmarker
    pose_model_path = BASE_DIR / "models" / "mediapipe" / "pose_landmarker_full.task"
    pose_opts = vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(pose_model_path)))
    pose_landmarker = vision.PoseLandmarker.create_from_options(pose_opts)
    rgb_point = cv2.cvtColor(frame_point, cv2.COLOR_BGR2RGB)
    mp_img_point = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_point)
    pose_res = pose_landmarker.detect(mp_img_point)
    print(f"  ✓ Pose Landmarker: Processed frame 150 successfully")

    # Frame 270 (Segment 3 - Text protection check)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 270)
    ret, frame_text = cap.read()
    assert ret, "Failed to read frame 270"
    cap.release()

    # Test EasyOCR
    easyocr_dir = BASE_DIR / "models" / "easyocr"
    reader = easyocr.Reader(['en'], gpu=False, model_storage_directory=str(easyocr_dir), download_enabled=False, verbose=False)
    ocr_res = reader.readtext(frame_text)
    detected_texts = [r[1] for r in ocr_res]
    print(f"  ✓ EasyOCR: Found {len(detected_texts)} text region(s) on frame 270: {detected_texts}")

    # Test Faster-Whisper on speech.wav
    whisper_dir = BASE_DIR / "models" / "whisper" / "base"
    whisper_model = WhisperModel(str(whisper_dir), device="cpu", compute_type="int8")
    audio_path = BASE_DIR / "assets" / "speech.wav"
    segments, info = whisper_model.transcribe(str(audio_path), word_timestamps=True)
    seg_list = list(segments)
    print(f"  ✓ Faster-Whisper: Transcribed {len(seg_list)} segment(s) from audio track:")
    for s in seg_list:
        print(f"      [{s.start:.1f}s - {s.end:.1f}s]: \"{s.text.strip()}\"")

    return True

def main():
    print("=" * 60)
    print("      CONTEXT-AWARE AUTO-REFRAME: PHASE 0 VERIFICATION      ")
    print("=" * 60)

    start_time = time.time()
    checks = [
        ("Python Dependencies", test_dependencies),
        ("Preloaded Model Weights", test_models_exist),
        ("FFmpeg Integration", test_ffmpeg),
        ("Safe Zones Configuration", test_safe_zones),
        ("Test Clip Integrity", test_test_clip),
        ("AI Models Inference Smoketest", test_model_inference)
    ]

    all_passed = True
    for name, fn in checks:
        try:
            passed = fn()
            if not passed:
                all_passed = False
                print(f"\n❌ [FAILED] Check '{name}' failed!")
                break
        except Exception as e:
            all_passed = False
            print(f"\n❌ [ERROR] Check '{name}' threw exception: {e}")
            import traceback
            traceback.print_exc()
            break

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    if all_passed:
        print(f"🎉 ALL PHASE 0 CHECKS PASSED PERFECTLY in {elapsed:.2f}s!")
        print("Setup is 100% complete and ready for Phase 1 & Phase 2!")
    else:
        print("❌ PHASE 0 VERIFICATION FAILED. Review errors above.")
    print("=" * 60)

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())

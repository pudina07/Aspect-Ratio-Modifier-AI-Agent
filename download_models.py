"""
Download and cache all AI models required for the Auto-Reframe pipeline up front.
Ensures zero runtime downloads during processing or live demo.
"""
import os
import sys
import time
import zipfile
import urllib.request
from pathlib import Path

# Force UTF-8 stdout/stderr for Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MEDIAPIPE_DIR = MODELS_DIR / "mediapipe"
WHISPER_DIR = MODELS_DIR / "whisper"
EASYOCR_DIR = MODELS_DIR / "easyocr"

MEDIAPIPE_MODELS = {
    "pose_landmarker_full.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "blaze_face_short_range.tflite": "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
    "face_detector.tflite": "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
}

EASYOCR_MODELS = {
    "craft_mlt_25k.zip": ("https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip", "craft_mlt_25k.pth"),
    "english_g2.zip": ("https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.zip", "english_g2.pth")
}

def download_file(url: str, dest_path: Path, desc: str = ""):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 1024:
        print(f"[EXISTS] {desc or dest_path.name} ({dest_path.stat().st_size / (1024*1024):.2f} MB)")
        return

    print(f"[DOWNLOADING] {desc or dest_path.name} from {url}...")
    start_time = time.time()
    
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)
    
    with urllib.request.urlopen(req) as response, open(dest_path, "wb") as out_file:
        total_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1MB chunks
        
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                percent = (downloaded / total_size) * 100
                print(f"  -> {downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB ({percent:.1f}%)", end="\r", flush=True)
                
    elapsed = time.time() - start_time
    print(f"\n[DONE] Saved {dest_path.name} ({downloaded / (1024*1024):.2f} MB in {elapsed:.1f}s)")

def download_mediapipe_models():
    print("\n--- 1. Downloading MediaPipe Models ---")
    for filename, url in MEDIAPIPE_MODELS.items():
        dest = MEDIAPIPE_DIR / filename
        download_file(url, dest, desc=filename)

def download_easyocr_models():
    print("\n--- 2. Pre-loading EasyOCR Models ---")
    EASYOCR_DIR.mkdir(parents=True, exist_ok=True)
    
    for zip_name, (url, expected_pth) in EASYOCR_MODELS.items():
        target_pth = EASYOCR_DIR / expected_pth
        if target_pth.exists() and target_pth.stat().st_size > 1024:
            print(f"[EXISTS] EasyOCR model {expected_pth} ({target_pth.stat().st_size / (1024*1024):.2f} MB)")
            continue
        
        zip_path = EASYOCR_DIR / zip_name
        download_file(url, zip_path, desc=zip_name)
        print(f"Extracting {zip_name} to {EASYOCR_DIR}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(EASYOCR_DIR)
        if zip_path.exists():
            zip_path.unlink()
        print(f"[DONE] Extracted {expected_pth}")

    # Verify EasyOCR loads with pre-downloaded weights without re-downloading
    import easyocr
    reader = easyocr.Reader(['en'], gpu=False, model_storage_directory=str(EASYOCR_DIR), download_enabled=False, verbose=False)
    print("[VERIFIED] EasyOCR Reader initialized successfully with local weights.")

def download_whisper_models():
    print("\n--- 3. Pre-loading Faster-Whisper Models ---")
    import faster_whisper
    WHISPER_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download base model (fast fallback and testing)
    print("\nDownloading Whisper 'base' model...")
    faster_whisper.download_model("base", output_dir=str(WHISPER_DIR / "base"))
    print("[DONE] Whisper 'base' model ready.")

    # Download distil-large-v3 model (production per plan)
    print("\nDownloading Whisper 'distil-large-v3' model...")
    try:
        faster_whisper.download_model("distil-large-v3", output_dir=str(WHISPER_DIR / "distil-large-v3"))
        print("[DONE] Whisper 'distil-large-v3' model ready.")
    except Exception as e:
        print(f"[WARN] distil-large-v3 note: {e}. Downloading 'small' model...")
        faster_whisper.download_model("small", output_dir=str(WHISPER_DIR / "small"))

def main():
    print("=== Auto-Reframe Phase 0 Model Preloader ===")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    download_mediapipe_models()
    download_easyocr_models()
    download_whisper_models()
    
    print("\n==========================================")
    print("ALL MODELS DOWNLOADED AND CACHED SUCCESSFULLY!")
    print("==========================================")

if __name__ == "__main__":
    main()

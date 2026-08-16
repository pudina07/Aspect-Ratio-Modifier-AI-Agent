"""
Generate the deliberate Phase 0 test clip (1920x1080, 16:9, 30fps) with synchronized audio.
Contains:
1. Segment 1 (0s - 3.2s): Talking-head center segment
2. Segment 2 (3.2s - 6.8s): Pointing gesture & vector toward an off-screen/off-center chart on the right
3. Segment 3 (6.8s - 10.5s): Edge-anchored text captions (top-right & bottom-left) stressing safe zones and OCR
"""
import os
import sys
import math
import subprocess
from pathlib import Path
import cv2
import numpy as np
from ffmpeg_utils import get_ffmpeg_exe, run_ffmpeg_command

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

def generate_speech_audio(wav_path: Path):
    """Generate TTS speech audio using PowerShell System.Speech."""
    ps_script = f"""
    Add-Type -AssemblyName System.Speech
    $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
    $synth.Rate = 0
    $synth.SetOutputToWaveFile('{wav_path.as_posix()}')
    $synth.Speak('Hello and welcome to the auto reframe demonstration. Look at this chart on the right side over here. Notice the key metric in the corner of your screen.')
    $synth.Dispose()
    """
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_script], check=True)
    print(f"[AUDIO] Generated {wav_path}")

def draw_speaker(frame, center_x, center_y, arm_state, mouth_open=0.0):
    """
    Draws a stylized presenter with head, face features, shoulders, torso, and arms.
    arm_state: ('rest') or ('pointing_right', progress 0..1)
    """
    # Studio lighting gradient on person
    body_color = (60, 60, 90)       # Dark slate suit
    shirt_color = (230, 230, 240)   # Light shirt
    skin_color = (180, 210, 240)    # Skin tone (BGR)
    hair_color = (40, 40, 50)       # Dark hair

    # Torso & Shoulders
    torso_top_y = center_y + 110
    torso_pts = np.array([
        [center_x - 170, torso_top_y + 120],  # Left shoulder
        [center_x + 170, torso_top_y + 120],  # Right shoulder
        [center_x + 220, 1080],               # Bottom right
        [center_x - 220, 1080]                # Bottom left
    ], np.int32)
    cv2.fillPoly(frame, [torso_pts], body_color)
    cv2.polylines(frame, [torso_pts], True, (40, 40, 60), 3)

    # Shirt collar (V-neck)
    v_pts = np.array([
        [center_x - 45, torso_top_y + 120],
        [center_x, torso_top_y + 240],
        [center_x + 45, torso_top_y + 120]
    ], np.int32)
    cv2.fillPoly(frame, [v_pts], shirt_color)

    # Neck
    cv2.rectangle(frame, (center_x - 35, center_y + 50), (center_x + 35, center_y + 130), skin_color, -1)

    # Head (oval)
    head_center = (center_x, center_y)
    head_axes = (75, 105)
    cv2.ellipse(frame, head_center, head_axes, 0, 0, 360, skin_color, -1)
    cv2.ellipse(frame, head_center, head_axes, 0, 0, 360, (140, 170, 200), 2)

    # Hair
    hair_pts = np.array([
        [center_x - 80, center_y - 20],
        [center_x - 85, center_y - 80],
        [center_x - 40, center_y - 120],
        [center_x + 40, center_y - 120],
        [center_x + 85, center_y - 80],
        [center_x + 80, center_y - 20],
        [center_x + 60, center_y - 65],
        [center_x - 60, center_y - 65]
    ], np.int32)
    cv2.fillPoly(frame, [hair_pts], hair_color)

    # Eyes & Eyebrows
    eye_y = center_y - 15
    cv2.circle(frame, (center_x - 30, eye_y), 9, (255, 255, 255), -1)
    cv2.circle(frame, (center_x + 30, eye_y), 9, (255, 255, 255), -1)
    cv2.circle(frame, (center_x - 28, eye_y), 4, (50, 40, 30), -1)
    cv2.circle(frame, (center_x + 32, eye_y), 4, (50, 40, 30), -1)
    # Eyebrows
    cv2.line(frame, (center_x - 42, eye_y - 16), (center_x - 18, eye_y - 14), hair_color, 4)
    cv2.line(frame, (center_x + 18, eye_y - 14), (center_x + 42, eye_y - 16), hair_color, 4)

    # Nose
    cv2.line(frame, (center_x, center_y - 5), (center_x - 6, center_y + 20), (140, 160, 190), 3)
    cv2.line(frame, (center_x - 6, center_y + 20), (center_x + 6, center_y + 20), (140, 160, 190), 3)

    # Mouth (animated with speech)
    mouth_y = center_y + 50
    mouth_h = int(6 + 10 * mouth_open)
    cv2.ellipse(frame, (center_x, mouth_y), (22, mouth_h), 0, 0, 360, (80, 80, 180), -1)

    # Arms
    left_shoulder = (center_x - 170, torso_top_y + 120)
    right_shoulder = (center_x + 170, torso_top_y + 120)

    # Left arm rests down
    left_elbow = (center_x - 210, torso_top_y + 300)
    left_wrist = (center_x - 190, torso_top_y + 450)
    cv2.line(frame, left_shoulder, left_elbow, body_color, 36)
    cv2.line(frame, left_elbow, left_wrist, body_color, 30)
    cv2.circle(frame, left_wrist, 18, skin_color, -1)

    # Right arm: rest OR pointing
    if arm_state[0] == "rest":
        right_elbow = (center_x + 210, torso_top_y + 300)
        right_wrist = (center_x + 190, torso_top_y + 450)
        cv2.line(frame, right_shoulder, right_elbow, body_color, 36)
        cv2.line(frame, right_elbow, right_wrist, body_color, 30)
        cv2.circle(frame, right_wrist, 18, skin_color, -1)
    elif arm_state[0] == "pointing_right":
        t = arm_state[1]  # 0 to 1
        # Ease out arm motion
        # Target pointing vector toward chart at (1650, 420)
        rest_elbow = np.array([center_x + 210, torso_top_y + 300])
        target_elbow = np.array([center_x + 360, torso_top_y + 100])
        elbow = (1 - t) * rest_elbow + t * target_elbow

        rest_wrist = np.array([center_x + 190, torso_top_y + 450])
        target_wrist = np.array([center_x + 580, torso_top_y - 20])
        wrist = (1 - t) * rest_wrist + t * target_wrist

        elbow_pt = (int(elbow[0]), int(elbow[1]))
        wrist_pt = (int(wrist[0]), int(wrist[1]))

        cv2.line(frame, right_shoulder, elbow_pt, body_color, 36)
        cv2.line(frame, elbow_pt, wrist_pt, body_color, 30)
        cv2.circle(frame, wrist_pt, 18, skin_color, -1)

        # Extended index finger pointing toward right edge
        if t > 0.4:
            finger_dir = np.array([1.0, -0.2])
            finger_dir /= np.linalg.norm(finger_dir)
            fingertip = wrist + finger_dir * (50 * t)
            cv2.line(frame, wrist_pt, (int(fingertip[0]), int(fingertip[1])), skin_color, 12)
            cv2.circle(frame, (int(fingertip[0]), int(fingertip[1])), 6, skin_color, -1)

def draw_chart(frame, alpha=1.0):
    """Draws a metric chart on the right side of the screen."""
    if alpha <= 0.01:
        return
    x, y, w, h = 1380, 240, 480, 360
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (35, 40, 50), -1)
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (100, 180, 255), 3)

    # Chart Header
    cv2.putText(overlay, "Q4 GROWTH METRICS", (x + 30, y + 55), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(overlay, "+84.6% ENGAGEMENT", (x + 30, y + 100), cv2.FONT_HERSHEY_DUPLEX, 1.1, (80, 230, 120), 2)

    # Bar chart columns
    bars = [("OCT", 140, (120, 120, 220)), ("NOV", 200, (100, 160, 240)), ("DEC", 270, (60, 220, 120))]
    base_y = y + 310
    col_w = 70
    gap = 50
    start_x = x + 60
    for i, (label, height, col) in enumerate(bars):
        bx = start_x + i * (col_w + gap)
        by = base_y - height
        cv2.rectangle(overlay, (bx, by), (bx + col_w, base_y), col, -1)
        cv2.rectangle(overlay, (bx, by), (bx + col_w, base_y), (255, 255, 255), 1)
        cv2.putText(overlay, label, (bx + 10, base_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

def draw_edge_text_overlays(frame, alpha=1.0):
    """Draws edge banners to test OCR detection and platform safe-zone protection."""
    if alpha <= 0.01:
        return
    overlay = frame.copy()

    # 1. Top-Right Protected Zone Banner
    tr_x, tr_y, tr_w, tr_h = 1320, 45, 560, 75
    cv2.rectangle(overlay, (tr_x, tr_y), (tr_x + tr_w, tr_y + tr_h), (20, 30, 180), -1)
    cv2.rectangle(overlay, (tr_x, tr_y), (tr_x + tr_w, tr_y + tr_h), (255, 255, 255), 2)
    cv2.putText(overlay, "CRITICAL: 95% RETENTION", (tr_x + 25, tr_y + 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)

    # 2. Bottom-Left Protected Zone Lower-Third
    bl_x, bl_y, bl_w, bl_h = 45, 940, 680, 85
    cv2.rectangle(overlay, (bl_x, bl_y), (bl_x + bl_w, bl_y + bl_h), (180, 80, 20), -1)
    cv2.rectangle(overlay, (bl_x, bl_y), (bl_x + bl_w, bl_y + bl_h), (255, 255, 255), 2)
    cv2.putText(overlay, "KEY TAKEAWAY: SUBSCRIBE NOW", (bl_x + 25, bl_y + 55), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

def generate_video():
    raw_video_path = ASSETS_DIR / "temp_raw_video.mp4"
    audio_path = ASSETS_DIR / "speech.wav"
    output_path = ASSETS_DIR / "test_clip_16_9.mp4"

    print("--- 1. Generating Audio Track ---")
    generate_speech_audio(audio_path)

    print("\n--- 2. Generating 1920x1080 60fps Video Stream ---")
    fps = 30
    width = 1920
    height = 1080
    total_duration_sec = 11.0
    total_frames = int(total_duration_sec * fps)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(raw_video_path), fourcc, fps, (width, height))

    # Background gradient setup
    bg_base = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        ratio = y / height
        # Modern studio blue-grey gradient
        b = int(45 + 30 * ratio)
        g = int(35 + 25 * ratio)
        r = int(30 + 20 * ratio)
        bg_base[y, :] = (b, g, r)

    speaker_center_x = 960
    speaker_center_y = 480

    for f in range(total_frames):
        t = f / fps
        frame = bg_base.copy()

        # Grid lines subtle studio backdrop
        cv2.line(frame, (0, 800), (width, 800), (45, 45, 55), 1)

        # Mouth movement sine wave during speech segments
        is_speaking = (0.0 <= t <= 2.9) or (3.6 <= t <= 6.2) or (6.9 <= t <= 9.5)
        mouth_open = (math.sin(t * 18) * 0.5 + 0.5) if is_speaking else 0.0

        # Segment 1: 0s to 3.2s -> Straight talking head, centered
        if t < 3.2:
            arm_state = ("rest", 0.0)
            draw_speaker(frame, speaker_center_x, speaker_center_y, arm_state, mouth_open)

        # Segment 2: 3.2s to 6.8s -> Pointing gesture to off-center chart on the right
        elif 3.2 <= t < 6.8:
            # Transition pointing in 3.2 to 3.8s, hold until 6.4s, return 6.4 to 6.8s
            if t < 3.8:
                point_t = (t - 3.2) / 0.6
            elif t < 6.4:
                point_t = 1.0
            else:
                point_t = 1.0 - (t - 6.4) / 0.4
            
            arm_state = ("pointing_right", point_t)
            draw_chart(frame, alpha=min(1.0, point_t * 1.2))
            draw_speaker(frame, speaker_center_x, speaker_center_y, arm_state, mouth_open)

        # Segment 3: 6.8s to 11.0s -> Edge text captions stressing safe zones + OCR
        else:
            arm_state = ("rest", 0.0)
            caption_alpha = min(1.0, (t - 6.8) / 0.5)
            draw_edge_text_overlays(frame, alpha=caption_alpha)
            draw_speaker(frame, speaker_center_x, speaker_center_y, arm_state, mouth_open)

        # Timestamp watermark for frame verification
        cv2.putText(frame, f"TC: {t:.2f}s | Frame: {f}", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 140, 160), 1)

        out.write(frame)

        if f % 60 == 0:
            print(f"  Rendered {f}/{total_frames} frames ({t:.1f}s / {total_duration_sec:.1f}s)...")

    out.release()
    print(f"[VIDEO] Raw video rendered to {raw_video_path}")

    print("\n--- 3. Muxing Audio & Video with FFmpeg ---")
    ffmpeg_exe = get_ffmpeg_exe()
    mux_cmd = [
        ffmpeg_exe, "-y",
        "-i", str(raw_video_path),
        "-i", str(audio_path),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        str(output_path)
    ]
    subprocess.run(mux_cmd, check=True)
    
    if raw_video_path.exists():
        raw_video_path.unlink()
        
    print(f"\n[DONE] High-fidelity test clip created at: {output_path.resolve()} ({output_path.stat().st_size / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    generate_video()

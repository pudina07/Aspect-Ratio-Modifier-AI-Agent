# Context-Aware Multi-Platform Auto-Reframe — Hackathon Build Plan v2

**Pain point recap:** 16:9 → 9:16 (TikTok/Reels/Shorts) *and* 1:1 (Instagram Feed), without losing the object a creator points to, without losing on-screen text near the edges, and without letting the crop drift into each platform's own UI safe zone.

---

## 1. hiii

---

## 2. Tech stack (quality-optimized, with justification)

| Component | Library | Why this one |
|---|---|---|
| Transcription | `faster-whisper` (large-v3 or distil-large-v3) run locally | Word-level timestamps, no API key/rate-limit risk mid-demo, CTranslate2 backend is materially faster than stock `openai-whisper` at equal accuracy. Fall back to the Whisper API only if you have zero local GPU. |
| Script analysis | Any strong chat LLM (e.g. GPT-4o-mini) with **structured JSON output enforced** | Cheap, fast, good enough at the "is this a pointing/reference cue" classification task. Enforce JSON mode so you don't burn time on brittle string parsing. |
| Pose + hand tracking | `mediapipe.tasks.vision` — **PoseLandmarker** (`pose_landmarker_full.task`) + **HandLandmarker** | Use the modern Tasks API, not the legacy `mp.solutions` API — it's what MediaPipe is actively maintained around now. Use the `full` model over `lite` for landmark accuracy; only drop to `lite` if you're missing your time budget. Note: on recent MediaPipe Python builds, the GPU delegate has shown inconsistent speedups over CPU — benchmark both for 30 seconds before committing, don't assume GPU wins. |
| Face fallback | MediaPipe **FaceDetector** (short-range model) | Cheap, always-on baseline so the speaker's face is never lost even if the transcript has no pointing cues in a stretch of video. |
| On-screen text detection | `EasyOCR` | Noticeably better recall than `pytesseract` on stylized/bold caption fonts creators actually use, still fast enough when you sample every 8–10th frame instead of every frame. |
| Smoothing | **One Euro Filter** (~30-line custom implementation, no dependency) | Purpose-built for exactly this problem: adaptive — stiff during slow motion (kills jitter), loose during fast motion (kills lag). Strictly better output quality than a moving average at the same code cost. |
| Rendering | `OpenCV` (`cv2.VideoWriter`) for per-frame numpy crop, then `ffmpeg -c:v copy -c:a aac` to mux audio back in | Your own v1 hackathon tip was correct — codify it as the *primary* path, not a fallback. Per-frame dynamic crops are painful in FFmpeg's `sendcmd`/`geq` filters and easy in OpenCV. |
| UI | `Streamlit` | Right call for a same-day demo. Kept as-is. |

---

## 3. Phase 0 — Setup (20 min) *(new)*

- Pin versions: `faster-whisper`, `mediapipe>=0.10.31`, `easyocr`, `opencv-python`, `streamlit`.
- Download models once, up front, not mid-pipeline: Whisper weights, `pose_landmarker_full.task`, `hand_landmarker.task`, `face_detector.tflite`, EasyOCR's English detector+recognizer weights. All of these are multi-hundred-MB downloads — doing this at Phase 5 while judges watch is a demo-killer.
- Pick your test clip deliberately. It should contain, in order: a straight talking-head segment, a moment where the creator points at an off-screen object, and a burned-in text caption near the frame edge. If your test clip doesn't stress all three failure modes, you won't know if the pipeline actually works until the live demo does.

---

## 4. Phase 1 — Architecture (30 min)

Same modular philosophy as v1 — each script reads one JSON and writes one JSON, so a failure in one stage doesn't take down the others. Pipeline order:

```
video.mp4
  → transcribe.py          → transcript.json
  → analyze_script.py      → focus_timeline.json
  → tracker.py             → raw_coords.json  (pose + hand + face landmarks)
  → ocr_pass.py             → text_regions.json   (protected zones)
  → smooth_coords.py        → final_coords_916.json + final_coords_11.json
  → render.py                → output_916.mp4 + output_11.mp4
  → app.py (Streamlit)       orchestrates all of the above
```

---

## 5. Phase 2 — Focus Timeline Generator (1.5–2 hrs)

**Step 1 — Transcribe** (same as v1, swap in faster-whisper):
> *AI Coder Prompt:* "Write `transcribe.py` using `faster-whisper` (model size `large-v3`, or `distil-large-v3` for speed). Extract audio with FFmpeg first. Output word-level timestamps to `transcript.json`."

**Step 2 — LLM pointing + reference analysis**, refined schema (adds direction hint, confidence, and a distinction between pointing at something in-frame vs. off-camera, since that's useful demo narrative for judges):
> *AI Coder Prompt:* "Send the transcript to the LLM with this system prompt: 'Identify timestamps where the speaker verbally references or points at something (e.g. "look at this", "here's the chart", "notice this"). For each, output start_time, end_time, focus ("speaker" or "object"), direction_hint ("left"/"right"/"center"/"unknown"), and confidence (0–1).' Enforce JSON output mode."

**Step 3 — Debounce the timeline** *(new, important)*: merge focus blocks under 1 second apart with the same direction; discard blocks shorter than 0.3s. Without this, a creator saying "here — no wait, here" produces a camera that whip-pans twice in one second, which looks like a bug, not a feature, on stage.

---

## 6. Phase 3 — Vision Tracking: Pose, Hands, Face, and Text (2.5 hrs)

**Step 4 — Baseline face track** (new, cheap, always runs): FaceDetector every 5th frame → gives you a fallback crop center for any stretch of video the LLM didn't flag, so the speaker is never lost by default.

**Step 5 — Pose + hand pointing vector** (upgrade from v1's wrist-only approach):
> *AI Coder Prompt:* "During 'object' blocks, run PoseLandmarker for the wrist (landmark 15/16) and HandLandmarker for the index fingertip (landmark 8). Compute the 2D vector from wrist → fingertip, and extrapolate a target point roughly 35–40% of the way from the fingertip toward the frame edge in that direction, clamped to frame bounds. Save wrist, fingertip, and extrapolated target per frame to `raw_coords.json`."

This is the single highest-leverage fix over v1: snapping the crop to the wrist alone tends to undershoot — the wrist is still near the body, not at the object. Extrapolating along the arm's vector gets you meaningfully closer to what's actually being pointed at, with zero extra model cost.

**Step 6 — OCR protected-region pass** *(new)*:
> *AI Coder Prompt:* "Write `ocr_pass.py` using EasyOCR. Sample every 8th frame. For each, detect text bounding boxes. Tag each frame range with any detected `protected_region` (x, y, w, h) so downstream crop selection knows not to fully exclude it. Save to `text_regions.json`."

---

## 7. Phase 4 — Dual-Aspect Coordinator & Smoothing (2 hrs)

This is where the two deliverables (9:16 and 1:1) diverge from a shared source of truth.

**Step 7 — Compute two crop tracks**, both driven by the same `raw_coords.json` + `text_regions.json`:
- **9:16 track:** 608×1080 window inside the 1920×1080 source. X pans per the focus target; Y mostly fixed.
- **1:1 track:** 1080×1080 window. Less horizontal room to work with, so define a fallback rule up front: if the pointed-at target and the speaker's face can't both fit in 1080px of width, bias toward keeping the face fully in frame and let the object be partially cropped, rather than splitting the difference and losing both.

**Step 8 — One Euro Filter smoothing** (replaces moving average):
> *AI Coder Prompt:* "Implement a One Euro Filter (mincutoff, beta, dcutoff parameters) and apply it per-axis to both the 9:16 and 1:1 coordinate tracks. Use a lower beta (more smoothing) during 'speaker' blocks, and a higher beta (more responsiveness) during 'object' transition blocks."

**Step 9 — Protected-region clamp:** if a text bounding box from `text_regions.json` would be more than ~50% cut off by the current crop, nudge the crop box toward including it, within what the smoothing pass allows. If it genuinely can't fit (text is far outside the crop's aspect ratio), that's a known limitation worth stating plainly in the demo rather than papering over.

**Step 10 — Transition interpolation** (kept from v1, upgraded): ease-in-out curve over ~15 frames instead of linear — linear pans read as mechanical; eased pans read as intentional camera work.

---

## 8. Phase 5 — Platform-Aware Rendering (1.5 hrs)

**Step 11 — Render** (OpenCV as primary path, not fallback, per your own v1 tip):
> *AI Coder Prompt:* "Write `render.py`. For each frame: create a blurred, scaled full-bleed background at the target aspect ratio, crop the region from `final_coords.json`, overlay it centered on the background, draw the platform's safe-zone rectangle (semi-transparent, toggleable — for internal QA, not baked into the shipped output). Write with `cv2.VideoWriter`, then mux the original audio track back in via FFmpeg (`-c:v copy -c:a aac`)."

**Safe-zone presets** — make these a small editable JSON config, *not* hardcoded constants. Published TikTok/Reels UI margins vary noticeably by source and change with app updates (icon rails and caption bars have both grown in recent app revisions), so treat the numbers below as sane defaults you can tune against a real screenshot the morning of the hackathon, not gospel:

| Platform | Canvas | Approx. top clear | Approx. bottom clear | Approx. right clear (icon rail) | Approx. left clear |
|---|---|---|---|---|---|
| TikTok (9:16) | 1080×1920 | ~130 px | ~320–400 px | ~150–180 px | ~50–60 px |
| Instagram Reels (9:16) | 1080×1920 | ~220 px | ~400–420 px | ~100 px | minimal |
| Instagram Feed (1:1) | 1080×1080 | minimal | ~8–10% (caption preview line) | minimal | minimal |

Making this a config file rather than constants is itself worth mentioning to judges — it's the difference between a demo hack and a tool that survives the next TikTok UI update.

---

## 9. Phase 6 — Streamlit UI & Polish (1 hr)

Same shape as v1, extended for dual output and platform selection:
> *AI Coder Prompt:* "Streamlit app `app.py`. Title: 'Context-Aware Auto-Reframe'. File uploader for MP4. Multi-select for target platforms (TikTok, Instagram Reels, Instagram Feed). 'Process Video' button runs the full pipeline via `subprocess.run` with `st.spinner` per stage. On completion, show the original video plus each generated output side-by-side with `st.video`, and a toggle to overlay the safe-zone guide for QA."

---

## 10. Time budget (v2, ~9.5 hrs core + buffer)

| Phase | Time |
|---|---|
| 0 — Setup | 20 min |
| 1 — Architecture | 30 min |
| 2 — Focus timeline | 1.5–2 hrs |
| 3 — Vision tracking + OCR | 2.5 hrs |
| 4 — Dual-aspect smoothing | 2 hrs |
| 5 — Platform rendering | 1.5 hrs |
| 6 — Streamlit UI | 1 hr |
| **Buffer — testing, re-runs, demo script** | **1.5–2 hrs** |
| **Total** | **~11–12 hrs** |

This is more than v1's original 8-hour estimate because the actual brief has more surface area (dual aspect ratio, per-platform safe zones, text protection) than v1 accounted for. If you're genuinely capped at 8 hours, cut in this order: drop 1:1 output first (keep 9:16 only), then drop OCR text protection, then fall back to `pose_landmarker_lite` — keep the pointing-vector fix and the One Euro filter regardless, they're cheap and they're your main quality differentiator.

---

## 11. Pre-demo QA checklist

- [ ] Run the full pipeline on a clip *not* used during development — catches overfitting to your one test video.
- [ ] Confirm both 9:16 and 1:1 outputs render without the crop box ever going out of source bounds (a silent black-frame bug is the most common failure mode here).
- [ ] Visually confirm the pointing-vector pan actually lands closer to the object than the old wrist-only approach would have, on at least one real "look at this" moment.
- [ ] Confirm the safe-zone overlay (QA mode) doesn't cover the speaker's face during a normal talking-head stretch — if it does, your presets are too aggressive.
- [ ] Time the full pipeline end-to-end on your actual demo clip so you know whether to run it live or show a pre-rendered result with the process narrated over it.

---

## 12. Demo narrative for judges (kept from v1, sharpened)

Don't open with code. Open with the side-by-side: existing tools (naive face-centered crop) versus yours, on the exact moment the creator points off-screen. Then walk the judges through *why* it works differently: the tool reads the transcript to know *when* something is being referenced, uses the arm's pointing vector — not just wrist position — to estimate *where*, protects any on-screen text so captions never get eaten, and renders two aspect ratios simultaneously with each platform's actual UI chrome respected, using a config file that survives the next time TikTok redesigns its icon rail. That combination — contextual, multi-target, multi-platform-aware — is the part no existing auto-reframe tool does.

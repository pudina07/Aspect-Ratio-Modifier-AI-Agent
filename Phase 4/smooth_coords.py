"""
pipeline/smooth_coords.py — Phase 4, Steps 7-10

Contract:  raw_coords.json + text_regions.json + focus_timeline.json
           ->  final_coords_916.json + final_coords_11.json

This is where the two deliverables (9:16 and 1:1) diverge from one
shared source of truth (raw_coords.json). Each track is built through
four layers, run in this order:

  1. Dense target construction + gap easing (Steps 7 & 10 combined).
     raw_coords.json is sparse by design — tracker.py's own docstring
     says so explicitly: "Frames where nothing was computed are omitted
     entirely — smooth_coords.py (Phase 4) interpolates/holds across the
     gaps, same as it already has to for the sparse face samples." So
     before anything else, every frame in the video needs a target
     value. Gaps are filled by holding the last known value and easing
     (cubic ease-in-out, Step 10) into the next known value over the
     last ~15 frames before it arrives, instead of a hard jump. This is
     also where the 1:1 track's face-priority fallback from Step 7
     lives: if centering on a pointing target would push the most
     recently seen face out of the crop window, the face wins.
  2. One Euro Filter (Step 8) on top of that dense signal, to smooth out
     the genuine per-frame jitter that comes from continuous landmark
     tracking during 'object' blocks (tracker.py runs pose+hand on
     *every* frame in those blocks, not sparsely) — with beta raised
     during 'object' blocks (responsive) and lowered during 'speaker'
     blocks (stable), per focus_timeline.json.
  3. Protected-region clamp (Step 9) against text_regions.json: nudges
     the crop, rate-limited so it can't itself reintroduce jitter,
     toward including on-screen text that would otherwise be more than
     half cut off. The nudge accumulates across the frames a caption is
     on screen and decays back out once it's gone, rather than resetting
     every frame.
  4. Final bounds clamp to keep the crop box inside the source frame.

Both tracks pan along whichever source axis has slack for their target
aspect ratio (x for both 9:16 and 1:1 against this plan's 1920x1080
source, matching the plan's 608x1080 / 1080x1080 windows) and use the
full extent of the other axis, so Y is fixed rather than "mostly fixed"
in the numbers this plan actually specifies.
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import stage_path                               # noqa: E402
from utils.io_json import load_json, save_json, fail_stage  # noqa: E402
from utils.one_euro import OneEuroFilter                    # noqa: E402

STAGE_NAME = "smooth_coords"

TRACKS = {"916": (9, 16), "11": (1, 1)}

TRANSITION_EASE_FRAMES = 15        # Step 10: "ease-in-out curve over ~15 frames"
MIN_CUTOFF = 1.0                   # One Euro Filter baseline cutoff
BETA_SPEAKER = 0.3                 # lower beta = more smoothing during 'speaker' blocks
BETA_OBJECT = 1.6                  # higher beta = more responsiveness during 'object' blocks
D_CUTOFF = 1.0                     # standard One Euro derivative cutoff
TEXT_COVERAGE_THRESHOLD = 0.5      # Step 9: "more than ~50% cut off" trigger
MAX_TEXT_NUDGE_PX_PER_FRAME = 8.0  # rate limit so the Step 9 clamp can't itself add jitter


def _crop_dims(source_w: int, source_h: int, aspect_w: int, aspect_h: int) -> tuple[int, int, str]:
    """Fit an (aspect_w:aspect_h) window inside (source_w, source_h),
    using the full extent of whichever source dimension the target
    aspect ratio has no slack on, and panning along the other. For this
    plan's 1920x1080 source, both the 9:16 and 1:1 targets land on
    height-fixed / x-panning (608x1080 and 1080x1080, Step 7) — the
    branch below just generalizes it in case a future source isn't
    16:9 landscape.
    """
    if source_w * aspect_h >= source_h * aspect_w:
        crop_h = source_h
        crop_w = min(source_w, round(crop_h * aspect_w / aspect_h))
        return crop_w, crop_h, "x"
    else:
        crop_w = source_w
        crop_h = min(source_h, round(crop_w * aspect_h / aspect_w))
        return crop_w, crop_h, "y"


def _ease_in_out(p: float) -> float:
    """Cubic smoothstep. Step 10: replaces v1's linear transition ramp —
    "linear pans read as mechanical; eased pans read as intentional
    camera work." """
    p = min(1.0, max(0.0, p))
    return p * p * (3.0 - 2.0 * p)


def _object_frame_mask(focus_timeline: dict, fps: float, frame_count: int) -> list[bool]:
    """Per-frame bool: True inside an 'object' focus_timeline block. Same
    seconds->frames conversion tracker.py's _object_frame_ranges does,
    kept as an independent local copy rather than an import: each stage
    is meant to stand alone per config.py's docstring (a failure in one
    stage shouldn't take down the others), so this stage doesn't reach
    into tracker.py's internals even though the logic is tiny and
    duplicated."""
    mask = [False] * frame_count
    for block in focus_timeline.get("blocks", []):
        if block.get("focus") != "object":
            continue
        start_f = max(0, int(round(block["start_time"] * fps)))
        end_f = min(frame_count - 1, int(round(block["end_time"] * fps)))
        for f in range(start_f, end_f + 1):
            mask[f] = True
    return mask


def _build_dense_target(raw_coords: dict, axis_index: int, crop_dim: float, source_dim: int) -> list[float]:
    """Steps 7 & 10. Walk raw_coords' sparse, time-ordered frame entries,
    turn each into a single target-coordinate keyframe along this track's
    panning axis (pointing target takes priority over face when both are
    present in the same entry, since that only happens inside an
    'object' block where pose+hand ran), then fill the full dense
    per-frame array by holding + easing across the gaps between
    keyframes.
    """
    meta = raw_coords["meta"]
    fps, frame_count = meta["fps"], meta["frame_count"]

    keyframes: list[tuple[int, float]] = []
    last_face = None
    for entry in raw_coords.get("frames", []):
        face = entry.get("face_center")
        if face is not None:
            last_face = face[axis_index]

        target_pt = entry.get("extrapolated_target")
        if target_pt is not None:
            val = target_pt[axis_index]
            # Step 7's face-priority fallback (spelled out for the 1:1
            # track in the plan, applied here to both): if centering on
            # the pointing target would push the most recently seen face
            # outside this crop window, keep the face in frame and let
            # the object be partially cropped instead of splitting the
            # difference and losing both. This naturally bites far more
            # often on the narrower 1:1 crop than 9:16, matching the
            # plan's rationale, without needing two separate code paths.
            #
            # Clamp toward the face's window rather than snapping val to
            # last_face outright — a hard snap discards the pointing
            # target entirely and reintroduces a jump-cut on borderline
            # frames (target just barely outside the face's window), even
            # though "partially cropped" means the crop should still lean
            # toward the target as far as the face-visible constraint
            # allows.
            if last_face is not None:
                lo, hi = last_face - crop_dim / 2.0, last_face + crop_dim / 2.0
                val = min(max(val, lo), hi)
        elif face is not None:
            val = face[axis_index]
        else:
            continue  # entry carried neither signal on this axis

        frame_idx = int(round(entry["t"] * fps))
        frame_idx = min(max(frame_idx, 0), frame_count - 1)
        keyframes.append((frame_idx, val))

    if not keyframes:
        # No face or pointing data anywhere in the video — hold dead
        # center rather than guessing.
        return [source_dim / 2.0] * frame_count

    dense = [0.0] * frame_count
    first_idx, first_val = keyframes[0]
    for f in range(0, first_idx + 1):
        dense[f] = first_val

    for (idx_a, val_a), (idx_b, val_b) in zip(keyframes, keyframes[1:]):
        gap = idx_b - idx_a
        if gap <= 0:
            continue  # duplicate/out-of-order timestamp; keep the earlier value
        ease_len = min(TRANSITION_EASE_FRAMES, gap)
        ease_start = idx_b - ease_len
        for f in range(idx_a + 1, ease_start):
            dense[f] = val_a
        for k in range(ease_len + 1):
            f = ease_start + k
            progress = (k / ease_len) if ease_len else 1.0
            dense[f] = val_a + (val_b - val_a) * _ease_in_out(progress)

    last_idx, last_val = keyframes[-1]
    for f in range(last_idx, frame_count):
        dense[f] = last_val

    return dense


def _smooth_with_one_euro(dense: list[float], fps: float, object_mask: list[bool]) -> list[float]:
    """Step 8. Adaptive beta per frame: 'object' blocks get the higher,
    more-responsive beta; everything else (including gaps we've already
    eased) gets the lower, smoother one."""
    if not dense:
        return []
    filt = OneEuroFilter(t0=0.0, x0=dense[0], min_cutoff=MIN_CUTOFF, beta=BETA_SPEAKER, d_cutoff=D_CUTOFF)
    out = [dense[0]]
    for i in range(1, len(dense)):
        t = i / fps
        beta = BETA_OBJECT if (i < len(object_mask) and object_mask[i]) else BETA_SPEAKER
        out.append(filt(t, dense[i], beta=beta))
    return out


def _decay_toward_zero(value: float, rate: float) -> float:
    if value > 0:
        return max(0.0, value - rate)
    if value < 0:
        return min(0.0, value + rate)
    return 0.0


def _apply_text_protection(centers: list[float], crop_dim: float, fps: float,
                            text_regions: dict, axis_index: int) -> tuple[list[float], int]:
    """Step 9. Maintains a persistent, rate-limited 'correction' offset
    on top of the smoothed baseline rather than nudging fresh from the
    baseline each frame — a caption on screen for two seconds needs the
    correction to accumulate across those frames to actually reach it,
    not repeat the same single-frame nudge every time. The correction
    decays back to zero (also rate-limited) once nothing needs
    protecting, so the crop returns to the plain smoothed track instead
    of snapping back.
    """
    regions_sorted = sorted(text_regions.get("regions", []), key=lambda r: r["t_start"])
    add_ptr = 0
    active: list[dict] = []
    unfixable_ids: set[int] = set()

    out = []
    correction = 0.0
    for i, base in enumerate(centers):
        t = i / fps
        while add_ptr < len(regions_sorted) and regions_sorted[add_ptr]["t_start"] <= t:
            active.append(regions_sorted[add_ptr])
            add_ptr += 1
        active[:] = [r for r in active if r["t_end"] >= t]

        center = base + correction
        worst_region, worst_cov = None, 1.0
        for r in active:
            r_lo = r["x"] if axis_index == 0 else r["y"]
            r_len = r["w"] if axis_index == 0 else r["h"]
            if r_len <= 0:
                continue
            r_hi = r_lo + r_len
            overlap = max(0.0, min(center + crop_dim / 2.0, r_hi) - max(center - crop_dim / 2.0, r_lo))
            cov = overlap / r_len
            if cov < worst_cov:
                worst_cov, worst_region = cov, r

        if worst_region is not None and worst_cov < TEXT_COVERAGE_THRESHOLD:
            r_lo = worst_region["x"] if axis_index == 0 else worst_region["y"]
            r_len = worst_region["w"] if axis_index == 0 else worst_region["h"]
            r_center = r_lo + r_len / 2.0
            desired = r_center - center
            step = max(-MAX_TEXT_NUDGE_PX_PER_FRAME, min(MAX_TEXT_NUDGE_PX_PER_FRAME, desired))
            correction += step
            if r_len > crop_dim:
                # Text is wider than the crop's aspect ratio allows —
                # can't ever fully include it. Best-effort center on it
                # anyway (still capped by the same rate limit above);
                # just flag it as a known limitation rather than pretend.
                unfixable_ids.add(id(worst_region))
        else:
            correction = _decay_toward_zero(correction, MAX_TEXT_NUDGE_PX_PER_FRAME)

        out.append(base + correction)

    return out, len(unfixable_ids)


def _build_track(aspect_w: int, aspect_h: int, raw_coords: dict, text_regions: dict,
                  focus_timeline: dict) -> dict:
    meta = raw_coords["meta"]
    source_w, source_h = meta["width"], meta["height"]
    fps, frame_count = meta["fps"], meta["frame_count"]

    crop_w, crop_h, pan_axis = _crop_dims(source_w, source_h, aspect_w, aspect_h)
    axis_index = 0 if pan_axis == "x" else 1
    source_dim = source_w if pan_axis == "x" else source_h
    crop_dim = crop_w if pan_axis == "x" else crop_h

    dense = _build_dense_target(raw_coords, axis_index, crop_dim, source_dim)

    object_mask = _object_frame_mask(focus_timeline, fps, frame_count)
    smoothed = _smooth_with_one_euro(dense, fps, object_mask)

    protected, unfixable = _apply_text_protection(smoothed, crop_dim, fps, text_regions, axis_index)

    frames_out = []
    for i, center in enumerate(protected):
        origin = center - crop_dim / 2.0
        origin = min(max(origin, 0.0), source_dim - crop_dim)
        if pan_axis == "x":
            crop_x, crop_y = origin, (source_h - crop_h) / 2.0
        else:
            crop_x, crop_y = (source_w - crop_w) / 2.0, origin
        frames_out.append({
            "t": round(i / fps, 3),
            "crop_x": int(round(crop_x)),
            "crop_y": int(round(crop_y)),
        })

    if unfixable:
        print(f"[{STAGE_NAME}] note: {unfixable} protected text region(s) wider than the "
              f"{aspect_w}:{aspect_h} crop window on this track — covered best-effort, "
              f"not fully fixable at this aspect ratio.", file=sys.stderr)

    return {
        "meta": {
            "crop_width": crop_w, "crop_height": crop_h,
            "source_width": source_w, "source_height": source_h,
            "fps": fps, "frame_count": frame_count,
        },
        "frames": frames_out,
    }


def run(raw_coords: dict, text_regions: dict, focus_timeline: dict) -> tuple[dict, dict]:
    """
    Returns (coords_916, coords_11), each shaped like:
        {"meta": {"crop_width": 608, "crop_height": 1080,
                  "source_width": 1920, "source_height": 1080,
                  "fps": 30.0, "frame_count": 900},
         "frames": [{"t": 4.13, "crop_x": 210, "crop_y": 0}, ...]}

    One dense entry per source video frame — render.py (Phase 5) can
    index straight off frame number without any further interpolation.
    """
    aw, ah = TRACKS["916"]
    coords_916 = _build_track(aw, ah, raw_coords, text_regions, focus_timeline)
    aw, ah = TRACKS["11"]
    coords_11 = _build_track(aw, ah, raw_coords, text_regions, focus_timeline)
    return coords_916, coords_11


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-coords", type=Path, default=stage_path("raw_coords.json"))
    parser.add_argument("--text-regions", type=Path, default=stage_path("text_regions.json"))
    parser.add_argument("--focus-timeline", type=Path, default=stage_path("focus_timeline.json"))
    parser.add_argument("--out-916", type=Path, default=stage_path("final_coords_916.json"))
    parser.add_argument("--out-11", type=Path, default=stage_path("final_coords_11.json"))
    args = parser.parse_args()

    try:
        raw_coords = load_json(args.raw_coords)
        text_regions = load_json(args.text_regions)
        focus_timeline = load_json(args.focus_timeline)

        coords_916, coords_11 = run(raw_coords, text_regions, focus_timeline)

        save_json(args.out_916, coords_916)
        save_json(args.out_11, coords_11)
        print(f"[{STAGE_NAME}] wrote {args.out_916} "
              f"({coords_916['meta']['crop_width']}x{coords_916['meta']['crop_height']}) and "
              f"{args.out_11} ({coords_11['meta']['crop_width']}x{coords_11['meta']['crop_height']}), "
              f"{len(coords_916['frames'])} frames each")
    except Exception as e:
        fail_stage(STAGE_NAME, e)


if __name__ == "__main__":
    main()

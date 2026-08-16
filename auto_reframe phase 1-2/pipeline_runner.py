"""
pipeline_runner.py — Phase 1: Architecture

The orchestration layer. Runs each stage as an isolated subprocess
(never an in-process import) so a crash in one stage — bad model
weights, an OOM, a stray exception — can't take the rest of the app
down or corrupt shared state. This is the direct implementation of the
plan's core principle: "each script reads one JSON and writes one JSON,
so a failure in one stage doesn't take down the others."

It schedules off config.PIPELINE_STAGES as a dependency graph rather
than a fixed top-to-bottom list: ocr_pass only needs video.mp4, so it
runs concurrently with the transcribe -> analyze_script -> tracker
chain instead of waiting behind it. Everything converges at
smooth_coords, then render.

Phase 6 will call run_pipeline() from inside Streamlit and wrap the
per-stage prints in st.spinner instead.
"""
import concurrent.futures
import subprocess
import sys
import time
from typing import NamedTuple

from config import PIPELINE_STAGES, DATA_DIR, BASE_DIR


class StageResult(NamedTuple):
    name: str
    ok: bool
    stderr: str


def _inputs_ready(stage: dict) -> bool:
    return all((DATA_DIR / f).exists() for f in stage["inputs"])


def run_stage(stage: dict) -> StageResult:
    """Run one stage script as its own process and capture the result.
    Never raises — a failed stage is reported back, not thrown, so the
    caller decides whether to halt the whole pipeline or keep going."""
    script_path = BASE_DIR / stage["script"]
    proc = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
    )
    return StageResult(name=stage["name"], ok=proc.returncode == 0, stderr=proc.stderr)


def run_pipeline(poll_interval: float = 0.2) -> list[StageResult]:
    """
    Runs every stage in PIPELINE_STAGES, launching each one as soon as
    its declared inputs exist on disk, and letting independent branches
    run in parallel. Requires video.mp4 to already be in DATA_DIR before
    this is called (app.py's job).

    Returns one StageResult per stage that actually ran. Does not raise
    on a stage failure — check `.ok` on each result. Does raise if the
    graph stalls (some stage's inputs never appear), which almost always
    means an upstream stage failed silently or video.mp4 is missing.
    """
    pending = {s["name"]: s for s in PIPELINE_STAGES}
    running: dict[str, concurrent.futures.Future] = {}
    results: list[StageResult] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        while pending or running:
            for name, stage in list(pending.items()):
                if _inputs_ready(stage):
                    print(f"[{name}] starting")
                    running[name] = pool.submit(run_stage, stage)
                    del pending[name]

            if not running:
                stuck = ", ".join(pending)
                raise RuntimeError(
                    f"Pipeline stalled — inputs never appeared for: {stuck}. "
                    f"Is video.mp4 in {DATA_DIR}? Did an upstream stage fail?"
                )

            done_names = [n for n, f in running.items() if f.done()]
            if not done_names:
                time.sleep(poll_interval)
                continue

            for name in done_names:
                result = running.pop(name).result()
                results.append(result)
                status = "OK" if result.ok else "FAILED"
                print(f"[{name}] {status}")
                if not result.ok:
                    print(result.stderr, file=sys.stderr)

    return results


if __name__ == "__main__":
    run_pipeline()

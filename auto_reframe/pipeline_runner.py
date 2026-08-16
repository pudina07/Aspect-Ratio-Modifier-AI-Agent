"""
pipeline_runner.py — Phase 1: Architecture & Concurrency Orchestrator

The orchestration layer. Runs each stage as an isolated subprocess
(never an in-process import) so a crash in one stage cannot corrupt shared
state or bring down the host application.

Key features:
1. Dynamic DAG scheduling: stages launch as soon as declared inputs exist.
2. Concurrency: independent stages (e.g., ocr_pass and transcribe) run in parallel.
3. Failure isolation: when an upstream stage fails, its downstream dependents are
   gracefully marked as SKIPPED rather than deadlocking the runner.
4. Clean workspace management: removes stale artifacts prior to a new run.
5. Explicit path arguments passed to each stage subprocess.
6. Progress reporting callbacks for CLI and UI spinners (Phase 6 Streamlit).
"""
import concurrent.futures
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

# Force UTF-8 encoding on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from config import PIPELINE_STAGES, DATA_DIR, BASE_DIR, get_downstream_stages


class StageStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


@dataclass
class StageResult:
    name: str
    status: StageStatus
    ok: bool
    duration: float = 0.0
    stdout: str = ""
    stderr: str = ""
    outputs: List[str] = field(default_factory=list)
    skip_reason: Optional[str] = None


@dataclass
class PipelineReport:
    ok: bool
    results: Dict[str, StageResult]
    total_duration: float
    failed_stages: List[str] = field(default_factory=list)
    skipped_stages: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Pipeline Execution Summary ({self.total_duration:.2f}s total):",
            "-" * 65,
            f"{'Stage':<18} | {'Status':<10} | {'Time (s)':<9} | {'Details'}",
            "-" * 65,
        ]
        for name, r in self.results.items():
            dur_str = f"{r.duration:.2f}" if r.duration > 0 else "-"
            details = ""
            if r.status == StageStatus.FAILED:
                err_snippet = r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "Non-zero exit code"
                details = f"ERR: {err_snippet[:40]}"
            elif r.status == StageStatus.SKIPPED:
                details = f"Skipped ({r.skip_reason or 'upstream dependency failed'})"
            elif r.status == StageStatus.SUCCESS:
                details = f"Outputs: {', '.join(r.outputs)}"
            lines.append(f"{name:<18} | {r.status.value:<10} | {dur_str:<9} | {details}")
        lines.append("-" * 65)
        lines.append(f"Overall Status: {'✅ SUCCESS' if self.ok else '❌ FAILED'}")
        return "\n".join(lines)


def clean_run_artifacts(data_dir: Optional[Path] = None, keep_video: bool = True) -> None:
    """
    Cleans up stale JSON, MP4, and tmp files in data_dir from previous runs.
    Keeps video.mp4 by default.
    """
    target_dir = data_dir or DATA_DIR
    if not target_dir.exists():
        return

    all_known_outputs = set()
    for s in PIPELINE_STAGES:
        all_known_outputs.update(s["outputs"])

    for item in target_dir.iterdir():
        if item.is_file():
            if item.name == "video.mp4" and keep_video:
                continue
            if item.name in all_known_outputs or item.suffix in (".tmp", ".json") or item.name.startswith("output_"):
                try:
                    item.unlink()
                except Exception:
                    pass


def _inputs_ready(stage: dict, data_dir: Path) -> bool:
    """Check if all inputs declared by this stage exist on disk."""
    return all((data_dir / f).exists() for f in stage["inputs"])


def _build_stage_cmd(stage: dict, data_dir: Path, use_mock: bool) -> List[str]:
    """Constructs explicit CLI arguments for a stage script based on data_dir."""
    script_path = BASE_DIR / stage["script"]
    cmd = [sys.executable, str(script_path)]
    name = stage["name"]

    if name == "transcribe":
        cmd.extend(["--video", str(data_dir / "video.mp4"), "--out", str(data_dir / "transcript.json")])
    elif name == "analyze_script":
        cmd.extend(["--transcript", str(data_dir / "transcript.json"), "--out", str(data_dir / "focus_timeline.json")])
    elif name == "tracker":
        cmd.extend([
            "--video", str(data_dir / "video.mp4"),
            "--focus-timeline", str(data_dir / "focus_timeline.json"),
            "--out", str(data_dir / "raw_coords.json")
        ])
    elif name == "ocr_pass":
        cmd.extend(["--video", str(data_dir / "video.mp4"), "--out", str(data_dir / "text_regions.json")])
    elif name == "smooth_coords":
        cmd.extend([
            "--raw-coords", str(data_dir / "raw_coords.json"),
            "--text-regions", str(data_dir / "text_regions.json"),
            "--focus-timeline", str(data_dir / "focus_timeline.json"),
            "--out-916", str(data_dir / "final_coords_916.json"),
            "--out-11", str(data_dir / "final_coords_11.json")
        ])
    elif name == "render":
        cmd.extend([
            "--video", str(data_dir / "video.mp4"),
            "--coords-916", str(data_dir / "final_coords_916.json"),
            "--coords-11", str(data_dir / "final_coords_11.json"),
            "--out-916", str(data_dir / "output_916.mp4"),
            "--out-11", str(data_dir / "output_11.mp4")
        ])

    if use_mock:
        cmd.append("--mock")
    return cmd


def run_stage(
    stage: dict,
    data_dir: Path,
    use_mock: bool = False,
) -> StageResult:
    """
    Run one stage script as an isolated subprocess.
    Returns StageResult with returncode, stdout, stderr, and elapsed time.
    """
    cmd = _build_stage_cmd(stage, data_dir, use_mock)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(BASE_DIR)
    env["PYTHONIOENCODING"] = "utf-8"

    start_t = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        duration = time.time() - start_t
        ok = (proc.returncode == 0)
        status = StageStatus.SUCCESS if ok else StageStatus.FAILED
        return StageResult(
            name=stage["name"],
            status=status,
            ok=ok,
            duration=duration,
            stdout=proc.stdout,
            stderr=proc.stderr,
            outputs=stage["outputs"] if ok else [],
        )
    except Exception as e:
        duration = time.time() - start_t
        return StageResult(
            name=stage["name"],
            status=StageStatus.FAILED,
            ok=False,
            duration=duration,
            stdout="",
            stderr=str(e),
            outputs=[],
        )


def run_pipeline(
    data_dir: Optional[Path] = None,
    poll_interval: float = 0.05,
    use_mock: bool = False,
    clean_workspace: bool = True,
    progress_callback: Optional[Callable[[str, StageStatus, Optional[str]], None]] = None,
) -> PipelineReport:
    """
    Runs every stage in PIPELINE_STAGES following the DAG dependency graph.
    Stages with satisfied inputs are executed concurrently using a ThreadPoolExecutor.
    """
    target_data_dir = data_dir or DATA_DIR
    target_data_dir.mkdir(parents=True, exist_ok=True)

    if clean_workspace:
        clean_run_artifacts(target_data_dir, keep_video=True)

    video_input = target_data_dir / "video.mp4"
    if not video_input.exists():
        raise FileNotFoundError(
            f"Input video 'video.mp4' not found in {target_data_dir}. "
            f"Stage video.mp4 before running the pipeline."
        )

    pending: Dict[str, dict] = {s["name"]: s for s in PIPELINE_STAGES}
    running: Dict[str, concurrent.futures.Future] = {}
    results: Dict[str, StageResult] = {}
    blocked_stages: Set[str] = set()

    start_total_t = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        while pending or running:
            # Check for newly ready stages
            for name, stage in list(pending.items()):
                if name in blocked_stages:
                    continue
                if _inputs_ready(stage, target_data_dir):
                    if progress_callback:
                        progress_callback(name, StageStatus.RUNNING, None)
                    print(f"[{name}] STARTING")
                    running[name] = pool.submit(run_stage, stage, target_data_dir, use_mock)
                    del pending[name]

            # If nothing is running and pending is not empty:
            # All remaining pending stages must be blocked
            if not running:
                if pending:
                    for name in list(pending.keys()):
                        reason = "Upstream dependency failed"
                        results[name] = StageResult(
                            name=name,
                            status=StageStatus.SKIPPED,
                            ok=False,
                            skip_reason=reason
                        )
                        if progress_callback:
                            progress_callback(name, StageStatus.SKIPPED, reason)
                        print(f"[{name}] SKIPPED ({reason})")
                        del pending[name]
                break

            done_names = [n for n, f in running.items() if f.done()]
            if not done_names:
                time.sleep(poll_interval)
                continue

            for name in done_names:
                fut = running.pop(name)
                res: StageResult = fut.result()
                results[name] = res

                if res.ok:
                    print(f"[{name}] OK ({res.duration:.2f}s)")
                    if progress_callback:
                        progress_callback(name, StageStatus.SUCCESS, None)
                else:
                    print(f"[{name}] FAILED ({res.duration:.2f}s)")
                    if res.stderr:
                        print(f"[{name}] stderr: {res.stderr.strip()}", file=sys.stderr)
                    if progress_callback:
                        progress_callback(name, StageStatus.FAILED, res.stderr)

                    # Propagate failure downstream: mark all dependent stages as blocked
                    downstream = get_downstream_stages(name)
                    for ds in downstream:
                        if ds in pending:
                            blocked_stages.add(ds)
                            skip_msg = f"Blocked by failed upstream stage '{name}'"
                            results[ds] = StageResult(
                                name=ds,
                                status=StageStatus.SKIPPED,
                                ok=False,
                                skip_reason=skip_msg
                            )
                            if progress_callback:
                                progress_callback(ds, StageStatus.SKIPPED, skip_msg)
                            print(f"[{ds}] SKIPPED ({skip_msg})")
                            del pending[ds]

    total_duration = time.time() - start_total_t
    all_ok = all(r.ok for r in results.values() if r.status != StageStatus.SKIPPED) and len(results) == len(PIPELINE_STAGES)
    failed = [name for name, r in results.items() if r.status == StageStatus.FAILED]
    skipped = [name for name, r in results.items() if r.status == StageStatus.SKIPPED]

    return PipelineReport(
        ok=all_ok,
        results=results,
        total_duration=total_duration,
        failed_stages=failed,
        skipped_stages=skipped,
    )


if __name__ == "__main__":
    report = run_pipeline()
    print("\n" + report.summary())

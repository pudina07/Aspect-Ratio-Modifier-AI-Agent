"""
pipeline_runner.py — Phases 1 to 5: Pipeline Orchestrator & Concurrency Governor

The core orchestration layer. Runs each stage as an isolated subprocess
so a crash in one stage — bad model weights, an OOM, a stray exception —
cannot crash the app or corrupt shared state.

Features:
- DAG dependency-driven concurrency: ocr_pass runs in parallel with transcribe -> analyze_script.
- Immediate failure isolation and downstream dependency pruning without pipeline stalls.
- Custom execution flags propagation (e.g. --mock, --qa-overlay).
- Stage timing and execution reporting.
- Workspace artifact cleanup.
"""
import argparse
import concurrent.futures
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

from config import (
    PIPELINE_STAGES, DATA_DIR, BASE_DIR,
    get_downstream_stages, validate_pipeline_dag
)


@dataclass
class StageResult:
    name: str
    ok: bool
    duration: float = 0.0
    stderr: str = ""
    stdout: str = ""
    skipped_reason: Optional[str] = None


@dataclass
class PipelineReport:
    results: List[StageResult] = field(default_factory=list)
    total_duration: float = 0.0

    @property
    def ok(self) -> bool:
        return all(r.ok for r in self.results if not r.skipped_reason) and len(self.results) > 0

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "           PIPELINE EXECUTION SUMMARY REPORT",
            "=" * 60,
        ]
        for r in self.results:
            if r.skipped_reason:
                status = "SKIPPED"
                details = f"({r.skipped_reason})"
            elif r.ok:
                status = "PASS"
                details = f"({r.duration:.2f}s)"
            else:
                status = "FAIL"
                details = f"({r.duration:.2f}s)"
            lines.append(f"  [{r.name:<16}] {status:<8} {details}")

        lines.append("-" * 60)
        overall = "COMPLETED SUCCESSFULLY 🟢" if self.ok else "FAILED 🔴"
        lines.append(f"  Overall Status : {overall}")
        lines.append(f"  Total Duration : {self.total_duration:.2f}s")
        lines.append("=" * 60)
        return "\n".join(lines)


def clean_run_artifacts(data_dir: Path, keep_video: bool = True) -> None:
    """Cleans previous intermediate JSON and MP4 artifacts from data directory."""
    if not data_dir.exists():
        return

    for item in data_dir.iterdir():
        if item.is_file():
            if keep_video and item.name == "video.mp4":
                continue
            if item.suffix in (".json", ".mp4", ".wav", ".tmp") or ".tmp" in item.name:
                try:
                    item.unlink()
                except Exception:
                    pass


def _inputs_ready(stage: dict, data_dir: Path) -> bool:
    return all((data_dir / f).exists() for f in stage["inputs"])


def run_stage(
    stage: dict,
    base_dir: Path,
    data_dir: Optional[Path] = None,
    mock: bool = False,
    extra_args: Optional[List[str]] = None
) -> StageResult:
    """Run one stage script as an isolated subprocess."""
    script_path = base_dir / stage["script"]
    cmd = [sys.executable, str(script_path)]
    if mock:
        cmd.append("--mock")
    if extra_args:
        cmd.extend(extra_args)

    env = os.environ.copy()
    if data_dir:
        env["AUTO_REFRAME_DATA_DIR"] = str(data_dir)

    start_time = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=base_dir,
            env=env,
            capture_output=True,
            text=True,
        )
        elapsed = time.time() - start_time
        return StageResult(
            name=stage["name"],
            ok=proc.returncode == 0,
            duration=elapsed,
            stderr=proc.stderr.strip(),
            stdout=proc.stdout.strip()
        )
    except Exception as e:
        elapsed = time.time() - start_time
        return StageResult(
            name=stage["name"],
            ok=False,
            duration=elapsed,
            stderr=str(e),
            stdout=""
        )


def run_pipeline(
    data_dir: Optional[Path] = None,
    mock: bool = False,
    use_mock: bool = False,
    clean_workspace: bool = False,
    poll_interval: float = 0.05,
    max_workers: int = 4
) -> PipelineReport:
    """
    Executes the full pipeline DAG.
    Launches stages as soon as their declared dependencies are satisfied.
    """
    is_mock = mock or use_mock
    target_data_dir = data_dir or DATA_DIR
    target_data_dir.mkdir(parents=True, exist_ok=True)

    if clean_workspace:
        clean_run_artifacts(target_data_dir, keep_video=True)

    validate_pipeline_dag()

    start_time = time.time()
    pending: Dict[str, dict] = {s["name"]: s for s in PIPELINE_STAGES}
    running: Dict[str, concurrent.futures.Future] = {}
    results: List[StageResult] = []
    blocked_stages: Set[str] = set()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        while pending or running:
            # Schedule ready stages
            for name, stage in list(pending.items()):
                if name in blocked_stages:
                    del pending[name]
                    continue

                if _inputs_ready(stage, target_data_dir):
                    print(f"[{name}] STARTING")
                    running[name] = pool.submit(run_stage, stage, BASE_DIR, target_data_dir, is_mock)
                    del pending[name]

            if not running and pending:
                stuck = [name for name in pending if name not in blocked_stages]
                if stuck:
                    raise RuntimeError(
                        f"Pipeline stalled — dependencies missing for: {', '.join(stuck)}. "
                        f"Check if video.mp4 exists in {target_data_dir} or if upstream stage failed."
                    )
                else:
                    break

            if not running and not pending:
                break

            # Poll running futures
            done_names = [n for n, f in running.items() if f.done()]
            if not done_names:
                time.sleep(poll_interval)
                continue

            for name in done_names:
                fut = running.pop(name)
                res = fut.result()
                results.append(res)
                status = "OK" if res.ok else "FAILED"
                print(f"[{name}] {status} ({res.duration:.2f}s)")
                if not res.ok:
                    if res.stderr:
                        print(f"  -> Error details:\n{res.stderr}", file=sys.stderr)
                    # Prune downstream stages that depend on this stage
                    cascading_blocked = get_downstream_stages(name)
                    for blocked_name in cascading_blocked:
                        if blocked_name not in blocked_stages:
                            blocked_stages.add(blocked_name)
                            skip_res = StageResult(
                                name=blocked_name,
                                ok=False,
                                duration=0.0,
                                skipped_reason=f"Blocked by failed upstream stage '{name}'"
                            )
                            results.append(skip_res)
                            print(f"[{blocked_name}] SKIPPED ({skip_res.skipped_reason})")

    total_elapsed = time.time() - start_time
    return PipelineReport(results=results, total_duration=total_elapsed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true", help="Run entire pipeline in mock mode")
    parser.add_argument("--video", type=Path, help="Optional input video to stage before running")
    parser.add_argument("--clean", action="store_true", help="Clean data directory before running")
    args = parser.parse_args()

    if args.video:
        import shutil
        target = DATA_DIR / "video.mp4"
        shutil.copy(args.video, target)
        print(f"Staged {args.video} to {target}")

    report = run_pipeline(mock=args.mock, clean_workspace=args.clean)
    print("\n" + report.summary())

    if not report.ok:
        sys.exit(1)


if __name__ == "__main__":
    main()

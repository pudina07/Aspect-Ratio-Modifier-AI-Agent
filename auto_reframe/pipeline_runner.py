"""
pipeline_runner.py — Phase 1 & 2: Pipeline Orchestrator & Concurrency Governor

The orchestration layer. Runs each stage as an isolated subprocess
so a crash in one stage — bad model weights, an OOM, a stray exception —
cannot crash the app or leave corrupted state.

Features:
- DAG dependency-driven concurrency: ocr_pass runs in parallel with transcribe -> analyze_script.
- Immediate failure isolation and downstream dependency pruning without pipeline stalls.
- Custom execution flags propagation (e.g. --mock).
- Stage timing and execution reporting.
"""
import argparse
import concurrent.futures
import os
import subprocess
import sys
import time
from dataclasses import dataclass
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
    poll_interval: float = 0.05,
    max_workers: int = 4
) -> List[StageResult]:
    """
    Executes the pipeline DAG.
    Launches stages as soon as dependencies are satisfied.
    """
    target_data_dir = data_dir or DATA_DIR
    validate_pipeline_dag()

    pending: Dict[str, dict] = {s["name"]: s for s in PIPELINE_STAGES}
    running: Dict[str, concurrent.futures.Future] = {}
    results: List[StageResult] = []
    blocked_stages: Set[str] = set()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        while pending or running:
            # Check for newly ready stages
            for name, stage in list(pending.items()):
                if name in blocked_stages:
                    del pending[name]
                    continue

                if _inputs_ready(stage, target_data_dir):
                    print(f"[{name}] STARTING")
                    running[name] = pool.submit(run_stage, stage, BASE_DIR, target_data_dir, mock)
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

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true", help="Run entire pipeline in mock mode")
    parser.add_argument("--video", type=Path, help="Optional input video to stage before running")
    args = parser.parse_args()

    if args.video:
        import shutil
        target = DATA_DIR / "video.mp4"
        shutil.copy(args.video, target)
        print(f"Staged {args.video} to {target}")

    results = run_pipeline(mock=args.mock)
    failed = [r for r in results if not r.ok and not r.skipped_reason]
    if failed:
        print(f"\nPipeline finished with {len(failed)} stage failures.")
        sys.exit(1)
    else:
        print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()

"""
run_tests.py — Master Test Runner & Quality Audit Suite (Phases 1 to 5)

Runs all unit, stress, mathematical invariant, and end-to-end integration tests:
1. test_contracts.py (Data contracts & schemas)
2. test_io_json.py (Atomic JSON I/O & NumPy serialization)
3. test_config.py (DAG structure & safe zones)
4. test_phase1_thorough.py (Phase 1 architectural foundations)
5. test_phase2_stress.py (Phase 2 speech STT & debouncing)
6. test_phase3_stress.py (Phase 3 MediaPipe tracking & OCR)
7. test_phase4_stress.py (Phase 4 dual-aspect smoothing & physics)
8. test_phase5_stress.py (Phase 5 platform rendering & compositing)
9. test_all_phases_integration.py (Cross-phase full integration)
"""
import importlib
import os
import sys
import time
from pathlib import Path

# Force UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PHASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PHASE_DIR.parent
if str(PHASE_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE_DIR))


TEST_MODULES = [
    ("Phase 1: Contracts & Data Types", "tests.test_contracts"),
    ("Phase 1: Atomic I/O & NumPy Serialization", "tests.test_io_json"),
    ("Phase 1: DAG Dependency Graph & Config", "tests.test_config"),
    ("Phase 1: Rigorous Architectural Stress", "tests.test_phase1_thorough"),
    ("Phase 2: Speech STT, NLP Cues & Debouncing", "tests.test_phase2_stress"),
    ("Phase 3: MediaPipe Vision & EasyOCR Text", "tests.test_phase3_stress"),
    ("Phase 4: Dual-Aspect Coordinator & Smoothing", "tests.test_phase4_stress"),
    ("Phase 5: Platform Rendering & Compositing", "tests.test_phase5_stress"),
    ("Phases 1-5: Full Cross-Phase Integration", "tests.test_all_phases_integration"),
]


def run_all_tests():
    print("=" * 75)
    print("🏛️  CONTEXT-AWARE AUTO-REFRAME: MASTER TEST SUITE (PHASES 1 TO 5)")
    print("=" * 75)

    passed_count = 0
    failed_count = 0
    results = []

    start_total = time.time()

    for name, module_name in TEST_MODULES:
        print(f"\n▶ Running: {name} ({module_name})...")
        t0 = time.time()
        try:
            mod = importlib.import_module(module_name)
            if hasattr(mod, "main"):
                mod.main()
            elif hasattr(mod, "run_all_stress_tests"):
                mod.run_all_stress_tests()
            elif hasattr(mod, "run_all_phase3_stress_tests"):
                mod.run_all_phase3_stress_tests()
            elif hasattr(mod, "run_full_integration_audit"):
                mod.run_full_integration_audit()
            else:
                for attr_name in dir(mod):
                    if attr_name.startswith("test_") and callable(getattr(mod, attr_name)):
                        getattr(mod, attr_name)()

            elapsed = time.time() - t0
            passed_count += 1
            results.append((name, True, elapsed, ""))
            print(f"  🟢 {name} PASSED ({elapsed:.2f}s)")
        except Exception as e:
            elapsed = time.time() - t0
            failed_count += 1
            results.append((name, False, elapsed, str(e)))
            print(f"  🔴 {name} FAILED ({elapsed:.2f}s): {e}")
            import traceback
            traceback.print_exc()

    total_time = time.time() - start_total

    print("\n" + "=" * 75)
    print("                    FINAL QUALITY CERTIFICATION REPORT")
    print("=" * 75)
    for name, ok, elapsed, err in results:
        status_str = "PASS 🟢" if ok else "FAIL 🔴"
        print(f"  {status_str:<12} | {name:<45} | {elapsed:.2f}s")
        if not ok and err:
            print(f"               Error: {err}")

    print("-" * 75)
    print(f"  Total Suites Executed : {len(TEST_MODULES)}")
    print(f"  Suites Passed         : {passed_count}")
    print(f"  Suites Failed         : {failed_count}")
    print(f"  Total Duration        : {total_time:.2f}s")
    print(f"  Overall Verdict       : {'CERTIFIED (100% PASS) 🟢' if failed_count == 0 else 'NEEDS WORK 🔴'}")
    print("=" * 75 + "\n")

    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(run_all_tests())

"""Rerun the complete project pipeline from raw data to final outputs.

Run from the project root with the project virtual environment:
    .venv\\Scripts\\python.exe .\\run_all.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Append new phases here as the project grows.
STEPS = [
    ("Phase 1 - data extraction and integrity check", PROJECT_ROOT / "src" / "phase1_data_check.py"),
    ("Phase 2 - preprocessing and healthy split", PROJECT_ROOT / "src" / "data.py"),
    ("Phase 2 - preprocessing sanity checks", PROJECT_ROOT / "src" / "sanity_checks.py"),
    ("Phase 2 - labeling and timing diagnostic", PROJECT_ROOT / "src" / "time_diagnostic.py"),
    ("Phase 3 - baseline models", PROJECT_ROOT / "src" / "baselines.py"),
    ("Phase 3 - summary report", PROJECT_ROOT / "src" / "phase3_summary.py"),
    ("Phase 4 - proposed TCN-BiLSTM training", PROJECT_ROOT / "src" / "train.py"),
    ("Phase 4 - summary report", PROJECT_ROOT / "src" / "phase4_summary.py"),
]


def main() -> int:
    print("=" * 80, flush=True)
    print("Running full pipeline", flush=True)
    print(f"Python: {sys.executable}", flush=True)
    print(f"Root  : {PROJECT_ROOT}", flush=True)
    print("=" * 80, flush=True)

    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"

    for label, script in STEPS:
        if not script.exists():
            print(f"\nERROR: missing pipeline step: {script}")
            return 1

        print(f"\n--- {label} ---", flush=True)
        completed = subprocess.run(
            [sys.executable, str(script)],
            cwd=PROJECT_ROOT,
            env=environment,
            check=False,
        )
        if completed.returncode != 0:
            print(f"\nPipeline stopped: {label} failed with exit code {completed.returncode}.")
            return completed.returncode

    print("\nPipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

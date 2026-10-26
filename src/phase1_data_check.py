"""Phase 1: extract GPVS-Faults and verify that all 16 CSV files load.

This script performs no cleaning, aggregation, modeling, or fault analysis.
It only checks the raw data contract and writes an inventory to results/.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = PROJECT_ROOT / "CSV_Files.zip"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CSV_DIR = RAW_DIR / "CSV_Files"
RESULTS_DIR = PROJECT_ROOT / "results"

TIME_COLUMN = "Time"
FEATURES = [
    "Ipv",
    "Vpv",
    "Vdc",
    "ia",
    "ib",
    "ic",
    "va",
    "vb",
    "vc",
    "Iabc",
    "If",
    "Vabc",
    "Vf",
]
EXPECTED_COLUMNS = [TIME_COLUMN] + FEATURES
MODES = {"L": "IPPT", "M": "MPPT"}
EXPECTED_STEMS = tuple(f"F{i}{suffix}" for i in range(8) for suffix in ("L", "M"))
ZIP_MEMBER_TEMPLATE = "CSV_Files/{stem}.csv"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    """Return the current git commit hash if available."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return completed.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def ensure_extracted() -> list[str]:
    """Extract missing or incomplete CSVs from the archive into data/raw/.

    Returns the list of stems that were written in this run.
    """
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"Dataset archive not found: {ZIP_PATH}")

    CSV_DIR.mkdir(parents=True, exist_ok=True)
    extracted: list[str] = []

    with zipfile.ZipFile(ZIP_PATH, "r") as archive:
        members = {info.filename: info for info in archive.infolist()}

        for stem in EXPECTED_STEMS:
            member_name = ZIP_MEMBER_TEMPLATE.format(stem=stem)
            if member_name not in members:
                raise FileNotFoundError(
                    f"{member_name} is missing from {ZIP_PATH.name}"
                )

            member = members[member_name]
            target = CSV_DIR / f"{stem}.csv"

            # Re-extract if absent or if the size does not match the archive.
            if target.exists() and target.stat().st_size == member.file_size:
                continue

            archive.extract(member, RAW_DIR)
            extracted.append(stem)

    return extracted


def check_file(path: Path, mode: str, healthy: bool) -> dict[str, object]:
    """Load one CSV and return a flat, JSON-safe validation record."""
    record: dict[str, object] = {
        "file": path.name,
        "stem": path.stem,
        "path": path.relative_to(PROJECT_ROOT).as_posix(),
        "mode": mode,
        "healthy": healthy,
        "loaded": False,
        "load_error": "",
        "rows": 0,
        "columns": 0,
        "columns_ok": False,
        "missing_columns": "",
        "extra_columns": "",
        "missing_values": 0,
        "infinite_values": 0,
        "duplicate_rows": 0,
        "time_strictly_increasing": False,
        "median_dt_s": None,
        "sampling_hz": None,
        "duration_s": None,
        "start_time_s": None,
        "end_time_s": None,
        "size_mb": round(path.stat().st_size / (1024 * 1024), 6),
    }

    try:
        frame = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001 - record the failure, do not abort
        record["load_error"] = f"{type(exc).__name__}: {exc}"
        return record

    record["loaded"] = True
    record["rows"] = int(len(frame))
    record["columns"] = int(frame.shape[1])

    actual_columns = [str(column) for column in frame.columns]
    missing_columns = [c for c in EXPECTED_COLUMNS if c not in actual_columns]
    extra_columns = [c for c in actual_columns if c not in EXPECTED_COLUMNS]
    record["missing_columns"] = ";".join(missing_columns)
    record["extra_columns"] = ";".join(extra_columns)
    record["columns_ok"] = actual_columns == EXPECTED_COLUMNS

    record["missing_values"] = int(frame.isna().sum().sum())
    record["duplicate_rows"] = int(frame.duplicated().sum())

    if all(column in frame.columns for column in FEATURES):
        numeric = frame[FEATURES].apply(pd.to_numeric, errors="coerce")
        record["infinite_values"] = int(np.isinf(numeric.to_numpy()).sum())

    if TIME_COLUMN in frame.columns:
        time_values = pd.to_numeric(frame[TIME_COLUMN], errors="coerce")
        diffs = time_values.diff().dropna()
        positive_diffs = diffs[diffs > 0]

        record["time_strictly_increasing"] = bool(
            len(time_values) > 1
            and not time_values.isna().any()
            and bool((diffs > 0).all())
        )

        if len(time_values) > 0 and not time_values.isna().any():
            record["start_time_s"] = float(time_values.iloc[0])
            record["end_time_s"] = float(time_values.iloc[-1])
            record["duration_s"] = float(time_values.iloc[-1] - time_values.iloc[0])

        if len(positive_diffs) > 0:
            median_dt = float(positive_diffs.median())
            record["median_dt_s"] = median_dt
            if median_dt > 0:
                record["sampling_hz"] = 1.0 / median_dt

    return record


def main() -> int:
    print("=" * 80)
    print("GPVS-Faults | Phase 1 data check")
    print("=" * 80)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Archive      : {ZIP_PATH}")

    if not ZIP_PATH.exists():
        print(f"ERROR: archive not found at {ZIP_PATH}")
        return 1

    print(f"Archive SHA256: {sha256_file(ZIP_PATH)}")
    extracted_now = ensure_extracted()
    print(f"Extracted now : {', '.join(extracted_now) if extracted_now else 'nothing (already present)'}")

    discovered = sorted(CSV_DIR.glob("*.csv"))
    discovered_stems = {path.stem for path in discovered}
    expected_stems = set(EXPECTED_STEMS)
    unexpected_stems = sorted(discovered_stems - expected_stems)
    missing_stems = sorted(expected_stems - discovered_stems)

    print(f"\nCSV directory: {CSV_DIR}")
    print(f"CSV files found: {len(discovered)} (expected {len(EXPECTED_STEMS)})")
    print("\nFile list:")
    for path in discovered:
        print(f"  {path.name:<10} {path.stat().st_size / (1024 * 1024):8.2f} MB")

    records: list[dict[str, object]] = []
    for stem in EXPECTED_STEMS:
        path = CSV_DIR / f"{stem}.csv"
        if not path.exists():
            continue
        mode = MODES[stem[-1]]
        healthy = stem[1] == "0"
        records.append(check_file(path, mode=mode, healthy=healthy))

    inventory = pd.DataFrame(records)
    if inventory.empty:
        print("\nERROR: no CSV files were loaded.")
        return 1

    inventory["missing_columns"] = inventory["missing_columns"].fillna("")
    inventory["extra_columns"] = inventory["extra_columns"].fillna("")
    inventory["load_error"] = inventory["load_error"].fillna("")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    inventory_path = RESULTS_DIR / "data_inventory.csv"
    inventory.to_csv(inventory_path, index=False)

    n_expected = len(EXPECTED_STEMS)
    n_found = len(discovered)
    n_loaded = int(inventory["loaded"].sum())
    all_present = n_found == n_expected and not missing_stems and not unexpected_stems
    all_loaded = n_loaded == n_expected
    all_columns_ok = bool(inventory["columns_ok"].all())
    all_time_increasing = bool(inventory["time_strictly_increasing"].all())
    total_rows = int(inventory["rows"].sum())
    total_missing = int(inventory["missing_values"].sum())
    total_infinite = int(inventory["infinite_values"].sum())
    total_duplicates = int(inventory["duplicate_rows"].sum())

    loaded_rows = inventory[inventory["loaded"]]
    rows_by_mode = {
        str(mode): int(group["rows"].sum())
        for mode, group in loaded_rows.groupby("mode", sort=True)
    }
    rows_by_class = {
        "healthy": int(loaded_rows.loc[loaded_rows["healthy"], "rows"].sum()),
        "fault": int(loaded_rows.loc[~loaded_rows["healthy"], "rows"].sum()),
    }

    summary = {
        "phase": "phase_1_data_check",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "archive": ZIP_PATH.name,
        "archive_sha256": sha256_file(ZIP_PATH),
        "archive_size_mb": round(ZIP_PATH.stat().st_size / (1024 * 1024), 6),
        "csv_directory": CSV_DIR.relative_to(PROJECT_ROOT).as_posix(),
        "expected_columns": EXPECTED_COLUMNS,
        "features": FEATURES,
        "expected_file_count": n_expected,
        "discovered_file_count": n_found,
        "loaded_file_count": n_loaded,
        "missing_stems": missing_stems,
        "unexpected_stems": unexpected_stems,
        "extracted_now": extracted_now,
        "checks": {
            "all_expected_files_present": bool(all_present),
            "all_files_loaded": bool(all_loaded),
            "all_columns_ok": bool(all_columns_ok),
            "all_time_strictly_increasing": bool(all_time_increasing),
        },
        "totals": {
            "rows": total_rows,
            "rows_by_mode": rows_by_mode,
            "rows_by_class": rows_by_class,
            "missing_values": total_missing,
            "infinite_values": total_infinite,
            "duplicate_rows": total_duplicates,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": package_version("numpy"),
            "pandas": package_version("pandas"),
            "scikit-learn": package_version("scikit-learn"),
            "tensorflow": package_version("tensorflow"),
            "matplotlib": package_version("matplotlib"),
            "seaborn": package_version("seaborn"),
            "scipy": package_version("scipy"),
        },
        "git_commit": git_commit(),
        "files": records,
    }

    summary_path = RESULTS_DIR / "data_check.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")

    print("\nPer-file checks:")
    header = (
        f"{'File':<8} {'Mode':<6} {'Rows':>10} {'Cols':>5} "
        f"{'Missing':>8} {'Inf':>5} {'Dup':>6} {'Fs(Hz)':>9} {'Duration(s)':>12}"
    )
    print(header)
    print("-" * len(header))
    for _, row in inventory.iterrows():
        if not bool(row["loaded"]):
            print(f"{row['file']:<8} LOAD FAILED: {row['load_error']}")
            continue
        print(
            f"{row['file']:<8} {row['mode']:<6} {int(row['rows']):>10,} "
            f"{int(row['columns']):>5} {int(row['missing_values']):>8,} "
            f"{int(row['infinite_values']):>5} {int(row['duplicate_rows']):>6,} "
            f"{float(row['sampling_hz']):>9.1f} {float(row['duration_s']):>12.3f}"
        )

    print("\nChecks:")
    print(f"  All expected CSVs present : {'YES' if all_present else 'NO'}")
    print(f"  All 16 CSVs loaded        : {'YES' if all_loaded else 'NO'}")
    print(f"  Columns match schema      : {'YES' if all_columns_ok else 'NO'}")
    print(f"  Time strictly increasing  : {'YES' if all_time_increasing else 'NO'}")
    print(f"  Total rows loaded         : {total_rows:,}")
    print(f"  Total missing values      : {total_missing:,}")
    print(f"  Total infinite values     : {total_infinite:,}")
    print(f"  Total duplicate rows      : {total_duplicates:,}")
    print(f"  Rows by mode              : {rows_by_mode}")
    print(f"  Rows by class             : {rows_by_class}")
    print(f"\nSaved: {inventory_path}")
    print(f"Saved: {summary_path}")

    ok = all_present and all_loaded and all_columns_ok and all_time_increasing
    if not ok:
        print("\nRESULT: FAILED - see checks above and results/data_check.json")
        return 1

    print("\nRESULT: PASS - all 16 CSVs loaded successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

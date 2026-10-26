"""Sanity checks for the Phase 2 preprocessed GPVS-Faults dataset.

This script rebuilds the dataset with ``src.data.build_dataset`` and asserts:

* all 16 files are present with the expected window shapes;
* no NaN or infinite values exist in windows, timestamps, or decimated rows;
* window counts follow the 100/10 windowing rule for every file;
* window timestamps come from the Time column and are aligned with
  ``Time[::10]``;
* the StandardScaler of each mode was fitted on healthy training rows only,
  and is applied to every file of that mode;
* train/val/test window indices are disjoint, chronological, separated by the
  configured 10-window gap, and share no raw samples;
* the healthy split sizes follow the 60/20/20 rule with the gap excluded.

It writes ``results/data_summary.csv`` (per-file window counts),
``results/split_summary.csv`` (per-split counts), and
``results/sanity_checks.json`` (machine-readable check results).

Run::

    .venv\\Scripts\\python.exe .\\src\\sanity_checks.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (  # noqa: E402
    FEATURES,
    MODES,
    RESULTS_DIR,
    SPLIT_FRACTIONS,
    SPLIT_GAP_WINDOWS,
    STEMS,
    STRIDE,
    TARGET_FS,
    WINDOW_SIZE,
    build_dataset,
    environment_info,
    file_counts_frame,
    git_commit,
    split_summary_frame,
    train_row_bounds,
    window_count,
)

CHECK_RESULTS: list[dict[str, object]] = []


def check(name: str, passed: bool, detail: str) -> bool:
    """Record and print one named assertion."""
    passed = bool(passed)
    CHECK_RESULTS.append({"name": name, "passed": passed, "detail": detail})
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    return passed


def format_table(frame: pd.DataFrame) -> str:
    """Render a summary table with plain ASCII formatting."""
    formatters: dict[str, object] = {}
    for column in ("native_rows", "decimated_rows", "windows", "first_window",
                   "last_window", "first_sample", "last_sample"):
        if column in frame.columns:
            formatters[column] = "{:,}".format
    if "measured_fs_hz" in frame.columns:
        formatters["measured_fs_hz"] = "{:.2f}".format
    for column in ("decimated_start_s", "decimated_end_s", "start_time_s",
                   "end_time_s"):
        if column in frame.columns:
            formatters[column] = "{:.4f}".format
    if "fraction_of_healthy_windows" in frame.columns:
        formatters["fraction_of_healthy_windows"] = "{:.4f}".format
    return frame.to_string(index=False, formatters=formatters)


def tree(values: object) -> list[np.ndarray]:
    """Flatten a dict of numpy arrays into a list for finite-value checks."""
    if isinstance(values, dict):
        arrays: list[np.ndarray] = []
        for value in values.values():
            arrays.extend(tree(value))
        return arrays
    if isinstance(values, np.ndarray):
        return [values]
    return []


def start_indices(n_samples: int) -> np.ndarray:
    """Window start indices used by ``src.data.make_windows``."""
    if n_samples < WINDOW_SIZE:
        return np.empty(0, dtype=np.int64)
    return np.arange(0, n_samples - WINDOW_SIZE + 1, STRIDE)


def main() -> int:
    print("=" * 80)
    print("GPVS-Faults | Phase 2 preprocessing sanity checks")
    print("=" * 80)
    print(f"Project root : {PROJECT_ROOT}")

    data = build_dataset(verbose=False)
    windows = data["windows"]
    times = data["times"]
    raw = data["raw_decimated"]
    decimated_time = data["time_decimated"]
    splits = data["splits"]
    scalers = data["scalers"]

    counts_frame = file_counts_frame(data)
    split_frame = split_summary_frame(data)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    counts_path = RESULTS_DIR / "data_summary.csv"
    split_path = RESULTS_DIR / "split_summary.csv"
    counts_frame.to_csv(counts_path, index=False)
    split_frame.to_csv(split_path, index=False)

    print("\nPer-file window counts:")
    print(format_table(counts_frame))
    print("\nHealthy split window counts:")
    print(format_table(split_frame))

    print("\nChecks:")

    # ------------------------------------------------------------------
    # Structure and shapes
    # ------------------------------------------------------------------
    expected_names = set(STEMS)
    all_names = (
        set(windows) == expected_names
        and set(times) == expected_names
        and set(raw) == expected_names
        and set(decimated_time) == expected_names
    )
    check(
        "all 16 files present",
        all_names,
        f"{len(windows)} window sets, {len(times)} time sets, "
        f"{len(raw)} decimated row sets",
    )

    check(
        "13 features in documented order",
        data["features"] == FEATURES,
        ", ".join(data["features"]),
    )

    shape_problems = [
        stem
        for stem in STEMS
        if windows[stem].ndim != 3
        or windows[stem].shape[1:] != (WINDOW_SIZE, len(FEATURES))
        or windows[stem].dtype != np.float32
        or times[stem].ndim != 2
        or times[stem].shape != (windows[stem].shape[0], 2)
        or raw[stem].shape[1] != len(FEATURES)
    ]
    check(
        "window shapes are (n, 100, 13) float32 with (n, 2) times",
        not shape_problems,
        "all files conform" if not shape_problems else str(shape_problems),
    )

    # ------------------------------------------------------------------
    # Finite values
    # ------------------------------------------------------------------
    numeric_arrays = (
        tree(windows) + tree(times) + tree(raw) + tree(decimated_time)
    )
    nan_total = sum(int(np.isnan(array).sum()) for array in numeric_arrays)
    inf_total = sum(int(np.isinf(array).sum()) for array in numeric_arrays)
    check(
        "no NaN values",
        nan_total == 0,
        f"{nan_total} NaN values in windows/times/decimated rows",
    )
    check(
        "no infinite values",
        inf_total == 0,
        f"{inf_total} infinite values in windows/times/decimated rows",
    )

    # ------------------------------------------------------------------
    # Window counts and decimated sampling rate
    # ------------------------------------------------------------------
    count_problems = [
        stem
        for stem in STEMS
        if windows[stem].shape[0] != window_count(len(raw[stem]))
    ]
    check(
        "window counts follow 1 + (n - 100) // 10",
        not count_problems,
        f"{sum(windows[stem].shape[0] for stem in STEMS):,} windows over 16 files"
        if not count_problems
        else str(count_problems),
    )

    fs_values = [
        1.0 / float(np.median(np.diff(decimated_time[stem]))) for stem in STEMS
    ]
    fs_ok = all(abs(fs - TARGET_FS) / TARGET_FS < 0.01 for fs in fs_values)
    check(
        "decimated sampling rate is 1 kHz",
        fs_ok,
        f"measured range {min(fs_values):.2f}-{max(fs_values):.2f} Hz",
    )

    # ------------------------------------------------------------------
    # Time alignment and window timestamps
    # ------------------------------------------------------------------
    start_ok = end_ok = monotonic_ok = True
    lengths_ok = True
    for stem in STEMS:
        starts = start_indices(len(raw[stem]))
        reference_start = decimated_time[stem][starts]
        reference_end = decimated_time[stem][starts + WINDOW_SIZE - 1]
        start_ok = start_ok and np.array_equal(times[stem][:, 0], reference_start)
        end_ok = end_ok and np.array_equal(times[stem][:, 1], reference_end)
        lengths_ok = lengths_ok and len(starts) == windows[stem].shape[0]
        if len(starts) > 1:
            monotonic_ok = monotonic_ok and bool(
                np.all(np.diff(times[stem][:, 0]) > 0)
            )

    check(
        "window start times equal Time[::10] at each window start",
        start_ok and lengths_ok,
        "all starts aligned with the decimated Time column",
    )
    check(
        "window end times equal the Time value of sample start+99",
        end_ok,
        "all end timestamps taken from the Time column",
    )
    check(
        "window start times strictly increase",
        monotonic_ok,
        "every file is in chronological order",
    )

    spans = np.concatenate([times[stem][:, 1] - times[stem][:, 0] for stem in STEMS])
    expected_span = (WINDOW_SIZE - 1) / TARGET_FS
    span_ok = abs(float(np.median(spans)) - expected_span) <= 0.001
    check(
        "median window span is 99 ms",
        span_ok,
        f"median {float(np.median(spans)):.6f} s, expected {expected_span:.3f} s",
    )

    # ------------------------------------------------------------------
    # Scaler fitted on healthy training rows only
    # ------------------------------------------------------------------
    scaler_details: dict[str, dict[str, float | int]] = {}
    for suffix, label in MODES.items():
        stem = f"F0{suffix}"
        scaler = scalers[suffix]
        start, stop = train_row_bounds(splits[suffix])
        fit_rows = raw[stem][start:stop]

        expected_mean = fit_rows.mean(axis=0)
        expected_std = fit_rows.std(axis=0, ddof=0)
        expected_scale = np.where(expected_std == 0.0, 1.0, expected_std)

        mean_ok = bool(
            np.allclose(scaler.mean_, expected_mean, rtol=1e-9, atol=1e-12)
        )
        scale_ok = bool(
            np.allclose(scaler.scale_, expected_scale, rtol=1e-9, atol=1e-12)
        )
        sample_count_ok = (
            int(scaler.n_samples_seen_) == fit_rows.shape[0]
            and fit_rows.shape[0] == stop - start
        )
        check(
            f"{label} scaler parameters equal healthy-train statistics",
            isinstance(scaler, StandardScaler)
            and scaler.n_features_in_ == len(FEATURES)
            and mean_ok
            and scale_ok
            and sample_count_ok,
            f"fitted on {int(scaler.n_samples_seen_):,} raw rows "
            f"[{start}, {stop}), 13 features",
        )

        train_rows = np.arange(start, stop)
        train_covered = np.unique(
            np.concatenate(
                [
                    np.arange(
                        int(index) * STRIDE,
                        int(index) * STRIDE + WINDOW_SIZE,
                    )
                    for index in splits[suffix]["train"]
                ]
            )
        )
        coverage_ok = np.array_equal(train_rows, train_covered)

        val_start = int(splits[suffix]["val"][0]) * STRIDE
        val_stop = int(splits[suffix]["val"][-1]) * STRIDE + WINDOW_SIZE
        test_start = int(splits[suffix]["test"][0]) * STRIDE
        test_stop = int(splits[suffix]["test"][-1]) * STRIDE + WINDOW_SIZE
        held_out = np.concatenate(
            (np.arange(val_start, val_stop), np.arange(test_start, test_stop))
        )
        shared_rows = int(np.intersect1d(train_rows, held_out).size)
        check(
            f"{label} scaler never saw validation/test rows",
            coverage_ok and shared_rows == 0,
            f"fit rows are exactly the train-window rows; "
            f"{shared_rows} shared raw rows with val/test",
        )

        whole_mean = raw[stem].mean(axis=0)
        whole_std = raw[stem].std(axis=0, ddof=0)
        whole_scale = np.where(whole_std == 0.0, 1.0, whole_std)
        max_mean_diff = float(np.max(np.abs(scaler.mean_ - whole_mean)))
        max_scale_diff = float(np.max(np.abs(scaler.scale_ - whole_scale)))
        print(
            f"[INFO] {label} scaler differs from whole-file statistics: "
            f"max |mean diff| = {max_mean_diff:.6g}, "
            f"max |scale diff| = {max_scale_diff:.6g}"
        )
        scaler_details[label] = {
            "fit_row_start": int(start),
            "fit_row_stop_exclusive": int(stop),
            "fit_rows": int(stop - start),
            "n_samples_seen": int(scaler.n_samples_seen_),
            "max_abs_mean_diff_vs_whole_file": max_mean_diff,
            "max_abs_scale_diff_vs_whole_file": max_scale_diff,
        }

    # Verify that every file's windows are standardized with its mode scaler.
    window_problems: list[str] = []
    max_window_diff = 0.0
    for stem in STEMS:
        count = windows[stem].shape[0]
        if count == 0:
            continue
        sampled = sorted({0, count // 2, count - 1})
        for index in sampled:
            row_start = index * STRIDE
            expected = scalers[stem[-1]].transform(
                raw[stem][row_start:row_start + WINDOW_SIZE]
            ).astype(np.float32)
            diff = float(np.max(np.abs(windows[stem][index] - expected)))
            max_window_diff = max(max_window_diff, diff)
            if not np.allclose(windows[stem][index], expected, rtol=1e-5, atol=1e-5):
                window_problems.append(f"{stem}[{index}]")
    check(
        "windows are standardized with their mode's scaler",
        not window_problems,
        f"checked first/middle/last window of all 16 files; "
        f"max abs difference {max_window_diff:.3g}",
    )

    # ------------------------------------------------------------------
    # Healthy split integrity
    # ------------------------------------------------------------------
    for suffix, label in MODES.items():
        stem = f"F0{suffix}"
        n_total = int(windows[stem].shape[0])
        n_train = int(np.floor(SPLIT_FRACTIONS[0] * n_total))
        n_val = int(np.floor(SPLIT_FRACTIONS[1] * n_total))
        n_test = n_total - n_train - n_val - 2 * SPLIT_GAP_WINDOWS

        sizes_ok = (
            splits[suffix]["train"].size == n_train
            and splits[suffix]["val"].size == n_val
            and splits[suffix]["test"].size == n_test
        )
        check(
            f"{label} split sizes follow 60/20/20 minus two 10-window gaps",
            sizes_ok,
            f"train={n_train:,}, val={n_val:,}, test={n_test:,}, "
            f"gap=2x{SPLIT_GAP_WINDOWS}, total windows={n_total:,}",
        )

        parts = {
            name: np.asarray(indices, dtype=np.int64)
            for name, indices in splits[suffix].items()
        }
        index_overlaps = {
            f"{a}/{b}": int(np.intersect1d(parts[a], parts[b]).size)
            for a, b in combinations(("train", "val", "test"), 2)
        }
        check(
            f"{label} train/val/test window indices are disjoint",
            all(value == 0 for value in index_overlaps.values()),
            str(index_overlaps),
        )

        boundary_gaps_ok = (
            int(parts["val"][0] - parts["train"][-1] - 1) == SPLIT_GAP_WINDOWS
            and int(parts["test"][0] - parts["val"][-1] - 1) == SPLIT_GAP_WINDOWS
        )
        check(
            f"{label} has a 10-window gap between consecutive parts",
            boundary_gaps_ok,
            f"val starts {SPLIT_GAP_WINDOWS} windows after train, "
            f"test starts {SPLIT_GAP_WINDOWS} windows after val",
        )

        all_indices = np.arange(n_total)
        used = np.concatenate(list(parts.values()))
        gap_indices = np.setdiff1d(all_indices, used)
        expected_gap = np.concatenate(
            (
                np.arange(n_train, n_train + SPLIT_GAP_WINDOWS),
                np.arange(
                    n_train + SPLIT_GAP_WINDOWS + n_val,
                    n_train + 2 * SPLIT_GAP_WINDOWS + n_val,
                ),
            )
        )
        check(
            f"{label} gap windows belong to no split",
            gap_indices.size == 2 * SPLIT_GAP_WINDOWS
            and np.array_equal(gap_indices, expected_gap),
            f"{gap_indices.size} gap windows excluded",
        )

        coverage = {
            name: np.arange(
                int(indices[0]) * STRIDE,
                int(indices[-1]) * STRIDE + WINDOW_SIZE,
            )
            for name, indices in parts.items()
        }
        sample_overlaps = {
            f"{a}/{b}": int(np.intersect1d(coverage[a], coverage[b]).size)
            for a, b in combinations(("train", "val", "test"), 2)
        }
        check(
            f"{label} train/val/test share no raw samples",
            all(value == 0 for value in sample_overlaps.values()),
            str(sample_overlaps),
        )

        chronological_ok = (
            times[stem][parts["train"][-1], 1] < times[stem][parts["val"][0], 0]
            and times[stem][parts["val"][-1], 1] < times[stem][parts["test"][0], 0]
        )
        check(
            f"{label} splits are chronological",
            chronological_ok,
            f"train ends {times[stem][parts['train'][-1], 1]:.3f} s, "
            f"val starts {times[stem][parts['val'][0], 0]:.3f} s, "
            f"test starts {times[stem][parts['test'][0], 0]:.3f} s",
        )

        span_start = float(times[stem][0, 0])
        span_end = float(times[stem][-1, 1])
        check(
            f"{label} split covers every healthy window exactly once",
            int(used.size + gap_indices.size) == n_total
            and int(np.unique(used).size) == int(used.size)
            and span_start < span_end,
            f"train+val+test+gap = {used.size + gap_indices.size:,} "
            f"of {n_total:,} windows",
        )

    # ------------------------------------------------------------------
    # Save machine-readable results
    # ------------------------------------------------------------------
    all_passed = all(bool(result["passed"]) for result in CHECK_RESULTS)
    sanity = {
        "phase": "phase_2_sanity_checks",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "environment": environment_info(),
        "all_passed": all_passed,
        "checks_passed": sum(bool(result["passed"]) for result in CHECK_RESULTS),
        "checks_total": len(CHECK_RESULTS),
        "checks": CHECK_RESULTS,
        "totals": {
            "files": len(STEMS),
            "windows_all_files": int(
                sum(windows[stem].shape[0] for stem in STEMS)
            ),
            "healthy_windows": {
                MODES[suffix]: int(windows[f"F0{suffix}"].shape[0])
                for suffix in ("L", "M")
            },
            "nan_values": nan_total,
            "infinite_values": inf_total,
            "decimated_hz_min": float(min(fs_values)),
            "decimated_hz_max": float(max(fs_values)),
        },
        "scaler_fit": scaler_details,
    }
    sanity_path = RESULTS_DIR / "sanity_checks.json"
    with sanity_path.open("w", encoding="utf-8") as handle:
        json.dump(sanity, handle, indent=2, allow_nan=False)
        handle.write("\n")

    print("\nTotals:")
    print(f"  files                 : {len(STEMS)}")
    print(
        "  windows (all files)   : "
        f"{sum(windows[stem].shape[0] for stem in STEMS):,}"
    )
    for suffix in ("L", "M"):
        print(
            f"  healthy {MODES[suffix]:<5} windows : "
            f"{windows[f'F0{suffix}'].shape[0]:,}"
        )
    print(f"  NaN values            : {nan_total:,}")
    print(f"  infinite values       : {inf_total:,}")
    print(f"\nSaved: {counts_path}")
    print(f"Saved: {split_path}")
    print(f"Saved: {sanity_path}")

    if not all_passed:
        failed = [
            str(result["name"])
            for result in CHECK_RESULTS
            if not result["passed"]
        ]
        print(f"\nRESULT: FAILED - {len(failed)} check(s): {failed}")
        raise AssertionError(f"{len(failed)} sanity check(s) failed: {failed}")

    print(f"\nRESULT: PASS - all {len(CHECK_RESULTS)} checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

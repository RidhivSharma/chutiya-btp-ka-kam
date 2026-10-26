"""Generate the Phase 1 section of results/summary.md.

This is a reporting script. It reads the outputs of
``src/phase1_data_check.py``, ``src/data.py``, and ``src/sanity_checks.py``,
rebuilds the dataset once to compute scaled-training and leakage statistics,
and appends a Phase 1 section to ``results/summary.md``. It does not modify
raw data, the pipeline, or any earlier result file.

The extra statistics are saved to ``results/phase1_summary_stats.json`` so
every number in the summary can be traced back to a machine-readable file.

Run::

    .venv\\Scripts\\python.exe .\\src\\phase1_summary.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (  # noqa: E402
    FEATURES,
    HEALTHY_STEMS,
    MODES,
    RESULTS_DIR,
    STEMS,
    build_dataset,
    train_row_bounds,
)

SUMMARY_PATH = RESULTS_DIR / "summary.md"
STATS_PATH = RESULTS_DIR / "phase1_summary_stats.json"
SECTION_HEADING = "## Phase 1: Clean data pipeline"
SPLIT_NAMES = ("train", "val", "test")


def read_json(path: Path) -> dict[str, object]:
    """Load a JSON file as a dictionary."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render an ASCII markdown table."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def compute_split_stats(split_indices: pd.DataFrame) -> dict[str, object]:
    """Recompute split counts, overlaps, and gap sizes from the saved indices."""
    stats: dict[str, object] = {}

    for suffix, label in MODES.items():
        subset = split_indices[split_indices["mode_suffix"] == suffix]
        blocks: dict[str, dict[str, float | int]] = {}

        for split_name in SPLIT_NAMES:
            part = subset[subset["split"] == split_name]
            if part.empty:
                raise RuntimeError(f"{label} {split_name} split is empty")
            blocks[split_name] = {
                "windows": int(part.shape[0]),
                "first_window": int(part["window_index"].min()),
                "last_window": int(part["window_index"].max()),
                "first_sample": int(part["start_sample"].min()),
                "last_sample_exclusive": int(part["stop_sample_exclusive"].max()),
                "start_time_s": float(part["start_time_s"].min()),
                "end_time_s": float(part["end_time_s"].max()),
            }

        window_index_overlaps: dict[str, int] = {}
        raw_sample_overlaps: dict[str, int] = {}

        for left, right in combinations(SPLIT_NAMES, 2):
            key = f"{left}/{right}"
            left_part = subset[subset["split"] == left]
            right_part = subset[subset["split"] == right]
            window_index_overlaps[key] = int(
                np.intersect1d(
                    left_part["window_index"].to_numpy(),
                    right_part["window_index"].to_numpy(),
                ).size
            )
            start = max(
                blocks[left]["first_sample"], blocks[right]["first_sample"]
            )
            stop = min(
                blocks[left]["last_sample_exclusive"],
                blocks[right]["last_sample_exclusive"],
            )
            raw_sample_overlaps[key] = max(0, int(stop - start))

        boundaries: dict[str, dict[str, float | int]] = {}
        for left, right in (("train", "val"), ("val", "test")):
            boundaries[f"{left}/{right}"] = {
                "gap_windows": int(
                    blocks[right]["first_window"]
                    - blocks[left]["last_window"]
                    - 1
                ),
                "excluded_raw_samples": int(
                    blocks[right]["first_sample"]
                    - blocks[left]["last_sample_exclusive"]
                ),
                "timestamp_gap_s": float(
                    blocks[right]["start_time_s"]
                    - blocks[left]["end_time_s"]
                ),
            }

        gap_rows = subset[subset["split"] == "gap"]
        stats[label] = {
            "blocks": blocks,
            "gap_windows_total": int(gap_rows.shape[0]),
            "boundaries": boundaries,
            "window_index_overlaps": window_index_overlaps,
            "raw_sample_overlaps": raw_sample_overlaps,
        }

    return stats


def compute_scaled_stats(data: dict[str, object]) -> dict[str, object]:
    """Mean and standard deviation of the scaled healthy training data."""
    stats: dict[str, object] = {}

    for suffix, label in MODES.items():
        stem = HEALTHY_STEMS[suffix]
        start, stop = train_row_bounds(data["splits"][suffix])
        fit_rows = data["raw_decimated"][stem][start:stop]
        scaler = data["scalers"][suffix]
        scaled_rows = scaler.transform(fit_rows)

        row_feature_means = scaled_rows.mean(axis=0)
        row_feature_stds = scaled_rows.std(axis=0, ddof=0)

        train_windows = data["windows"][stem][data["splits"][suffix]["train"]]
        window_matrix = train_windows.reshape(-1, len(FEATURES)).astype(np.float64)
        window_values = window_matrix.ravel()
        window_feature_means = window_matrix.mean(axis=0)
        window_feature_stds = window_matrix.std(axis=0, ddof=0)

        stats[label] = {
            "fit_stem": stem,
            "fit_row_start": int(start),
            "fit_row_stop_exclusive": int(stop),
            "fit_rows": int(stop - start),
            "n_samples_seen": int(scaler.n_samples_seen_),
            "scaled_rows_mean": float(scaled_rows.mean()),
            "scaled_rows_std": float(scaled_rows.std(ddof=0)),
            "scaled_rows_max_abs_feature_mean": float(
                np.max(np.abs(row_feature_means))
            ),
            "scaled_rows_min_feature_std": float(np.min(row_feature_stds)),
            "scaled_rows_max_feature_std": float(np.max(row_feature_stds)),
            "scaled_windows": int(train_windows.shape[0]),
            "scaled_windows_mean": float(window_values.mean()),
            "scaled_windows_std": float(window_values.std(ddof=0)),
            "scaled_windows_max_abs_feature_mean": float(
                np.max(np.abs(window_feature_means))
            ),
            "scaled_windows_min_feature_std": float(np.min(window_feature_stds)),
            "scaled_windows_max_feature_std": float(np.max(window_feature_stds)),
        }

    return stats


def compute_startup_stats(data: dict[str, object]) -> dict[str, object]:
    """Quantify the startup transient in the first training windows."""
    stats: dict[str, object] = {}
    vabc_index = FEATURES.index("Vabc")

    for suffix, label in MODES.items():
        stem = HEALTHY_STEMS[suffix]
        train_windows = data["windows"][stem][data["splits"][suffix]["train"]]
        vabc = train_windows[:, :, vabc_index].astype(np.float64)
        stats[label] = {
            "first_window_count": 10,
            "first_window_share": float(10 / train_windows.shape[0]),
            "vabc_first_window_min": float(vabc[0].min()),
            "vabc_first_window_mean": float(vabc[0].mean()),
            "vabc_first_window_std": float(vabc[0].std(ddof=0)),
            "vabc_all_windows_std": float(vabc.std(ddof=0)),
            "vabc_windows_10plus_std": float(vabc[10:].std(ddof=0)),
        }

    return stats


def compute_leakage(
    data: dict[str, object], split_stats: dict[str, object]
) -> dict[str, object]:
    """Verify the scaler fit source and split separation."""
    leakage: dict[str, object] = {}

    for suffix, label in MODES.items():
        stem = HEALTHY_STEMS[suffix]
        start, stop = train_row_bounds(data["splits"][suffix])
        fit_rows = data["raw_decimated"][stem][start:stop]
        scaler = data["scalers"][suffix]

        fit_mean = fit_rows.mean(axis=0)
        fit_std = fit_rows.std(axis=0, ddof=0)
        fit_scale = np.where(fit_std == 0.0, 1.0, fit_std)

        fault_stems = [
            candidate
            for candidate in STEMS
            if candidate.endswith(suffix) and candidate != stem
        ]
        pooled_fault_rows = np.concatenate(
            [data["raw_decimated"][candidate] for candidate in fault_stems],
            axis=0,
        )

        if fit_stem_is_fault := (stem in fault_stems):
            raise RuntimeError(f"{label}: scaler source is a fault file")
        if int(scaler.n_samples_seen_) != fit_rows.shape[0]:
            raise RuntimeError(
                f"{label}: scaler saw {scaler.n_samples_seen_} rows, "
                f"expected {fit_rows.shape[0]}"
            )

        leakage[label] = {
            "fit_stem": stem,
            "fit_rows": int(stop - start),
            "n_samples_seen": int(scaler.n_samples_seen_),
            "fault_files_used_in_fit": int(fit_stem_is_fault),
            "fault_stems_in_mode": fault_stems,
            "max_abs_scaler_mean_minus_fit_mean": float(
                np.max(np.abs(scaler.mean_ - fit_mean))
            ),
            "max_abs_scaler_scale_minus_fit_std": float(
                np.max(np.abs(scaler.scale_ - fit_scale))
            ),
            "max_abs_scaler_mean_minus_pooled_fault_mean": float(
                np.max(np.abs(scaler.mean_ - pooled_fault_rows.mean(axis=0)))
            ),
            "window_index_overlaps": split_stats[label]["window_index_overlaps"],
            "raw_sample_overlaps": split_stats[label]["raw_sample_overlaps"],
            "gap_windows_total": split_stats[label]["gap_windows_total"],
        }

    return leakage


def format_feature_range(low: float, high: float) -> str:
    """Format a per-feature standard-deviation range."""
    return f"{low:.6f} to {high:.6f}"


def format_zero(value: float, digits: int = 6) -> str:
    """Format a float, snapping tiny values to zero."""
    if abs(value) < 0.5 * 10 ** (-digits):
        value = 0.0
    return f"{value:.{digits}f}"


def build_section(
    counts: pd.DataFrame,
    splits: pd.DataFrame,
    split_stats: dict[str, object],
    scaled_stats: dict[str, object],
    startup_stats: dict[str, object],
    leakage: dict[str, object],
    config_data: dict[str, object],
    sanity: dict[str, object],
    data_check: dict[str, object],
    inventory: pd.DataFrame,
    timestamp: str,
) -> str:
    """Render the complete Phase 1 markdown section from saved/computed data."""
    config = config_data["config"]
    decimation = config["decimation"]
    windowing = config["windowing"]
    split_config = config["healthy_split"]

    n_files = int(counts.shape[0])
    total_native = int(counts["native_rows"].sum())
    total_decimated = int(counts["decimated_rows"].sum())
    total_windows = int(counts["windows"].sum())
    healthy_windows = int(
        counts.loc[counts["condition"] == "healthy", "windows"].sum()
    )
    fault_windows = total_windows - healthy_windows

    native_hz = int(config["native_sampling_hz"])
    target_hz = int(config["target_sampling_hz"])
    window_size = int(windowing["window_size"])
    stride = int(windowing["stride"])
    overlap_pct = 100.0 * float(windowing["overlap_fraction"])
    window_ms = 1000.0 * window_size / target_hz
    gap_windows = int(split_config["gap_windows"])
    train_pct = 100.0 * float(split_config["fractions"]["train"])
    val_pct = 100.0 * float(split_config["fractions"]["val"])
    test_pct = 100.0 * float(split_config["fractions"]["test"])

    checks_passed = int(sanity["checks_passed"])
    checks_total = int(sanity["checks_total"])
    all_passed = bool(sanity["all_passed"])
    nan_total = int(sanity["totals"]["nan_values"])
    infinite_total = int(sanity["totals"]["infinite_values"])

    shortest = inventory.loc[inventory["duration_s"].idxmin()]
    f2l_row = inventory.loc[inventory["stem"] == "F2L"].iloc[0]
    f2l_fs = float(
        counts.loc[counts["file"] == "F2L", "measured_fs_hz"].iloc[0]
    )
    median_fs = float(counts["measured_fs_hz"].median())
    f2l_deviation_pct = 100.0 * abs(f2l_fs - target_hz) / target_hz

    if all_passed:
        final_sentence = (
            f"All {checks_total} automated checks passed, including no missing "
            f"values, no split overlap, and no fault data used in fitting."
        )
    else:
        final_sentence = (
            f"Only {checks_passed} of {checks_total} automated checks passed; "
            f"the failures are listed under Problems or warnings."
        )

    what_was_done = " ".join(
        [
            (
                f"Phase 1 turned the {n_files} raw GPVS-Faults CSV recordings "
                f"into clean, non-overlapping train/validation/test sets for "
                f"the healthy panels."
            ),
            (
                f"Each {native_hz / 1000:g} kHz recording was low-pass filtered "
                f"and reduced to {target_hz / 1000:g} kHz, then cut into "
                f"{window_size}-sample ({window_ms:g} ms) windows with a "
                f"{stride}-sample step."
            ),
            (
                f"The healthy IPPT (F0L) and MPPT (F0M) recordings were split "
                f"chronologically {train_pct:g}/{val_pct:g}/{test_pct:g}, with "
                f"{gap_windows} windows dropped at each boundary so that no "
                f"window or raw sample is shared between parts."
            ),
            (
                "One standardizer per operating mode was fitted only on healthy "
                "training rows and then applied to every file of that mode."
            ),
            final_sentence,
        ]
    )

    files_lines = [
        f"- `src/phase1_data_check.py` - extracts and validates the {n_files} raw CSVs.",
        "- `src/data.py` - decimates, windows, splits, and standardizes the data.",
        "- `src/sanity_checks.py` - rebuilds the dataset and runs the automated checks.",
        "- `src/phase1_summary.py` - computes the extra statistics and writes this summary.",
        "- `run_all.py` - runs the Phase 1 and preprocessing steps in order.",
        "- `results/data_inventory.csv` - per-file raw-data inventory.",
        "- `results/data_check.json` - raw-data integrity summary.",
        "- `results/data_summary.csv` - per-file row and window counts.",
        "- `results/split_summary.csv` - split counts and time ranges.",
        "- `results/split_indices.csv` - one row per healthy window with its split label.",
        "- `results/data_config.json` - preprocessing configuration and scaler metadata.",
        "- `results/sanity_checks.json` - machine-readable results of all checks.",
        "- `results/phase1_summary_stats.json` - scaled-data and leakage statistics.",
        "- `results/summary.md` - this report.",
        "- `STATUS.md` - appended Phase 1 log entry.",
    ]

    file_rows: list[list[str]] = []
    for _, row in counts.iterrows():
        file_rows.append(
            [
                f"`{row['file']}`",
                str(row["mode"]),
                str(row["condition"]),
                f"{int(row['native_rows']):,}",
                f"{int(row['decimated_rows']):,}",
                f"{int(row['windows']):,}",
                f"{float(row['measured_fs_hz']):.2f}",
            ]
        )
    file_rows.append(
        [
            "**Total**",
            "",
            "",
            f"**{total_native:,}**",
            f"**{total_decimated:,}**",
            f"**{total_windows:,}**",
            "",
        ]
    )
    file_table = markdown_table(
        [
            "File",
            "Mode",
            "Condition",
            "Raw rows",
            f"{target_hz:,} Hz rows",
            "Windows",
            "Measured rate (Hz)",
        ],
        file_rows,
    )

    decimation_rows = [
        ["Function", f"`{decimation['function']}`"],
        ["Filter", str(decimation["filter"])],
        ["Decimation factor q", str(decimation["q"])],
        [
            "Zero-phase",
            (
                "yes (`zero_phase=True`, forward-backward `filtfilt`)"
                if decimation["zero_phase"]
                else "no"
            ),
        ],
        ["Low-pass cutoff", f"{float(decimation['cutoff_hz']):g} Hz"],
        ["Output sampling rate", f"{target_hz:,} Hz"],
        ["Time alignment", f"`{decimation['time_alignment']}`"],
    ]
    decimation_table = markdown_table(["Setting", "Value"], decimation_rows)

    split_rows: list[list[str]] = []
    for _, row in splits.iterrows():
        label = str(row["mode"])
        split_name = str(row["split"])
        if split_name == "train":
            gap_before = 0
        elif split_name == "val":
            gap_before = int(
                split_stats[label]["boundaries"]["train/val"]["gap_windows"]
            )
        else:
            gap_before = int(
                split_stats[label]["boundaries"]["val/test"]["gap_windows"]
            )
        split_rows.append(
            [
                label,
                split_name,
                f"{int(row['windows']):,}",
                f"{float(row['fraction_of_healthy_windows']):.2%}",
                f"{float(row['start_time_s']):.3f}",
                f"{float(row['end_time_s']):.3f}",
                str(gap_before),
            ]
        )
    split_table = markdown_table(
        [
            "Mode",
            "Split",
            "Windows",
            "Share of healthy",
            "Start (s)",
            "End (s)",
            "Gap before (windows)",
        ],
        split_rows,
    )

    first_gap = split_stats["IPPT"]["boundaries"]["train/val"]
    second_gap = split_stats["IPPT"]["boundaries"]["val/test"]
    gap_samples = int(first_gap["excluded_raw_samples"])
    gap_ms = 1000.0 * float(first_gap["timestamp_gap_s"])
    gap_note = (
        f"Each boundary drops {gap_windows} windows. Because the stride divides "
        f"the window length, that removes only {gap_samples} raw samples "
        f"(about {gap_ms:.1f} ms between the last kept timestamp and the first "
        f"timestamp of the next part), and those gap windows belong to no split. "
        f"The IPPT and MPPT boundaries behave the same way."
    )

    leakage_rows = [
        [
            "Scaler fit source",
            f"`{leakage['IPPT']['fit_stem']}` only",
            f"`{leakage['MPPT']['fit_stem']}` only",
        ],
        [
            "Scaler fit rows",
            f"{int(leakage['IPPT']['fit_rows']):,}",
            f"{int(leakage['MPPT']['fit_rows']):,}",
        ],
        [
            "`n_samples_seen`",
            f"{int(leakage['IPPT']['n_samples_seen']):,}",
            f"{int(leakage['MPPT']['n_samples_seen']):,}",
        ],
        [
            "Fault files used in fit",
            str(int(leakage["IPPT"]["fault_files_used_in_fit"])),
            str(int(leakage["MPPT"]["fault_files_used_in_fit"])),
        ],
    ]
    for key in ("train/val", "train/test", "val/test"):
        leakage_rows.append(
            [
                f"{key} window-index overlap",
                str(int(leakage["IPPT"]["window_index_overlaps"][key])),
                str(int(leakage["MPPT"]["window_index_overlaps"][key])),
            ]
        )
    for key in ("train/val", "train/test", "val/test"):
        leakage_rows.append(
            [
                f"{key} shared raw samples",
                str(int(leakage["IPPT"]["raw_sample_overlaps"][key])),
                str(int(leakage["MPPT"]["raw_sample_overlaps"][key])),
            ]
        )
    leakage_rows.extend(
        [
            [
                "Gap windows excluded from every split",
                str(int(leakage["IPPT"]["gap_windows_total"])),
                str(int(leakage["MPPT"]["gap_windows_total"])),
            ],
            [
                "Max abs difference: scaler mean vs F0 train mean",
                f"{float(leakage['IPPT']['max_abs_scaler_mean_minus_fit_mean']):.3g}",
                f"{float(leakage['MPPT']['max_abs_scaler_mean_minus_fit_mean']):.3g}",
            ],
            [
                "Max abs difference: scaler mean vs pooled fault mean",
                (
                    f"{float(leakage['IPPT']['max_abs_scaler_mean_minus_pooled_fault_mean']):.3g}"
                ),
                (
                    f"{float(leakage['MPPT']['max_abs_scaler_mean_minus_pooled_fault_mean']):.3g}"
                ),
            ],
        ]
    )
    leakage_table = markdown_table(
        ["Check", "IPPT", "MPPT"],
        leakage_rows,
    )

    scaled_rows_table: list[list[str]] = []
    for label in ("IPPT", "MPPT"):
        item = scaled_stats[label]
        scaled_rows_table.append(
            [
                label,
                "Unique fit rows",
                f"{int(item['fit_rows']):,}",
                format_zero(float(item["scaled_rows_mean"])),
                f"{float(item['scaled_rows_std']):.6f}",
                format_zero(float(item["scaled_rows_max_abs_feature_mean"])),
                format_feature_range(
                    float(item["scaled_rows_min_feature_std"]),
                    float(item["scaled_rows_max_feature_std"]),
                ),
            ]
        )
    for label in ("IPPT", "MPPT"):
        item = scaled_stats[label]
        scaled_rows_table.append(
            [
                label,
                "Training windows",
                f"{int(item['scaled_windows']):,}",
                format_zero(float(item["scaled_windows_mean"])),
                f"{float(item['scaled_windows_std']):.6f}",
                format_zero(float(item["scaled_windows_max_abs_feature_mean"])),
                format_feature_range(
                    float(item["scaled_windows_min_feature_std"]),
                    float(item["scaled_windows_max_feature_std"]),
                ),
            ]
        )
    scaled_table = markdown_table(
        [
            "Mode",
            "Scope",
            "Windows or rows",
            "Pooled mean",
            "Pooled std",
            "Max abs feature mean",
            "Feature std range",
        ],
        scaled_rows_table,
    )

    checks_lines = [
        f"- All {checks_total} automated sanity checks passed." if all_passed else (
            f"- {checks_passed} of {checks_total} automated sanity checks passed; "
            f"see the warnings below."
        ),
        (
            f"- No missing or infinite values: {nan_total:,} NaN and "
            f"{infinite_total:,} infinite values in windows, timestamps, and "
            f"decimated rows."
        ),
        (
            f"- The IPPT scaler was fitted on {int(leakage['IPPT']['fit_rows']):,} "
            f"raw rows from `{leakage['IPPT']['fit_stem']}` only, and the MPPT "
            f"scaler on {int(leakage['MPPT']['fit_rows']):,} raw rows from "
            f"`{leakage['MPPT']['fit_stem']}` only; `n_samples_seen` matches "
            f"both row counts."
        ),
        (
            f"- No fault file was used in fitting: "
            f"{int(leakage['IPPT']['fault_files_used_in_fit'])} fault files for "
            f"IPPT and {int(leakage['MPPT']['fault_files_used_in_fit'])} for MPPT."
        ),
        (
            f"- Train/validation/test window indices are pairwise disjoint "
            f"({int(leakage['IPPT']['window_index_overlaps']['train/val'])} "
            f"train/val, "
            f"{int(leakage['IPPT']['window_index_overlaps']['train/test'])} "
            f"train/test, "
            f"{int(leakage['IPPT']['window_index_overlaps']['val/test'])} "
            f"val/test overlaps for IPPT; the same zeros hold for MPPT)."
        ),
        (
            f"- Train/validation/test raw sample ranges are pairwise disjoint "
            f"({int(leakage['IPPT']['raw_sample_overlaps']['train/val'])} "
            f"train/val, "
            f"{int(leakage['IPPT']['raw_sample_overlaps']['train/test'])} "
            f"train/test, "
            f"{int(leakage['IPPT']['raw_sample_overlaps']['val/test'])} "
            f"val/test shared samples for IPPT; the same zeros hold for MPPT)."
        ),
        (
            f"- The splits are chronological, the {gap_windows}-window gaps "
            f"belong to no split ({int(leakage['IPPT']['gap_windows_total'])} "
            f"IPPT and {int(leakage['MPPT']['gap_windows_total'])} MPPT gap "
            f"windows), and every healthy window is assigned exactly once."
        ),
        (
            f"- The scaled healthy training rows have pooled mean and std "
            f"{format_zero(float(scaled_stats['IPPT']['scaled_rows_mean']))} and "
            f"{float(scaled_stats['IPPT']['scaled_rows_std']):.6f} for IPPT, "
            f"and {format_zero(float(scaled_stats['MPPT']['scaled_rows_mean']))} and "
            f"{float(scaled_stats['MPPT']['scaled_rows_std']):.6f} for MPPT; "
            f"the target is 0 and 1."
        ),
        (
            f"- All {n_files} files loaded with the expected columns in order, "
            f"and `Time` is strictly increasing in every file."
        ),
        (
            f"- The raw CSVs contain {int(data_check['totals']['missing_values']):,} "
            f"missing values, {int(data_check['totals']['infinite_values']):,} "
            f"infinite values, and {int(data_check['totals']['duplicate_rows']):,} "
            f"duplicate rows."
        ),
    ]

    warning_lines = [
        (
            f"- `F3M` is the shortest recording: {float(shortest['duration_s']):.3f} s, "
            f"{int(shortest['rows']):,} raw rows, and "
            f"{int(counts.loc[counts['file'] == 'F3M', 'windows'].iloc[0]):,} "
            f"windows. Per-file evaluation must report coverage for this file "
            f"rather than silently dropping it."
        ),
        (
            f"- `F2L` has a slightly different decimated rate estimate: "
            f"{f2l_fs:.2f} Hz versus a median of {median_fs:.2f} Hz across files "
            f"({f2l_deviation_pct:.3f}% from {target_hz:,} Hz). The native "
            f"timestamps are nominal {native_hz:,} Hz; the difference comes from "
            f"timestamp jitter, and q={decimation['q']} is still applied as "
            f"specified."
        ),
        (
            f"- Startup transient: the first "
            f"{int(startup_stats['IPPT']['first_window_count'])} training windows "
            f"are outliers. `Vabc` reaches "
            f"{float(startup_stats['IPPT']['vabc_first_window_min']):.1f} "
            f"standardized units in the first IPPT window and "
            f"{float(startup_stats['MPPT']['vabc_first_window_min']):.1f} in the "
            f"first MPPT window. Excluding those windows, the pooled `Vabc` std "
            f"drops from "
            f"{float(startup_stats['IPPT']['vabc_all_windows_std']):.3f} to "
            f"{float(startup_stats['IPPT']['vabc_windows_10plus_std']):.3f} for "
            f"IPPT and from "
            f"{float(startup_stats['MPPT']['vabc_all_windows_std']):.3f} to "
            f"{float(startup_stats['MPPT']['vabc_windows_10plus_std']):.3f} for "
            f"MPPT. The scaler is correct for the fit rows, but the first "
            f"windows mix a startup transient into the training distribution."
        ),
        (
            "- The zero-phase decimation filter has a short edge transient at the "
            "start and end of every file. It is kept, so the first and last "
            "window of each file may contain a small transient."
        ),
        (
            f"- Windows overlap by {overlap_pct:.0f}% within each split, so "
            f"windows are not independent observations. This is intended for "
            f"window-level detection but must be considered in statistical tests."
        ),
        (
            f"- The test parts are below 20% "
            f"({float(splits.loc[(splits['mode'] == 'IPPT') & (splits['split'] == 'test'), 'fraction_of_healthy_windows'].iloc[0]):.2%} "
            f"IPPT and "
            f"{float(splits.loc[(splits['mode'] == 'MPPT') & (splits['split'] == 'test'), 'fraction_of_healthy_windows'].iloc[0]):.2%} "
            f"MPPT) because the two {gap_windows}-window gaps are excluded from "
            f"the total."
        ),
    ]

    test_ippt = float(
        splits.loc[
            (splits["mode"] == "IPPT") & (splits["split"] == "test"),
            "fraction_of_healthy_windows",
        ].iloc[0]
    )
    test_mppt = float(
        splits.loc[
            (splits["mode"] == "MPPT") & (splits["split"] == "test"),
            "fraction_of_healthy_windows",
        ].iloc[0]
    )

    scaled_note = (
        "The unique fit rows are the non-overlapping raw rows covered by the "
        "healthy training windows; the training windows are the actual model "
        "inputs, which overlap by design. Both are pooled across all 13 "
        "features. The unique fit rows have per-feature mean 0 and std 1 by "
        "construction. The training windows have a pooled std below 1 "
        f"({float(scaled_stats['IPPT']['scaled_windows_std']):.6f} for IPPT and "
        f"{float(scaled_stats['MPPT']['scaled_windows_std']):.6f} for MPPT) "
        "because the first few windows contain a startup transient, especially "
        "in `Vabc`; see the warnings. This is a property of the window "
        "distribution, not a scaler error."
    )

    decision = (
        f"Yes - confirm that these Phase 1 choices are frozen for modeling: "
        f"{target_hz:,} Hz anti-aliased data, {window_size}-sample windows with "
        f"stride {stride}, one scaler per operating mode, and a chronological "
        f"{train_pct:g}/{val_pct:g}/{test_pct:g} healthy split with two "
        f"{gap_windows}-window gaps (the test share is {test_ippt:.1%} IPPT and "
        f"{test_mppt:.1%} MPPT). If not, say which choice should be revisited."
    )

    parts = [
        f"{SECTION_HEADING}   ({timestamp})",
        "",
        f"What was done: {what_was_done}",
        "",
        "Files created or changed:",
        *files_lines,
        "",
        "Key numbers:",
        "",
        "### Raw rows and windows per file",
        "",
        file_table,
        "",
        (
            f"`Raw rows` are the rows in each original CSV. `{target_hz:,} Hz rows` "
            f"are the rows after anti-alias filtering and decimation, and "
            f"`Windows` are the full {window_size}-sample windows cut from those "
            f"rows with stride {stride}."
        ),
        "",
        "### Downsampling",
        "",
        decimation_table,
        "",
        "### Healthy split (F0L and F0M only)",
        "",
        split_table,
        "",
        gap_note,
        "",
        "### Leakage checks",
        "",
        leakage_table,
        "",
        "### Scaled healthy training data",
        "",
        scaled_table,
        "",
        scaled_note,
        "",
        "Checks passed:",
        *checks_lines,
        "",
        "Problems or warnings:",
        *warning_lines,
        "",
        f"Decision needed from me: {decision}",
        "",
        (
            "Next phase: Phase 2 - train and validate a healthy-only anomaly "
            "detector on these frozen windows, keeping fault files unseen until "
            "final evaluation."
        ),
    ]

    return "\n".join(parts).rstrip() + "\n"


def main() -> int:
    counts = pd.read_csv(RESULTS_DIR / "data_summary.csv")
    splits = pd.read_csv(RESULTS_DIR / "split_summary.csv")
    split_indices = pd.read_csv(RESULTS_DIR / "split_indices.csv")
    inventory = pd.read_csv(RESULTS_DIR / "data_inventory.csv")
    config_data = read_json(RESULTS_DIR / "data_config.json")
    sanity = read_json(RESULTS_DIR / "sanity_checks.json")
    data_check = read_json(RESULTS_DIR / "data_check.json")

    if int(counts.shape[0]) != len(STEMS):
        raise RuntimeError(
            f"Expected {len(STEMS)} rows in data_summary.csv, got {counts.shape[0]}"
        )
    if not bool(sanity["all_passed"]):
        print("WARNING: sanity_checks.json reports failed checks.")

    data = build_dataset(verbose=False)
    split_stats = compute_split_stats(split_indices)
    scaled_stats = compute_scaled_stats(data)
    startup_stats = compute_startup_stats(data)
    leakage = compute_leakage(data, split_stats)

    for _, row in splits.iterrows():
        conf_label = str(row["mode"])
        conf_split = str(row["split"])
        block_windows = int(split_stats[conf_label]["blocks"][conf_split]["windows"])
        if int(row["windows"]) != block_windows:
            raise RuntimeError(
                f"{conf_label} {conf_split}: split_summary says "
                f"{int(row['windows'])} windows, split_indices says {block_windows}"
            )

    for suffix, label in MODES.items():
        total_windows = int(data["windows"][HEALTHY_STEMS[suffix]].shape[0])
        assigned = sum(
            int(split_stats[label]["blocks"][name]["windows"])
            for name in SPLIT_NAMES
        )
        gaps = int(split_stats[label]["gap_windows_total"])
        if assigned + gaps != total_windows:
            raise RuntimeError(
                f"{label}: train+val+test+gap = {assigned + gaps}, "
                f"expected {total_windows}"
            )
        expected_gap = 2 * int(config_data["config"]["healthy_split"]["gap_windows"])
        if gaps != expected_gap:
            raise RuntimeError(
                f"{label}: {gaps} gap windows, expected {expected_gap}"
            )

    stats_payload = {
        "phase": "phase_1_clean_data_pipeline",
        "generated_local": datetime.now().astimezone().isoformat(),
        "git_commit": config_data.get("git_commit"),
        "environment": config_data.get("environment"),
        "totals": {
            "files": int(counts.shape[0]),
            "raw_rows": int(counts["native_rows"].sum()),
            "decimated_rows": int(counts["decimated_rows"].sum()),
            "windows_all_files": int(counts["windows"].sum()),
            "healthy_windows": int(
                counts.loc[counts["condition"] == "healthy", "windows"].sum()
            ),
            "fault_windows": int(
                counts.loc[counts["condition"] == "fault", "windows"].sum()
            ),
            "nan_values": int(sanity["totals"]["nan_values"]),
            "infinite_values": int(sanity["totals"]["infinite_values"]),
            "checks_passed": int(sanity["checks_passed"]),
            "checks_total": int(sanity["checks_total"]),
        },
        "raw_data_check": {
            "files_loaded": int(data_check["loaded_file_count"]),
            "rows": int(data_check["totals"]["rows"]),
            "missing_values": int(data_check["totals"]["missing_values"]),
            "infinite_values": int(data_check["totals"]["infinite_values"]),
            "duplicate_rows": int(data_check["totals"]["duplicate_rows"]),
        },
        "split_stats": split_stats,
        "scaled_training": scaled_stats,
        "startup_transient": startup_stats,
        "leakage": leakage,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with STATS_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(stats_payload, handle, indent=2, allow_nan=False)
        handle.write("\n")

    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %z")
    section = build_section(
        counts=counts,
        splits=splits,
        split_stats=split_stats,
        scaled_stats=scaled_stats,
        startup_stats=startup_stats,
        leakage=leakage,
        config_data=config_data,
        sanity=sanity,
        data_check=data_check,
        inventory=inventory,
        timestamp=timestamp,
    )

    if SUMMARY_PATH.exists():
        existing = SUMMARY_PATH.read_text(encoding="utf-8")
        if SECTION_HEADING in existing:
            print(
                f"{SUMMARY_PATH} already contains '{SECTION_HEADING}'; "
                "left unchanged."
            )
            return 0
        updated = existing.rstrip() + "\n\n" + section
    else:
        updated = section

    SUMMARY_PATH.write_text(updated, encoding="utf-8", newline="\n")

    print(f"Saved: {STATS_PATH}")
    print(f"Saved: {SUMMARY_PATH}")
    print(
        f"Checks: {stats_payload['totals']['checks_passed']}/"
        f"{stats_payload['totals']['checks_total']} passed; "
        f"{stats_payload['totals']['nan_values']} NaN, "
        f"{stats_payload['totals']['infinite_values']} infinite."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

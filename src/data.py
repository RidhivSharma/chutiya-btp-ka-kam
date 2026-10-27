"""Load, decimate, window, split, and standardize the GPVS-Faults dataset.

This module reproduces Section 2 of ``Untitled0.ipynb`` with the leakage and
signal-processing fixes agreed for the paper:

1. The notebook decimated with ``df.iloc[::10]``. That skips samples without
   low-pass filtering and aliases 500-5000 Hz content into the 0-500 Hz band.
   We keep the same 1 kHz target rate but filter first with
   ``scipy.signal.decimate(q=10, zero_phase=True)`` (order-8 Chebyshev type I,
   cutoff 0.8 * 500 = 400 Hz). ``zero_phase=True`` uses ``filtfilt``, so there
   is no net phase shift: decimated sample ``i`` corresponds to native sample
   ``i * 10`` and ``Time[::10]`` stays aligned with it. The ``filtfilt`` edge
   transient is confined to the first/last few decimated samples and is not
   trimmed, so the very first and last windows contain a short transient.
2. Windows of 100 samples with stride 10 (90 % overlap), as in the notebook.
   Each window records the timestamps of its first and last sample.
3. The healthy F0L / F0M windows are split chronologically 60/20/20 per mode.
   Ten windows are omitted between train/validation and validation/test so
   that windows from different parts share no samples.
4. One StandardScaler per operating mode is fitted only on the raw decimated
   rows covered by that mode's healthy *training* windows. The scalers are
   then applied to every file of the corresponding mode. Fault and healthy
   validation/test rows are never used to fit any statistic.
5. The result is a dictionary with ``windows[name]`` of shape
   ``(n_windows, 100, 13)`` (standardized, float32) and ``times[name]`` of
   shape ``(n_windows, 2)`` (start_time, end_time in seconds).

The pipeline has no stochastic component, so no random seed is involved.
Raw CSVs are read from ``data/raw/CSV_Files`` and are never modified.

Run directly to write ``results/data_config.json`` and
``results/split_indices.csv``::

    .venv\\Scripts\\python.exe .\\src\\data.py
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import decimate as scipy_decimate
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = PROJECT_ROOT / "data" / "raw" / "CSV_Files"
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
EXPECTED_COLUMNS = [TIME_COLUMN, *FEATURES]

MODES = {"L": "IPPT", "M": "MPPT"}
STEMS = tuple(f"F{i}{suffix}" for i in range(8) for suffix in ("L", "M"))
HEALTHY_STEMS = {"L": "F0L", "M": "F0M"}

NATIVE_FS = 10_000
TARGET_FS = 1_000
DOWNSAMPLE_FACTOR = NATIVE_FS // TARGET_FS  # 10

WINDOW_SIZE = 100
STRIDE = 10
SPLIT_FRACTIONS = (0.6, 0.2, 0.2)
SPLIT_GAP_WINDOWS = 10

# Label definitions used by the Phase 2 timing diagnostic. They are additive:
# none of the preprocessing outputs above depends on them.
LABEL_MODES = ("whole_file", "time_resolved")
HEALTHY_LABEL = 0
FAULT_LABEL = 1
PRE_FAULT_LABEL = -1
INJECTION_OVERLAP_LABEL = -2
FAULT_START_S = 7.0
FAULT_END_S = 8.0


def package_version(name: str) -> str | None:
    """Return an installed package version, or None if it is absent."""
    try:
        return version(name)
    except PackageNotFoundError:
        return None


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


def environment_info() -> dict[str, str | None]:
    """Return the Python and package versions used by the pipeline."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": package_version("numpy"),
        "pandas": package_version("pandas"),
        "scipy": package_version("scipy"),
        "scikit-learn": package_version("scikit-learn"),
    }


def load_file(stem: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one CSV and return (features, time) as float64 arrays.

    Phase 1 found no missing or infinite values. Rather than silently
    interpolating, this function refuses to continue if the raw contract is
    violated, so that a future bad file cannot change the pipeline unnoticed.
    """
    path = CSV_DIR / f"{stem}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")

    frame = pd.read_csv(path)
    actual_columns = [str(column) for column in frame.columns]
    if actual_columns != EXPECTED_COLUMNS:
        raise ValueError(
            f"{path.name}: unexpected columns. "
            f"Expected {EXPECTED_COLUMNS}, got {actual_columns}"
        )

    values = frame[FEATURES].to_numpy(dtype=np.float64)
    time = frame[TIME_COLUMN].to_numpy(dtype=np.float64)

    if not np.isfinite(values).all():
        n_bad = int((~np.isfinite(values)).sum())
        raise ValueError(f"{path.name}: {n_bad} non-finite feature values")
    if not np.isfinite(time).all():
        raise ValueError(f"{path.name}: non-finite Time values")
    if len(time) < 2 or not np.all(np.diff(time) > 0):
        raise ValueError(f"{path.name}: Time is not strictly increasing")

    measured_fs = 1.0 / float(np.median(np.diff(time)))
    if not np.isclose(measured_fs, NATIVE_FS, rtol=0.01):
        raise ValueError(
            f"{path.name}: expected {NATIVE_FS} Hz, measured {measured_fs:.2f} Hz"
        )

    return values, time


def decimate_file(
    values: np.ndarray, time: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Anti-alias and decimate 10 kHz -> 1 kHz, keeping Time aligned.

    ``scipy.signal.decimate`` applies an order-8 Chebyshev type I low-pass
    filter with cutoff ``0.8 * (TARGET_FS / 2) = 400 Hz`` before taking every
    10th sample. With ``zero_phase=True`` the filter runs forwards and
    backwards, so the output has no net delay; output row ``i`` corresponds to
    native row ``i * 10`` and ``time[::10]`` is the matching Time vector.
    This is the anti-aliasing fix over the notebook's plain ``iloc[::10]``.

    The IIR filter has an edge transient of a few samples. It is deliberately
    kept (not trimmed) so the timestamps remain exactly the native sample
    times; only the first/last window can be affected.
    """
    decimated = scipy_decimate(
        values,
        q=DOWNSAMPLE_FACTOR,
        ftype="iir",
        axis=0,
        zero_phase=True,
    )
    decimated_time = time[::DOWNSAMPLE_FACTOR].copy()

    if decimated.shape[0] != decimated_time.size:
        raise RuntimeError(
            "Decimated signal and Time column are misaligned: "
            f"{decimated.shape[0]} samples vs {decimated_time.size} timestamps"
        )
    if decimated.shape[1] != len(FEATURES):
        raise RuntimeError(
            f"Expected {len(FEATURES)} features, got {decimated.shape[1]}"
        )
    if not np.all(np.diff(decimated_time) > 0):
        raise RuntimeError("Decimated Time is not strictly increasing")

    measured_fs = 1.0 / float(np.median(np.diff(decimated_time)))
    if not np.isclose(measured_fs, TARGET_FS, rtol=0.01):
        raise RuntimeError(
            f"Expected {TARGET_FS} Hz after decimation, measured {measured_fs:.2f} Hz"
        )

    return decimated, decimated_time


def window_count(n_samples: int) -> int:
    """Number of full windows that fit in ``n_samples`` with the fixed stride."""
    if n_samples < WINDOW_SIZE:
        return 0
    return 1 + (n_samples - WINDOW_SIZE) // STRIDE


def make_windows(
    values: np.ndarray, time: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Cut ``(n, 13)`` rows into ``(n_windows, 100, 13)`` windows plus times.

    The returned time matrix has one row per window: ``start_time`` is the
    Time value of the first sample and ``end_time`` that of the last sample
    (not an exclusive boundary).
    """
    count = window_count(len(values))
    if count == 0:
        return (
            np.empty((0, WINDOW_SIZE, len(FEATURES)), dtype=np.float32),
            np.empty((0, 2), dtype=np.float64),
        )

    starts = np.arange(0, len(values) - WINDOW_SIZE + 1, STRIDE)
    offsets = np.arange(WINDOW_SIZE)
    windows = values[starts[:, None] + offsets[None, :]]

    window_times = np.column_stack(
        (time[starts], time[starts + WINDOW_SIZE - 1])
    )
    if windows.shape[0] != count:
        raise RuntimeError(
            f"Window count mismatch: {windows.shape[0]} vs expected {count}"
        )

    return windows.astype(np.float32), window_times.astype(np.float64)


@lru_cache(maxsize=len(STEMS))
def _window_time_bounds(name: str) -> np.ndarray:
    """Return cached (start_time, end_time) for every window of one file.

    Only the ``Time`` column is needed for labelling. Anti-alias filtering does
    not change timestamps: ``decimate_file`` returns ``time[::10]``, so slicing
    the validated Time column here gives exactly the same window bounds as
    ``build_dataset`` without decimating all 13 feature columns again.
    """
    if name not in STEMS:
        raise ValueError(f"Unknown dataset file: {name!r}")

    _, time = load_file(name)
    decimated_time = time[::DOWNSAMPLE_FACTOR].copy()
    count = window_count(decimated_time.size)
    if count == 0:
        return np.empty((0, 2), dtype=np.float64)

    starts = np.arange(
        0, decimated_time.size - WINDOW_SIZE + 1, STRIDE
    )
    return np.column_stack(
        (decimated_time[starts], decimated_time[starts + WINDOW_SIZE - 1])
    ).astype(np.float64)


def get_labels(
    name: str,
    label_mode: str,
    times: np.ndarray | None = None,
) -> np.ndarray:
    """Return one integer label per window of ``name``.

    Parameters
    ----------
    name
        File stem, one of ``STEMS`` (for example ``"F3M"``).
    label_mode
        ``"whole_file"`` labels every window of a fault file as faulty (1),
        reproducing the notebook/reference-paper convention. ``"time_resolved"``
        uses the documented fault-injection interval of 7.0-8.0 s:

        * ``-1``: the window ends before 7.0 s (pre-fault; excluded from the
          fault class but retained for false-alarm analysis),
        * ``1``: the window starts after 8.0 s (post-injection fault),
        * ``-2``: the window overlaps 7.0-8.0 s and is dropped from both the
          pre-fault and fault sets.

        Healthy files (``F0L``/``F0M``) are labelled 0 in both modes.
    times
        Optional ``(n_windows, 2)`` array of (start_time, end_time) seconds.
        When omitted, the bounds are read from the file's Time column. Passing
        ``data["times"][name]`` avoids a second read and guarantees that labels
        and model inputs refer to identical windows.

    A file that ends before 7.0 s (``F3M``) gets only ``-1`` labels in
    ``"time_resolved"`` mode; no exception is raised and no post-injection
    share is invented downstream.
    """
    if name not in STEMS:
        raise ValueError(f"Unknown dataset file: {name!r}")
    if label_mode not in LABEL_MODES:
        raise ValueError(
            f"Unknown label_mode {label_mode!r}; expected one of {LABEL_MODES}"
        )

    if times is None:
        bounds = _window_time_bounds(name)
    else:
        bounds = np.asarray(times, dtype=np.float64)
        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError(
                f"{name}: times must have shape (n_windows, 2), "
                f"got {bounds.shape}"
            )
        if not np.isfinite(bounds).all():
            raise ValueError(f"{name}: times contain non-finite values")
        if np.any(bounds[:, 1] < bounds[:, 0]):
            raise ValueError(f"{name}: a window ends before it starts")

    n_windows = int(bounds.shape[0])
    if name[1] == "0":
        return np.zeros(n_windows, dtype=np.int8)
    if label_mode == "whole_file":
        return np.ones(n_windows, dtype=np.int8)

    labels = np.full(n_windows, INJECTION_OVERLAP_LABEL, dtype=np.int8)
    pre_fault = bounds[:, 1] < FAULT_START_S
    post_fault = bounds[:, 0] > FAULT_END_S
    if np.any(pre_fault & post_fault):
        raise RuntimeError(
            f"{name}: a window cannot both end before {FAULT_START_S} s "
            f"and start after {FAULT_END_S} s"
        )
    labels[pre_fault] = PRE_FAULT_LABEL
    labels[post_fault] = FAULT_LABEL
    return labels


def make_healthy_split(
    n_total: int,
) -> dict[str, np.ndarray]:
    """Chronological 60/20/20 split of F0 windows with a 10-window embargo.

    Ten windows are dropped after the training block and ten after the
    validation block. This prevents any training window from sharing raw
    samples with a validation or test window. The dropped windows are why the
    test set is slightly below 20 % of all windows.
    """
    n_train = int(np.floor(SPLIT_FRACTIONS[0] * n_total))
    n_val = int(np.floor(SPLIT_FRACTIONS[1] * n_total))
    n_test = n_total - n_train - n_val - 2 * SPLIT_GAP_WINDOWS

    if min(n_train, n_val, n_test) <= 0:
        raise ValueError(
            f"Not enough windows ({n_total}) for a 60/20/20 split "
            f"with a {SPLIT_GAP_WINDOWS}-window gap"
        )

    train = np.arange(0, n_train, dtype=np.int64)
    val = np.arange(
        n_train + SPLIT_GAP_WINDOWS,
        n_train + SPLIT_GAP_WINDOWS + n_val,
        dtype=np.int64,
    )
    test = np.arange(
        n_train + SPLIT_GAP_WINDOWS + n_val + SPLIT_GAP_WINDOWS,
        n_total,
        dtype=np.int64,
    )

    if len(test) != n_test:
        raise RuntimeError(
            f"Split size mismatch: test has {len(test)} windows, expected {n_test}"
        )
    for part in (train, val, test):
        if not np.all(np.diff(part) == 1):
            raise RuntimeError("Split indices must be contiguous")

    return {"train": train, "val": val, "test": test}


def train_row_bounds(
    splits: dict[str, np.ndarray], stride: int = STRIDE
) -> tuple[int, int]:
    """Return the half-open decimated-row range covered by training windows.

    Stride 10 divides the 100-sample window, so the union of the training
    windows is one contiguous block. Rows are used once each when fitting the
    scaler, never once per overlapping window.
    """
    train = splits["train"]
    if train.size == 0 or train[0] != 0 or not np.all(np.diff(train) == 1):
        raise ValueError("Expected a contiguous training block starting at 0")
    start = int(train[0]) * stride
    stop = int(train[-1]) * stride + WINDOW_SIZE
    return start, stop


def build_dataset(verbose: bool = False) -> dict[str, object]:
    """Build the full windowed dataset and healthy split.

    Returns a dictionary with:
      windows          name -> (n_windows, 100, 13) float32, standardized
      times            name -> (n_windows, 2) [start_time, end_time] seconds
      splits           "L"/"M" -> {"train", "val", "test"} window indices
      scalers          "L"/"M" -> fitted StandardScaler
      raw_decimated    name -> (n_decimated, 13) float64, unscaled
      time_decimated   name -> (n_decimated,) float64 seconds
      native_rows      name -> int, rows in the raw CSV
      features         list of 13 feature names
      config           preprocessing constants and metadata
    """
    raw_decimated: dict[str, np.ndarray] = {}
    time_decimated: dict[str, np.ndarray] = {}
    native_rows: dict[str, int] = {}

    for stem in STEMS:
        values, time = load_file(stem)
        decimated, decimated_time = decimate_file(values, time)
        raw_decimated[stem] = decimated
        time_decimated[stem] = decimated_time
        native_rows[stem] = int(values.shape[0])
        if verbose:
            print(
                f"  {stem}: {values.shape[0]:>8,} native -> "
                f"{decimated.shape[0]:>7,} decimated rows"
            )

    splits: dict[str, dict[str, np.ndarray]] = {}
    for suffix in ("L", "M"):
        stem = HEALTHY_STEMS[suffix]
        splits[suffix] = make_healthy_split(window_count(len(raw_decimated[stem])))

    scalers: dict[str, StandardScaler] = {}
    for suffix in ("L", "M"):
        stem = HEALTHY_STEMS[suffix]
        start, stop = train_row_bounds(splits[suffix])
        fit_rows = raw_decimated[stem][start:stop]
        scaler = StandardScaler().fit(fit_rows)
        if int(scaler.n_samples_seen_) != fit_rows.shape[0]:
            raise RuntimeError(
                f"{stem}: scaler saw {scaler.n_samples_seen_} rows, "
                f"expected {fit_rows.shape[0]}"
            )
        scalers[suffix] = scaler

    windows: dict[str, np.ndarray] = {}
    times: dict[str, np.ndarray] = {}
    for stem in STEMS:
        # Scaling the rows before windowing matches the notebook's order and
        # ensures every window is standardized with its mode's healthy-train
        # scaler, including all seven fault files.
        scaled = scalers[stem[-1]].transform(raw_decimated[stem])
        windows[stem], times[stem] = make_windows(scaled, time_decimated[stem])

    config = {
        "dataset": "GPVS-Faults",
        "stochastic": False,
        "seed": None,
        "seed_note": "No random operation is performed in this module.",
        "native_sampling_hz": NATIVE_FS,
        "target_sampling_hz": TARGET_FS,
        "downsample_factor": DOWNSAMPLE_FACTOR,
        "decimation": {
            "function": "scipy.signal.decimate",
            "q": DOWNSAMPLE_FACTOR,
            "ftype": "iir",
            "filter": "order-8 Chebyshev type I, 0.05 dB ripple",
            "cutoff_hz": 0.8 * TARGET_FS / 2.0,
            "zero_phase": True,
            "axis": 0,
            "time_alignment": "Time[::10]",
            "note": (
                "filtfilt gives zero net phase shift, so decimated sample i "
                "corresponds to native sample i*10. Edge transients are kept."
            ),
            "replaces_notebook_choice": "df.iloc[::10] without anti-aliasing",
        },
        "windowing": {
            "window_size": WINDOW_SIZE,
            "stride": STRIDE,
            "overlap_samples": WINDOW_SIZE - STRIDE,
            "overlap_fraction": (WINDOW_SIZE - STRIDE) / WINDOW_SIZE,
            "values": "standardized with the mode's healthy-training scaler",
            "end_time": "timestamp of the last sample in the window",
            "windows_per_file": "1 + (n_decimated - window_size) // stride",
        },
        "healthy_split": {
            "basis": "chronological order of F0L/F0M windows",
            "fractions": {
                "train": SPLIT_FRACTIONS[0],
                "val": SPLIT_FRACTIONS[1],
                "test": SPLIT_FRACTIONS[2],
            },
            "gap_windows": SPLIT_GAP_WINDOWS,
            "note": (
                "Ten windows are dropped between train/val and between "
                "val/test, so the parts share no raw samples. Gap windows are "
                "not part of any split and test is slightly below 20%."
            ),
        },
        "scaling": {
            "class": "sklearn.preprocessing.StandardScaler",
            "scope": "one scaler per operating mode",
            "fit_data": (
                "unique raw decimated rows covered by the healthy training "
                "windows of that mode (F0L for IPPT, F0M for MPPT)"
            ),
            "applied_to": "all 16 files of the corresponding mode",
            "ddof": 0,
        },
    }

    return {
        "windows": windows,
        "times": times,
        "splits": splits,
        "scalers": scalers,
        "raw_decimated": raw_decimated,
        "time_decimated": time_decimated,
        "native_rows": native_rows,
        "features": list(FEATURES),
        "config": config,
    }


def file_counts_frame(data: dict[str, object]) -> pd.DataFrame:
    """Per-file row and window counts, computed from the built arrays."""
    windows = data["windows"]
    time_decimated = data["time_decimated"]
    native_rows = data["native_rows"]

    rows: list[dict[str, object]] = []
    for stem in STEMS:
        decimated_time = time_decimated[stem]
        measured_fs = 1.0 / float(np.median(np.diff(decimated_time)))
        rows.append(
            {
                "file": stem,
                "mode": MODES[stem[-1]],
                "condition": "healthy" if stem[1] == "0" else "fault",
                "native_rows": int(native_rows[stem]),
                "decimated_rows": int(decimated_time.size),
                "windows": int(windows[stem].shape[0]),
                "measured_fs_hz": measured_fs,
                "decimated_start_s": float(decimated_time[0]),
                "decimated_end_s": float(decimated_time[-1]),
            }
        )
    return pd.DataFrame(rows)


def split_summary_frame(data: dict[str, object]) -> pd.DataFrame:
    """Window counts and time bounds for each healthy split."""
    windows = data["windows"]
    times = data["times"]
    rows: list[dict[str, object]] = []

    for suffix in ("L", "M"):
        stem = HEALTHY_STEMS[suffix]
        total = int(windows[stem].shape[0])
        for split_name in ("train", "val", "test"):
            indices = data["splits"][suffix][split_name]
            rows.append(
                {
                    "mode": MODES[suffix],
                    "split": split_name,
                    "windows": int(indices.size),
                    "fraction_of_healthy_windows": float(indices.size / total),
                    "first_window": int(indices[0]),
                    "last_window": int(indices[-1]),
                    "first_sample": int(indices[0]) * STRIDE,
                    "last_sample": int(indices[-1]) * STRIDE + WINDOW_SIZE - 1,
                    "start_time_s": float(times[stem][indices[0], 0]),
                    "end_time_s": float(times[stem][indices[-1], 1]),
                }
            )
    return pd.DataFrame(rows)


def split_indices_frame(data: dict[str, object]) -> pd.DataFrame:
    """One row per healthy window, labelled train/val/test/gap."""
    windows = data["windows"]
    times = data["times"]
    rows: list[dict[str, object]] = []

    for suffix in ("L", "M"):
        stem = HEALTHY_STEMS[suffix]
        total = int(windows[stem].shape[0])
        labels = np.full(total, "gap", dtype=object)
        for split_name, indices in data["splits"][suffix].items():
            labels[indices] = split_name

        for window_index in range(total):
            start_sample = window_index * STRIDE
            stop_sample = start_sample + WINDOW_SIZE
            rows.append(
                {
                    "mode": MODES[suffix],
                    "mode_suffix": suffix,
                    "file": stem,
                    "split": str(labels[window_index]),
                    "window_index": window_index,
                    "start_sample": start_sample,
                    "stop_sample_exclusive": stop_sample,
                    "start_time_s": float(times[stem][window_index, 0]),
                    "end_time_s": float(times[stem][window_index, 1]),
                }
            )
    return pd.DataFrame(rows)


def build_metadata(data: dict[str, object]) -> dict[str, object]:
    """Assemble JSON-safe run metadata for results/data_config.json."""
    scalers = data["scalers"]
    windows = data["windows"]

    scalar_metadata: dict[str, object] = {}
    for suffix in ("L", "M"):
        start, stop = train_row_bounds(data["splits"][suffix])
        scaler = scalers[suffix]
        scalar_metadata[MODES[suffix]] = {
            "fit_row_start": start,
            "fit_row_stop_exclusive": stop,
            "fit_rows": int(stop - start),
            "n_samples_seen": int(scaler.n_samples_seen_),
            "n_features_in": int(scaler.n_features_in_),
            "mean": [float(value) for value in scaler.mean_],
            "scale": [float(value) for value in scaler.scale_],
        }

    split_metadata: dict[str, object] = {}
    for suffix in ("L", "M"):
        stem = HEALTHY_STEMS[suffix]
        total = int(windows[stem].shape[0])
        split_metadata[MODES[suffix]] = {
            split_name: {
                "count": int(indices.size),
                "fraction_of_healthy_windows": float(indices.size / total),
                "first_window": int(indices[0]),
                "last_window": int(indices[-1]),
            }
            for split_name, indices in data["splits"][suffix].items()
        }

    return {
        "phase": "phase_2_preprocessing",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "environment": environment_info(),
        "features": data["features"],
        "config": data["config"],
        "counts": file_counts_frame(data).to_dict(orient="records"),
        "split_counts": split_metadata,
        "scalers": scalar_metadata,
    }


def main() -> int:
    print("=" * 80)
    print("GPVS-Faults | Phase 2 preprocessing")
    print("=" * 80)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"CSV directory: {CSV_DIR}")

    data = build_dataset(verbose=True)
    counts = file_counts_frame(data)
    splits = split_summary_frame(data)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    split_indices_path = RESULTS_DIR / "split_indices.csv"
    metadata_path = RESULTS_DIR / "data_config.json"
    split_indices_frame(data).to_csv(split_indices_path, index=False)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(build_metadata(data), handle, indent=2, allow_nan=False)
        handle.write("\n")

    print("\nPer-file window counts:")
    print(
        counts.to_string(
            index=False,
            formatters={
                "native_rows": "{:,}".format,
                "decimated_rows": "{:,}".format,
                "windows": "{:,}".format,
                "measured_fs_hz": "{:.2f}".format,
                "decimated_start_s": "{:.4f}".format,
                "decimated_end_s": "{:.4f}".format,
            },
        )
    )

    print("\nHealthy split counts:")
    print(
        splits.to_string(
            index=False,
            formatters={
                "windows": "{:,}".format,
                "fraction_of_healthy_windows": "{:.4f}".format,
                "first_sample": "{:,}".format,
                "last_sample": "{:,}".format,
                "start_time_s": "{:.4f}".format,
                "end_time_s": "{:.4f}".format,
            },
        )
    )

    print(f"\nSaved: {split_indices_path}")
    print(f"Saved: {metadata_path}")
    print("\nRESULT: dataset built.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

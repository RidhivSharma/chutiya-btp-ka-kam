"""Phase 3: reference baselines on the frozen Phase 2 pipeline.

The four scikit-learn baselines are fitted on the standardized 1 kHz healthy
*train* rows of one operating mode. A row is one time step with the 13 sensor
features. The 90 %-overlapping training windows are unrolled to their unique
rows, so each 1 kHz time step is used once rather than ten times; this is the
same row set on which the Phase 2 ``StandardScaler`` was fitted.

A window score is the mean of the model's per-time-step anomaly score over the
100 samples in that window. ``-decision_function`` is used so that higher means
more anomalous. Because consecutive windows overlap, each file's per-row score
is computed once and then averaged over the window's 100-row slice; this is
mathematically identical to scoring every window row separately. The
equivalence is checked on sample windows for every model, mode, and file.

The fifth reference is the Phase 2 Mahalanobis placeholder, imported unchanged
from ``src/time_diagnostic.py``. It operates on window means, exactly as in
Phase 2, and is labelled "Mahalanobis (window-mean)". It is included to show
whether a learned model adds value over a trivial detector.

No fault file and no healthy test window is used to fit a model or to choose a
threshold. For every model and mode the threshold is the 99th percentile of
the window scores on healthy *validation* windows. A window is flagged when
``score > threshold``, matching Phase 2.

Outputs
-------
``results/scores_baseline_<model>_<mode>.npz``
    One archive per model and mode with exactly the keys ``scores``,
    ``start_time``, ``end_time``, ``threshold`` and ``name``. ``name`` is the
    file or file/part label (``F0L_train``, ``F0L_val``, ``F0L_test``,
    ``F1L`` ...). This is the format the proposed model will use in Phase 5.
``results/baseline_sanity.csv``
    Healthy validation/test flag rates, per-fault whole-file flag shares, and
    per-fault pre-injection versus post-8 s median scores. No final metrics.
``results/baselines_metadata.json``
    Model settings, training/scoring times, thresholds, checks, environment,
    and the git commit.

Run::

    .venv\\Scripts\\python.exe .\\src\\baselines.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (  # noqa: E402
    FAULT_END_S,
    FAULT_START_S,
    FEATURES,
    HEALTHY_STEMS,
    MODES,
    RESULTS_DIR,
    STEMS,
    STRIDE,
    WINDOW_SIZE,
    build_dataset,
    environment_info,
    git_commit,
    train_row_bounds,
)
from src.time_diagnostic import (  # noqa: E402
    RIDGE_RELATIVE,
    STARTUP_END_S,
    THRESHOLD_QUANTILE,
    fit_placeholder,
    score_means,
)

SEED = 0
SUBSAMPLE_CAP = 20_000
FAULT_STEMS_BY_MODE = {
    "L": tuple(f"F{i}L" for i in range(1, 8)),
    "M": tuple(f"F{i}M" for i in range(1, 8)),
}
SANITY_PATH = RESULTS_DIR / "baseline_sanity.csv"
METADATA_PATH = RESULTS_DIR / "baselines_metadata.json"

CHECK_RESULTS: list[dict[str, object]] = []


def check(name: str, passed: bool, detail: str) -> bool:
    """Record and print one named assertion."""
    passed = bool(passed)
    CHECK_RESULTS.append({"name": name, "passed": passed, "detail": detail})
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    return passed


# ---------------------------------------------------------------------------
# Model definitions. Every setting is fixed by the phase specification.
# ---------------------------------------------------------------------------


def make_elliptic_envelope() -> EllipticEnvelope:
    return EllipticEnvelope(
        support_fraction=0.999,
        contamination=0.1,
        assume_centered=False,
        random_state=SEED,
    )


def make_isolation_forest() -> IsolationForest:
    return IsolationForest(n_estimators=100, random_state=SEED)


def make_lof() -> LocalOutlierFactor:
    return LocalOutlierFactor(n_neighbors=20, novelty=True)


def make_one_class_svm() -> OneClassSVM:
    # nu is fixed at 0.1 as specified. Fault data are never used to tune it.
    return OneClassSVM(kernel="rbf", gamma="scale", nu=0.1)


MODEL_SPECS: dict[str, dict[str, object]] = {
    "elliptic_envelope": {
        "label": "EllipticEnvelope",
        "factory": make_elliptic_envelope,
        "input": "time_step_rows",
        "params": {
            "support_fraction": 0.999,
            "contamination": 0.1,
            "assume_centered": False,
            "random_state": SEED,
        },
    },
    "isolation_forest": {
        "label": "IsolationForest",
        "factory": make_isolation_forest,
        "input": "time_step_rows",
        "params": {"n_estimators": 100, "random_state": SEED},
    },
    "lof": {
        "label": "LocalOutlierFactor",
        "factory": make_lof,
        "input": "time_step_rows",
        "params": {"n_neighbors": 20, "novelty": True},
    },
    "one_class_svm": {
        "label": "OneClassSVM",
        "factory": make_one_class_svm,
        "input": "time_step_rows",
        "params": {"kernel": "rbf", "gamma": "scale", "nu": 0.1},
    },
    "mahalanobis_window_mean": {
        "label": "Mahalanobis (window-mean)",
        "factory": None,
        "input": "window_means",
        "params": {
            "covariance_ddof": 1,
            "ridge_relative": RIDGE_RELATIVE,
            "pseudo_inverse": "numpy.linalg.pinv(..., hermitian=True)",
            "source": "src/time_diagnostic.py fit_placeholder/score_means",
        },
    },
}
MODEL_ORDER = (
    "elliptic_envelope",
    "isolation_forest",
    "lof",
    "one_class_svm",
    "mahalanobis_window_mean",
)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def scaled_rows_for_mode(
    data: dict[str, object], suffix: str
) -> dict[str, np.ndarray]:
    """Apply the mode's healthy-train scaler to every row of its files."""
    scaler = data["scalers"][suffix]
    stems = (HEALTHY_STEMS[suffix], *FAULT_STEMS_BY_MODE[suffix])
    scaled: dict[str, np.ndarray] = {}
    for stem in stems:
        values = scaler.transform(data["raw_decimated"][stem])
        scaled[stem] = np.asarray(values, dtype=np.float32)
    return scaled


def check_scaled_row_alignment(
    data: dict[str, object], suffix: str, scaled: dict[str, np.ndarray]
) -> list[str]:
    """Verify that window rows are exactly the cached scaled rows."""
    problems: list[str] = []
    for stem in (HEALTHY_STEMS[suffix], *FAULT_STEMS_BY_MODE[suffix]):
        windows = data["windows"][stem]
        count = int(windows.shape[0])
        for window_index in (0, count // 2, count - 1):
            start = window_index * STRIDE
            expected = scaled[stem][start : start + WINDOW_SIZE]
            if not np.array_equal(windows[window_index], expected):
                problems.append(f"{stem}[{window_index}]")
    return problems


def healthy_train_rows(
    data: dict[str, object], suffix: str, scaled: dict[str, np.ndarray]
) -> np.ndarray:
    """Unique standardized 1 kHz rows covered by the healthy train windows."""
    stem = HEALTHY_STEMS[suffix]
    start, stop = train_row_bounds(data["splits"][suffix])
    rows = scaled[stem][start:stop]
    expected_shape = (stop - start, len(FEATURES))
    if rows.shape != expected_shape:
        raise RuntimeError(
            f"{stem}: expected train rows {expected_shape}, got {rows.shape}"
        )
    return rows


def subsample_rows(
    rows: np.ndarray, cap: int = SUBSAMPLE_CAP, seed: int = SEED
) -> tuple[np.ndarray, bool]:
    """Deterministically cap the fit rows; no subsampling is needed here."""
    if rows.shape[0] <= cap:
        return rows, False
    rng = np.random.RandomState(seed)
    indices = np.sort(rng.choice(rows.shape[0], size=cap, replace=False))
    return rows[indices], True


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def row_anomaly_scores(model, rows: np.ndarray) -> np.ndarray:
    """Per-time-step anomaly score; higher means more anomalous."""
    decision = model.decision_function(rows)
    scores = -np.asarray(decision, dtype=np.float64).reshape(-1)
    if scores.shape[0] != rows.shape[0]:
        raise RuntimeError(
            f"decision_function returned {scores.shape[0]} scores "
            f"for {rows.shape[0]} rows"
        )
    if not np.isfinite(scores).all():
        raise RuntimeError("decision_function produced non-finite scores")
    return scores


def window_scores_from_row_scores(
    row_scores: np.ndarray, n_windows: int
) -> np.ndarray:
    """Mean of the per-row scores over each 100-sample window."""
    starts = np.arange(n_windows, dtype=np.int64) * STRIDE
    stops = starts + WINDOW_SIZE
    if starts.size == 0:
        return np.empty(0, dtype=np.float64)
    if int(stops[-1]) > row_scores.size:
        raise RuntimeError(
            f"Window slice ends at row {int(stops[-1])}, "
            f"but only {row_scores.size} rows were scored"
        )
    cumulative = np.concatenate(
        (np.zeros(1, dtype=np.float64), np.cumsum(row_scores, dtype=np.float64))
    )
    return (cumulative[stops] - cumulative[starts]) / float(WINDOW_SIZE)


def check_window_score_equivalence(
    model,
    rows: np.ndarray,
    row_scores: np.ndarray,
    n_windows: int,
    label: str,
) -> list[str]:
    """Check the cached-row shortcut against direct window scoring."""
    computed = window_scores_from_row_scores(row_scores, n_windows)
    problems: list[str] = []
    for window_index in (0, n_windows // 2, n_windows - 1):
        start = window_index * STRIDE
        window = rows[start : start + WINDOW_SIZE]
        direct = float(np.mean(-model.decision_function(window)))
        # Batch and single-window decision_function can differ at round-off
        # level (about 1e-7 for LOF), so the check uses a numerical tolerance.
        if not np.isclose(direct, computed[window_index], rtol=1e-5, atol=1e-8):
            problems.append(
                f"{label}[{window_index}] direct={direct:.12g} "
                f"cached={computed[window_index]:.12g}"
            )
    return problems


def make_group(
    name: str,
    scores: np.ndarray,
    times: np.ndarray,
    file: str,
    part: str,
) -> dict[str, object]:
    return {
        "name": name,
        "scores": np.asarray(scores, dtype=np.float64),
        "start_time": np.asarray(times[:, 0], dtype=np.float64),
        "end_time": np.asarray(times[:, 1], dtype=np.float64),
        "file": file,
        "part": part,
    }


def score_file(model, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Score every unique row once, then reduce to overlapping windows."""
    row_scores = row_anomaly_scores(model, rows)
    n_windows = 1 + (rows.shape[0] - WINDOW_SIZE) // STRIDE
    return row_scores, window_scores_from_row_scores(row_scores, n_windows)


def build_time_step_groups(
    data: dict[str, object],
    suffix: str,
    model,
    scaled: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    """Score healthy train/val/test parts and all seven fault files."""
    healthy_stem = HEALTHY_STEMS[suffix]
    splits = data["splits"][suffix]
    groups: list[dict[str, object]] = []

    healthy_rows = scaled[healthy_stem]
    healthy_row_scores, healthy_window_scores = score_file(model, healthy_rows)
    problems = check_window_score_equivalence(
        model,
        healthy_rows,
        healthy_row_scores,
        int(data["windows"][healthy_stem].shape[0]),
        healthy_stem,
    )
    if problems:
        raise RuntimeError("Window-score equivalence failed: " + "; ".join(problems))

    for part in ("train", "val", "test"):
        indices = np.asarray(splits[part], dtype=np.int64)
        groups.append(
            make_group(
                f"{healthy_stem}_{part}",
                healthy_window_scores[indices],
                data["times"][healthy_stem][indices],
                healthy_stem,
                part,
            )
        )

    for fault in FAULT_STEMS_BY_MODE[suffix]:
        rows = scaled[fault]
        row_scores, window_scores = score_file(model, rows)
        problems = check_window_score_equivalence(
            model,
            rows,
            row_scores,
            int(data["windows"][fault].shape[0]),
            fault,
        )
        if problems:
            raise RuntimeError(
                "Window-score equivalence failed: " + "; ".join(problems)
            )
        groups.append(
            make_group(
                fault,
                window_scores,
                data["times"][fault],
                fault,
                "whole_file",
            )
        )

    return groups


def build_mahalanobis_groups(
    data: dict[str, object], suffix: str, detector: dict[str, object]
) -> list[dict[str, object]]:
    """Score every window with the Phase 2 window-mean placeholder."""
    windows = data["windows"]
    healthy_stem = HEALTHY_STEMS[suffix]
    splits = data["splits"][suffix]
    groups: list[dict[str, object]] = []

    healthy_means = windows[healthy_stem].mean(axis=1, dtype=np.float64)
    healthy_scores = score_means(healthy_means, detector)
    for part in ("train", "val", "test"):
        indices = np.asarray(splits[part], dtype=np.int64)
        groups.append(
            make_group(
                f"{healthy_stem}_{part}",
                healthy_scores[indices],
                data["times"][healthy_stem][indices],
                healthy_stem,
                part,
            )
        )

    for fault in FAULT_STEMS_BY_MODE[suffix]:
        means = windows[fault].mean(axis=1, dtype=np.float64)
        scores = score_means(means, detector)
        groups.append(
            make_group(
                fault,
                scores,
                data["times"][fault],
                fault,
                "whole_file",
            )
        )

    return groups


def validation_threshold(
    groups: list[dict[str, object]], healthy_stem: str
) -> float:
    """99th percentile of healthy validation window scores only."""
    validation = next(
        group for group in groups if group["name"] == f"{healthy_stem}_val"
    )
    scores = np.asarray(validation["scores"], dtype=np.float64)
    if scores.size == 0:
        raise RuntimeError("No healthy validation windows to set the threshold")
    return float(np.quantile(scores, THRESHOLD_QUANTILE))


def assemble_arrays(
    groups: list[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate groups into the Phase 5 archive layout."""
    scores = np.concatenate([group["scores"] for group in groups])
    start_time = np.concatenate([group["start_time"] for group in groups])
    end_time = np.concatenate([group["end_time"] for group in groups])
    names = np.concatenate(
        [
            np.full(
                np.asarray(group["scores"]).size,
                str(group["name"]),
                dtype="U32",
            )
            for group in groups
        ]
    )
    return scores, start_time, end_time, names


def save_score_archive(
    path: Path,
    scores: np.ndarray,
    start_time: np.ndarray,
    end_time: np.ndarray,
    threshold: float,
    names: np.ndarray,
) -> None:
    """Save exactly the five keys the proposed model will use in Phase 5."""
    np.savez_compressed(
        path,
        scores=np.asarray(scores, dtype=np.float64),
        start_time=np.asarray(start_time, dtype=np.float64),
        end_time=np.asarray(end_time, dtype=np.float64),
        threshold=np.float64(threshold),
        name=np.asarray(names, dtype="U32"),
    )


# ---------------------------------------------------------------------------
# One model/mode run
# ---------------------------------------------------------------------------


def run_model_mode(
    data: dict[str, object],
    model_key: str,
    suffix: str,
    scaled: dict[str, np.ndarray],
) -> dict[str, object]:
    """Fit one model on one mode's healthy train rows, then score all files."""
    spec = MODEL_SPECS[model_key]
    mode = MODES[suffix]
    healthy_stem = HEALTHY_STEMS[suffix]
    print(f"\n  {spec['label']} | {mode}")

    if model_key == "mahalanobis_window_mean":
        train_indices = np.asarray(
            data["splits"][suffix]["train"], dtype=np.int64
        )
        train_means = data["windows"][healthy_stem][train_indices].mean(
            axis=1, dtype=np.float64
        )
        start_clock = time.perf_counter()
        detector = fit_placeholder(train_means)
        training_time = time.perf_counter() - start_clock
        start_clock = time.perf_counter()
        groups = build_mahalanobis_groups(data, suffix, detector)
        scoring_time = time.perf_counter() - start_clock
        train_units = int(train_means.shape[0])
        subsample_applied = False
    else:
        rows = healthy_train_rows(data, suffix, scaled)
        fit_rows, subsample_applied = subsample_rows(rows)
        model = spec["factory"]()
        start_clock = time.perf_counter()
        model.fit(fit_rows)
        training_time = time.perf_counter() - start_clock
        start_clock = time.perf_counter()
        groups = build_time_step_groups(data, suffix, model, scaled)
        scoring_time = time.perf_counter() - start_clock
        train_units = int(fit_rows.shape[0])

    threshold = validation_threshold(groups, healthy_stem)
    scores, start_time, end_time, names = assemble_arrays(groups)
    if not np.isfinite(scores).all():
        raise RuntimeError(f"{model_key}/{mode}: non-finite window scores")
    if not np.isfinite(start_time).all() or not np.isfinite(end_time).all():
        raise RuntimeError(f"{model_key}/{mode}: non-finite window times")

    path = RESULTS_DIR / f"scores_baseline_{model_key}_{mode}.npz"
    save_score_archive(path, scores, start_time, end_time, threshold, names)

    record: dict[str, object] = {
        "model": model_key,
        "model_label": str(spec["label"]),
        "mode": mode,
        "mode_suffix": suffix,
        "healthy_file": healthy_stem,
        "input_kind": str(spec["input"]),
        "train_units": train_units,
        "subsample_applied": bool(subsample_applied),
        "training_time_s": float(training_time),
        "scoring_time_s": float(scoring_time),
        "threshold": float(threshold),
        "windows": int(scores.size),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "score_nan": int(np.isnan(scores).sum()),
        "npz_path": str(path.relative_to(PROJECT_ROOT)),
        "model_params": spec["params"],
    }

    print(
        f"    train={train_units:,} {spec['input']} "
        f"subsample={subsample_applied} | "
        f"fit={training_time:.2f}s score={scoring_time:.2f}s | "
        f"threshold={threshold:.6g} | windows={scores.size:,} | "
        f"score range {scores.min():.4g} to {scores.max():.4g}"
    )
    print(f"    saved {path.name}")
    return record


# ---------------------------------------------------------------------------
# Validation and sanity tables
# ---------------------------------------------------------------------------


def expected_group_counts(
    data: dict[str, object], suffix: str
) -> dict[str, int]:
    """Expected number of windows in every saved group."""
    healthy_stem = HEALTHY_STEMS[suffix]
    counts = {
        f"{healthy_stem}_{part}": int(
            np.asarray(data["splits"][suffix][part]).size
        )
        for part in ("train", "val", "test")
    }
    for fault in FAULT_STEMS_BY_MODE[suffix]:
        counts[fault] = int(data["windows"][fault].shape[0])
    return counts


def validate_archives(
    data: dict[str, object], records: list[dict[str, object]]
) -> None:
    """Check every archive's keys, lengths, groups, and threshold provenance."""
    expected_keys = {"scores", "start_time", "end_time", "threshold", "name"}
    all_keys_ok = True
    all_lengths_ok = True
    all_finite = True
    all_thresholds_ok = True
    all_groups_ok = True
    total_windows = 0

    for record in records:
        path = PROJECT_ROOT / str(record["npz_path"])
        with np.load(path, allow_pickle=False) as archive:
            keys = set(archive.files)
            scores = np.asarray(archive["scores"], dtype=np.float64)
            start_time = np.asarray(archive["start_time"], dtype=np.float64)
            end_time = np.asarray(archive["end_time"], dtype=np.float64)
            threshold = float(np.asarray(archive["threshold"]))
            names = np.asarray(archive["name"]).astype(str)

        all_keys_ok = all_keys_ok and keys == expected_keys
        all_lengths_ok = all_lengths_ok and (
            scores.shape == start_time.shape == end_time.shape == names.shape
        )
        all_finite = all_finite and bool(
            np.isfinite(scores).all()
            and np.isfinite(start_time).all()
            and np.isfinite(end_time).all()
            and np.isfinite(threshold)
        )

        expected_counts = expected_group_counts(data, str(record["mode_suffix"]))
        observed_counts = {
            name: int(np.sum(names == name)) for name in np.unique(names)
        }
        all_groups_ok = all_groups_ok and observed_counts == expected_counts

        validation_name = f"{record['healthy_file']}_val"
        validation_scores = scores[names == validation_name]
        recomputed = float(
            np.quantile(validation_scores, THRESHOLD_QUANTILE)
        )
        all_thresholds_ok = all_thresholds_ok and bool(
            np.isclose(threshold, recomputed, rtol=1e-12, atol=1e-12)
        )
        total_windows += int(scores.size)

    check(
        "every archive has exactly the five Phase 5 keys",
        all_keys_ok,
        "keys = scores, start_time, end_time, threshold, name",
    )
    check(
        "score/time/name arrays have identical lengths",
        all_lengths_ok,
        f"{len(records)} archives checked",
    )
    check(
        "all scores, times, and thresholds are finite",
        all_finite,
        f"{total_windows:,} window scores across {len(records)} archives",
    )
    check(
        "every group has the expected window count",
        all_groups_ok,
        "healthy train/val/test and seven fault files per mode",
    )
    check(
        "every threshold equals the healthy-validation p99 recomputed from disk",
        all_thresholds_ok,
        "threshold uses no test or fault window",
    )


def safe_median(values: np.ndarray) -> float:
    """Median with an explicit NaN for an empty group."""
    return float(np.median(values)) if values.size > 0 else float("nan")


def share(numerator: int, denominator: int) -> float:
    """Fraction with an explicit NaN when the denominator is zero."""
    return float(numerator / denominator) if denominator > 0 else float("nan")


def build_sanity_frame(records: list[dict[str, object]]) -> pd.DataFrame:
    """Healthy flag rates and per-fault whole-file/pre/post sanity numbers."""
    rows: list[dict[str, object]] = []

    for record in records:
        path = PROJECT_ROOT / str(record["npz_path"])
        with np.load(path, allow_pickle=False) as archive:
            scores = np.asarray(archive["scores"], dtype=np.float64)
            start_time = np.asarray(archive["start_time"], dtype=np.float64)
            end_time = np.asarray(archive["end_time"], dtype=np.float64)
            threshold = float(np.asarray(archive["threshold"]))
            names = np.asarray(archive["name"]).astype(str)

        healthy_stem = str(record["healthy_file"])
        base = {
            "model": record["model"],
            "model_label": record["model_label"],
            "mode": record["mode"],
            "threshold": threshold,
            "input_kind": record["input_kind"],
            "train_units": record["train_units"],
            "subsample_applied": record["subsample_applied"],
            "training_time_s": record["training_time_s"],
            "scoring_time_s": record["scoring_time_s"],
        }

        for part in ("val", "test"):
            mask = names == f"{healthy_stem}_{part}"
            group_scores = scores[mask]
            flagged = int(np.sum(group_scores > threshold))
            rows.append(
                {
                    **base,
                    "section": "healthy",
                    "file": healthy_stem,
                    "part": part,
                    "fault": "",
                    "windows": int(group_scores.size),
                    "flagged": flagged,
                    "flag_rate": share(flagged, int(group_scores.size)),
                    "median_score": safe_median(group_scores),
                    "pre_injection_windows": np.nan,
                    "median_pre_injection_score": np.nan,
                    "post_injection_windows": np.nan,
                    "median_after_8s_score": np.nan,
                    "median_ratio_after8_over_pre": np.nan,
                    "no_post_injection_data": False,
                }
            )

        for fault in FAULT_STEMS_BY_MODE[str(record["mode_suffix"])]:
            mask = names == fault
            group_scores = scores[mask]
            group_start = start_time[mask]
            group_end = end_time[mask]
            flagged = int(np.sum(group_scores > threshold))

            pre_mask = (group_end < FAULT_START_S) & (
                group_start >= STARTUP_END_S
            )
            post_mask = group_start > FAULT_END_S
            median_pre = safe_median(group_scores[pre_mask])
            median_post = safe_median(group_scores[post_mask])
            ratio = (
                float(median_post / median_pre)
                if np.isfinite(median_pre)
                and np.isfinite(median_post)
                and median_pre > 0.0
                else float("nan")
            )
            no_post = int(post_mask.sum()) == 0

            rows.append(
                {
                    **base,
                    "section": "fault",
                    "file": fault,
                    "part": "whole_file",
                    "fault": fault[:2],
                    "windows": int(group_scores.size),
                    "flagged": flagged,
                    "flag_rate": share(flagged, int(group_scores.size)),
                    "median_score": safe_median(group_scores),
                    "pre_injection_windows": int(pre_mask.sum()),
                    "median_pre_injection_score": median_pre,
                    "post_injection_windows": int(post_mask.sum()),
                    "median_after_8s_score": median_post,
                    "median_ratio_after8_over_pre": ratio,
                    "no_post_injection_data": bool(no_post),
                }
            )

    frame = pd.DataFrame(rows)
    column_order = [
        "model",
        "model_label",
        "mode",
        "section",
        "file",
        "part",
        "fault",
        "windows",
        "flagged",
        "flag_rate",
        "median_score",
        "pre_injection_windows",
        "median_pre_injection_score",
        "post_injection_windows",
        "median_after_8s_score",
        "median_ratio_after8_over_pre",
        "no_post_injection_data",
        "threshold",
        "input_kind",
        "train_units",
        "subsample_applied",
        "training_time_s",
        "scoring_time_s",
    ]
    return frame[column_order]


def validate_sanity_frame(frame: pd.DataFrame) -> None:
    """Check the requested sanity table for consistency and F3M handling."""
    expected_rows = len(MODEL_ORDER) * 2 * (2 + 7)
    check(
        "sanity table has 5 models x 2 modes x 9 groups",
        frame.shape[0] == expected_rows,
        f"{frame.shape[0]} rows, expected {expected_rows}",
    )

    flag_rate_ok = True
    for row in frame.itertuples():
        expected = (
            float(row.flagged / row.windows) if row.windows > 0 else float("nan")
        )
        if not np.isclose(float(row.flag_rate), expected, rtol=0.0, atol=1e-12):
            flag_rate_ok = False
    check(
        "every flag rate equals flagged/windows",
        flag_rate_ok,
        "healthy and fault rows",
    )

    f3m = frame[frame["file"] == "F3M"]
    f3m_ok = bool(
        (f3m["post_injection_windows"] == 0).all()
        and f3m["no_post_injection_data"].all()
        and f3m["median_after_8s_score"].isna().all()
    )
    check(
        "F3M is marked as having no post-injection data",
        f3m_ok,
        f"{f3m.shape[0]} model/mode rows; post windows = 0",
    )

    val_rows = frame[(frame["section"] == "healthy") & (frame["part"] == "val")]
    val_ok = bool((val_rows["flag_rate"] <= 0.02).all())
    check(
        "healthy validation flag rate is at most 2%",
        val_ok,
        ", ".join(
            f"{row.mode}/{row.model_label}={float(row.flag_rate):.4f}"
            for row in val_rows.itertuples()
        ),
    )


def phase2_comparison(
    data: dict[str, object], records: list[dict[str, object]]
) -> None:
    """Check that the imported Mahalanobis placeholder reproduces Phase 2."""
    phase2_json = RESULTS_DIR / "time_diagnostic.json"
    phase2_csv = RESULTS_DIR / "time_diagnostic_scores.csv"
    if not phase2_json.exists() or not phase2_csv.exists():
        check(
            "Mahalanobis reference reproduces Phase 2",
            False,
            "Phase 2 score files are missing",
        )
        return

    with phase2_json.open("r", encoding="utf-8") as handle:
        phase2_meta = json.load(handle)
    phase2_scores = pd.read_csv(phase2_csv)

    threshold_ok = True
    scores_ok = True
    compared_windows = 0
    details: list[str] = []

    for record in records:
        if record["model"] != "mahalanobis_window_mean":
            continue
        mode = str(record["mode"])
        suffix = str(record["mode_suffix"])
        healthy_stem = str(record["healthy_file"])
        expected_threshold = float(phase2_meta["modes"][mode]["threshold"])
        threshold_ok = threshold_ok and bool(
            np.isclose(
                float(record["threshold"]),
                expected_threshold,
                rtol=1e-12,
                atol=1e-12,
            )
        )

        path = PROJECT_ROOT / str(record["npz_path"])
        with np.load(path, allow_pickle=False) as archive:
            scores = np.asarray(archive["scores"], dtype=np.float64)
            names = np.asarray(archive["name"]).astype(str)

        for part in ("train", "val", "test"):
            indices = np.asarray(data["splits"][suffix][part], dtype=np.int64)
            expected = (
                phase2_scores[phase2_scores["file"] == healthy_stem]
                .sort_values("window_index")["score"]
                .to_numpy(dtype=np.float64)[indices]
            )
            actual = scores[names == f"{healthy_stem}_{part}"]
            compared_windows += int(actual.size)
            scores_ok = scores_ok and bool(
                np.allclose(actual, expected, rtol=1e-9, atol=1e-12)
            )

        for fault in FAULT_STEMS_BY_MODE[suffix]:
            expected = (
                phase2_scores[phase2_scores["file"] == fault]
                .sort_values("window_index")["score"]
                .to_numpy(dtype=np.float64)
            )
            actual = scores[names == fault]
            compared_windows += int(actual.size)
            scores_ok = scores_ok and bool(
                np.allclose(actual, expected, rtol=1e-9, atol=1e-12)
            )

        details.append(
            f"{mode}: threshold {float(record['threshold']):.6g} "
            f"vs Phase 2 {expected_threshold:.6g}"
        )

    check(
        "Mahalanobis reference reuses the Phase 2 threshold unchanged",
        threshold_ok,
        "; ".join(details),
    )
    check(
        "Mahalanobis reference reuses the Phase 2 window scores unchanged",
        scores_ok,
        f"all {compared_windows:,} scored windows match "
        "time_diagnostic_scores.csv",
    )


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def build_metadata(
    data: dict[str, object],
    records: list[dict[str, object]],
    sanity: pd.DataFrame,
) -> dict[str, object]:
    """Assemble JSON-safe run metadata and check results."""
    return {
        "phase": "phase_3_baselines",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "environment": environment_info(),
        "seed": SEED,
        "seed_note": (
            "The phase specification fixes random_state=0 for the two "
            "stochastic baselines (EllipticEnvelope and IsolationForest); "
            "LOF, OneClassSVM, and the Mahalanobis placeholder are "
            "deterministic. No multi-seed sweep is performed because the "
            "baseline settings are frozen and no fault data are used."
        ),
        "label_convention": {
            "primary": "whole_file",
            "secondary": "time_resolved",
            "f3m": "no post-injection data; always reported as such",
            "source": "AGENTS.md section 2a (decided 2026-10-27)",
        },
        "training_data": (
            "unique standardized 1 kHz rows covered by the healthy training "
            "windows of the same mode (F0L for IPPT, F0M for MPPT)"
        ),
        "subsample_cap": SUBSAMPLE_CAP,
        "threshold_rule": (
            "99th percentile of window scores on healthy validation windows"
        ),
        "threshold_quantile": THRESHOLD_QUANTILE,
        "flag_rule": "score > threshold",
        "window_score_rule": (
            "mean of -decision_function over the 100 time steps in a window; "
            "the Mahalanobis reference uses its Phase 2 window-mean distance"
        ),
        "models": {
            key: {
                "label": spec["label"],
                "input": spec["input"],
                "params": spec["params"],
            }
            for key, spec in MODEL_SPECS.items()
        },
        "runs": records,
        "sanity_csv": str(SANITY_PATH.relative_to(PROJECT_ROOT)),
        "sanity_rows": int(sanity.shape[0]),
        "checks_passed": sum(
            bool(result["passed"]) for result in CHECK_RESULTS
        ),
        "checks_total": len(CHECK_RESULTS),
        "checks": CHECK_RESULTS,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    print("=" * 80)
    print("GPVS-Faults | Phase 3 reference baselines")
    print("=" * 80)
    print(f"Project root : {PROJECT_ROOT}")

    data = build_dataset(verbose=False)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\nPreparing standardized rows and checking window alignment:")
    scaled_by_mode: dict[str, dict[str, np.ndarray]] = {}
    for suffix, mode in MODES.items():
        scaled_by_mode[suffix] = scaled_rows_for_mode(data, suffix)
        problems = check_scaled_row_alignment(
            data, suffix, scaled_by_mode[suffix]
        )
        check(
            f"cached scaled rows match {mode} windows",
            not problems,
            (
                "first, middle, and last window of every file"
                if not problems
                else ", ".join(problems)
            ),
        )

    print("\nFitting baselines on healthy training rows only:")
    records: list[dict[str, object]] = []
    for model_key in MODEL_ORDER:
        for suffix in ("L", "M"):
            records.append(
                run_model_mode(
                    data, model_key, suffix, scaled_by_mode[suffix]
                )
            )

    print("\nArchive checks:")
    validate_archives(data, records)

    print("\nMahalanobis comparison with Phase 2:")
    phase2_comparison(data, records)

    print("\nBuilding sanity table from the saved archives:")
    sanity = build_sanity_frame(records)
    validate_sanity_frame(sanity)
    sanity.to_csv(SANITY_PATH, index=False)
    print(f"  saved {SANITY_PATH}")

    metadata = build_metadata(data, records, sanity)
    with METADATA_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(metadata, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"  saved {METADATA_PATH}")

    print("\nTraining time per model and mode:")
    timing = sanity[
        [
            "model",
            "model_label",
            "mode",
            "input_kind",
            "train_units",
            "subsample_applied",
            "training_time_s",
            "scoring_time_s",
            "threshold",
        ]
    ].drop_duplicates()
    print(timing.to_string(index=False))

    print("\nHealthy flag rates (validation and test):")
    healthy = sanity[sanity["section"] == "healthy"]
    print(
        healthy[
            ["model_label", "mode", "part", "windows", "flagged", "flag_rate"]
        ].to_string(index=False)
    )

    print("\nPer-fault whole-file flag share and post/pre median ratio:")
    fault = sanity[sanity["section"] == "fault"]
    print(
        fault[
            [
                "model_label",
                "mode",
                "file",
                "windows",
                "flagged",
                "flag_rate",
                "median_pre_injection_score",
                "median_after_8s_score",
                "median_ratio_after8_over_pre",
                "no_post_injection_data",
            ]
        ].to_string(index=False, na_rep="NA")
    )

    all_passed = all(bool(result["passed"]) for result in CHECK_RESULTS)
    print(
        f"\nRESULT: {'PASS' if all_passed else 'FAIL'} - "
        f"{sum(bool(result['passed']) for result in CHECK_RESULTS)}"
        f"/{len(CHECK_RESULTS)} checks passed."
    )
    if not all_passed:
        failed = [
            str(result["name"])
            for result in CHECK_RESULTS
            if not result["passed"]
        ]
        raise AssertionError(f"Baseline check(s) failed: {failed}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

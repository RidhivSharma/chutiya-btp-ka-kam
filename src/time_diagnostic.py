"""Phase 2 diagnostic: when do fault files actually start to look anomalous?

This script does **not** build the final detector. It uses one deliberately
simple placeholder so that the window labels inherited from the notebook can be
checked against the documented fault-injection interval (7-8 s):

* for each operating mode, the mean and covariance of the 13-dimensional
  window-mean vectors are estimated from healthy *training* windows only;
* the score of a window is the Mahalanobis distance of its window mean, using a
  pseudo-inverse of the covariance plus a small relative ridge;
* the threshold is the 99th percentile of the scores on healthy *validation*
  windows.

The placeholder never sees a fault file while being fitted or thresholded, and
it is not tuned using fault results. Fault files are scored only to produce the
diagnostics requested for this phase.

Outputs
-------
``results/time_diagnostic_scores.csv``
    One row per window of all 16 files: score, threshold, flag, timestamps, and
    both label modes.
``results/label_diagnostic.csv``
    Per fault and mode: pre-injection, startup, injection-window and
    post-injection flag shares.
``results/healthy_false_alarm.csv``
    Flag rates on healthy train/validation/test windows for reference.
``results/time_diagnostic.json``
    Detector definition, fitted parameters, thresholds, and check results.
``figures/score_vs_time_<file>.png``
    Score versus window start time for each of the 14 fault files.
``figures/score_vs_time_grid.png``
    Seven faults by two modes summary grid.

Run::

    .venv\\Scripts\\python.exe .\\src\\time_diagnostic.py
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

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
    build_dataset,
    environment_info,
    get_labels,
    git_commit,
)

FIGURES_DIR = PROJECT_ROOT / "figures"

DETECTOR_NAME = "placeholder_window_mean_mahalanobis"
RIDGE_RELATIVE = 1e-6
THRESHOLD_QUANTILE = 0.99
STARTUP_END_S = 0.5

FAULT_STEMS = tuple(f"F{i}{suffix}" for i in range(1, 8) for suffix in ("L", "M"))
SEGMENT_NAMES = {
    -1: "pre_fault",
    -2: "injection_overlap",
    0: "healthy",
    1: "post_fault",
}

CHECK_RESULTS: list[dict[str, object]] = []


def check(name: str, passed: bool, detail: str) -> bool:
    """Record and print one named assertion."""
    passed = bool(passed)
    CHECK_RESULTS.append({"name": name, "passed": passed, "detail": detail})
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    return passed


def fit_placeholder(train_means: np.ndarray) -> dict[str, object]:
    """Fit the window-mean mean/covariance placeholder on healthy train data."""
    if train_means.ndim != 2 or train_means.shape[1] != len(FEATURES):
        raise ValueError(
            f"Expected (n_windows, {len(FEATURES)}) train means, "
            f"got {train_means.shape}"
        )
    if train_means.shape[0] <= len(FEATURES):
        raise ValueError("Not enough healthy training windows for a covariance")

    center = train_means.mean(axis=0)
    covariance = np.cov(train_means, rowvar=False, ddof=1)
    n_features = covariance.shape[0]
    mean_variance = float(np.trace(covariance) / n_features)
    if not np.isfinite(mean_variance) or mean_variance <= 0.0:
        raise RuntimeError(f"Degenerate window-mean covariance: {mean_variance}")

    ridge = RIDGE_RELATIVE * mean_variance
    regularized = covariance + ridge * np.eye(n_features)
    precision = np.linalg.pinv(regularized, hermitian=True)

    if not np.isfinite(precision).all():
        raise RuntimeError("Pseudo-inverse produced non-finite values")

    return {
        "center": center,
        "covariance": covariance,
        "mean_variance": mean_variance,
        "ridge": ridge,
        "regularized_covariance": regularized,
        "precision": precision,
        "condition_number": float(np.linalg.cond(regularized)),
    }


def score_means(means: np.ndarray, detector: dict[str, object]) -> np.ndarray:
    """Mahalanobis distance of each window mean under a fitted placeholder."""
    center = np.asarray(detector["center"], dtype=np.float64)
    precision = np.asarray(detector["precision"], dtype=np.float64)
    difference = np.asarray(means, dtype=np.float64) - center
    quadratic = np.einsum(
        "ij,jk,ik->i", difference, precision, difference, optimize=True
    )
    # Round-off can make an exactly-zero quadratic form slightly negative.
    return np.sqrt(np.maximum(quadratic, 0.0))


def share(numerator: int, denominator: int) -> float:
    """Fraction with an explicit NaN when the denominator is zero."""
    return float(numerator / denominator) if denominator > 0 else float("nan")


def build_score_frame(
    data: dict[str, object],
    detectors: dict[str, dict[str, object]],
) -> pd.DataFrame:
    """Score every window of every file and attach both label modes."""
    windows = data["windows"]
    times = data["times"]
    rows: list[pd.DataFrame] = []

    for stem in STEMS:
        suffix = stem[-1]
        detector = detectors[suffix]
        threshold = float(detector["threshold"])

        window_means = windows[stem].mean(axis=1, dtype=np.float64)
        scores = score_means(window_means, detector)
        bounds = np.asarray(times[stem], dtype=np.float64)
        time_labels = get_labels(stem, "time_resolved", times=bounds)
        whole_labels = get_labels(stem, "whole_file", times=bounds)

        rows.append(
            pd.DataFrame(
                {
                    "mode": MODES[suffix],
                    "mode_suffix": suffix,
                    "file": stem,
                    "fault": stem[:2],
                    "condition": "healthy" if stem[1] == "0" else "fault",
                    "window_index": np.arange(bounds.shape[0], dtype=np.int64),
                    "start_time_s": bounds[:, 0],
                    "end_time_s": bounds[:, 1],
                    "score": scores,
                    "threshold": threshold,
                    "flagged": scores > threshold,
                    "label_time_resolved": time_labels,
                    "label_whole_file": whole_labels,
                    "segment": [
                        SEGMENT_NAMES[int(label)] for label in time_labels
                    ],
                }
            )
        )

    return pd.concat(rows, ignore_index=True)


def build_label_diagnostic(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize where the placeholder flags windows inside each fault file."""
    rows: list[dict[str, object]] = []

    for stem in FAULT_STEMS:
        group = frame[frame["file"] == stem].sort_values("window_index")
        if group.empty:
            raise RuntimeError(f"No scored windows for {stem}")

        flagged = group["flagged"].to_numpy(dtype=bool)
        start_time = group["start_time_s"].to_numpy(dtype=np.float64)
        scores = group["score"].to_numpy(dtype=np.float64)

        pre = group["segment"].eq("pre_fault").to_numpy()
        overlap = group["segment"].eq("injection_overlap").to_numpy()
        post = group["segment"].eq("post_fault").to_numpy()
        startup = start_time < STARTUP_END_S
        pre_excluding_startup = pre & ~startup

        n_pre = int(pre.sum())
        n_pre_excl = int(pre_excluding_startup.sum())
        n_startup = int(startup.sum())
        n_overlap = int(overlap.sum())
        n_post = int(post.sum())

        flagged_pre_excl = int(flagged[pre_excluding_startup].sum())
        flagged_startup = int(flagged[startup].sum())
        flagged_overlap = int(flagged[overlap].sum())
        flagged_post = int(flagged[post].sum())

        median_pre = (
            float(np.median(scores[pre_excluding_startup]))
            if n_pre_excl > 0
            else float("nan")
        )
        median_startup = (
            float(np.median(scores[startup]))
            if n_startup > 0
            else float("nan")
        )
        median_overlap = (
            float(np.median(scores[overlap]))
            if n_overlap > 0
            else float("nan")
        )
        median_post = (
            float(np.median(scores[post]))
            if n_post > 0
            else float("nan")
        )
        threshold = float(group["threshold"].iloc[0])

        no_post = n_post == 0
        rows.append(
            {
                "mode": str(group["mode"].iloc[0]),
                "fault": stem[:2],
                "file": stem,
                "windows_total": int(group.shape[0]),
                "windows_before_7s": n_pre,
                "windows_before_7s_excl_startup": n_pre_excl,
                "flagged_before_7s_excl_startup": flagged_pre_excl,
                "share_flagged_before_7s_excl_startup": share(
                    flagged_pre_excl, n_pre_excl
                ),
                "startup_windows": n_startup,
                "flagged_startup": flagged_startup,
                "share_flagged_startup_first_0p5s": share(
                    flagged_startup, n_startup
                ),
                "windows_7_to_8s": n_overlap,
                "flagged_7_to_8s": flagged_overlap,
                "share_flagged_7_to_8s": share(flagged_overlap, n_overlap),
                "windows_after_8s": n_post,
                "flagged_after_8s": flagged_post,
                "share_flagged_after_8s": share(flagged_post, n_post),
                "median_score_before_7s_excl_startup": median_pre,
                "median_score_startup_first_0p5s": median_startup,
                "median_score_7_to_8s": median_overlap,
                "median_score_after_8s": median_post,
                "median_score_before_7s_over_threshold": (
                    float(median_pre / threshold)
                    if np.isfinite(median_pre) and threshold > 0.0
                    else float("nan")
                ),
                "median_score_ratio_7_to_8_over_before_7": (
                    float(median_overlap / median_pre)
                    if np.isfinite(median_overlap)
                    and np.isfinite(median_pre)
                    and median_pre > 0.0
                    else float("nan")
                ),
                "median_score_ratio_after_8_over_before_7": (
                    float(median_post / median_pre)
                    if np.isfinite(median_post)
                    and np.isfinite(median_pre)
                    and median_pre > 0.0
                    else float("nan")
                ),
                "no_post_injection_data": bool(no_post),
                "post_injection_note": (
                    "no post-injection data" if no_post else ""
                ),
            }
        )

    return pd.DataFrame(rows)


def build_healthy_reference(
    frame: pd.DataFrame,
    data: dict[str, object],
) -> pd.DataFrame:
    """False-alarm rates on the healthy train/validation/test windows."""
    rows: list[dict[str, object]] = []
    splits = data["splits"]

    for suffix, mode in MODES.items():
        stem = HEALTHY_STEMS[suffix]
        group = frame[frame["file"] == stem].set_index("window_index")
        for split_name in ("train", "val", "test"):
            indices = np.asarray(splits[suffix][split_name], dtype=np.int64)
            subset = group.loc[indices]
            flagged = subset["flagged"].to_numpy(dtype=bool)
            scores = subset["score"].to_numpy(dtype=np.float64)
            n_windows = int(subset.shape[0])
            n_flagged = int(flagged.sum())
            rows.append(
                {
                    "mode": mode,
                    "split": split_name,
                    "file": stem,
                    "windows": n_windows,
                    "flagged": n_flagged,
                    "false_alarm_rate": share(n_flagged, n_windows),
                    "threshold": float(subset["threshold"].iloc[0]),
                    "median_score": float(np.median(scores)),
                    "p99_score": float(np.quantile(scores, THRESHOLD_QUANTILE)),
                    "first_start_time_s": float(
                        subset["start_time_s"].iloc[0]
                    ),
                    "last_end_time_s": float(subset["end_time_s"].iloc[-1]),
                }
            )

    return pd.DataFrame(rows)


def score_axis_limits(scores: np.ndarray, threshold: float) -> tuple[float, float]:
    """Choose log-axis limits that keep zero scores from breaking the plot."""
    positive = scores[np.isfinite(scores) & (scores > 0.0)]
    candidates = np.concatenate((positive, np.array([threshold])))
    candidates = candidates[np.isfinite(candidates) & (candidates > 0.0)]
    if candidates.size == 0:
        return 1e-6, 1.0
    bottom = max(float(candidates.min()) / 10.0, 1e-12)
    top = float(candidates.max()) * 2.0
    if top <= bottom:
        top = bottom * 10.0
    return bottom, top


def draw_panel(
    ax: plt.Axes,
    subset: pd.DataFrame,
    title: str,
    show_legend: bool,
    show_xlabel: bool,
    show_ylabel: bool,
    marker_size: float = 6.0,
) -> None:
    """Draw one score-versus-time panel from saved score rows."""
    subset = subset.sort_values("window_index")
    x = subset["start_time_s"].to_numpy(dtype=np.float64)
    y = subset["score"].to_numpy(dtype=np.float64)
    flagged = subset["flagged"].to_numpy(dtype=bool)
    threshold = float(subset["threshold"].iloc[0])

    ax.axvspan(
        FAULT_START_S,
        FAULT_END_S,
        color="#ff7f0e",
        alpha=0.12,
        linewidth=0,
        zorder=0,
    )
    ax.axvline(FAULT_START_S, color="#ff7f0e", linestyle="--", linewidth=1.0)
    ax.axvline(FAULT_END_S, color="#ff7f0e", linestyle="--", linewidth=1.0)
    ax.axhline(threshold, color="black", linestyle="--", linewidth=1.1)

    visible = y > 0.0
    if np.any(~flagged & visible):
        ax.scatter(
            x[~flagged & visible],
            y[~flagged & visible],
            s=marker_size,
            color="#1f77b4",
            alpha=0.55,
            linewidths=0,
            label="below threshold",
            zorder=2,
        )
    if np.any(flagged & visible):
        ax.scatter(
            x[flagged & visible],
            y[flagged & visible],
            s=marker_size * 1.4,
            color="#d62728",
            alpha=0.75,
            linewidths=0,
            label="flagged",
            zorder=3,
        )

    ax.set_yscale("log")
    bottom, top = score_axis_limits(y, threshold)
    ax.set_ylim(bottom, top)
    right = max(FAULT_END_S + 0.2, float(np.max(subset["end_time_s"])) + 0.05)
    ax.set_xlim(0.0, right)
    ax.grid(True, which="both", alpha=0.20, linewidth=0.4)
    ax.set_title(title, fontsize=9)
    if show_xlabel:
        ax.set_xlabel("Window start time (s)", fontsize=8)
    if show_ylabel:
        ax.set_ylabel("Mahalanobis distance (log scale)", fontsize=8)
    ax.tick_params(labelsize=7)

    if np.any(y <= 0.0):
        ax.text(
            0.02,
            0.02,
            f"{int(np.sum(y <= 0.0))} zero score(s) not shown",
            transform=ax.transAxes,
            fontsize=6,
            color="#555555",
        )

    no_post = int(subset["segment"].eq("post_fault").sum()) == 0
    if no_post:
        ax.text(
            0.98,
            0.95,
            "no post-injection data",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            color="#8c2d04",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )

    if show_legend:
        ax.legend(
            loc="upper left",
            fontsize=6.5,
            framealpha=0.9,
            markerscale=1.8,
        )


def plot_individual_figures(frame: pd.DataFrame) -> list[Path]:
    """Write one score-versus-time figure for each fault file."""
    paths: list[Path] = []
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for stem in FAULT_STEMS:
        subset = frame[frame["file"] == stem]
        suffix = stem[-1]
        fig, ax = plt.subplots(figsize=(9.0, 4.6))
        draw_panel(
            ax,
            subset,
            f"{stem} ({MODES[suffix]}): placeholder score vs window start time",
            show_legend=True,
            show_xlabel=True,
            show_ylabel=True,
        )
        fig.tight_layout()
        path = FIGURES_DIR / f"score_vs_time_{stem}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_summary_grid(frame: pd.DataFrame) -> Path:
    """Write the 7-fault by 2-mode summary grid."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    faults = [f"F{i}" for i in range(1, 8)]
    suffixes = ("L", "M")

    fig, axes = plt.subplots(
        len(faults),
        len(suffixes),
        figsize=(12.5, 18.0),
        sharex=False,
    )

    for row, fault in enumerate(faults):
        for column, suffix in enumerate(suffixes):
            ax = axes[row, column]
            stem = f"{fault}{suffix}"
            subset = frame[frame["file"] == stem]
            draw_panel(
                ax,
                subset,
                f"{stem} ({MODES[suffix]})",
                show_legend=False,
                show_xlabel=row == len(faults) - 1,
                show_ylabel=column == 0,
                marker_size=3.0,
            )

    handles = [
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.1,
               label=f"healthy-validation p{int(THRESHOLD_QUANTILE * 100)} threshold"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#d62728",
               markeredgecolor="none", markersize=4, label="flagged"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#1f77b4",
               markeredgecolor="none", markersize=4, label="below threshold"),
        Patch(facecolor="#ff7f0e", alpha=0.20, label="documented 7-8 s injection"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, 0.997),
    )
    fig.suptitle(
        "Placeholder window-mean Mahalanobis score by fault and operating mode",
        fontsize=12,
        y=0.999,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
    path = FIGURES_DIR / "score_vs_time_grid.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def build_metadata(
    data: dict[str, object],
    detectors: dict[str, dict[str, object]],
    label_frame: pd.DataFrame,
    healthy_frame: pd.DataFrame,
) -> dict[str, object]:
    """Assemble the JSON-safe diagnostic definition and fitted parameters."""
    splits = data["splits"]
    mode_metadata: dict[str, object] = {}

    for suffix, mode in MODES.items():
        stem = HEALTHY_STEMS[suffix]
        detector = detectors[suffix]
        precision = np.asarray(detector["precision"], dtype=np.float64)
        regularized = np.asarray(
            detector["regularized_covariance"], dtype=np.float64
        )
        mode_metadata[mode] = {
            "healthy_file": stem,
            "train_windows": int(splits[suffix]["train"].size),
            "validation_windows": int(splits[suffix]["val"].size),
            "test_windows": int(splits[suffix]["test"].size),
            "threshold": float(detector["threshold"]),
            "validation_flagged": int(detector["validation_flagged"]),
            "validation_flag_rate": float(detector["validation_flag_rate"]),
            "test_flagged": int(detector["test_flagged"]),
            "test_false_alarm_rate": float(detector["test_false_alarm_rate"]),
            "covariance_ddof": 1,
            "mean_window_variance": float(detector["mean_variance"]),
            "ridge": float(detector["ridge"]),
            "regularized_condition_number": float(
                detector["condition_number"]
            ),
            "train_mean": [float(value) for value in detector["center"]],
            "covariance": [
                [float(value) for value in row]
                for row in np.asarray(detector["covariance"])
            ],
            "regularized_covariance": [
                [float(value) for value in row] for row in regularized
            ],
            "precision": [
                [float(value) for value in row] for row in precision
            ],
        }

    f3m = label_frame[label_frame["file"] == "F3M"].iloc[0]
    checks_passed = sum(bool(result["passed"]) for result in CHECK_RESULTS)

    return {
        "phase": "phase_2_labeling_time_diagnostic",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "environment": environment_info(),
        "stochastic": False,
        "seed": None,
        "seed_note": (
            "No random operation is performed. The placeholder is a "
            "deterministic mean/covariance calculation, and no fault data "
            "enter the fit or the threshold."
        ),
        "detector": {
            "name": DETECTOR_NAME,
            "purpose": "diagnostic only; not the final anomaly detector",
            "input": (
                "mean over the 100 time samples of each mode-standardized "
                "13-feature window"
            ),
            "fit_data": (
                "healthy training windows only (F0L for IPPT, F0M for MPPT)"
            ),
            "covariance_ddof": 1,
            "score": "sqrt((x-mu)^T pinv(C + ridge*I) (x-mu))",
            "pseudo_inverse": "numpy.linalg.pinv(..., hermitian=True)",
            "ridge_relative": RIDGE_RELATIVE,
            "ridge_rule": "ridge = 1e-6 * mean(diag(covariance))",
            "threshold": (
                "99th percentile of the placeholder score on healthy "
                "validation windows"
            ),
            "threshold_quantile": THRESHOLD_QUANTILE,
            "threshold_comparison": "score > threshold",
            "fault_files_used_for_fit": 0,
            "fault_files_used_for_threshold": 0,
        },
        "labels": {
            "whole_file": {
                "healthy": 0,
                "fault": 1,
                "scope": "every window of F1..F7",
            },
            "time_resolved": {
                "healthy": 0,
                "pre_fault": -1,
                "injection_overlap_dropped": -2,
                "post_injection_fault": 1,
                "pre_fault_rule": "window end_time < 7.0 s",
                "post_fault_rule": "window start_time > 8.0 s",
                "overlap_rule": "otherwise (window intersects 7.0-8.0 s)",
            },
            "fault_injection_interval_s": [FAULT_START_S, FAULT_END_S],
            "startup_end_s": STARTUP_END_S,
            "startup_definition": "window start_time < 0.5 s",
        },
        "modes": mode_metadata,
        "label_diagnostic_rows": int(label_frame.shape[0]),
        "healthy_reference_rows": int(healthy_frame.shape[0]),
        "f3m": {
            "windows": int(f3m["windows_total"]),
            "windows_before_7s": int(f3m["windows_before_7s"]),
            "windows_7_to_8s": int(f3m["windows_7_to_8s"]),
            "windows_after_8s": int(f3m["windows_after_8s"]),
            "no_post_injection_data": bool(f3m["no_post_injection_data"]),
            "note": str(f3m["post_injection_note"]),
        },
        "checks_passed": checks_passed,
        "checks_total": len(CHECK_RESULTS),
        "checks": CHECK_RESULTS,
    }


def main() -> int:
    print("=" * 80)
    print("GPVS-Faults | Phase 2 labeling and timing diagnostic")
    print("=" * 80)
    print(f"Project root : {PROJECT_ROOT}")

    data = build_dataset(verbose=False)
    windows = data["windows"]
    times = data["times"]
    splits = data["splits"]

    print("\nFitting the placeholder on healthy training windows only:")
    detectors: dict[str, dict[str, object]] = {}
    for suffix, mode in MODES.items():
        stem = HEALTHY_STEMS[suffix]
        train_indices = np.asarray(splits[suffix]["train"], dtype=np.int64)
        val_indices = np.asarray(splits[suffix]["val"], dtype=np.int64)
        test_indices = np.asarray(splits[suffix]["test"], dtype=np.int64)

        train_means = windows[stem][train_indices].mean(
            axis=1, dtype=np.float64
        )
        val_means = windows[stem][val_indices].mean(axis=1, dtype=np.float64)
        test_means = windows[stem][test_indices].mean(axis=1, dtype=np.float64)

        detector = fit_placeholder(train_means)
        val_scores = score_means(val_means, detector)
        test_scores = score_means(test_means, detector)
        threshold = float(np.quantile(val_scores, THRESHOLD_QUANTILE))
        validation_flagged = int(np.sum(val_scores > threshold))
        test_flagged = int(np.sum(test_scores > threshold))

        detector["threshold"] = threshold
        detector["validation_scores"] = val_scores
        detector["validation_flagged"] = validation_flagged
        detector["validation_flag_rate"] = share(
            validation_flagged, val_scores.size
        )
        detector["test_scores"] = test_scores
        detector["test_flagged"] = test_flagged
        detector["test_false_alarm_rate"] = share(
            test_flagged, test_scores.size
        )
        detectors[suffix] = detector

        print(
            f"  {mode:<5} train={train_indices.size:,} "
            f"val={val_indices.size:,} test={test_indices.size:,} | "
            f"ridge={detector['ridge']:.6g} "
            f"cond={detector['condition_number']:.3g} | "
            f"threshold={threshold:.6g} | "
            f"val flagged={validation_flagged} "
            f"({detector['validation_flag_rate']:.4f}) | "
            f"test flagged={test_flagged} "
            f"({detector['test_false_alarm_rate']:.4f})"
        )

    print("\nScoring all 16 files (fault files are not used in fitting):")
    score_frame = build_score_frame(data, detectors)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    scores_path = RESULTS_DIR / "time_diagnostic_scores.csv"
    score_frame.to_csv(scores_path, index=False)
    print(f"  {score_frame.shape[0]:,} window scores written")

    # Read the saved file back so every downstream figure and table is derived
    # from the persisted, auditable numbers rather than in-memory state.
    saved_scores = pd.read_csv(scores_path)
    saved_scores["flagged"] = saved_scores["flagged"].astype(bool)

    label_frame = build_label_diagnostic(saved_scores)
    healthy_frame = build_healthy_reference(saved_scores, data)

    label_path = RESULTS_DIR / "label_diagnostic.csv"
    healthy_path = RESULTS_DIR / "healthy_false_alarm.csv"
    label_frame.to_csv(label_path, index=False)
    healthy_frame.to_csv(healthy_path, index=False)

    # ------------------------------------------------------------------
    # Checks
    # ------------------------------------------------------------------
    print("\nChecks:")
    expected_rows = int(sum(windows[stem].shape[0] for stem in STEMS))
    check(
        "every window of all 16 files was scored",
        saved_scores.shape[0] == expected_rows,
        f"{saved_scores.shape[0]:,} rows, expected {expected_rows:,}",
    )

    check(
        "all scores are finite and non-negative",
        bool(np.isfinite(saved_scores["score"]).all())
        and bool((saved_scores["score"] >= 0.0).all()),
        f"score range {saved_scores['score'].min():.6g} to "
        f"{saved_scores['score'].max():.6g}",
    )

    valid_labels = {-2, -1, 0, 1}
    observed_labels = set(
        int(value) for value in saved_scores["label_time_resolved"].unique()
    )
    check(
        "time-resolved labels use only -2/-1/0/1",
        observed_labels.issubset(valid_labels),
        f"observed {sorted(observed_labels)}",
    )

    label_mismatches: list[str] = []
    for stem in STEMS:
        subset = saved_scores[saved_scores["file"] == stem].sort_values(
            "window_index"
        )
        expected = get_labels(
            stem,
            "time_resolved",
            times=subset[["start_time_s", "end_time_s"]].to_numpy(),
        )
        if not np.array_equal(
            subset["label_time_resolved"].to_numpy(dtype=np.int8), expected
        ):
            label_mismatches.append(stem)
    check(
        "saved time-resolved labels match get_labels after CSV round-trip",
        not label_mismatches,
        "all files agree" if not label_mismatches else str(label_mismatches),
    )

    segment_counts_ok = True
    for stem in FAULT_STEMS:
        subset = saved_scores[saved_scores["file"] == stem]
        counts = subset["segment"].value_counts()
        total = int(
            counts.get("pre_fault", 0)
            + counts.get("injection_overlap", 0)
            + counts.get("post_fault", 0)
        )
        segment_counts_ok = segment_counts_ok and total == subset.shape[0]
    check(
        "fault windows partition into pre/overlap/post segments",
        segment_counts_ok,
        "pre + injection-overlap + post equals the file window count",
    )

    f3m_row = label_frame[label_frame["file"] == "F3M"].iloc[0]
    check(
        "F3M has no post-injection or 7-8 s window and is handled",
        int(f3m_row["windows_after_8s"]) == 0
        and int(f3m_row["windows_7_to_8s"]) == 0
        and bool(f3m_row["no_post_injection_data"]),
        f"{int(f3m_row['windows_before_7s'])} pre-fault windows, "
        "0 overlap, 0 post; row written without crashing",
    )

    healthy_files_ok = set(healthy_frame["file"].unique()) == set(
        HEALTHY_STEMS.values()
    )
    check(
        "healthy reference contains only F0L/F0M",
        healthy_files_ok,
        ", ".join(sorted(healthy_frame["file"].unique())),
    )

    test_rows = healthy_frame[healthy_frame["split"] == "test"]
    test_far_ok = all(
        math.isclose(
            float(row.false_alarm_rate),
            float(row.flagged / row.windows),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for row in test_rows.itertuples()
    )
    check(
        "healthy test false-alarm rate equals flagged/windows",
        test_far_ok,
        "; ".join(
            f"{row.mode}: {int(row.flagged)}/{int(row.windows)}="
            f"{row.false_alarm_rate:.4f}"
            for row in test_rows.itertuples()
        ),
    )

    threshold_ok = all(
        np.isfinite(detector["threshold"]) and float(detector["threshold"]) > 0
        for detector in detectors.values()
    )
    check(
        "both thresholds are finite and positive",
        threshold_ok,
        ", ".join(
            f"{MODES[suffix]}={detectors[suffix]['threshold']:.6g}"
            for suffix in ("L", "M")
        ),
    )

    # ------------------------------------------------------------------
    # Figures, derived from the saved score CSV
    # ------------------------------------------------------------------
    print("\nWriting figures from the saved score table:")
    individual_paths = plot_individual_figures(saved_scores)
    grid_path = plot_summary_grid(saved_scores)
    print(f"  {len(individual_paths)} individual fault figures")
    print(f"  {grid_path.name}")

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    metadata = build_metadata(
        data, detectors, label_frame, healthy_frame
    )
    metadata_path = RESULTS_DIR / "time_diagnostic.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, allow_nan=False)
        handle.write("\n")

    print("\nPer-fault flag shares (fractions, not percent):")
    display_columns = [
        "file",
        "windows_before_7s",
        "share_flagged_before_7s_excl_startup",
        "share_flagged_startup_first_0p5s",
        "share_flagged_7_to_8s",
        "share_flagged_after_8s",
        "median_score_before_7s_over_threshold",
        "median_score_ratio_after_8_over_before_7",
        "no_post_injection_data",
    ]
    display = label_frame[display_columns].copy()
    print(
        display.to_string(
            index=False,
            formatters={
                "share_flagged_before_7s_excl_startup": "{:.4f}".format,
                "share_flagged_startup_first_0p5s": "{:.4f}".format,
                "share_flagged_7_to_8s": "{:.4f}".format,
                "share_flagged_after_8s": "{:.4f}".format,
                "median_score_before_7s_over_threshold": "{:.1f}".format,
                "median_score_ratio_after_8_over_before_7": "{:.2f}".format,
            },
            na_rep="NA",
        )
    )

    print("\nSaved:")
    for path in (
        scores_path,
        label_path,
        healthy_path,
        metadata_path,
        grid_path,
        *individual_paths,
    ):
        print(f"  {path}")

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
        raise AssertionError(f"Diagnostic check(s) failed: {failed}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

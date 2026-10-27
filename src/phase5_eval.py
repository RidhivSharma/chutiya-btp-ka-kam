"""Phase 5: final evaluation, computed only from the saved score archives.

This module reads ``results/scores_baseline_*.npz`` and
``results/scores_proposed_*.npz`` and nothing else. It never trains a model,
never changes a score, never changes k=3, never changes a normalization, and
never recomputes a threshold except in the explicitly requested sensitivity
analysis (percentiles of the saved healthy validation scores).

Primary view (``whole_file``)
    Every window of a fault file is labelled 1 and every healthy test window of
    the same mode is labelled 0.  For each model, mode and fault file the
    threshold stored in the archive is applied and TPR, FPR, accuracy,
    precision, F1 and AUC are computed.  The AUC uses the continuous scores.
    Each cell is bootstrapped 200 times with n = number of healthy test windows
    drawn from each class, so every resample has equal class sizes.

Secondary view (within-run change test)
    For each fault file with post-injection data, pre-injection windows
    (end < 7.0 s and start >= 0.5 s) are labelled 0 and post-injection windows
    (start >= 8.0 s) are labelled 1, both from the same file.  AUC and its
    bootstrap interval are computed with equal class sizes
    (n = min(n_pre, n_post)).  The share of pre-injection windows flagged by
    the stored threshold is also saved.

Bootstrap convention
    ``BOOTSTRAP_SEED`` is fixed.  Independent ``SeedSequence`` streams are
    derived from it using the analysis, mode and fault index, so each table
    cell is reproducible independently of loop order.  The same draws are
    shared by all models and by the three proposed-model seeds within a
    mode/fault, which makes the seed aggregation paired.  Intervals are the
    2.5th and 97.5th percentiles of the bootstrap distribution.

Proposed-model seed aggregation
    Each proposed variant is evaluated per seed (0, 1, 2).  A per-fault value
    is the mean of the three per-seed bootstrap means; the reported seed
    standard deviation is the standard deviation of those three means
    (ddof = 1).  The 95% interval is taken from the seed-averaged bootstrap
    distribution (the three per-seed distributions are averaged resample by
    resample).

Zero-division convention
    Precision is 0.0 when no window is flagged, and F1 is 0.0 when precision
    plus recall is 0.0.  This matches ``zero_division=0``.

Outputs
-------
``results/per_fault_primary_<mode>.csv``
``results/average_primary_<mode>.csv``
``results/within_run_<mode>.csv``
``results/within_run_average_<mode>.csv``
``results/threshold_sensitivity.csv``
``results/paper_reference.csv``
``results/phase5_checks.json``
``results/phase5_metadata.json``

Run::

    .venv\\Scripts\\python.exe .\\src\\phase5_eval.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import environment_info, git_commit  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "results"

MODES = (("IPPT", "L"), ("MPPT", "M"))
SEEDS = (0, 1, 2)
FAULTS = tuple(f"F{i}" for i in range(1, 8))
METRICS = ("tpr", "fpr", "accuracy", "precision", "f1", "auc")
METRIC_LABELS = {
    "tpr": "TPR",
    "fpr": "FPR",
    "accuracy": "Accuracy",
    "precision": "Precision",
    "f1": "F1",
    "auc": "AUC",
}

BOOTSTRAP_RESAMPLES = 200
BOOTSTRAP_SEED = 20261028
CI_PERCENTILES = (2.5, 97.5)
SEED_STD_DDOF = 1
FLAG_RULE = "score > threshold"
PRE_START_S = 0.5
FAULT_START_S = 7.0
FAULT_END_S = 8.0
SENSITIVITY_QUANTILES = (95.0, 97.5, 99.0, 99.5)

# Analysis codes used inside the bootstrap SeedSequence.
ANALYSIS_PRIMARY = 1
ANALYSIS_WITHIN = 2

MODEL_SPECS = (
    {
        "key": "proposed_plain",
        "label": "Proposed (plain)",
        "type": "proposed",
        "score": "plain",
    },
    {
        "key": "proposed_featnorm",
        "label": "Proposed (featnorm)",
        "type": "proposed",
        "score": "featnorm",
    },
    {
        "key": "elliptic_envelope",
        "label": "EllipticEnvelope",
        "type": "baseline",
        "file": "elliptic_envelope",
    },
    {
        "key": "isolation_forest",
        "label": "IsolationForest",
        "type": "baseline",
        "file": "isolation_forest",
    },
    {
        "key": "lof",
        "label": "LocalOutlierFactor",
        "type": "baseline",
        "file": "lof",
    },
    {
        "key": "one_class_svm",
        "label": "OneClassSVM",
        "type": "baseline",
        "file": "one_class_svm",
    },
    {
        "key": "mahalanobis_window_mean",
        "label": "Mahalanobis (window-mean)",
        "type": "baseline",
        "file": "mahalanobis_window_mean",
    },
)
MODEL_BY_KEY = {spec["key"]: spec for spec in MODEL_SPECS}
MODEL_BY_LABEL = {spec["label"]: spec for spec in MODEL_SPECS}

SENSITIVITY_MODEL_KEYS = (
    "proposed_plain",
    "proposed_featnorm",
    "elliptic_envelope",
    "mahalanobis_window_mean",
)

# Published averages from the reference paper. These are external reference
# values transcribed from the paper, not numbers computed by this pipeline.
# They are saved with an explicit protocol warning and must never be compared
# as if they were produced on the same windows.
PAPER_PROTOCOL_NOTE = (
    "reference only, different protocol: the paper scores single 10 kHz "
    "samples; we score 100-sample windows at 1 kHz"
)
PAPER_REFERENCE = (
    # mode, model, paper label, TPR, FPR, Accuracy, Precision, F1, AUC
    ("MPPT", "IsolationForest", "iForest", 0.7046, 0.0073, 0.8486, 0.9890, 0.8369, 0.8486),
    ("MPPT", "LocalOutlierFactor", "LOF", 0.9765, 0.0676, 0.9520, 0.9624, 0.9569, 0.9423),
    ("MPPT", "OneClassSVM", "1SVM", 0.4917, 0.1156, 0.7124, 0.7148, 0.6073, 0.7124),
    ("MPPT", "EllipticEnvelope", "EE", 0.8923, 0.0000, 0.9468, 0.9998, 0.9433, 0.9468),
    ("IPPT", "IsolationForest", "iForest", 0.7624, 0.1137, 0.8161, 0.8999, 0.8053, 0.8161),
    ("IPPT", "LocalOutlierFactor", "LOF", 0.9697, 0.4625, 0.8308, 0.7315, 0.8120, 0.7536),
    ("IPPT", "OneClassSVM", "1SVM", 0.5370, 0.1471, 0.6954, 0.7809, 0.6612, 0.6954),
    ("IPPT", "EllipticEnvelope", "EE", 0.8796, 0.0683, 0.9354, 0.9759, 0.9300, 0.9354),
)

CHECK_RECORDS: list[dict[str, object]] = []
BOOTSTRAP_AUDIT: dict[str, object] = {
    "draws": 0,
    "unequal_class_resamples": 0,
    "resample_count_200": 0,
    "other_resample_count": 0,
    "class_sizes": set(),
}
PRIMARY_COLUMNS: list[str] = []
AVERAGE_COLUMNS: list[str] = []
WITHIN_COLUMNS: list[str] = []
WITHIN_AVERAGE_COLUMNS: list[str] = []
SENSITIVITY_COLUMNS: list[str] = []


def build_column_lists() -> None:
    """Build the explicit column order used by every output table."""
    primary = [
        "model",
        "model_type",
        "mode",
        "fault",
        "seed_count",
        "n_healthy_test",
        "n_fault_windows",
        "file_end_time_s",
        "threshold_mean",
        "threshold_seed_std",
        "healthy_test_false_alarm_rate",
        "healthy_test_false_alarm_rate_seed_std",
        "has_post_injection_data",
        "post_injection_windows",
        "post_injection_status",
        "post_injection_note",
    ]
    for metric in METRICS:
        primary.extend(
            [
                f"{metric}_point",
                f"{metric}_mean",
                f"{metric}_ci_lo",
                f"{metric}_ci_hi",
                f"{metric}_seed_std",
            ]
        )
        for seed in SEEDS:
            primary.extend(
                [
                    f"{metric}_seed{seed}_mean",
                    f"{metric}_seed{seed}_ci_lo",
                    f"{metric}_seed{seed}_ci_hi",
                ]
            )
    PRIMARY_COLUMNS.extend(primary)

    average = [
        "model",
        "model_type",
        "mode",
        "n_faults",
        "faults_included",
        "n_healthy_test",
        "threshold_mean",
        "threshold_seed_std",
        "healthy_test_false_alarm_rate",
        "healthy_test_false_alarm_rate_seed_std",
    ]
    for metric in METRICS:
        average.extend(
            [
                f"{metric}_mean",
                f"{metric}_ci_lo",
                f"{metric}_ci_hi",
                f"{metric}_std_across_faults",
                f"{metric}_min",
                f"{metric}_max",
                f"{metric}_seed_std",
            ]
        )
    average.extend(["excl_f3_n_faults", "excl_f3_faults_included"])
    for metric in METRICS:
        average.extend(
            [
                f"excl_f3_{metric}_mean",
                f"excl_f3_{metric}_ci_lo",
                f"excl_f3_{metric}_ci_hi",
                f"excl_f3_{metric}_std_across_faults",
                f"excl_f3_{metric}_min",
                f"excl_f3_{metric}_max",
                f"excl_f3_{metric}_seed_std",
            ]
        )
    AVERAGE_COLUMNS.extend(average)

    within = [
        "model",
        "model_type",
        "mode",
        "fault",
        "seed_count",
        "n_pre_injection",
        "n_post_injection",
        "threshold_mean",
        "threshold_seed_std",
        "pre_injection_flag_share",
        "pre_injection_flag_share_seed_std",
        "healthy_test_false_alarm_rate",
        "healthy_test_false_alarm_rate_seed_std",
        "auc_point",
        "auc_mean",
        "auc_ci_lo",
        "auc_ci_hi",
        "auc_seed_std",
    ]
    for seed in SEEDS:
        within.extend(
            [
                f"auc_seed{seed}_mean",
                f"auc_seed{seed}_ci_lo",
                f"auc_seed{seed}_ci_hi",
            ]
        )
    WITHIN_COLUMNS.extend(within)

    within_average = [
        "model",
        "model_type",
        "mode",
        "n_faults",
        "faults_included",
        "threshold_mean",
        "threshold_seed_std",
        "pre_injection_flag_share",
        "pre_injection_flag_share_min",
        "pre_injection_flag_share_max",
        "healthy_test_false_alarm_rate",
        "healthy_test_false_alarm_rate_seed_std",
        "auc_mean",
        "auc_ci_lo",
        "auc_ci_hi",
        "auc_std_across_faults",
        "auc_min",
        "auc_max",
        "auc_seed_std",
    ]
    WITHIN_AVERAGE_COLUMNS.extend(within_average)

    sensitivity = [
        "model",
        "model_type",
        "mode",
        "quantile",
        "seed_count",
        "n_faults",
        "n_faults_excl_f3",
        "threshold",
        "threshold_seed_std",
        "healthy_test_false_alarm_rate",
        "healthy_test_false_alarm_rate_seed_std",
        "average_tpr",
        "average_tpr_seed_std",
        "average_tpr_excl_f3",
        "average_tpr_excl_f3_seed_std",
    ]
    for seed in SEEDS:
        sensitivity.extend(
            [
                f"threshold_seed{seed}",
                f"healthy_test_false_alarm_rate_seed{seed}",
                f"average_tpr_seed{seed}",
            ]
        )
    SENSITIVITY_COLUMNS.extend(sensitivity)


def check(name: str, passed: bool, detail: str) -> bool:
    """Record and print one named assertion."""
    passed = bool(passed)
    CHECK_RECORDS.append({"name": name, "passed": passed, "detail": detail})
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    return passed


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def mode_index(mode: str) -> int:
    return [item[0] for item in MODES].index(mode)


def suffix_for_mode(mode: str) -> str:
    for candidate_mode, suffix in MODES:
        if candidate_mode == mode:
            return suffix
    raise ValueError(f"Unknown mode: {mode}")


def fault_stems(mode: str) -> tuple[str, ...]:
    suffix = suffix_for_mode(mode)
    return tuple(f"{fault}{suffix}" for fault in FAULTS)


def archive_paths(spec: dict[str, object], mode: str) -> list[Path]:
    if spec["type"] == "proposed":
        return [
            RESULTS_DIR
            / f"scores_proposed_{spec['score']}_{mode}_seed{seed}.npz"
            for seed in SEEDS
        ]
    return [RESULTS_DIR / f"scores_baseline_{spec['file']}_{mode}.npz"]


def expected_parts(mode: str) -> tuple[str, ...]:
    suffix = suffix_for_mode(mode)
    healthy = f"F0{suffix}"
    return (
        f"{healthy}_train",
        f"{healthy}_val",
        f"{healthy}_test",
        *fault_stems(mode),
    )


@lru_cache(maxsize=None)
def load_archive(path_text: str) -> dict[str, object]:
    """Load and validate one archive; extra keys such as feature_errors are allowed."""
    path = Path(path_text)
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase 5 archive: {path}")

    with np.load(path, allow_pickle=False) as handle:
        keys = set(handle.files)
        required = {"scores", "start_time", "end_time", "threshold", "name"}
        missing = required - keys
        if missing:
            raise ValueError(f"{path.name}: missing keys {sorted(missing)}")
        scores = np.asarray(handle["scores"], dtype=np.float64).reshape(-1)
        start = np.asarray(handle["start_time"], dtype=np.float64).reshape(-1)
        end = np.asarray(handle["end_time"], dtype=np.float64).reshape(-1)
        threshold = float(np.asarray(handle["threshold"]).reshape(()))
        names = np.asarray(handle["name"]).astype(str).reshape(-1)

    if not (scores.size == start.size == end.size == names.size):
        raise ValueError(f"{path.name}: archive arrays have different lengths")
    if scores.size == 0:
        raise ValueError(f"{path.name}: archive is empty")
    if not np.isfinite(scores).all():
        raise ValueError(f"{path.name}: non-finite scores")
    if not np.isfinite(start).all() or not np.isfinite(end).all():
        raise ValueError(f"{path.name}: non-finite window times")
    if np.any(end < start):
        raise ValueError(f"{path.name}: a window ends before it starts")
    if not np.isfinite(threshold):
        raise ValueError(f"{path.name}: non-finite threshold")

    return {
        "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "scores": scores,
        "start_time": start,
        "end_time": end,
        "threshold": threshold,
        "name": names,
        "sha256": sha256_of(path),
    }


def part_mask(archive: dict[str, object], part: str) -> np.ndarray:
    return archive["name"] == part


def part_scores(archive: dict[str, object], part: str) -> np.ndarray:
    mask = part_mask(archive, part)
    if not mask.any():
        raise ValueError(f"{archive['path']}: part {part!r} is absent")
    return archive["scores"][mask]


def part_times(archive: dict[str, object], part: str) -> tuple[np.ndarray, np.ndarray]:
    mask = part_mask(archive, part)
    return archive["start_time"][mask], archive["end_time"][mask]


def metric_values(
    negative_scores: np.ndarray,
    positive_scores: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """TPR, FPR, accuracy, precision, F1 and AUC for negative (0) vs positive (1)."""
    negative_scores = np.asarray(negative_scores, dtype=np.float64).reshape(-1)
    positive_scores = np.asarray(positive_scores, dtype=np.float64).reshape(-1)
    if negative_scores.size == 0 or positive_scores.size == 0:
        raise ValueError("Both classes must contain at least one window")

    negative_flag = negative_scores > threshold
    positive_flag = positive_scores > threshold
    tp = float(np.count_nonzero(positive_flag))
    fn = float(positive_scores.size - tp)
    fp = float(np.count_nonzero(negative_flag))
    tn = float(negative_scores.size - fp)

    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (
        2.0 * precision * tpr / (precision + tpr)
        if (precision + tpr) > 0
        else 0.0
    )

    labels = np.concatenate(
        (np.zeros(negative_scores.size, dtype=np.int8), np.ones(positive_scores.size, dtype=np.int8))
    )
    scores = np.concatenate((negative_scores, positive_scores))
    auc = float(roc_auc_score(labels, scores))

    return {
        "tpr": float(tpr),
        "fpr": float(fpr),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "f1": float(f1),
        "auc": auc,
    }


def bootstrap_metrics(
    negative_scores: np.ndarray,
    positive_scores: np.ndarray,
    threshold: float,
    index_negative: np.ndarray,
    index_positive: np.ndarray,
) -> dict[str, np.ndarray]:
    """Metric distributions over paired equal-size bootstrap resamples."""
    if index_negative.shape != index_positive.shape:
        raise ValueError("Bootstrap resamples must have equal class sizes")
    n_resamples = index_negative.shape[0]
    result = {metric: np.empty(n_resamples, dtype=np.float64) for metric in METRICS}

    for resample in range(n_resamples):
        neg = negative_scores[index_negative[resample]]
        pos = positive_scores[index_positive[resample]]
        if neg.size != pos.size:
            BOOTSTRAP_AUDIT["unequal_class_resamples"] += 1
        BOOTSTRAP_AUDIT["draws"] += 1
        if n_resamples == BOOTSTRAP_RESAMPLES:
            BOOTSTRAP_AUDIT["resample_count_200"] += 1
        else:
            BOOTSTRAP_AUDIT["other_resample_count"] += 1
        BOOTSTRAP_AUDIT["class_sizes"].add(int(neg.size))
        values = metric_values(neg, pos, threshold)
        for metric in METRICS:
            result[metric][resample] = values[metric]
    return result


def bootstrap_auc(
    negative_scores: np.ndarray,
    positive_scores: np.ndarray,
    index_negative: np.ndarray,
    index_positive: np.ndarray,
) -> np.ndarray:
    if index_negative.shape != index_positive.shape:
        raise ValueError("Bootstrap resamples must have equal class sizes")
    result = np.empty(index_negative.shape[0], dtype=np.float64)
    for resample in range(index_negative.shape[0]):
        neg = negative_scores[index_negative[resample]]
        pos = positive_scores[index_positive[resample]]
        if neg.size != pos.size:
            BOOTSTRAP_AUDIT["unequal_class_resamples"] += 1
        BOOTSTRAP_AUDIT["draws"] += 1
        if index_negative.shape[0] == BOOTSTRAP_RESAMPLES:
            BOOTSTRAP_AUDIT["resample_count_200"] += 1
        else:
            BOOTSTRAP_AUDIT["other_resample_count"] += 1
        BOOTSTRAP_AUDIT["class_sizes"].add(int(neg.size))
        labels = np.concatenate(
            (np.zeros(neg.size, dtype=np.int8), np.ones(pos.size, dtype=np.int8))
        )
        result[resample] = roc_auc_score(labels, np.concatenate((neg, pos)))
    return result


def draw_bootstrap_indices(
    n_negative: int,
    n_positive: int,
    n_draw: int,
    analysis: int,
    mode: str,
    fault_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Equal-size draws from each class, reproducible from the fixed base seed."""
    sequence = np.random.SeedSequence(
        [BOOTSTRAP_SEED, int(analysis), mode_index(mode), int(fault_index)]
    )
    rng = np.random.default_rng(sequence)
    index_negative = rng.integers(
        0, n_negative, size=(BOOTSTRAP_RESAMPLES, n_draw), dtype=np.int64
    )
    index_positive = rng.integers(
        0, n_positive, size=(BOOTSTRAP_RESAMPLES, n_draw), dtype=np.int64
    )
    return index_negative, index_positive


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(values, CI_PERCENTILES)
    return float(low), float(high)


def aggregate_seed_distributions(
    distributions: list[dict[str, np.ndarray]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Average per-seed bootstrap distributions resample by resample.

    Returns the aggregated distributions, per-seed means, per-seed low bounds,
    and per-seed high bounds.
    """
    combined: dict[str, np.ndarray] = {}
    seed_means: dict[str, np.ndarray] = {}
    seed_low: dict[str, np.ndarray] = {}
    seed_high: dict[str, np.ndarray] = {}
    for metric in METRICS:
        stacked = np.stack([item[metric] for item in distributions], axis=0)
        combined[metric] = stacked.mean(axis=0)
        seed_means[metric] = np.array([float(item[metric].mean()) for item in distributions])
        bounds = [percentile_interval(item[metric]) for item in distributions]
        seed_low[metric] = np.array([bound[0] for bound in bounds])
        seed_high[metric] = np.array([bound[1] for bound in bounds])
    return combined, seed_means, seed_low, seed_high


def seed_std_of_means(seed_means: np.ndarray) -> float:
    if seed_means.size < 2:
        return float("nan")
    return float(np.std(seed_means, ddof=SEED_STD_DDOF))


def post_injection_masks(
    start_time: np.ndarray, end_time: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    pre = (end_time < FAULT_START_S) & (start_time >= PRE_START_S)
    post = start_time >= FAULT_END_S
    return pre, post


def aux_values(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=SEED_STD_DDOF))
        if values.size > 1
        else float("nan"),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def evaluate_primary() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Compute per-fault and averaged primary whole-file tables."""
    per_fault_frames: dict[str, pd.DataFrame] = {}
    average_frames: dict[str, pd.DataFrame] = {}
    distributions: dict[tuple[str, str], dict[str, dict[str, np.ndarray]]] = {}

    for mode, suffix in MODES:
        healthy_part = f"F0{suffix}_test"
        faults = fault_stems(mode)
        rows: list[dict[str, object]] = []
        print(f"\nPrimary whole-file evaluation: {mode}")
        for spec in MODEL_SPECS:
            paths = archive_paths(spec, mode)
            archives = [load_archive(str(path)) for path in paths]
            reference = archives[0]
            expected = set(expected_parts(mode))
            if set(reference["name"].tolist()) != expected:
                raise ValueError(
                    f"{reference['path']}: unexpected part names "
                    f"{sorted(set(reference['name'].tolist()) - expected)}"
                )
            test_scores_by_seed = [part_scores(archive, healthy_part) for archive in archives]
            n_healthy = int(test_scores_by_seed[0].size)
            thresholds = np.array([archive["threshold"] for archive in archives])
            test_far_seed = np.array(
                [
                    float(np.mean(scores > threshold))
                    for scores, threshold in zip(test_scores_by_seed, thresholds)
                ]
            )
            distributions[(spec["key"], mode)] = {}

            for fault_index, fault in enumerate(faults):
                fault_scores_by_seed = [
                    part_scores(archive, fault) for archive in archives
                ]
                n_fault = int(fault_scores_by_seed[0].size)
                index_negative, index_positive = draw_bootstrap_indices(
                    n_healthy,
                    n_fault,
                    n_healthy,
                    ANALYSIS_PRIMARY,
                    mode,
                    fault_index,
                )

                point_by_seed: list[dict[str, float]] = []
                distribution_by_seed: list[dict[str, np.ndarray]] = []
                for scores, threshold in zip(fault_scores_by_seed, thresholds):
                    point_by_seed.append(
                        metric_values(test_scores_by_seed[len(point_by_seed)], scores, float(threshold))
                    )
                    distribution_by_seed.append(
                        bootstrap_metrics(
                            test_scores_by_seed[len(distribution_by_seed)],
                            scores,
                            float(threshold),
                            index_negative,
                            index_positive,
                        )
                    )

                combined, seed_means, seed_low, seed_high = aggregate_seed_distributions(
                    distribution_by_seed
                )
                distributions[(spec["key"], mode)][fault] = combined

                start_time, end_time = part_times(reference, fault)
                pre_mask, post_mask = post_injection_masks(start_time, end_time)
                n_post = int(np.count_nonzero(post_mask))
                has_post = n_post > 0
                file_end = float(np.max(end_time))
                if fault == "F3M":
                    if has_post:
                        raise RuntimeError(f"{fault}: expected no post-injection windows")
                    status = "no_post_injection_data"
                    note = (
                        f"no post-injection data: file ends at {file_end:.3f} s, "
                        f"before the 7.0-8.0 s injection interval"
                    )
                else:
                    if not has_post:
                        raise RuntimeError(f"{fault}: expected post-injection windows")
                    status = "post_injection_data_available"
                    note = ""

                row: dict[str, object] = {
                    "model": spec["label"],
                    "model_type": spec["type"],
                    "mode": mode,
                    "fault": fault,
                    "seed_count": len(archives),
                    "n_healthy_test": n_healthy,
                    "n_fault_windows": n_fault,
                    "file_end_time_s": file_end,
                    "threshold_mean": float(np.mean(thresholds)),
                    "threshold_seed_std": aux_values(thresholds)["std"],
                    "healthy_test_false_alarm_rate": float(np.mean(test_far_seed)),
                    "healthy_test_false_alarm_rate_seed_std": aux_values(test_far_seed)["std"],
                    "has_post_injection_data": bool(has_post),
                    "post_injection_windows": n_post,
                    "post_injection_status": status,
                    "post_injection_note": note,
                }
                for metric in METRICS:
                    row[f"{metric}_point"] = float(
                        np.mean([point[metric] for point in point_by_seed])
                    )
                    row[f"{metric}_mean"] = float(combined[metric].mean())
                    low, high = percentile_interval(combined[metric])
                    row[f"{metric}_ci_lo"] = low
                    row[f"{metric}_ci_hi"] = high
                    row[f"{metric}_seed_std"] = seed_std_of_means(seed_means[metric])
                    if spec["type"] == "proposed":
                        for seed_position, seed in enumerate(SEEDS):
                            row[f"{metric}_seed{seed}_mean"] = float(
                                seed_means[metric][seed_position]
                            )
                            row[f"{metric}_seed{seed}_ci_lo"] = float(
                                seed_low[metric][seed_position]
                            )
                            row[f"{metric}_seed{seed}_ci_hi"] = float(
                                seed_high[metric][seed_position]
                            )
                rows.append(row)
            print(f"  {spec['label']}: done")

        frame = pd.DataFrame(rows).reindex(columns=PRIMARY_COLUMNS)
        per_fault_frames[mode] = frame
        average_frames[mode] = build_primary_average(frame, distributions, mode)
    return per_fault_frames, average_frames


def build_primary_average(
    per_fault: pd.DataFrame,
    distributions: dict[tuple[str, str], dict[str, dict[str, np.ndarray]]],
    mode: str,
) -> pd.DataFrame:
    suffix = suffix_for_mode(mode)
    all_faults = list(fault_stems(mode))
    excl_f3 = [fault for fault in all_faults if fault != f"F3{suffix}"]
    rows: list[dict[str, object]] = []

    for spec in MODEL_SPECS:
        subset = per_fault[per_fault["model"] == spec["label"]]
        if len(subset) != len(all_faults):
            raise RuntimeError(f"{spec['label']}/{mode}: expected 7 fault rows")

        row: dict[str, object] = {
            "model": spec["label"],
            "model_type": spec["type"],
            "mode": mode,
            "n_faults": len(all_faults),
            "faults_included": ",".join(all_faults),
            "n_healthy_test": int(subset.iloc[0]["n_healthy_test"]),
            "threshold_mean": float(subset.iloc[0]["threshold_mean"]),
            "threshold_seed_std": float(subset.iloc[0]["threshold_seed_std"]),
            "healthy_test_false_alarm_rate": float(
                subset.iloc[0]["healthy_test_false_alarm_rate"]
            ),
            "healthy_test_false_alarm_rate_seed_std": float(
                subset.iloc[0]["healthy_test_false_alarm_rate_seed_std"]
            ),
        }

        for prefix, faults in (("", all_faults), ("excl_f3_", excl_f3)):
            if prefix:
                row["excl_f3_n_faults"] = len(faults)
                row["excl_f3_faults_included"] = ",".join(faults)
            fault_distributions = [
                distributions[(spec["key"], mode)][fault] for fault in faults
            ]
            fault_rows = subset[subset["fault"].isin(faults)].set_index("fault")
            for metric in METRICS:
                stacked = np.stack(
                    [item[metric] for item in fault_distributions], axis=0
                )
                averaged = stacked.mean(axis=0)
                per_fault_means = np.array(
                    [float(fault_rows.loc[fault, f"{metric}_mean"]) for fault in faults]
                )
                row[f"{prefix}{metric}_mean"] = float(averaged.mean())
                low, high = percentile_interval(averaged)
                row[f"{prefix}{metric}_ci_lo"] = low
                row[f"{prefix}{metric}_ci_hi"] = high
                row[f"{prefix}{metric}_std_across_faults"] = float(
                    np.std(per_fault_means, ddof=0)
                )
                row[f"{prefix}{metric}_min"] = float(np.min(per_fault_means))
                row[f"{prefix}{metric}_max"] = float(np.max(per_fault_means))
                if spec["type"] == "proposed":
                    per_seed_average = [
                        float(
                            np.mean(
                                [
                                    float(fault_rows.loc[fault, f"{metric}_seed{seed}_mean"])
                                    for fault in faults
                                ]
                            )
                        )
                        for seed in SEEDS
                    ]
                    row[f"{prefix}{metric}_seed_std"] = seed_std_of_means(
                        np.array(per_seed_average)
                    )
        rows.append(row)

    return pd.DataFrame(rows).reindex(columns=AVERAGE_COLUMNS)


def evaluate_within_run() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Compute the secondary within-run change test."""
    per_fault_frames: dict[str, pd.DataFrame] = {}
    average_frames: dict[str, pd.DataFrame] = {}
    distributions: dict[tuple[str, str], dict[str, np.ndarray]] = {}

    for mode, suffix in MODES:
        faults = fault_stems(mode)
        healthy_part = f"F0{suffix}_test"
        rows: list[dict[str, object]] = []
        print(f"\nSecondary within-run evaluation: {mode}")
        for spec in MODEL_SPECS:
            archives = [
                load_archive(str(path)) for path in archive_paths(spec, mode)
            ]
            reference = archives[0]
            thresholds = np.array([archive["threshold"] for archive in archives])
            test_far_seed = np.array(
                [
                    float(
                        np.mean(
                            part_scores(archive, healthy_part) > archive["threshold"]
                        )
                    )
                    for archive in archives
                ]
            )
            distributions[(spec["key"], mode)] = {}

            for fault_index, fault in enumerate(faults):
                start_time, end_time = part_times(reference, fault)
                pre_mask, post_mask = post_injection_masks(start_time, end_time)
                n_pre = int(np.count_nonzero(pre_mask))
                n_post = int(np.count_nonzero(post_mask))
                if n_post == 0:
                    if fault != "F3M":
                        raise RuntimeError(f"{fault}: no post-injection windows")
                    continue
                if n_pre == 0:
                    raise RuntimeError(f"{fault}: no pre-injection windows")

                n_draw = min(n_pre, n_post)
                index_pre, index_post = draw_bootstrap_indices(
                    n_pre,
                    n_post,
                    n_draw,
                    ANALYSIS_WITHIN,
                    mode,
                    fault_index,
                )

                point_auc_by_seed: list[float] = []
                distribution_by_seed: list[np.ndarray] = []
                pre_flag_by_seed: list[float] = []
                for archive in archives:
                    scores = part_scores(archive, fault)
                    pre_scores = scores[pre_mask]
                    post_scores = scores[post_mask]
                    labels = np.concatenate(
                        (
                            np.zeros(pre_scores.size, dtype=np.int8),
                            np.ones(post_scores.size, dtype=np.int8),
                        )
                    )
                    point_auc_by_seed.append(
                        float(
                            roc_auc_score(
                                labels, np.concatenate((pre_scores, post_scores))
                            )
                        )
                    )
                    distribution_by_seed.append(
                        bootstrap_auc(
                            pre_scores, post_scores, index_pre, index_post
                        )
                    )
                    pre_flag_by_seed.append(
                        float(np.mean(pre_scores > archive["threshold"]))
                    )

                stacked = np.stack(distribution_by_seed, axis=0)
                combined = stacked.mean(axis=0)
                distributions[(spec["key"], mode)][fault] = combined
                seed_means = np.array([float(item.mean()) for item in distribution_by_seed])
                bounds = [percentile_interval(item) for item in distribution_by_seed]

                row: dict[str, object] = {
                    "model": spec["label"],
                    "model_type": spec["type"],
                    "mode": mode,
                    "fault": fault,
                    "seed_count": len(archives),
                    "n_pre_injection": n_pre,
                    "n_post_injection": n_post,
                    "threshold_mean": float(np.mean(thresholds)),
                    "threshold_seed_std": aux_values(thresholds)["std"],
                    "pre_injection_flag_share": float(np.mean(pre_flag_by_seed)),
                    "pre_injection_flag_share_seed_std": aux_values(
                        np.array(pre_flag_by_seed)
                    )["std"],
                    "healthy_test_false_alarm_rate": float(np.mean(test_far_seed)),
                    "healthy_test_false_alarm_rate_seed_std": aux_values(
                        test_far_seed
                    )["std"],
                    "auc_point": float(np.mean(point_auc_by_seed)),
                    "auc_mean": float(combined.mean()),
                    "auc_ci_lo": percentile_interval(combined)[0],
                    "auc_ci_hi": percentile_interval(combined)[1],
                    "auc_seed_std": seed_std_of_means(seed_means),
                }
                if spec["type"] == "proposed":
                    for seed_position, seed in enumerate(SEEDS):
                        row[f"auc_seed{seed}_mean"] = float(seed_means[seed_position])
                        row[f"auc_seed{seed}_ci_lo"] = float(bounds[seed_position][0])
                        row[f"auc_seed{seed}_ci_hi"] = float(bounds[seed_position][1])
                rows.append(row)
            print(f"  {spec['label']}: done")

        frame = pd.DataFrame(rows).reindex(columns=WITHIN_COLUMNS)
        per_fault_frames[mode] = frame
        average_frames[mode] = build_within_average(frame, distributions, mode)
    return per_fault_frames, average_frames


def build_within_average(
    per_fault: pd.DataFrame,
    distributions: dict[tuple[str, str], dict[str, np.ndarray]],
    mode: str,
) -> pd.DataFrame:
    faults = [fault for fault in fault_stems(mode) if fault != "F3M"]
    rows: list[dict[str, object]] = []

    for spec in MODEL_SPECS:
        subset = per_fault[per_fault["model"] == spec["label"]].set_index("fault")
        if sorted(subset.index.tolist()) != sorted(faults):
            raise RuntimeError(
                f"{spec['label']}/{mode}: within-run faults differ from expectation"
            )
        stacked = np.stack(
            [distributions[(spec["key"], mode)][fault] for fault in faults], axis=0
        )
        averaged = stacked.mean(axis=0)
        per_fault_auc = np.array(
            [float(subset.loc[fault, "auc_mean"]) for fault in faults]
        )
        pre_rates = np.array(
            [float(subset.loc[fault, "pre_injection_flag_share"]) for fault in faults]
        )
        row: dict[str, object] = {
            "model": spec["label"],
            "model_type": spec["type"],
            "mode": mode,
            "n_faults": len(faults),
            "faults_included": ",".join(faults),
            "threshold_mean": float(subset.iloc[0]["threshold_mean"]),
            "threshold_seed_std": float(subset.iloc[0]["threshold_seed_std"]),
            "pre_injection_flag_share": float(np.mean(pre_rates)),
            "pre_injection_flag_share_min": float(np.min(pre_rates)),
            "pre_injection_flag_share_max": float(np.max(pre_rates)),
            "healthy_test_false_alarm_rate": float(
                subset.iloc[0]["healthy_test_false_alarm_rate"]
            ),
            "healthy_test_false_alarm_rate_seed_std": float(
                subset.iloc[0]["healthy_test_false_alarm_rate_seed_std"]
            ),
            "auc_mean": float(averaged.mean()),
            "auc_ci_lo": percentile_interval(averaged)[0],
            "auc_ci_hi": percentile_interval(averaged)[1],
            "auc_std_across_faults": float(np.std(per_fault_auc, ddof=0)),
            "auc_min": float(np.min(per_fault_auc)),
            "auc_max": float(np.max(per_fault_auc)),
        }
        if spec["type"] == "proposed":
            per_seed_average = [
                float(
                    np.mean(
                        [
                            float(subset.loc[fault, f"auc_seed{seed}_mean"])
                            for fault in faults
                        ]
                    )
                )
                for seed in SEEDS
            ]
            row["auc_seed_std"] = seed_std_of_means(np.array(per_seed_average))
        rows.append(row)
    return pd.DataFrame(rows).reindex(columns=WITHIN_AVERAGE_COLUMNS)


def evaluate_threshold_sensitivity() -> pd.DataFrame:
    """Recompute thresholds from saved healthy validation scores."""
    rows: list[dict[str, object]] = []
    for model_key in SENSITIVITY_MODEL_KEYS:
        spec = MODEL_BY_KEY[model_key]
        for mode, suffix in MODES:
            archives = [load_archive(str(path)) for path in archive_paths(spec, mode)]
            val_scores = [part_scores(archive, f"F0{suffix}_val") for archive in archives]
            test_scores = [part_scores(archive, f"F0{suffix}_test") for archive in archives]
            faults = fault_stems(mode)
            fault_scores = [
                [part_scores(archive, fault) for archive in archives] for fault in faults
            ]
            for quantile in SENSITIVITY_QUANTILES:
                thresholds = np.array(
                    [float(np.percentile(scores, quantile)) for scores in val_scores]
                )
                far = np.array(
                    [
                        float(np.mean(test > threshold))
                        for test, threshold in zip(test_scores, thresholds)
                    ]
                )
                tpr_by_fault = np.array(
                    [
                        [
                            float(
                                np.mean(
                                    fault_scores[fault_position][seed_position]
                                    > thresholds[seed_position]
                                )
                            )
                            for seed_position in range(len(archives))
                        ]
                        for fault_position in range(len(faults))
                    ]
                )
                average_tpr_by_seed = tpr_by_fault.mean(axis=0)
                excl_positions = [
                    position
                    for position, fault in enumerate(faults)
                    if fault != f"F3{suffix}"
                ]
                average_tpr_excl_by_seed = tpr_by_fault[excl_positions].mean(axis=0)

                row: dict[str, object] = {
                    "model": spec["label"],
                    "model_type": spec["type"],
                    "mode": mode,
                    "quantile": quantile,
                    "seed_count": len(archives),
                    "n_faults": len(faults),
                    "n_faults_excl_f3": len(excl_positions),
                    "threshold": float(np.mean(thresholds)),
                    "threshold_seed_std": aux_values(thresholds)["std"],
                    "healthy_test_false_alarm_rate": float(np.mean(far)),
                    "healthy_test_false_alarm_rate_seed_std": aux_values(far)["std"],
                    "average_tpr": float(np.mean(average_tpr_by_seed)),
                    "average_tpr_seed_std": aux_values(average_tpr_by_seed)["std"],
                    "average_tpr_excl_f3": float(
                        np.mean(average_tpr_excl_by_seed)
                    ),
                    "average_tpr_excl_f3_seed_std": aux_values(
                        average_tpr_excl_by_seed
                    )["std"],
                }
                if spec["type"] == "proposed":
                    for seed_position, seed in enumerate(SEEDS):
                        row[f"threshold_seed{seed}"] = float(
                            thresholds[seed_position]
                        )
                        row[f"healthy_test_false_alarm_rate_seed{seed}"] = float(
                            far[seed_position]
                        )
                        row[f"average_tpr_seed{seed}"] = float(
                            average_tpr_by_seed[seed_position]
                        )
                rows.append(row)
    return pd.DataFrame(rows).reindex(columns=SENSITIVITY_COLUMNS)


def paper_reference_frame() -> pd.DataFrame:
    rows = []
    for mode, model, paper_label, tpr, fpr, accuracy, precision, f1, auc in PAPER_REFERENCE:
        rows.append(
            {
                "mode": mode,
                "model": model,
                "paper_label": paper_label,
                "tpr": tpr,
                "fpr": fpr,
                "accuracy": accuracy,
                "precision": precision,
                "f1": f1,
                "auc": auc,
                "source": "reference paper published averages",
                "protocol_note": PAPER_PROTOCOL_NOTE,
            }
        )
    return pd.DataFrame(rows)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, float_format="%.17g", lineterminator="\n")


def close_enough(left: object, right: object, rtol: float = 1e-9, atol: float = 1e-12) -> bool:
    left_missing = pd.isna(left)
    right_missing = pd.isna(right)
    if left_missing or right_missing:
        return bool(left_missing and right_missing)
    return bool(np.isclose(float(left), float(right), rtol=rtol, atol=atol))


def check_archives() -> None:
    """Archive contract, threshold provenance and validation/test separation."""
    paths = []
    for spec in MODEL_SPECS:
        for mode, _ in MODES:
            paths.extend(archive_paths(spec, mode))
    if len(paths) != 22:
        check("archive_count", False, f"found {len(paths)} paths, expected 22")
        return
    check("archive_count", True, f"{len(paths)} archives: 10 baseline + 12 proposed")

    missing = [str(path) for path in paths if not path.exists()]
    check("archives_exist", not missing, f"{len(paths) - len(missing)}/{len(paths)} present")
    if missing:
        return

    finite = 0
    complete = 0
    threshold_diffs: list[float] = []
    disjoint = 0
    val_counts: list[int] = []
    test_counts: list[int] = []
    for spec in MODEL_SPECS:
        for mode, suffix in MODES:
            expected = set(expected_parts(mode))
            for path in archive_paths(spec, mode):
                archive = load_archive(str(path))
                if np.isfinite(archive["scores"]).all():
                    finite += 1
                names = set(archive["name"].tolist())
                if names == expected:
                    complete += 1
                val_scores = part_scores(archive, f"F0{suffix}_val")
                threshold_diffs.append(abs(float(np.percentile(val_scores, 99.0)) - archive["threshold"]))
                val_start, val_end = part_times(archive, f"F0{suffix}_val")
                test_start, test_end = part_times(archive, f"F0{suffix}_test")
                val_counts.append(val_start.size)
                test_counts.append(test_start.size)
                if float(np.max(val_end)) < float(np.min(test_start)):
                    disjoint += 1

    check("archive_scores_finite", finite == len(paths), f"{finite}/{len(paths)} archives all finite")
    check(
        "archive_part_names_complete",
        complete == len(paths),
        f"{complete}/{len(paths)} archives contain exactly the expected 10 parts",
    )
    check(
        "threshold_equals_healthy_validation_p99",
        max(threshold_diffs) == 0.0,
        f"{len(threshold_diffs)} thresholds; max |stored - p99(F0_val)| = {max(threshold_diffs):.3g}",
    )
    check(
        "validation_precedes_test_without_overlap",
        disjoint == len(paths),
        f"{disjoint}/{len(paths)} archives: max validation end < min test start",
    )


def check_recomputation(
    per_fault_primary: dict[str, pd.DataFrame],
    within_run: dict[str, pd.DataFrame],
) -> None:
    """Recompute every saved per-fault metric and bootstrap summary from archives."""
    primary_values = 0
    primary_problems: list[str] = []
    within_values = 0
    within_problems: list[str] = []

    for mode, suffix in MODES:
        faults = fault_stems(mode)
        healthy_part = f"F0{suffix}_test"
        for spec in MODEL_SPECS:
            archives = [load_archive(str(path)) for path in archive_paths(spec, mode)]
            reference = archives[0]
            thresholds = np.array([archive["threshold"] for archive in archives])
            test_scores_by_seed = [part_scores(archive, healthy_part) for archive in archives]
            n_healthy = int(test_scores_by_seed[0].size)

            for fault_index, fault in enumerate(faults):
                index_negative, index_positive = draw_bootstrap_indices(
                    n_healthy,
                    int(part_scores(reference, fault).size),
                    n_healthy,
                    ANALYSIS_PRIMARY,
                    mode,
                    fault_index,
                )
                fault_scores_by_seed = [
                    part_scores(archive, fault) for archive in archives
                ]
                point_by_seed = [
                    metric_values(test_scores_by_seed[position], fault_scores_by_seed[position], float(thresholds[position]))
                    for position in range(len(archives))
                ]
                distribution_by_seed = [
                    bootstrap_metrics(
                        test_scores_by_seed[position],
                        fault_scores_by_seed[position],
                        float(thresholds[position]),
                        index_negative,
                        index_positive,
                    )
                    for position in range(len(archives))
                ]
                combined, seed_means, seed_low, seed_high = aggregate_seed_distributions(
                    distribution_by_seed
                )

                row = per_fault_primary[mode]
                row = row[(row["model"] == spec["label"]) & (row["fault"] == fault)]
                if len(row) != 1:
                    primary_problems.append(f"{spec['label']}/{mode}/{fault}: row count {len(row)}")
                    continue
                row = row.iloc[0]
                for metric in METRICS:
                    comparisons = [
                        (f"{metric}_point", np.mean([item[metric] for item in point_by_seed])),
                        (f"{metric}_mean", combined[metric].mean()),
                        (f"{metric}_ci_lo", percentile_interval(combined[metric])[0]),
                        (f"{metric}_ci_hi", percentile_interval(combined[metric])[1]),
                    ]
                    if spec["type"] == "proposed":
                        comparisons.append((f"{metric}_seed_std", seed_std_of_means(seed_means[metric])))
                        for seed_position, seed in enumerate(SEEDS):
                            comparisons.extend(
                                [
                                    (f"{metric}_seed{seed}_mean", seed_means[metric][seed_position]),
                                    (f"{metric}_seed{seed}_ci_lo", seed_low[metric][seed_position]),
                                    (f"{metric}_seed{seed}_ci_hi", seed_high[metric][seed_position]),
                                ]
                            )
                    for column, expected in comparisons:
                        primary_values += 1
                        if not close_enough(row[column], expected):
                            primary_problems.append(
                                f"{spec['label']}/{mode}/{fault}/{column}: "
                                f"saved={row[column]} recomputed={expected}"
                            )

                # Secondary recomputation for this fault, if it has post data.
                start_time, end_time = part_times(reference, fault)
                pre_mask, post_mask = post_injection_masks(start_time, end_time)
                if not post_mask.any():
                    continue
                n_pre = int(np.count_nonzero(pre_mask))
                n_post = int(np.count_nonzero(post_mask))
                index_pre, index_post = draw_bootstrap_indices(
                    n_pre,
                    n_post,
                    min(n_pre, n_post),
                    ANALYSIS_WITHIN,
                    mode,
                    fault_index,
                )
                point_auc_by_seed = []
                distribution_by_seed = []
                pre_flag_by_seed = []
                for archive in archives:
                    scores = part_scores(archive, fault)
                    pre_scores = scores[pre_mask]
                    post_scores = scores[post_mask]
                    labels = np.concatenate(
                        (np.zeros(pre_scores.size, dtype=np.int8), np.ones(post_scores.size, dtype=np.int8))
                    )
                    point_auc_by_seed.append(
                        float(roc_auc_score(labels, np.concatenate((pre_scores, post_scores))))
                    )
                    distribution_by_seed.append(
                        bootstrap_auc(pre_scores, post_scores, index_pre, index_post)
                    )
                    pre_flag_by_seed.append(float(np.mean(pre_scores > archive["threshold"])))
                stacked = np.stack(distribution_by_seed, axis=0)
                auc_combined = stacked.mean(axis=0)
                auc_seed_means = np.array([float(item.mean()) for item in distribution_by_seed])
                auc_bounds = [percentile_interval(item) for item in distribution_by_seed]

                row = within_run[mode]
                row = row[(row["model"] == spec["label"]) & (row["fault"] == fault)]
                if len(row) != 1:
                    within_problems.append(f"{spec['label']}/{mode}/{fault}: within row count {len(row)}")
                    continue
                row = row.iloc[0]
                comparisons = [
                    ("auc_point", np.mean(point_auc_by_seed)),
                    ("auc_mean", float(auc_combined.mean())),
                    ("auc_ci_lo", percentile_interval(auc_combined)[0]),
                    ("auc_ci_hi", percentile_interval(auc_combined)[1]),
                    ("pre_injection_flag_share", float(np.mean(pre_flag_by_seed))),
                ]
                if spec["type"] == "proposed":
                    comparisons.append(("auc_seed_std", seed_std_of_means(auc_seed_means)))
                    for seed_position, seed in enumerate(SEEDS):
                        comparisons.extend(
                            [
                                (f"auc_seed{seed}_mean", auc_seed_means[seed_position]),
                                (f"auc_seed{seed}_ci_lo", auc_bounds[seed_position][0]),
                                (f"auc_seed{seed}_ci_hi", auc_bounds[seed_position][1]),
                            ]
                        )
                for column, expected in comparisons:
                    within_values += 1
                    if not close_enough(row[column], expected):
                        within_problems.append(
                            f"{spec['label']}/{mode}/{fault}/{column}: "
                            f"saved={row[column]} recomputed={expected}"
                        )

    check(
        "primary_metrics_recomputed_from_archives",
        not primary_problems,
        f"{primary_values} values recomputed; {len(primary_problems)} mismatches",
    )
    if primary_problems:
        print("  first primary mismatches: " + "; ".join(primary_problems[:3]))
    check(
        "within_run_metrics_recomputed_from_archives",
        not within_problems,
        f"{within_values} values recomputed; {len(within_problems)} mismatches",
    )
    if within_problems:
        print("  first within-run mismatches: " + "; ".join(within_problems[:3]))


def check_averages(
    per_fault_primary: dict[str, pd.DataFrame],
    average_primary: dict[str, pd.DataFrame],
    within_run: dict[str, pd.DataFrame],
    within_average: dict[str, pd.DataFrame],
) -> None:
    suffix = {"IPPT": "L", "MPPT": "M"}
    primary_values = 0
    primary_problems: list[str] = []
    within_values = 0
    within_problems: list[str] = []

    for mode in ("IPPT", "MPPT"):
        all_faults = list(fault_stems(mode))
        excl_f3 = [fault for fault in all_faults if fault != f"F3{suffix[mode]}"]
        for spec in MODEL_SPECS:
            average_row = average_primary[mode]
            average_row = average_row[average_row["model"] == spec["label"]]
            if len(average_row) != 1:
                primary_problems.append(f"{spec['label']}/{mode}: average row count {len(average_row)}")
                continue
            average_row = average_row.iloc[0]
            subset = per_fault_primary[mode]
            subset = subset[subset["model"] == spec["label"]].set_index("fault")

            for prefix, faults in (("", all_faults), ("excl_f3_", excl_f3)):
                for metric in METRICS:
                    values = np.array(
                        [float(subset.loc[fault, f"{metric}_mean"]) for fault in faults]
                    )
                    comparisons = [
                        (f"{prefix}{metric}_mean", float(np.mean(values))),
                        (f"{prefix}{metric}_std_across_faults", float(np.std(values, ddof=0))),
                        (f"{prefix}{metric}_min", float(np.min(values))),
                        (f"{prefix}{metric}_max", float(np.max(values))),
                    ]
                    for column, expected in comparisons:
                        primary_values += 1
                        if not close_enough(average_row[column], expected):
                            primary_problems.append(
                                f"{spec['label']}/{mode}/{column}: "
                                f"saved={average_row[column]} recomputed={expected}"
                            )

            within_subset = within_run[mode]
            within_subset = within_subset[within_subset["model"] == spec["label"]].set_index("fault")
            within_faults = [
                fault for fault in all_faults if fault != "F3M"
            ]
            within_average_row = within_average[mode]
            within_average_row = within_average_row[
                within_average_row["model"] == spec["label"]
            ]
            if len(within_average_row) != 1:
                within_problems.append(f"{spec['label']}/{mode}: within average row count {len(within_average_row)}")
                continue
            within_average_row = within_average_row.iloc[0]
            values = np.array(
                [float(within_subset.loc[fault, "auc_mean"]) for fault in within_faults]
            )
            pre_rates = np.array(
                [
                    float(within_subset.loc[fault, "pre_injection_flag_share"])
                    for fault in within_faults
                ]
            )
            comparisons = [
                ("auc_mean", float(np.mean(values))),
                ("auc_std_across_faults", float(np.std(values, ddof=0))),
                ("auc_min", float(np.min(values))),
                ("auc_max", float(np.max(values))),
                ("pre_injection_flag_share", float(np.mean(pre_rates))),
                ("pre_injection_flag_share_min", float(np.min(pre_rates))),
                ("pre_injection_flag_share_max", float(np.max(pre_rates))),
            ]
            for column, expected in comparisons:
                within_values += 1
                if not close_enough(within_average_row[column], expected):
                    within_problems.append(
                        f"{spec['label']}/{mode}/{column}: "
                        f"saved={within_average_row[column]} recomputed={expected}"
                    )

    check(
        "average_primary_equals_mean_of_per_fault",
        not primary_problems,
        f"{primary_values} values recomputed; {len(primary_problems)} mismatches",
    )
    if primary_problems:
        print("  first average mismatches: " + "; ".join(primary_problems[:3]))
    check(
        "within_run_average_equals_mean_of_per_fault",
        not within_problems,
        f"{within_values} values recomputed; {len(within_problems)} mismatches",
    )
    if within_problems:
        print("  first within-run average mismatches: " + "; ".join(within_problems[:3]))


def check_f3m(
    per_fault_primary: dict[str, pd.DataFrame],
    within_run: dict[str, pd.DataFrame],
) -> None:
    primary = per_fault_primary["MPPT"]
    row = primary[primary["fault"] == "F3M"]
    ok = (
        len(row) == len(MODEL_SPECS)
        and (row["model"].nunique() == len(MODEL_SPECS))
        and not row["has_post_injection_data"].astype(bool).any()
        and (row["post_injection_windows"].astype(int) == 0).all()
        and (row["post_injection_status"] == "no_post_injection_data").all()
        and row["post_injection_note"].astype(str).str.contains("no post-injection").all()
        and (row["n_fault_windows"].astype(int) > 0).all()
    )
    detail = (
        f"F3M appears for all {len(row)} models with 0 post-injection windows "
        "and an explicit note"
    )
    if len(row) == 1:
        detail += f" (n_fault_windows={int(row.iloc[0]['n_fault_windows'])})"
    check("f3m_marked_no_post_injection_in_primary", ok, detail)

    within_faults = set(within_run["MPPT"]["fault"].tolist())
    check(
        "f3m_excluded_from_within_run",
        "F3M" not in within_faults,
        f"MPPT within-run faults: {','.join(sorted(within_faults))}",
    )
    post_ok = bool((within_run["MPPT"]["n_post_injection"] > 0).all()) and bool(
        (within_run["MPPT"]["n_pre_injection"] > 0).all()
    )
    check(
        "within_run_has_both_classes",
        post_ok,
        "every within-run row has n_pre > 0 and n_post > 0",
    )
    check(
        "f3l_has_post_injection_data",
        "F3L" in set(within_run["IPPT"]["fault"].tolist()),
        "F3L is present in the IPPT within-run table (file ends after 8 s)",
    )


def check_threshold_sensitivity(
    sensitivity: pd.DataFrame,
    per_fault_primary: dict[str, pd.DataFrame],
) -> None:
    problems: list[str] = []
    values = 0
    for model_key in SENSITIVITY_MODEL_KEYS:
        spec = MODEL_BY_KEY[model_key]
        for mode, suffix in MODES:
            archives = [load_archive(str(path)) for path in archive_paths(spec, mode)]
            thresholds = np.array([archive["threshold"] for archive in archives])
            faults = fault_stems(mode)
            row = sensitivity[
                (sensitivity["model"] == spec["label"])
                & (sensitivity["mode"] == mode)
                & (sensitivity["quantile"] == 99.0)
            ]
            if len(row) != 1:
                problems.append(f"{spec['label']}/{mode}/q99 row count {len(row)}")
                continue
            row = row.iloc[0]
            far_seed = np.array(
                [
                    float(np.mean(part_scores(archive, f"F0{suffix}_test") > archive["threshold"]))
                    for archive in archives
                ]
            )
            tpr_seed = np.array(
                [
                    [
                        float(np.mean(part_scores(archive, fault) > archive["threshold"]))
                        for archive in archives
                    ]
                    for fault in faults
                ]
            ).mean(axis=0)
            comparisons = [
                ("threshold", float(np.mean(thresholds))),
                ("healthy_test_false_alarm_rate", float(np.mean(far_seed))),
                ("average_tpr", float(np.mean(tpr_seed))),
            ]
            for column, expected in comparisons:
                values += 1
                if not close_enough(row[column], expected):
                    problems.append(
                        f"{spec['label']}/{mode}/{column}: saved={row[column]} recomputed={expected}"
                    )

            expected_quantiles = set(SENSITIVITY_QUANTILES)
            found_quantiles = set(
                sensitivity[
                    (sensitivity["model"] == spec["label"])
                    & (sensitivity["mode"] == mode)
                ]["quantile"].tolist()
            )
            if found_quantiles != expected_quantiles:
                problems.append(
                    f"{spec['label']}/{mode}: quantiles {sorted(found_quantiles)}"
                )

            primary = per_fault_primary[mode]
            primary = primary[primary["model"] == spec["label"]]
            primary_tpr = np.array(
                [float(primary[primary["fault"] == fault]["tpr_point"].iloc[0]) for fault in faults]
            )
            values += 1
            if not close_enough(row["average_tpr"], float(np.mean(primary_tpr))):
                problems.append(
                    f"{spec['label']}/{mode}: q99 average TPR differs from primary point TPR"
                )
    check(
        "threshold_sensitivity_recomputed_and_consistent_at_q99",
        not problems,
        f"{values} values checked; {len(problems)} problems",
    )
    if problems:
        print("  first sensitivity problems: " + "; ".join(problems[:3]))


def check_reporting_contract(
    per_fault_primary: dict[str, pd.DataFrame],
    average_primary: dict[str, pd.DataFrame],
    within_run: dict[str, pd.DataFrame],
    within_average: dict[str, pd.DataFrame],
    sensitivity: pd.DataFrame,
) -> None:
    labels = set(per_fault_primary["IPPT"]["model"].tolist()) | set(
        per_fault_primary["MPPT"]["model"].tolist()
    )
    check(
        "both_proposed_variants_reported",
        {"Proposed (plain)", "Proposed (featnorm)"}.issubset(labels),
        f"{len(labels)} model labels: {', '.join(sorted(labels))}",
    )

    missing: list[str] = []
    for mode in ("IPPT", "MPPT"):
        for frame_name, frame in (
            ("per_fault_primary", per_fault_primary[mode]),
            ("average_primary", average_primary[mode]),
            ("within_run", within_run[mode]),
            ("within_run_average", within_average[mode]),
        ):
            if "healthy_test_false_alarm_rate" not in frame.columns:
                missing.append(f"{frame_name}_{mode}: column absent")
            elif frame["healthy_test_false_alarm_rate"].isna().any():
                missing.append(f"{frame_name}_{mode}: NaN")
    if "healthy_test_false_alarm_rate" not in sensitivity.columns:
        missing.append("threshold_sensitivity: column absent")
    elif sensitivity["healthy_test_false_alarm_rate"].isna().any():
        missing.append("threshold_sensitivity: NaN")
    check(
        "healthy_test_false_alarm_rate_in_every_table",
        not missing,
        "present and non-missing in all Phase 5 performance tables"
        if not missing
        else "; ".join(missing),
    )


def check_bootstrap_audit() -> None:
    draws = int(BOOTSTRAP_AUDIT["draws"])
    unequal = int(BOOTSTRAP_AUDIT["unequal_class_resamples"])
    wrong_count = int(BOOTSTRAP_AUDIT["other_resample_count"])
    sizes = sorted(int(size) for size in BOOTSTRAP_AUDIT["class_sizes"])
    check(
        "bootstrap_equal_class_sizes_every_resample",
        draws > 0 and unequal == 0,
        f"{draws} resamples; {unequal} with unequal class sizes; "
        f"per-class sizes seen: {sizes[:3]}{'...' if len(sizes) > 3 else ''}",
    )
    check(
        "bootstrap_200_resamples_everywhere",
        wrong_count == 0,
        f"{int(BOOTSTRAP_AUDIT['resample_count_200'])} draws at 200 resamples; "
        f"{wrong_count} at other counts",
    )


def check_paper_reference(paper: pd.DataFrame) -> None:
    ok = (
        len(paper) == len(PAPER_REFERENCE)
        and paper["mode"].isin(["IPPT", "MPPT"]).all()
        and paper["protocol_note"].eq(PAPER_PROTOCOL_NOTE).all()
        and paper[["tpr", "fpr", "accuracy", "precision", "f1", "auc"]]
        .apply(lambda column: column.between(0.0, 1.0).all())
        .all()
    )
    check(
        "paper_reference_table_complete_and_labelled",
        bool(ok),
        f"{len(paper)} rows; every row carries the different-protocol note",
    )


def check_seed_columns(per_fault_primary: dict[str, pd.DataFrame]) -> None:
    problems: list[str] = []
    for mode in ("IPPT", "MPPT"):
        frame = per_fault_primary[mode]
        for spec in MODEL_SPECS:
            subset = frame[frame["model"] == spec["label"]]
            if subset.empty:
                problems.append(f"{spec['label']}/{mode}: absent")
                continue
            if spec["type"] == "proposed":
                if not (subset["seed_count"] == 3).all():
                    problems.append(f"{spec['label']}/{mode}: seed_count != 3")
                for metric in METRICS:
                    for seed in SEEDS:
                        column = f"{metric}_seed{seed}_mean"
                        if column not in subset.columns or subset[column].isna().any():
                            problems.append(f"{spec['label']}/{mode}: missing {column}")
            else:
                if not (subset["seed_count"] == 1).all():
                    problems.append(f"{spec['label']}/{mode}: seed_count != 1")
    check(
        "proposed_seed_detail_and_single_seed_baselines",
        not problems,
        "3 seed columns for both proposed variants; 1-seed baselines"
        if not problems
        else "; ".join(problems[:3]),
    )


def main() -> int:
    print("=" * 80)
    print("GPVS-Faults | Phase 5 final evaluation")
    print("=" * 80)
    print("Inputs: saved score archives only. No model is trained.")
    build_column_lists()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    per_fault_primary, average_primary = evaluate_primary()
    within_run, within_average = evaluate_within_run()
    sensitivity = evaluate_threshold_sensitivity()
    paper = paper_reference_frame()

    for mode in ("IPPT", "MPPT"):
        write_csv(per_fault_primary[mode], RESULTS_DIR / f"per_fault_primary_{mode}.csv")
        write_csv(average_primary[mode], RESULTS_DIR / f"average_primary_{mode}.csv")
        write_csv(within_run[mode], RESULTS_DIR / f"within_run_{mode}.csv")
        write_csv(within_average[mode], RESULTS_DIR / f"within_run_average_{mode}.csv")
    write_csv(sensitivity, RESULTS_DIR / "threshold_sensitivity.csv")
    # The paper values are published to four decimals; write them at that
    # precision instead of the full binary representation used for computed
    # results. They are external reference values, not pipeline results.
    paper.to_csv(
        RESULTS_DIR / "paper_reference.csv",
        index=False,
        float_format="%.4f",
        lineterminator="\n",
    )

    print("\nChecks")
    check_archives()
    check_recomputation(per_fault_primary, within_run)
    check_averages(
        per_fault_primary, average_primary, within_run, within_average
    )
    check_f3m(per_fault_primary, within_run)
    check_threshold_sensitivity(sensitivity, per_fault_primary)
    check_reporting_contract(
        per_fault_primary, average_primary, within_run, within_average, sensitivity
    )
    check_bootstrap_audit()
    check_paper_reference(paper)
    check_seed_columns(per_fault_primary)

    passed = sum(1 for record in CHECK_RECORDS if record["passed"])
    total = len(CHECK_RECORDS)
    print(f"\nChecks passed: {passed}/{total}")

    checks = {
        "phase": "phase_5_final_evaluation",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "checks_passed": passed,
        "checks_total": total,
        "checks": CHECK_RECORDS,
        "bootstrap": {
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "audit_draws": int(BOOTSTRAP_AUDIT["draws"]),
            "audit_unequal": int(BOOTSTRAP_AUDIT["unequal_class_resamples"]),
            "audit_resample_count_200": int(BOOTSTRAP_AUDIT["resample_count_200"]),
        },
    }
    with (RESULTS_DIR / "phase5_checks.json").open("w", encoding="utf-8") as handle:
        json.dump(checks, handle, indent=2)
        handle.write("\n")

    metadata = {
        "phase": "phase_5_final_evaluation",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "environment": environment_info(),
        "inputs": {
            "description": "saved score archives only; no model, raw data or fault score recomputation",
            "archives": [
                {
                    "path": load_archive(str(path))["path"],
                    "sha256": load_archive(str(path))["sha256"],
                }
                for spec in MODEL_SPECS
                for mode, _ in MODES
                for path in archive_paths(spec, mode)
            ],
        },
        "primary": {
            "label_view": "whole_file",
            "negative": "healthy F0*_test windows",
            "positive": "all windows of the fault file",
            "threshold": "stored in the archive, 99th percentile of healthy validation scores",
            "flag_rule": FLAG_RULE,
            "bootstrap": {
                "resamples": BOOTSTRAP_RESAMPLES,
                "equal_class_sizes": "n = number of healthy test windows drawn from each class",
                "seed": BOOTSTRAP_SEED,
                "ci": "2.5th and 97.5th percentiles",
            },
        },
        "secondary": {
            "label_view": "within_run change test",
            "pre": (
                f"same fault file, window end < {FAULT_START_S} s and "
                f"start >= {PRE_START_S} s"
            ),
            "post": f"same fault file, window start >= {FAULT_END_S} s",
            "bootstrap": {
                "resamples": BOOTSTRAP_RESAMPLES,
                "equal_class_sizes": "n = min(number pre, number post)",
                "seed": BOOTSTRAP_SEED,
                "ci": "2.5th and 97.5th percentiles",
            },
        },
        "proposed_aggregation": {
            "seeds": list(SEEDS),
            "per_fault_mean": "mean of the three per-seed bootstrap means",
            "seed_std": f"standard deviation of the three per-seed means (ddof={SEED_STD_DDOF})",
            "interval": "percentiles of the seed-averaged bootstrap distribution",
        },
        "threshold_sensitivity": {
            "quantiles": list(SENSITIVITY_QUANTILES),
            "source": "recomputed from saved healthy validation scores in each archive",
        },
        "zero_division": "precision=0 when no window is flagged; F1=0 when precision+recall=0",
        "checks_passed": passed,
        "checks_total": total,
    }
    with (RESULTS_DIR / "phase5_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")

    print(f"\nSaved: {RESULTS_DIR / 'per_fault_primary_IPPT.csv'}")
    print(f"Saved: {RESULTS_DIR / 'average_primary_MPPT.csv'}")
    print(f"Saved: {RESULTS_DIR / 'within_run_average_MPPT.csv'}")
    print(f"Saved: {RESULTS_DIR / 'threshold_sensitivity.csv'}")
    print(f"Saved: {RESULTS_DIR / 'paper_reference.csv'}")
    print(f"Saved: {RESULTS_DIR / 'phase5_checks.json'}")
    print(f"Saved: {RESULTS_DIR / 'phase5_metadata.json'}")
    print(f"\nRESULT: Phase 5 metrics computed; {passed}/{total} checks passed.")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Append the Phase 4 section to results/summary.md and STATUS.md.

This is a reporting script. It reads the saved Phase 4 outputs
(``results/proposed_training.csv``, ``results/proposed_sanity.csv``,
``results/proposed_metadata.json``) plus the frozen Phase 3 sanity table
(``results/baseline_sanity.csv``), computes every descriptive number used in
the report, saves them to ``results/phase4_summary_stats.json``, and appends a
Phase 4 section in the format required by ``AGENTS.md``. Earlier summary
sections and earlier STATUS entries are never modified.

Reporting conventions (not detector settings):

* clear rise = post-8 s median score at least 2.0x the same file's
  pre-injection median;
* weak rise = 1.2x to 2.0x;
* flat or lower = below 1.2x;
* very early stop = early stopping before 20% of the run's epoch cap
  (20 epochs for the 100-epoch runs here);
* failed to converge = the best validation loss did not improve on the first
  epoch's validation loss.

No final metrics (AUROC, precision, recall, etc.) are computed here.

Run after ``src/train.py``::

    .venv\\Scripts\\python.exe .\\src\\phase4_summary.py
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import RESULTS_DIR  # noqa: E402

SUMMARY_PATH = RESULTS_DIR / "summary.md"
STATUS_PATH = PROJECT_ROOT / "STATUS.md"
TRAINING_PATH = RESULTS_DIR / "proposed_training.csv"
SANITY_PATH = RESULTS_DIR / "proposed_sanity.csv"
METADATA_PATH = RESULTS_DIR / "proposed_metadata.json"
BASELINE_SANITY_PATH = RESULTS_DIR / "baseline_sanity.csv"
STATS_PATH = RESULTS_DIR / "phase4_summary_stats.json"

SECTION_HEADING = "## Phase 4: Proposed healthy-only TCN-BiLSTM autoencoder"
STATUS_HEADING = "## 2026-10-27 - Phase 4: proposed TCN-BiLSTM autoencoder"

CLEAR_RISE_RATIO = 2.0
WEAK_RISE_RATIO = 1.2
VERY_EARLY_FRACTION = 0.20
TARGET_FAULTS = ("F4L", "F4M", "F6L", "F6M", "F7L", "F7M")
SCORE_KEYS = {
    "plain": "proposed_plain",
    "featnorm": "proposed_featnorm",
}
MODEL_LABELS = {
    "plain": "TCN-BiLSTM (plain)",
    "featnorm": "TCN-BiLSTM (featnorm)",
}
MODE_ORDER = ("MPPT", "IPPT")


def read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def fmt_pct(value: object) -> str:
    number = float(value)
    if not np.isfinite(number):
        return "NA"
    return f"{100.0 * number:.2f}%"


def fmt_pct_range(low: object, high: object) -> str:
    return f"{fmt_pct(low)}-{fmt_pct(high)}"


def fmt_ratio(value: object) -> str:
    number = float(value)
    if not np.isfinite(number):
        return "NA"
    return f"{number:.2f}x"


def fmt_ratio_range(low: object, high: object) -> str:
    return f"{fmt_ratio(low)}-{fmt_ratio(high)}"


def fmt_score(value: object) -> str:
    number = float(value)
    if not np.isfinite(number):
        return "NA"
    return f"{number:.6g}"


def fmt_loss(value: object) -> str:
    return f"{float(value):.6f}"


def fmt_minutes(seconds: object) -> str:
    return f"{float(seconds) / 60.0:.1f}"


def fmt_bool(value: object) -> str:
    return "yes" if bool(value) else "no"


def name_list(names: list[str]) -> str:
    if not names:
        return "none"
    return ", ".join(f"`{name}`" for name in names)


def append_unique(path: Path, heading: str, section: str) -> bool:
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if heading in existing:
            print(f"{path} already contains '{heading}'; left unchanged.")
            return False
        updated = existing.rstrip() + "\n\n" + section
    else:
        updated = section
    path.write_text(updated, encoding="utf-8", newline="\n")
    return True


def classify_ratio(ratio: object, no_post: bool) -> str:
    if no_post:
        return "no_post"
    number = float(ratio)
    if not np.isfinite(number):
        return "undefined"
    if number >= CLEAR_RISE_RATIO:
        return "clear"
    if number >= WEAK_RISE_RATIO:
        return "weak"
    return "flat"


def class_counts(classes: list[str]) -> str:
    if not classes:
        return "NA"
    counts = {key: classes.count(key) for key in sorted(set(classes))}
    return ", ".join(f"{key} {count}/{len(classes)}" for key, count in counts.items())


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    training = pd.read_csv(TRAINING_PATH)
    sanity = pd.read_csv(SANITY_PATH)
    baseline = pd.read_csv(BASELINE_SANITY_PATH)
    meta = read_json(METADATA_PATH)
    return training, sanity, baseline, meta


def build_stats(
    training: pd.DataFrame,
    sanity: pd.DataFrame,
    baseline: pd.DataFrame,
    meta: dict[str, object],
) -> dict[str, object]:
    """Compute every descriptive fact used in the report from saved files."""
    seeds = sorted(int(seed) for seed in meta["seeds"])
    modes = [mode for mode in MODE_ORDER if mode in set(sanity["mode"])]
    if len(modes) != 2:
        raise RuntimeError(f"Expected both modes in sanity table, got {modes}")
    if training.shape[0] != len(meta["runs"]):
        raise RuntimeError("Training CSV and metadata run counts differ")
    if sorted(training["seed"].unique()) != seeds:
        raise RuntimeError("Training CSV seeds do not match metadata")

    # Healthy validation/test false-alarm aggregates.
    healthy: list[dict[str, object]] = []
    for score_kind, model in SCORE_KEYS.items():
        for mode in modes:
            subset = sanity[
                (sanity["model"] == model) & (sanity["mode"] == mode)
            ]
            val = subset[
                (subset["section"] == "healthy") & (subset["part"] == "val")
            ].sort_values("seed")
            test = subset[
                (subset["section"] == "healthy") & (subset["part"] == "test")
            ].sort_values("seed")
            if val.shape[0] != len(seeds) or test.shape[0] != len(seeds):
                raise RuntimeError(
                    f"{score_kind}/{mode}: expected {len(seeds)} val/test rows"
                )

            def seed_rows(frame: pd.DataFrame) -> list[dict[str, object]]:
                return [
                    {
                        "seed": int(row.seed),
                        "flagged": int(row.flagged),
                        "windows": int(row.windows),
                        "flag_rate": float(row.flag_rate),
                    }
                    for row in frame.itertuples()
                ]

            val_rows = seed_rows(val)
            test_rows = seed_rows(test)
            healthy.append(
                {
                    "score": score_kind,
                    "model": model,
                    "model_label": MODEL_LABELS[score_kind],
                    "mode": mode,
                    "val_windows": int(val.iloc[0]["windows"]),
                    "test_windows": int(test.iloc[0]["windows"]),
                    "val_mean": float(np.mean([r["flag_rate"] for r in val_rows])),
                    "val_min": float(np.min([r["flag_rate"] for r in val_rows])),
                    "val_max": float(np.max([r["flag_rate"] for r in val_rows])),
                    "test_mean": float(
                        np.mean([r["flag_rate"] for r in test_rows])
                    ),
                    "test_min": float(np.min([r["flag_rate"] for r in test_rows])),
                    "test_max": float(np.max([r["flag_rate"] for r in test_rows])),
                    "val_by_seed": val_rows,
                    "test_by_seed": test_rows,
                }
            )

    # Per-fault aggregates for every score/mode/file.
    faults: list[dict[str, object]] = []
    for score_kind, model in SCORE_KEYS.items():
        for mode in modes:
            subset = sanity[
                (sanity["model"] == model)
                & (sanity["mode"] == mode)
                & (sanity["section"] == "fault")
            ]
            for file in sorted(subset["file"].unique()):
                rows = subset[subset["file"] == file].sort_values("seed")
                if rows.shape[0] != len(seeds):
                    raise RuntimeError(
                        f"{score_kind}/{mode}/{file}: expected {len(seeds)} rows"
                    )
                ratios = [
                    float(value) if np.isfinite(float(value)) else None
                    for value in rows["median_ratio_after8_over_pre"]
                ]
                no_post = bool(rows["no_post_injection_data"].all())
                classes = [
                    classify_ratio(ratio, no_post)
                    for ratio in rows["median_ratio_after8_over_pre"]
                ]
                finite_ratios = [value for value in ratios if value is not None]

                baseline_rows = baseline[
                    (baseline["mode"] == mode)
                    & (baseline["file"] == file)
                    & (baseline["section"] == "fault")
                ]
                baseline_ratios = [
                    float(value)
                    for value in baseline_rows["median_ratio_after8_over_pre"]
                    if np.isfinite(float(value))
                ]
                baseline_clear = [
                    str(row.model_label)
                    for row in baseline_rows.itertuples()
                    if np.isfinite(float(row.median_ratio_after8_over_pre))
                    and float(row.median_ratio_after8_over_pre) >= CLEAR_RISE_RATIO
                ]

                faults.append(
                    {
                        "score": score_kind,
                        "model": model,
                        "model_label": MODEL_LABELS[score_kind],
                        "mode": mode,
                        "file": str(file),
                        "no_post": no_post,
                        "ratios_by_seed": [
                            {"seed": int(seed), "ratio": ratio}
                            for seed, ratio in zip(rows["seed"], ratios)
                        ],
                        "ratio_mean": (
                            float(np.mean(finite_ratios))
                            if finite_ratios
                            else None
                        ),
                        "ratio_min": (
                            float(np.min(finite_ratios))
                            if finite_ratios
                            else None
                        ),
                        "ratio_max": (
                            float(np.max(finite_ratios))
                            if finite_ratios
                            else None
                        ),
                        "classes": classes,
                        "class_counts": class_counts(classes),
                        "flag_rate_mean": float(rows["flag_rate"].mean()),
                        "flag_rate_min": float(rows["flag_rate"].min()),
                        "flag_rate_max": float(rows["flag_rate"].max()),
                        "baseline_post_available": int(
                            (~baseline_rows["no_post_injection_data"]).sum()
                        ),
                        "baseline_clear_count": len(baseline_clear),
                        "baseline_clear_labels": baseline_clear,
                        "baseline_ratio_min": (
                            float(np.min(baseline_ratios))
                            if baseline_ratios
                            else None
                        ),
                        "baseline_ratio_max": (
                            float(np.max(baseline_ratios))
                            if baseline_ratios
                            else None
                        ),
                    }
                )

    # Seed-consistency aggregates.
    consistency: list[dict[str, object]] = []
    for score_kind, model in SCORE_KEYS.items():
        for mode in modes:
            subset = sanity[
                (sanity["model"] == model) & (sanity["mode"] == mode)
            ]
            thresholds = (
                subset.groupby("seed")["threshold"].first().sort_index()
            )
            fault = subset[subset["section"] == "fault"]

            flag_ranges: list[float] = []
            ratio_spreads: list[dict[str, object]] = []
            agreement = 0
            agreement_denominator = 0
            for file, group in fault.groupby("file"):
                flag_ranges.append(
                    float(group["flag_rate"].max() - group["flag_rate"].min())
                )
                ratios = [
                    float(value)
                    for value in group["median_ratio_after8_over_pre"]
                    if np.isfinite(float(value))
                ]
                if len(ratios) >= 2 and min(ratios) > 0.0:
                    ratio_spreads.append(
                        {
                            "file": str(file),
                            "factor": float(max(ratios) / min(ratios)),
                            "ratio_min": float(min(ratios)),
                            "ratio_max": float(max(ratios)),
                        }
                    )
                no_post = bool(group["no_post_injection_data"].all())
                if no_post:
                    continue
                agreement_denominator += 1
                classes = {
                    classify_ratio(value, False)
                    for value in group["median_ratio_after8_over_pre"]
                }
                if len(classes) == 1:
                    agreement += 1

            worst_spread = (
                max(ratio_spreads, key=lambda item: float(item["factor"]))
                if ratio_spreads
                else None
            )
            consistency.append(
                {
                    "score": score_kind,
                    "mode": mode,
                    "threshold_mean": float(thresholds.mean()),
                    "threshold_min": float(thresholds.min()),
                    "threshold_max": float(thresholds.max()),
                    "threshold_range": float(thresholds.max() - thresholds.min()),
                    "threshold_cv": (
                        float(thresholds.std(ddof=0) / thresholds.mean())
                        if float(thresholds.mean()) != 0.0
                        else None
                    ),
                    "flag_share_max_range": (
                        float(max(flag_ranges)) if flag_ranges else None
                    ),
                    "ratio_spread_max_factor": (
                        float(worst_spread["factor"]) if worst_spread else None
                    ),
                    "ratio_spread_file": (
                        str(worst_spread["file"]) if worst_spread else None
                    ),
                    "ratio_spread_ratio_min": (
                        float(worst_spread["ratio_min"]) if worst_spread else None
                    ),
                    "ratio_spread_ratio_max": (
                        float(worst_spread["ratio_max"]) if worst_spread else None
                    ),
                    "class_agreement_count": int(agreement),
                    "class_agreement_denominator": int(agreement_denominator),
                }
            )

    # Early stopping and convergence facts.
    very_early_rows = []
    no_improvement_rows = []
    peaked_first_rows = []
    stopped_early_rows = []
    for row in training.itertuples():
        label = f"{row.mode}/seed{int(row.seed)}"
        if int(row.epochs_run) <= int(
            math.floor(VERY_EARLY_FRACTION * int(row.max_epochs))
        ):
            very_early_rows.append(label)
        if not bool(row.improved_over_initial):
            no_improvement_rows.append(label)
        if int(row.best_epoch) == 1:
            peaked_first_rows.append(label)
        if bool(row.stopped_early):
            stopped_early_rows.append(label)

    total_training_time = float(training["training_time_s"].sum())
    total_scoring_time = float(training["scoring_time_s"].sum())
    windows_total = int(
        sum(int(run["windows"]) for run in meta["runs"]) * len(SCORE_KEYS)
    )

    stats = {
        "phase": "phase_4_proposed_summary",
        "generated_local": datetime.now().astimezone().isoformat(),
        "date": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M"),
        "git_commit": meta.get("git_commit"),
        "environment": meta.get("environment"),
        "source_files": {
            "proposed_training": "results/proposed_training.csv",
            "proposed_sanity": "results/proposed_sanity.csv",
            "proposed_metadata": "results/proposed_metadata.json",
            "baseline_sanity": "results/baseline_sanity.csv",
        },
        "conventions": {
            "clear_rise_ratio": CLEAR_RISE_RATIO,
            "weak_rise_ratio": WEAK_RISE_RATIO,
            "very_early_fraction": VERY_EARLY_FRACTION,
            "failed_to_converge": (
                "best validation loss did not improve on epoch 1"
            ),
        },
        "totals": {
            "runs": int(len(meta["runs"])),
            "modes": len(modes),
            "seeds": len(seeds),
            "archives": int(len(meta["runs"]) * len(SCORE_KEYS)),
            "windows_total": windows_total,
            "fault_files": int(
                sanity[sanity["section"] == "fault"]["file"].nunique()
            ),
            "min_mean_fault_flag_share": float(
                min(float(row["flag_rate_mean"]) for row in faults)
            ),
            "max_mean_fault_flag_share": float(
                max(float(row["flag_rate_mean"]) for row in faults)
            ),
            "min_seed_fault_flag_share": float(
                min(float(row["flag_rate_min"]) for row in faults)
            ),
            "max_seed_fault_flag_share": float(
                max(float(row["flag_rate_max"]) for row in faults)
            ),
            "checks_passed": int(meta["checks_passed"]),
            "checks_total": int(meta["checks_total"]),
            "model_params": sorted(
                {int(run["model_params"]) for run in meta["runs"]}
            ),
            "device": meta.get("device"),
            "tensorflow": meta.get("tensorflow"),
            "keras": meta.get("keras"),
            "total_training_time_s": total_training_time,
            "total_scoring_time_s": total_scoring_time,
            "very_early_threshold_epochs": int(
                math.floor(
                    VERY_EARLY_FRACTION
                    * int(training["max_epochs"].min())
                )
            ),
        },
        "training": [
            {
                "mode": str(row.mode),
                "seed": int(row.seed),
                "epochs_run": int(row.epochs_run),
                "max_epochs": int(row.max_epochs),
                "best_epoch": int(row.best_epoch),
                "best_val_loss": float(row.best_val_loss),
                "final_val_loss": float(row.final_val_loss),
                "initial_val_loss": float(row.initial_val_loss),
                "stopped_early": bool(row.stopped_early),
                "improved_over_initial": bool(row.improved_over_initial),
                "train_windows": int(row.train_windows),
                "val_windows": int(row.val_windows),
                "model_params": int(row.model_params),
                "training_time_s": float(row.training_time_s),
                "scoring_time_s": float(row.scoring_time_s),
                "device": str(row.device),
            }
            for row in training.itertuples()
        ],
        "healthy": healthy,
        "faults": faults,
        "consistency": consistency,
        "early_stopping": {
            "stopped_early": stopped_early_rows,
            "very_early": very_early_rows,
            "no_improvement": no_improvement_rows,
            "peaked_first": peaked_first_rows,
            "epochs_run_min": int(training["epochs_run"].min()),
            "epochs_run_max": int(training["epochs_run"].max()),
            "best_epoch_min": int(training["best_epoch"].min()),
            "best_epoch_max": int(training["best_epoch"].max()),
        },
    }
    return stats


def target_rows(stats: dict[str, object]) -> list[dict[str, object]]:
    rows = [
        row
        for row in stats["faults"]
        if str(row["file"]) in TARGET_FAULTS
    ]
    order = {file: index for index, file in enumerate(TARGET_FAULTS)}
    score_order = {"plain": 0, "featnorm": 1}
    rows.sort(
        key=lambda row: (
            score_order[str(row["score"])],
            order[str(row["file"])],
        )
    )
    return rows


def build_section(stats: dict[str, object]) -> str:
    """Render the Phase 4 markdown section from computed facts."""
    totals = stats["totals"]
    training = stats["training"]
    healthy = stats["healthy"]
    targets = target_rows(stats)
    consistency = stats["consistency"]
    early = stats["early_stopping"]
    conventions = stats["conventions"]

    training_rows = [
        [
            str(row["mode"]),
            str(row["seed"]),
            str(row["epochs_run"]),
            str(row["best_epoch"]),
            fmt_loss(row["best_val_loss"]),
            fmt_loss(row["final_val_loss"]),
            fmt_bool(row["stopped_early"]),
            fmt_minutes(row["training_time_s"]),
        ]
        for row in training
    ]
    training_table = markdown_table(
        [
            "Mode",
            "Seed",
            "Epochs run",
            "Best epoch",
            "Best val loss",
            "Final val loss",
            "Stopped early",
            "Training time (min)",
        ],
        training_rows,
    )

    healthy_rows = []
    for row in healthy:
        test_counts = "; ".join(
            f"s{item['seed']}: {item['flagged']}/{item['windows']}"
            for item in row["test_by_seed"]
        )
        healthy_rows.append(
            [
                str(row["score"]),
                str(row["mode"]),
                f"{fmt_pct(row['val_mean'])} "
                f"({fmt_pct_range(row['val_min'], row['val_max'])})",
                f"{fmt_pct(row['test_mean'])} "
                f"({fmt_pct_range(row['test_min'], row['test_max'])})",
                test_counts,
            ]
        )
    healthy_table = markdown_table(
        [
            "Score",
            "Mode",
            "Validation FAR mean (range)",
            "Test FAR mean (range)",
            "Test false alarms by seed",
        ],
        healthy_rows,
    )

    target_table_rows = []
    for row in targets:
        ratio_cells = " / ".join(
            fmt_ratio(item["ratio"]) for item in row["ratios_by_seed"]
        )
        if row["no_post"]:
            ratio_cells = "no post data"
            mean_range = "no post data"
        else:
            mean_range = (
                f"{fmt_ratio(row['ratio_mean'])} "
                f"({fmt_ratio_range(row['ratio_min'], row['ratio_max'])})"
            )
        baseline_cell = (
            f"{int(row['baseline_clear_count'])}/"
            f"{int(row['baseline_post_available'])}"
        )
        if row["baseline_clear_labels"]:
            baseline_cell += " (" + ", ".join(row["baseline_clear_labels"]) + ")"
        target_table_rows.append(
            [
                str(row["score"]),
                str(row["file"]),
                str(row["mode"]),
                ratio_cells,
                mean_range,
                str(row["class_counts"]),
                baseline_cell,
            ]
        )
    target_table = markdown_table(
        [
            "Score",
            "Fault file",
            "Mode",
            "Post/pre ratio seed 0 / 1 / 2",
            "Mean (range)",
            "Rise class across seeds",
            "Baselines with clear rise",
        ],
        target_table_rows,
    )

    consistency_rows = []
    for row in consistency:
        consistency_rows.append(
            [
                str(row["score"]),
                str(row["mode"]),
                f"{fmt_score(row['threshold_mean'])} "
                f"({fmt_score(row['threshold_min'])}-"
                f"{fmt_score(row['threshold_max'])})",
                (
                    fmt_pct(row["threshold_cv"])
                    if row["threshold_cv"] is not None
                    else "NA"
                ),
                f"{100.0 * float(row['flag_share_max_range']):.2f} pp",
                (
                    fmt_ratio(row["ratio_spread_max_factor"])
                    if row["ratio_spread_max_factor"] is not None
                    else "NA"
                ),
                f"{int(row['class_agreement_count'])}/"
                f"{int(row['class_agreement_denominator'])}",
            ]
        )
    consistency_table = markdown_table(
        [
            "Score",
            "Mode",
            "Validation threshold mean (range)",
            "Threshold CV",
            "Max per-file flag-share range across seeds",
            "Max per-file ratio spread across seeds (max/min)",
            "Same rise class across seeds",
        ],
        consistency_rows,
    )

    # Data-driven prose for (a) healthy false alarms.
    healthy_sentences = []
    for row in healthy:
        healthy_sentences.append(
            f"{row['score']} {row['mode']}: validation "
            f"{fmt_pct(row['val_mean'])} "
            f"({fmt_pct_range(row['val_min'], row['val_max'])}), "
            f"test {fmt_pct(row['test_mean'])} "
            f"({fmt_pct_range(row['test_min'], row['test_max'])})"
        )
    test_above_val = [
        f"{row['score']} {row['mode']}"
        for row in healthy
        if float(row["test_mean"]) > float(row["val_mean"])
    ]
    answer_a = (
        "(a) Healthy test false-alarm rates versus the validation rates "
        "(mean and range across the three seeds): "
        + "; ".join(healthy_sentences)
        + ". The test rate is above the validation rate for "
        + (", ".join(test_above_val) if test_above_val else "none")
        + "."
    )

    # Data-driven prose for (b) F4/F6/F7 post-8 s rise versus baselines.
    target_sentences = []
    for score_kind in ("plain", "featnorm"):
        rows = [row for row in targets if row["score"] == score_kind]
        clear_files = [
            str(row["file"])
            for row in rows
            if "clear" in [str(item) for item in row["classes"]]
        ]
        target_sentences.append(
            f"{score_kind} clear-rise files: "
            + (name_list(clear_files) if clear_files else "none")
        )
    baseline_sentences = []
    for row in targets:
        if row["score"] != "plain":
            continue
        baseline_sentences.append(
            f"`{row['file']}` {int(row['baseline_clear_count'])}/"
            f"{int(row['baseline_post_available'])}"
        )
    any_proposed_clear = [
        f"{row['score']} {row['file']}"
        for row in targets
        if "clear" in [str(item) for item in row["classes"]]
    ]
    answer_b = (
        "(b) Post-8 s rise for F4, F6 and F7: "
        + "; ".join(target_sentences)
        + ". Baseline clear-rise counts on the same six files are "
        + ", ".join(baseline_sentences)
        + " (clear rise is >= 2.0x the same file's pre-injection median). "
        + (
            "Proposed clear-rise target cases: "
            + ", ".join(any_proposed_clear)
            + ". "
            if any_proposed_clear
            else "No proposed score shows a clear rise on any target file. "
        )
        + "`F3M` has no post-injection data and is excluded from this comparison."
    )

    # Data-driven prose for (c) seed consistency.
    max_threshold_cv = max(
        float(row["threshold_cv"])
        for row in consistency
        if row["threshold_cv"] is not None
    )
    max_flag_range = max(
        float(row["flag_share_max_range"]) for row in consistency
    )
    spread_rows = [
        row
        for row in consistency
        if row["ratio_spread_max_factor"] is not None
    ]
    worst_spread = (
        max(
            spread_rows,
            key=lambda row: float(row["ratio_spread_max_factor"]),
        )
        if spread_rows
        else None
    )
    agreement_total = sum(
        int(row["class_agreement_count"]) for row in consistency
    )
    agreement_denominator = sum(
        int(row["class_agreement_denominator"]) for row in consistency
    )
    spread_text = (
        "the largest spread between the smallest and largest seed "
        "post/pre ratio is "
        f"{fmt_ratio(worst_spread['ratio_spread_max_factor'])} "
        f"(`{worst_spread['score']}` `{worst_spread['mode']}` "
        f"`{worst_spread['ratio_spread_file']}`, "
        f"{fmt_ratio(worst_spread['ratio_spread_ratio_min'])}-"
        f"{fmt_ratio(worst_spread['ratio_spread_ratio_max'])})"
        if worst_spread is not None
        else "no file had two finite positive seed ratios to compare"
    )
    answer_c = (
        "(c) Seed consistency: the largest threshold coefficient of variation "
        f"is {fmt_pct(max_threshold_cv)}; the largest per-file flag-share "
        f"range is {100.0 * max_flag_range:.2f} percentage points; "
        f"{spread_text}. The three seeds give the same "
        "clear/weak/flat rise class in "
        f"{agreement_total}/{agreement_denominator} score/mode/file "
        "comparisons with post-injection data."
    )

    # Data-driven prose for (d) early stopping and convergence.
    early_bits = []
    if early["very_early"]:
        early_bits.append(
            "Runs stopped before the very-early cut-off "
            f"(<= {int(totals['very_early_threshold_epochs'])} epochs): "
            + name_list(list(early["very_early"]))
        )
    else:
        early_bits.append(
            "No run stopped before the very-early cut-off "
            f"(<= {int(totals['very_early_threshold_epochs'])} epochs)"
        )
    if early["no_improvement"]:
        early_bits.append(
            "best validation loss did not improve on epoch 1: "
            + name_list(list(early["no_improvement"]))
        )
    else:
        early_bits.append(
            "every run improved on its first-epoch validation loss"
        )
    if early["peaked_first"]:
        early_bits.append(
            "best epoch was 1: " + name_list(list(early["peaked_first"]))
        )
    else:
        early_bits.append("no run peaked at epoch 1")
    answer_d = (
        "(d) Early stopping and convergence: epochs run ranged "
        f"{int(early['epochs_run_min'])}-{int(early['epochs_run_max'])} and "
        f"best epochs ranged {int(early['best_epoch_min'])}-"
        f"{int(early['best_epoch_max'])}. "
        + "; ".join(early_bits)
        + f". Total measured training time was "
        f"{fmt_minutes(totals['total_training_time_s'])} minutes on "
        f"{totals['device']}."
    )

    # Whole-file saturation description, if any.
    min_fault_flag = float(totals["min_mean_fault_flag_share"])
    max_fault_flag = float(totals["max_mean_fault_flag_share"])
    min_seed_fault_flag = float(totals["min_seed_fault_flag_share"])
    saturated = min_fault_flag >= 0.95

    what_was_done = " ".join(
        [
            (
                "The proposed TCN-BiLSTM autoencoder was copied unchanged "
                "from `Untitled0.ipynb` and trained separately for MPPT and "
                "IPPT with seeds 0, 1 and 2."
            ),
            (
                "Each run used only the healthy training windows of its mode "
                "for fitting and the healthy validation windows for early "
                "stopping; no fault file influenced training, normalization, "
                "threshold selection, or any model decision."
            ),
            (
                "Two scores were saved for every window of every file: the "
                "plain mean squared reconstruction error and a fixed k=3 "
                "feature-normalized score whose per-feature mean and standard "
                "deviation come only from healthy validation windows."
            ),
            (
                "The threshold of each score is the 99th percentile of that "
                "score on healthy validation windows; the fixed k=3 was set "
                "before any fault scoring and was not tuned."
            ),
            (
                "The training script is resumable through the saved model "
                "file and writes one progress line per epoch to "
                "`logs/train.log`. Only sanity outputs were produced; no final "
                "classification metrics were computed."
            ),
        ]
    )

    files_lines = [
        (
            "- `src/model.py` - the notebook's `tcn_residual_block` and "
            "`build_tcn_bilstm_autoencoder` copied without architecture changes."
        ),
        (
            "- `src/train.py` - resumable healthy-only training, two-score "
            "scoring, archive saving, sanity/training tables, and checks."
        ),
        (
            "- `src/phase4_summary.py` - reads saved results and writes this "
            "section plus the STATUS entry."
        ),
        "- `run_all.py` - added the Phase 4 training and summary steps.",
        (
            "- `models/proposed_tcn_bilstm_<mode>_seed<k>.keras` - "
            f"{int(totals['runs'])} trained models."
        ),
        (
            "- `results/history_<mode>_seed<k>.json` and "
            "`figures/training_curve_<mode>_seed<k>.png` - one history and "
            "training curve per run."
        ),
        (
            "- `results/scores_proposed_<score>_<mode>_seed<k>.npz` - "
            f"{int(totals['archives'])} archives with the five baseline keys "
            "plus `feature_errors` of shape `(n_windows, 13)`."
        ),
        (
            "- `results/proposed_sanity.csv` - healthy validation/test flag "
            "rates, per-fault whole-file shares, and pre/post-8 s medians for "
            "both scores and all seeds."
        ),
        (
            "- `results/proposed_training.csv` - epochs run, best epoch, and "
            "best/final validation loss per mode and seed."
        ),
        (
            "- `results/proposed_run_<mode>_seed<k>.json` - per-run "
            "configuration, thresholds, normalization statistics, timing, and "
            "check results."
        ),
        (
            "- `results/proposed_metadata.json` - configuration, thresholds, "
            "normalization statistics, check results, environment, git commit."
        ),
        "- `results/phase4_summary_stats.json` - the facts used in this report.",
        "- `results/summary.md` - this Phase 4 section.",
        "- `STATUS.md` - appended Phase 4 log entry.",
        "- `logs/train.log` - per-epoch progress and resumability log.",
    ]

    checks_lines = [
        (
            f"- The proposed-model run reports "
            f"{int(totals['checks_passed'])}/{int(totals['checks_total'])} "
            "checks passed."
        ),
        (
            f"- All {int(totals['archives'])} archives contain exactly the "
            "five baseline keys plus `feature_errors`, and all "
            f"{int(totals['windows_total']):,} saved window scores are finite."
        ),
        (
            "- Every threshold equals the 99th percentile of healthy "
            "validation scores recomputed from the saved archive; no healthy "
            "test or fault window was used."
        ),
        (
            "- Training and validation arrays are exactly the frozen healthy "
            "F0L/F0M splits (SHA-256 verified), and no fault file contributed "
            "a window."
        ),
        (
            "- The feature-error mean/std used for `featnorm` match the "
            "healthy validation windows exactly when recomputed from each "
            "saved model."
        ),
        (
            "- Batch scoring agrees with one-window-at-a-time scoring on "
            "sample windows for every run, mode, and file."
        ),
        (
            "- `plain` equals the mean of the saved feature errors, and "
            "`featnorm` equals the fixed k=3 formula recomputed from the saved "
            "feature errors."
        ),
        (
            "- Every flag rate in `proposed_sanity.csv` equals "
            "flagged/windows; `F3M` is marked as having zero post-injection "
            "windows for every score and seed."
        ),
        (
            "- The frozen Phase 1-3 modules `src/data.py` and `src/baselines.py` "
            "were not edited; `Untitled0.ipynb` was not modified."
        ),
    ]

    warning_lines = [
        (
            (
                "- No TensorFlow GPU was found; all training used "
                if str(totals["device"]) == "/CPU:0"
                else "- Training used "
            )
            + f"`{totals['device']}`. Measured training time was "
            f"{fmt_minutes(totals['total_training_time_s'])} minutes for six "
            "runs."
        ),
        (
            "- Windows overlap by 90%, so window-level observations are not "
            "independent; the sanity table is descriptive and has no "
            "confidence intervals or significance tests."
        ),
        (
            "- The frozen preprocessing keeps the `filtfilt` edge transient "
            "and the startup transient in the first windows, so those windows "
            "are learned as normal healthy behavior."
        ),
        (
            "- Mean whole-file fault flag shares per score/mode/file range "
            f"from {fmt_pct(min_fault_flag)} to {fmt_pct(max_fault_flag)}; the "
            f"lowest single-seed share is {fmt_pct(min_seed_fault_flag)}. "
            + (
                "The whole-file view is therefore saturated and cannot "
                "separate methods; the within-run pre/post comparison is the "
                "discriminating view."
                if saturated
                else "The whole-file view is close to saturation but not uniform."
            )
        ),
        (
            "- Post/pre ratios in the tens, hundreds, or thousands occur when "
            "the post-injection score is far outside the healthy range; they "
            "are descriptive magnitudes, not calibrated effect sizes, and the "
            "exact ratio varies with seed."
        ),
        (
            "- The healthy recording drifts over time; a threshold fitted on "
            "the validation block can transfer imperfectly to the later test "
            "block. The test false-alarm rates above are the relevant "
            "measurement, not a guarantee."
        ),
        (
            "- `F3M` ends before 7 s and has no post-injection data. It is "
            "always reported as such and is excluded from the post-8 s "
            "comparison; no denominator was invented for it."
        ),
        (
            "- Comparisons with `baseline_sanity.csv` are descriptive. The "
            "baselines use the same frozen windows and threshold rule but "
            "different model classes; no statistical test was run."
        ),
        (
            "- No AUROC, precision, recall, F1, or other final metric was "
            "computed in this phase. The saved archives are the input to the "
            "Phase 5 evaluation."
        ),
    ]

    decision_text = (
        "No. The architecture, fixed k=3, validation normalization, and "
        "threshold rule are frozen by this phase; Phase 5 can consume the "
        "saved archives without retuning anything after seeing fault sanity."
    )

    return "\n".join(
        [
            f"{SECTION_HEADING}   ({stats['date']})",
            "",
            f"What was done: {what_was_done}",
            "",
            "Files created or changed:",
            *files_lines,
            "",
            "Key numbers:",
            "",
            "### Training runs",
            "",
            training_table,
            "",
            "Training time is the measured `fit` call only. The best epoch is "
            "the `argmin` of the saved validation-loss history; the model file "
            "contains the restored best weights.",
            "",
            "### Healthy false alarms",
            "",
            healthy_table,
            "",
            "A false alarm is a healthy window with score > threshold. "
            "Validation windows set the threshold; test windows are later in "
            "time and were never used for fitting or thresholding.",
            "",
            "### F4, F6 and F7: post-8 s rise versus baselines",
            "",
            target_table,
            "",
            "The proposed ratio is the median score of windows starting after "
            "8 s divided by the median score of pre-injection windows ending "
            "before 7 s, excluding the first 0.5 s. Baseline counts come from "
            "`baseline_sanity.csv`; the denominator is the number of baseline "
            "models with post-injection data for that file.",
            "",
            "### Seed consistency",
            "",
            consistency_table,
            "",
            "### Plain answers",
            "",
            f"- {answer_a}",
            f"- {answer_b}",
            f"- {answer_c}",
            f"- {answer_d}",
            "",
            "Checks passed:",
            *checks_lines,
            "",
            "Problems or warnings:",
            *warning_lines,
            "",
            f"Decision needed from me: {decision_text}",
            "",
            "Next phase: Phase 5 - final whole-file and time-resolved "
            "evaluation of the saved proposed-model score archives.",
            "",
        ]
    )


def build_status_entry(stats: dict[str, object]) -> str:
    totals = stats["totals"]
    healthy = stats["healthy"]
    early = stats["early_stopping"]
    targets = target_rows(stats)

    healthy_bits = "; ".join(
        f"{row['score']} {row['mode']} test {fmt_pct(row['test_mean'])} "
        f"({fmt_pct_range(row['test_min'], row['test_max'])}) vs validation "
        f"{fmt_pct(row['val_mean'])}"
        for row in healthy
    )
    target_bits = "; ".join(
        f"{row['score']} `{row['file']}` {row['class_counts']}"
        for row in targets
    )
    device_note = (
        "No GPU was available."
        if str(totals["device"]) == "/CPU:0"
        else f"Training used `{totals['device']}`."
    )
    return "\n".join(
        [
            STATUS_HEADING,
            "",
            "**What was done.** Copied the notebook's TCN-BiLSTM autoencoder "
            "unchanged into `src/model.py` and trained it healthy-only for "
            "MPPT and IPPT with seeds 0, 1 and 2. Each run used only the "
            "healthy training windows for fitting and healthy validation "
            "windows for early stopping. Two window scores (`plain` and fixed "
            "k=3 `featnorm`) were saved for every file, with thresholds at the "
            "99th percentile of healthy validation scores. No fault file was "
            "used for training, normalization, thresholds, or model selection.",
            "",
            "**Files created/changed.** `src/model.py`, `src/train.py`, "
            "`src/phase4_summary.py`, `run_all.py`, "
            "`models/proposed_tcn_bilstm_<mode>_seed<k>.keras` (6), "
            "`results/history_<mode>_seed<k>.json` (6), "
            "`figures/training_curve_<mode>_seed<k>.png` (6), "
            "`results/scores_proposed_<score>_<mode>_seed<k>.npz` (12), "
            "`results/proposed_sanity.csv`, `results/proposed_training.csv`, "
            "`results/proposed_run_<mode>_seed<k>.json` (6), "
            "`results/proposed_metadata.json`, "
            "`results/phase4_summary_stats.json`, `logs/train.log`, "
            "`results/summary.md`, `STATUS.md`. Frozen `src/data.py`, "
            "`src/baselines.py`, and `Untitled0.ipynb` were not modified.",
            "",
            "**Key numbers (computed by `src/train.py` and "
            "`src/phase4_summary.py`).**",
            f"- {int(totals['runs'])} runs, {int(totals['archives'])} "
            f"archives, {int(totals['windows_total']):,} saved window scores; "
            f"{int(totals['checks_passed'])}/{int(totals['checks_total'])} "
            "checks passed.",
            f"- Epochs run {int(early['epochs_run_min'])}-"
            f"{int(early['epochs_run_max'])}; best epochs "
            f"{int(early['best_epoch_min'])}-{int(early['best_epoch_max'])}; "
            f"total training time "
            f"{fmt_minutes(totals['total_training_time_s'])} min on "
            f"`{totals['device']}`.",
            f"- Healthy test false-alarm rates: {healthy_bits}.",
            f"- F4/F6/F7 rise classes: {target_bits}.",
            (
                "- Seed consistency: largest threshold CV "
                f"{fmt_pct(max(float(row['threshold_cv']) for row in stats['consistency'] if row['threshold_cv'] is not None))}; "
                "rise-class agreement "
                f"{sum(int(row['class_agreement_count']) for row in stats['consistency'])}/"
                f"{sum(int(row['class_agreement_denominator']) for row in stats['consistency'])}; "
                "largest single-file seed ratio spread "
                f"{fmt_ratio(max(float(row['ratio_spread_max_factor']) for row in stats['consistency'] if row['ratio_spread_max_factor'] is not None))}."
            ),
            "",
            "**Open issues/risks.** "
            + device_note
            + " Mean whole-file fault flag shares are "
            f"{fmt_pct(totals['min_mean_fault_flag_share'])}-"
            f"{fmt_pct(totals['max_mean_fault_flag_share'])} (lowest single-seed "
            f"share {fmt_pct(totals['min_seed_fault_flag_share'])}), so the "
            "whole-file view is close to saturation; the pre/post-8 s "
            "comparison is the discriminating sanity view. `F3M` has no "
            "post-injection data. "
            "Windows overlap by 90%, and the healthy validation-to-test drift "
            "affects false-alarm transfer. No final metrics were computed.",
            "",
        ]
    )


def main() -> int:
    print("=" * 80)
    print("GPVS-Faults | Phase 4 summary report")
    print("=" * 80)

    training, sanity, baseline, meta = load_inputs()
    stats = build_stats(training, sanity, baseline, meta)
    with STATS_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(stats, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Saved {STATS_PATH}")

    section = build_section(stats)
    appended = append_unique(SUMMARY_PATH, SECTION_HEADING, section)
    if appended:
        print(f"Appended Phase 4 section to {SUMMARY_PATH}")
    else:
        print(f"{SUMMARY_PATH} already has the Phase 4 section.")

    status_entry = build_status_entry(stats)
    status_appended = append_unique(STATUS_PATH, STATUS_HEADING, status_entry)
    if status_appended:
        print(f"Appended Phase 4 entry to {STATUS_PATH}")
    else:
        print(f"{STATUS_PATH} already has the Phase 4 entry.")

    print("\nPhase 4 section preview:")
    print(section)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

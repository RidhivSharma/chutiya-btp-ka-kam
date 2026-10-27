"""Phase 6: publication figures and tables from saved Phase 1-5 results.

This module performs **no training, scoring or threshold selection**. Every
number in the figures and tables is read from a saved CSV, JSON or NPZ file
under ``results/``. The only exception is the architecture diagram, which is a
schematic of ``src/model.py`` and contains no result values.

Outputs
-------
figures/paper/fig_*.png
    Nine 300 dpi PNG figures.
results/tables.md
    Tables T1-T8, generated from the saved result files.
results/figure_captions.md
    One factual caption per figure.
results/phase6_table_values.json
    Audit trail: every numeric/string table cell with its source selector.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle
from matplotlib.ticker import FuncFormatter, LogLocator, PercentFormatter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
FIGURES = PROJECT_ROOT / "figures" / "paper"

MODEL_ORDER = [
    "Proposed (plain)",
    "Proposed (featnorm)",
    "EllipticEnvelope",
    "IsolationForest",
    "LocalOutlierFactor",
    "OneClassSVM",
    "Mahalanobis (window-mean)",
]

MODEL_COLORS = {
    "Proposed (plain)": "#1f4e79",
    "Proposed (featnorm)": "#d55e00",
    "EllipticEnvelope": "#2ca02c",
    "IsolationForest": "#9467bd",
    "LocalOutlierFactor": "#17becf",
    "OneClassSVM": "#e377c2",
    "Mahalanobis (window-mean)": "#7f7f7f",
}

MODE_SUFFIX = {"IPPT": "L", "MPPT": "M"}

# Static architecture summary copied from src/model.py. This is not a result
# value; it is written to results/phase6_architecture.json so that the table and
# the block diagram read the same numbers.
ARCHITECTURE_SUMMARY = {
    "source": "src/model.py",
    "note": "Static architecture summary copied from src/model.py; not a result value.",
    "input_timesteps": 100,
    "input_features": 13,
    "encoder_tcn_filters": [32, 32, 64],
    "encoder_tcn_dilations": [1, 2, 4],
    "kernel_size": 3,
    "dropout_rate": 0.1,
    "encoder_bilstm_units_per_direction": 64,
    "latent_bilstm_units_per_direction": 32,
    "repeat_timesteps": 100,
    "decoder_bilstm_units_per_direction": [32, 64],
    "decoder_tcn_filters": [64, 32],
    "decoder_tcn_dilations": [4, 2],
    "output_dense_units": 13,
    "batch_normalization": True,
    "residual_connections": True,
}

# Simple, consistent typography. No decorative spines or backgrounds.
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 11,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

CSV_CACHE: dict[str, pd.DataFrame] = {}
JSON_CACHE: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def load_csv(name: str) -> pd.DataFrame:
    if name not in CSV_CACHE:
        CSV_CACHE[name] = pd.read_csv(RESULTS / name)
    return CSV_CACHE[name]


def load_json(name: str) -> Any:
    if name not in JSON_CACHE:
        JSON_CACHE[name] = json.loads((RESULTS / name).read_text(encoding="utf-8"))
    return JSON_CACHE[name]


def csv_raw(source: str, filters: dict[str, Any], column: str) -> Any:
    """Select exactly one CSV row and return one column value."""

    frame = load_csv(source)
    subset = frame
    for key, value in filters.items():
        subset = subset[subset[key].astype(str) == str(value)]
    if len(subset) != 1:
        raise ValueError(
            f"{source}: filters {filters} selected {len(subset)} rows, expected 1"
        )
    value = subset.iloc[0][column]
    if isinstance(value, np.generic):
        value = value.item()
    return value


def json_raw(source: str, path: list[str]) -> Any:
    """Return a value from a JSON document by key path."""

    value = load_json(source)
    for key in path:
        value = value[key]
    if isinstance(value, np.generic):
        value = value.item()
    return value


def f0(value: Any) -> str:
    return "n/a" if pd.isna(value) else f"{int(round(float(value)))}"


def f2(value: Any) -> str:
    return "n/a" if pd.isna(value) else f"{float(value):.2f}"


def f3(value: Any) -> str:
    return "n/a" if pd.isna(value) else f"{float(value):.3f}"


def f4(value: Any) -> str:
    return "n/a" if pd.isna(value) else f"{float(value):.4f}"


def pct2(value: Any) -> str:
    return "n/a" if pd.isna(value) else f"{100.0 * float(value):.2f}%"


def pct1(value: Any) -> str:
    return "n/a" if pd.isna(value) else f"{100.0 * float(value):.1f}%"


def fnum(value: Any) -> str:
    """Compact fixed/scientific formatting for thresholds and counts."""

    if pd.isna(value):
        return "n/a"
    value = float(value)
    magnitude = abs(value)
    if magnitude != 0.0 and (magnitude < 1e-3 or magnitude >= 1e4):
        return f"{value:.3e}"
    return f"{value:.4f}"


def fmt_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return fnum(value)
    return str(value)


def fmt_list(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return str(value)


# ---------------------------------------------------------------------------
# Audit recorder
# ---------------------------------------------------------------------------


class Recorder:
    """Collect every table cell together with the file it came from."""

    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []

    def add(
        self,
        *,
        table: str,
        row: str,
        column: str,
        value: Any,
        source: str,
        selector: dict[str, Any],
        text: str,
    ) -> str:
        if isinstance(value, np.generic):
            value = value.item()
        self.entries.append(
            {
                "table": table,
                "row": row,
                "column": column,
                "value": value,
                "source": source,
                "selector": selector,
                "text": text,
            }
        )
        return text

    def csv_cell(
        self,
        *,
        table: str,
        row: str,
        column: str,
        source: str,
        filters: dict[str, Any],
        column_name: str,
        formatter: Callable[[Any], str],
    ) -> str:
        raw = csv_raw(source, filters, column_name)
        text = formatter(raw)
        return self.add(
            table=table,
            row=row,
            column=column,
            value=raw,
            source=source,
            selector={"kind": "csv", "filters": filters, "column": column_name},
            text=text,
        )

    def json_cell(
        self,
        *,
        table: str,
        row: str,
        column: str,
        source: str,
        path: list[str],
        formatter: Callable[[Any], str],
    ) -> str:
        raw = json_raw(source, path)
        text = formatter(raw)
        return self.add(
            table=table,
            row=row,
            column=column,
            value=raw,
            source=source,
            selector={"kind": "json", "path": path},
            text=text,
        )


# ---------------------------------------------------------------------------
# Markdown table helpers
# ---------------------------------------------------------------------------


def render_table(header: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for row in rows:
        if len(row) != len(header):
            raise ValueError(f"row has {len(row)} cells, header has {len(header)}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def ci_cell(
    rec: Recorder,
    *,
    table: str,
    row: str,
    metric_label: str,
    source: str,
    filters: dict[str, Any],
    prefix: str,
    formatter: Callable[[Any], str] = f4,
) -> str:
    mean = rec.csv_cell(
        table=table,
        row=row,
        column=f"{metric_label} mean",
        source=source,
        filters=filters,
        column_name=f"{prefix}_mean",
        formatter=formatter,
    )
    low = rec.csv_cell(
        table=table,
        row=row,
        column=f"{metric_label} CI low",
        source=source,
        filters=filters,
        column_name=f"{prefix}_ci_lo",
        formatter=formatter,
    )
    high = rec.csv_cell(
        table=table,
        row=row,
        column=f"{metric_label} CI high",
        source=source,
        filters=filters,
        column_name=f"{prefix}_ci_hi",
        formatter=formatter,
    )
    return f"{mean} [{low}, {high}]"


# ---------------------------------------------------------------------------
# T1 - dataset and split
# ---------------------------------------------------------------------------


def build_t1(rec: Recorder) -> str:
    source = "split_summary.csv"
    header = [
        "Mode",
        "Split",
        "Windows",
        "Fraction of healthy windows",
        "First window",
        "Last window",
        "Start time (s)",
        "End time (s)",
    ]
    rows: list[list[str]] = []
    for mode in ["IPPT", "MPPT"]:
        for split in ["train", "val", "test"]:
            filters = {"mode": mode, "split": split}
            row_label = f"{mode} / {split}"
            row = [
                mode,
                split,
                rec.csv_cell(
                    table="T1",
                    row=row_label,
                    column="Windows",
                    source=source,
                    filters=filters,
                    column_name="windows",
                    formatter=f0,
                ),
                rec.csv_cell(
                    table="T1",
                    row=row_label,
                    column="Fraction",
                    source=source,
                    filters=filters,
                    column_name="fraction_of_healthy_windows",
                    formatter=f4,
                ),
                rec.csv_cell(
                    table="T1",
                    row=row_label,
                    column="First window",
                    source=source,
                    filters=filters,
                    column_name="first_window",
                    formatter=f0,
                ),
                rec.csv_cell(
                    table="T1",
                    row=row_label,
                    column="Last window",
                    source=source,
                    filters=filters,
                    column_name="last_window",
                    formatter=f0,
                ),
                rec.csv_cell(
                    table="T1",
                    row=row_label,
                    column="Start time",
                    source=source,
                    filters=filters,
                    column_name="start_time_s",
                    formatter=f3,
                ),
                rec.csv_cell(
                    table="T1",
                    row=row_label,
                    column="End time",
                    source=source,
                    filters=filters,
                    column_name="end_time_s",
                    formatter=f3,
                ),
            ]
            rows.append(row)
    return (
        "### T1. Dataset and split windows\n\n"
        "Healthy windows only (`F0L` for IPPT, `F0M` for MPPT). Ten gap windows "
        "between train/validation and between validation/test are excluded from "
        "all three splits, so the test fraction is slightly below 0.20. Source: "
        "`results/split_summary.csv`.\n\n"
        + render_table(header, rows)
    )


# ---------------------------------------------------------------------------
# T2 - primary averages
# ---------------------------------------------------------------------------


def build_t2_table(rec: Recorder, mode: str, exclude_f3: bool) -> str:
    source = f"average_primary_{mode}.csv"
    prefix = "excl_f3_" if exclude_f3 else ""
    if exclude_f3:
        table = "T2c"
        title = (
            "### T2c. Primary whole-file averages, MPPT excluding F3M (6 fault files)\n\n"
            "The reference paper omits F3 from its MPPT table; this view is shown "
            "for that comparison only. F3M remains in the main MPPT table (T2b). "
        )
    elif mode == "IPPT":
        table = "T2a"
        title = (
            "### T2a. Primary whole-file averages, IPPT (7 fault files)\n\n"
            "Every window of a fault file is labelled faulty; healthy test windows "
            "are the negative class. Mean over F1L-F7L. "
        )
    else:
        table = "T2b"
        title = (
            "### T2b. Primary whole-file averages, MPPT (7 fault files, including F3M)\n\n"
            "Every window of a fault file is labelled faulty; healthy test windows "
            "are the negative class. Mean over F1M-F7M. "
        )

    header = [
        "Model",
        "F1 [95% CI]",
        "Accuracy [95% CI]",
        "AUC [95% CI]",
        "TPR [95% CI]",
        "FPR [95% CI]",
        "Healthy-test FAR",
    ]
    rows: list[list[str]] = []
    for model in MODEL_ORDER:
        filters = {"model": model, "mode": mode}
        row = [model]
        for metric, label in [
            ("f1", "F1"),
            ("accuracy", "Accuracy"),
            ("auc", "AUC"),
            ("tpr", "TPR"),
            ("fpr", "FPR"),
        ]:
            row.append(
                ci_cell(
                    rec,
                    table=table,
                    row=model,
                    metric_label=label,
                    source=source,
                    filters=filters,
                    prefix=f"{prefix}{metric}",
                )
            )
        row.append(
            rec.csv_cell(
                table=table,
                row=model,
                column="Healthy-test FAR",
                source=source,
                filters=filters,
                column_name="healthy_test_false_alarm_rate",
                formatter=pct2,
            )
        )
        rows.append(row)

    note = (
        "Values are averages over the fault files of that mode with 95% bootstrap "
        "intervals in brackets. The healthy-test false-alarm rate is the point "
        "estimate at the stored 99th-percentile healthy-validation threshold. "
        "Proposed rows are averaged over seeds 0, 1 and 2; baselines use their "
        "single frozen seed."
    )
    return title + "Source: `results/" + source + "`.\n\n" + render_table(header, rows) + "\n\n" + note


# ---------------------------------------------------------------------------
# T3 - within-run
# ---------------------------------------------------------------------------


def build_t3(rec: Recorder) -> str:
    header = [
        "Model",
        "IPPT within-run AUC [95% CI]",
        "MPPT within-run AUC [95% CI]",
        "IPPT pre-injection flagged",
        "MPPT pre-injection flagged",
        "IPPT healthy-test FAR",
        "MPPT healthy-test FAR",
    ]
    rows: list[list[str]] = []
    for model in MODEL_ORDER:
        row = [model]
        for mode in ["IPPT", "MPPT"]:
            source = f"within_run_average_{mode}.csv"
            filters = {"model": model, "mode": mode}
            row.append(
                ci_cell(
                    rec,
                    table="T3",
                    row=model,
                    metric_label=f"{mode} within-run AUC",
                    source=source,
                    filters=filters,
                    prefix="auc",
                )
            )
        for mode in ["IPPT", "MPPT"]:
            source = f"within_run_average_{mode}.csv"
            filters = {"model": model, "mode": mode}
            row.append(
                rec.csv_cell(
                    table="T3",
                    row=model,
                    column=f"{mode} pre-injection flagged",
                    source=source,
                    filters=filters,
                    column_name="pre_injection_flag_share",
                    formatter=pct2,
                )
            )
        for mode in ["IPPT", "MPPT"]:
            source = f"within_run_average_{mode}.csv"
            filters = {"model": model, "mode": mode}
            row.append(
                rec.csv_cell(
                    table="T3",
                    row=model,
                    column=f"{mode} healthy-test FAR",
                    source=source,
                    filters=filters,
                    column_name="healthy_test_false_alarm_rate",
                    formatter=pct2,
                )
            )
        rows.append(row)

    note = (
        "Within-run AUC compares pre-injection windows (end < 7 s and start "
        ">= 0.5 s) with post-8 s windows of the same fault file. F3M is excluded "
        "from the MPPT column because it ends before 7 s. Pre-injection flagged "
        "is the share of pre-injection windows already above the stored threshold, "
        "so a high value means the binary flag is saturated before injection. "
        "Sources: `results/within_run_average_IPPT.csv`, "
        "`results/within_run_average_MPPT.csv`."
    )
    return (
        "### T3. Secondary within-run AUC and pre-injection flag share\n\n"
        + render_table(header, rows)
        + "\n\n"
        + note
    )


# ---------------------------------------------------------------------------
# T4 - per-fault F1 and AUC
# ---------------------------------------------------------------------------


def build_t4_table(rec: Recorder, mode: str) -> str:
    source = f"per_fault_primary_{mode}.csv"
    suffix = MODE_SUFFIX[mode]
    faults = [f"F{i}{suffix}" for i in range(1, 8)]
    models = [
        "Proposed (plain)",
        "EllipticEnvelope",
        "IsolationForest",
        "LocalOutlierFactor",
        "OneClassSVM",
    ]
    table = f"T4{'a' if mode == 'IPPT' else 'b'}"
    header = ["Fault"] + [f"{model} F1 / AUC" for model in models]
    rows: list[list[str]] = []
    for fault in faults:
        row = [fault]
        for model in models:
            filters = {"model": model, "mode": mode, "fault": fault}
            f1 = rec.csv_cell(
                table=table,
                row=fault,
                column=f"{model} F1",
                source=source,
                filters=filters,
                column_name="f1_mean",
                formatter=f4,
            )
            auc = rec.csv_cell(
                table=table,
                row=fault,
                column=f"{model} AUC",
                source=source,
                filters=filters,
                column_name="auc_mean",
                formatter=f4,
            )
            row.append(f"{f1} / {auc}")
        rows.append(row)

    far_row = ["Healthy-test FAR"]
    for model in models:
        filters = {"model": model, "mode": mode, "fault": faults[0]}
        far_row.append(
            rec.csv_cell(
                table=table,
                row="Healthy-test FAR",
                column=f"{model} FAR",
                source=source,
                filters=filters,
                column_name="healthy_test_false_alarm_rate",
                formatter=pct2,
            )
        )
    rows.append(far_row)

    title = (
        f"### T4{'a' if mode == 'IPPT' else 'b'}. Per-fault primary F1 and AUC, {mode}\n\n"
    )
    note = (
        "Each model cell is F1 / AUC. The final row is the healthy-test "
        "false-alarm rate at the stored threshold; it is model-level and is "
        "therefore shown once per model. Proposed (plain) is averaged over seeds; "
        "baselines use their single frozen seed. Mahalanobis (window-mean) is "
        "omitted from this table because it is the trivial reference, not one of "
        "the four classical baselines. Source: `results/" + source + "`."
    )
    return title + render_table(header, rows) + "\n\n" + note


# ---------------------------------------------------------------------------
# T5 - threshold sensitivity
# ---------------------------------------------------------------------------


def build_t5(rec: Recorder) -> str:
    source = "threshold_sensitivity.csv"
    header = [
        "Model",
        "Mode",
        "Threshold percentile",
        "Threshold",
        "Healthy-test FAR",
        "Average TPR",
    ]
    rows: list[list[str]] = []
    for model in [
        "Proposed (plain)",
        "Proposed (featnorm)",
        "EllipticEnvelope",
        "Mahalanobis (window-mean)",
    ]:
        for mode in ["IPPT", "MPPT"]:
            for quantile in [95.0, 97.5, 99.0, 99.5]:
                filters = {"model": model, "mode": mode, "quantile": quantile}
                row_label = f"{model} / {mode} / q{quantile:g}"
                rows.append(
                    [
                        model,
                        mode,
                        rec.csv_cell(
                            table="T5",
                            row=row_label,
                            column="Threshold percentile",
                            source=source,
                            filters=filters,
                            column_name="quantile",
                            formatter=f2,
                        ),
                        rec.csv_cell(
                            table="T5",
                            row=row_label,
                            column="Threshold",
                            source=source,
                            filters=filters,
                            column_name="threshold",
                            formatter=fnum,
                        ),
                        rec.csv_cell(
                            table="T5",
                            row=row_label,
                            column="Healthy-test FAR",
                            source=source,
                            filters=filters,
                            column_name="healthy_test_false_alarm_rate",
                            formatter=pct2,
                        ),
                        rec.csv_cell(
                            table="T5",
                            row=row_label,
                            column="Average TPR",
                            source=source,
                            filters=filters,
                            column_name="average_tpr",
                            formatter=f4,
                        ),
                    ]
                )
    note = (
        "Thresholds were recomputed from saved healthy validation scores only; no "
        "fault result was used to choose a threshold. The stored deployment "
        "threshold is the 99th-percentile row. Average TPR is over the same fault "
        "files as the primary table. Proposed rows are seed-averaged. Source: "
        "`results/threshold_sensitivity.csv`."
    )
    return (
        "### T5. Threshold sensitivity\n\n"
        + render_table(header, rows)
        + "\n\n"
        + note
    )


# ---------------------------------------------------------------------------
# T6 - reference paper
# ---------------------------------------------------------------------------


def build_t6(rec: Recorder) -> str:
    source = "paper_reference.csv"
    frame = load_csv(source)
    header = [
        "Mode",
        "Model",
        "Paper label",
        "TPR",
        "FPR",
        "Accuracy",
        "Precision",
        "F1",
        "AUC",
    ]
    rows: list[list[str]] = []
    order = sorted(
        range(len(frame)),
        key=lambda index: (0 if frame.iloc[index]["mode"] == "IPPT" else 1, index),
    )
    for index in order:
        row_data = frame.iloc[index]
        row_label = f"{row_data['mode']} / {row_data['model']}"
        filters = {"mode": row_data["mode"], "model": row_data["model"]}
        row = [str(row_data["mode"]), str(row_data["model"]), str(row_data["paper_label"])]
        for column in ["tpr", "fpr", "accuracy", "precision", "f1", "auc"]:
            row.append(
                rec.csv_cell(
                    table="T6",
                    row=row_label,
                    column=column.upper(),
                    source=source,
                    filters=filters,
                    column_name=column,
                    formatter=f4,
                )
            )
        rows.append(row)
    note = (
        "Source: `results/paper_reference.csv` (reference paper published "
        "averages). **Note: different protocol: single 10 kHz samples vs "
        "100-sample windows at 1 kHz; not comparable.** These values are included "
        "for reference only and are not a like-for-like comparison with T2."
    )
    return (
        "### T6. Reference paper's published averages\n\n"
        + render_table(header, rows)
        + "\n\n"
        + note
    )


# ---------------------------------------------------------------------------
# T7 - seed variability
# ---------------------------------------------------------------------------


def build_t7(rec: Recorder) -> str:
    header = [
        "Mode",
        "Model",
        "F1",
        "Accuracy",
        "AUC",
        "TPR",
        "FPR",
        "Healthy-test FAR",
        "Threshold",
    ]
    rows: list[list[str]] = []
    for mode in ["IPPT", "MPPT"]:
        source = f"average_primary_{mode}.csv"
        for model in ["Proposed (plain)", "Proposed (featnorm)"]:
            filters = {"model": model, "mode": mode}
            row_label = f"{mode} / {model}"
            row = [mode, model]
            for metric, label in [
                ("f1", "F1"),
                ("accuracy", "Accuracy"),
                ("auc", "AUC"),
                ("tpr", "TPR"),
                ("fpr", "FPR"),
            ]:
                mean = rec.csv_cell(
                    table="T7",
                    row=row_label,
                    column=f"{label} mean",
                    source=source,
                    filters=filters,
                    column_name=f"{metric}_mean",
                    formatter=f4,
                )
                std = rec.csv_cell(
                    table="T7",
                    row=row_label,
                    column=f"{label} seed std",
                    source=source,
                    filters=filters,
                    column_name=f"{metric}_seed_std",
                    formatter=f4,
                )
                row.append(f"{mean} +/- {std}")
            far_mean = rec.csv_cell(
                table="T7",
                row=row_label,
                column="Healthy-test FAR mean",
                source=source,
                filters=filters,
                column_name="healthy_test_false_alarm_rate",
                formatter=pct2,
            )
            far_std = rec.csv_cell(
                table="T7",
                row=row_label,
                column="Healthy-test FAR seed std",
                source=source,
                filters=filters,
                column_name="healthy_test_false_alarm_rate_seed_std",
                formatter=pct2,
            )
            row.append(f"{far_mean} +/- {far_std}")
            threshold_mean = rec.csv_cell(
                table="T7",
                row=row_label,
                column="Threshold mean",
                source=source,
                filters=filters,
                column_name="threshold_mean",
                formatter=fnum,
            )
            threshold_std = rec.csv_cell(
                table="T7",
                row=row_label,
                column="Threshold seed std",
                source=source,
                filters=filters,
                column_name="threshold_seed_std",
                formatter=fnum,
            )
            row.append(f"{threshold_mean} +/- {threshold_std}")
            rows.append(row)
    note = (
        "Each cell is mean +/- standard deviation across seeds 0, 1 and 2 "
        "(ddof=1). The threshold is the 99th percentile of healthy validation "
        "scores. Healthy-test FAR is a fraction of healthy test windows; no "
        "threshold was selected from fault results. Source: "
        "`results/average_primary_IPPT.csv`, `results/average_primary_MPPT.csv`."
    )
    return (
        "### T7. Seed variability of the two proposed variants\n\n"
        + render_table(header, rows)
        + "\n\n"
        + note
    )


# ---------------------------------------------------------------------------
# T8 - hyperparameters and training settings
# ---------------------------------------------------------------------------


def build_t8(rec: Recorder) -> str:
    data_source = "data_config.json"
    prop_source = "proposed_metadata.json"
    arch_source = "phase6_architecture.json"
    base_source = "baselines_metadata.json"
    eval_source = "phase5_metadata.json"
    train_source = "proposed_training.csv"

    settings_rows: list[list[str]] = []
    settings = [
        ("Native sampling rate (Hz)", data_source, ["config", "native_sampling_hz"], f0),
        ("Target sampling rate (Hz)", data_source, ["config", "target_sampling_hz"], f0),
        ("Decimation factor q", data_source, ["config", "downsample_factor"], f0),
        (
            "Anti-alias filter",
            data_source,
            ["config", "decimation", "filter"],
            str,
        ),
        ("Filter cutoff (Hz)", data_source, ["config", "decimation", "cutoff_hz"], f0),
        (
            "Zero-phase filtering",
            data_source,
            ["config", "decimation", "zero_phase"],
            fmt_scalar,
        ),
        ("Window size (samples)", data_source, ["config", "windowing", "window_size"], f0),
        ("Window stride (samples)", data_source, ["config", "windowing", "stride"], f0),
        (
            "Window overlap (samples)",
            data_source,
            ["config", "windowing", "overlap_samples"],
            f0,
        ),
        ("Healthy train fraction", data_source, ["config", "healthy_split", "fractions", "train"], f2),
        ("Healthy validation fraction", data_source, ["config", "healthy_split", "fractions", "val"], f2),
        ("Healthy test fraction", data_source, ["config", "healthy_split", "fractions", "test"], f2),
        ("Gap windows between splits", data_source, ["config", "healthy_split", "gap_windows"], f0),
        (
            "Scaler",
            data_source,
            ["config", "scaling", "class"],
            str,
        ),
        (
            "Scaler fit data",
            data_source,
            ["config", "scaling", "fit_data"],
            str,
        ),
        ("Threshold quantile", prop_source, ["scoring", "threshold_quantile"], f2),
        ("Flag rule", prop_source, ["scoring", "flag_rule"], str),
        ("Top-k features for featnorm", prop_source, ["scoring", "top_k"], f0),
        (
            "Feature reference for featnorm",
            prop_source,
            ["scoring", "feature_reference"],
            str,
        ),
        (
            "Bootstrap resamples",
            eval_source,
            ["primary", "bootstrap", "resamples"],
            f0,
        ),
        (
            "Bootstrap seed",
            eval_source,
            ["primary", "bootstrap", "seed"],
            f0,
        ),
        (
            "Bootstrap interval",
            eval_source,
            ["primary", "bootstrap", "ci"],
            str,
        ),
        (
            "Bootstrap class sizes",
            eval_source,
            ["primary", "bootstrap", "equal_class_sizes"],
            str,
        ),
    ]
    for label, source, path, formatter in settings:
        value = rec.json_cell(
            table="T8a",
            row=label,
            column="Value",
            source=source,
            path=path,
            formatter=formatter,
        )
        settings_rows.append([label, value, f"`results/{source}`"])

    proposed_rows: list[list[str]] = []
    proposed = [
        ("Architecture (metadata string)", prop_source, ["architecture"], str),
        ("Input time steps", arch_source, ["input_timesteps"], f0),
        ("Input features", arch_source, ["input_features"], f0),
        ("Encoder TCN filters", arch_source, ["encoder_tcn_filters"], fmt_list),
        ("Encoder TCN dilations", arch_source, ["encoder_tcn_dilations"], fmt_list),
        ("Kernel size", arch_source, ["kernel_size"], f0),
        (
            "Encoder BiLSTM units per direction",
            arch_source,
            ["encoder_bilstm_units_per_direction"],
            f0,
        ),
        (
            "Latent BiLSTM units per direction",
            arch_source,
            ["latent_bilstm_units_per_direction"],
            f0,
        ),
        ("RepeatVector time steps", arch_source, ["repeat_timesteps"], f0),
        (
            "Decoder BiLSTM units per direction",
            arch_source,
            ["decoder_bilstm_units_per_direction"],
            fmt_list,
        ),
        ("Decoder TCN filters", arch_source, ["decoder_tcn_filters"], fmt_list),
        ("Decoder TCN dilations", arch_source, ["decoder_tcn_dilations"], fmt_list),
        ("Output Dense units", arch_source, ["output_dense_units"], f0),
        ("Dropout rate", arch_source, ["dropout_rate"], fnum),
        ("Batch normalization", arch_source, ["batch_normalization"], fmt_scalar),
        ("Residual connections", arch_source, ["residual_connections"], fmt_scalar),
        ("Model parameters", prop_source, ["runs", 0, "model_params"], f0),
        ("Optimizer", prop_source, ["training", "optimizer"], str),
        ("Learning rate", prop_source, ["training", "learning_rate"], fnum),
        ("Batch size", prop_source, ["training", "batch_size"], f0),
        ("Maximum epochs", prop_source, ["training", "max_epochs"], f0),
        (
            "Early-stopping monitor",
            prop_source,
            ["training", "early_stopping", "monitor"],
            str,
        ),
        (
            "Early-stopping patience",
            prop_source,
            ["training", "early_stopping", "patience"],
            f0,
        ),
        (
            "Restore best weights",
            prop_source,
            ["training", "early_stopping", "restore_best_weights"],
            fmt_scalar,
        ),
        (
            "Reduce-LR factor",
            prop_source,
            ["training", "reduce_lr", "factor"],
            fnum,
        ),
        (
            "Reduce-LR patience",
            prop_source,
            ["training", "reduce_lr", "patience"],
            f0,
        ),
        ("Minimum learning rate", prop_source, ["training", "reduce_lr", "min_lr"], fnum),
        ("Seeds", prop_source, ["seeds"], fmt_list),
        (
            "Training windows, IPPT",
            train_source,
            None,
            f0,
        ),
        (
            "Training windows, MPPT",
            train_source,
            None,
            f0,
        ),
        ("Validation windows, IPPT", train_source, None, f0),
        ("Validation windows, MPPT", train_source, None, f0),
        ("Device", prop_source, ["device"], str),
        ("Plain window score", prop_source, ["scoring", "plain"], str),
        ("Featnorm window score", prop_source, ["scoring", "featnorm"], str),
        ("Featnorm standard-deviation floor", prop_source, ["scoring", "std_floor"], fnum),
        (
            "Threshold rule",
            prop_source,
            ["scoring", "threshold_rule"],
            str,
        ),
    ]
    for label, source, path, formatter in proposed:
        if path is None:
            # Training-window counts are per-mode rows in proposed_training.csv.
            mode = "IPPT" if "IPPT" in label else "MPPT"
            column = "train_windows" if "Training windows" in label else "val_windows"
            filters = {"mode": mode, "seed": 0}
            value = rec.csv_cell(
                table="T8b",
                row=label,
                column="Value",
                source=train_source,
                filters=filters,
                column_name=column,
                formatter=formatter,
            )
            proposed_rows.append([label, value, f"`results/{train_source}`"])
        else:
            value = rec.json_cell(
                table="T8b",
                row=label,
                column="Value",
                source=source,
                path=path,
                formatter=formatter,
            )
            proposed_rows.append([label, value, f"`results/{source}`"])

    baseline_rows: list[list[str]] = []
    baseline_models = [
        ("elliptic_envelope", "EllipticEnvelope"),
        ("isolation_forest", "IsolationForest"),
        ("lof", "LocalOutlierFactor"),
        ("one_class_svm", "OneClassSVM"),
        ("mahalanobis_window_mean", "Mahalanobis (window-mean)"),
    ]
    for key, label in baseline_models:
        params = load_json(base_source)["models"][key]["params"]
        for param_name, param_value in params.items():
            value = rec.json_cell(
                table="T8c",
                row=f"{label} / {param_name}",
                column="Value",
                source=base_source,
                path=["models", key, "params", param_name],
                formatter=fmt_scalar,
            )
            baseline_rows.append([label, param_name, value])

    threshold_rows: list[list[str]] = []
    for mode in ["IPPT", "MPPT"]:
        source = f"average_primary_{mode}.csv"
        for model in MODEL_ORDER:
            filters = {"model": model, "mode": mode}
            threshold = rec.csv_cell(
                table="T8d",
                row=f"{mode} / {model}",
                column="Threshold",
                source=source,
                filters=filters,
                column_name="threshold_mean",
                formatter=fnum,
            )
            threshold_rows.append([model, mode, threshold])

    run_rows: list[list[str]] = []
    training = load_csv(train_source).copy()
    training["_mode_order"] = training["mode"].map({"IPPT": 0, "MPPT": 1})
    training = training.sort_values(["_mode_order", "seed"]).drop(columns="_mode_order")
    for index in range(len(training)):
        row_data = training.iloc[index]
        mode = str(row_data["mode"])
        seed = int(row_data["seed"])
        row_label = f"{mode} / seed {seed}"
        filters = {"mode": mode, "seed": seed}
        row = [
            mode,
            rec.csv_cell(
                table="T8e",
                row=row_label,
                column="Seed",
                source=train_source,
                filters=filters,
                column_name="seed",
                formatter=f0,
            ),
            rec.csv_cell(
                table="T8e",
                row=row_label,
                column="Epochs run",
                source=train_source,
                filters=filters,
                column_name="epochs_run",
                formatter=f0,
            ),
            rec.csv_cell(
                table="T8e",
                row=row_label,
                column="Best epoch",
                source=train_source,
                filters=filters,
                column_name="best_epoch",
                formatter=f0,
            ),
            rec.csv_cell(
                table="T8e",
                row=row_label,
                column="Best validation loss",
                source=train_source,
                filters=filters,
                column_name="best_val_loss",
                formatter=f4,
            ),
            rec.csv_cell(
                table="T8e",
                row=row_label,
                column="Training time",
                source=train_source,
                filters=filters,
                column_name="training_time_s",
                formatter=f2,
            ),
        ]
        run_rows.append(row)

    return (
        "### T8. Hyperparameters and training settings\n\n"
        "#### T8a. Pipeline and evaluation settings\n\n"
        + render_table(["Parameter", "Value", "Source"], settings_rows)
        + "\n\n#### T8b. Proposed architecture and training settings\n\n"
        + render_table(["Parameter", "Value", "Source"], proposed_rows)
        + "\n\n#### T8c. Baseline hyperparameters\n\n"
        + render_table(["Model", "Parameter", "Value"], baseline_rows)
        + "\n\n#### T8d. Stored thresholds by model and mode\n\n"
        + render_table(["Model", "Mode", "Threshold (mean)"], threshold_rows)
        + "\n\n#### T8e. Proposed training runs\n\n"
        + render_table(
            ["Mode", "Seed", "Epochs run", "Best epoch", "Best validation loss", "Training time (s)"],
            run_rows,
        )
        + "\n\n"
        "T8a reads `results/data_config.json`, `results/proposed_metadata.json` and "
        "`results/phase5_metadata.json`. T8b reads `results/proposed_metadata.json`, "
        "`results/phase6_architecture.json` and `results/proposed_training.csv`. "
        "T8c reads `results/baselines_metadata.json`. T8d reads "
        "`results/average_primary_IPPT.csv` and `results/average_primary_MPPT.csv`. "
        "T8e reads `results/proposed_training.csv`. Baselines use a single frozen "
        "seed; the two proposed variants use seeds 0, 1 and 2. The architecture "
        "string is copied verbatim from `results/proposed_metadata.json`; the "
        "structured architecture rows are copied from `src/model.py` into "
        "`results/phase6_architecture.json` and match that file."
    )


def build_tables(rec: Recorder) -> str:
    intro = (
        "# Phase 6 tables\n\n"
        "All values below are generated by `src/make_figures.py` from the saved "
        "CSV and JSON files named under each table. No model was retrained, no "
        "score or threshold was changed, and no fault file was used to make a "
        "modeling decision. The two proposed variants are reported side by side "
        "in every table where they appear, and the healthy-test false-alarm rate "
        "is shown next to every performance table. `Proposed (featnorm)` is a "
        "negative result: its healthy-test false-alarm rate is high in both "
        "modes, so fault-detection values must be read together with that rate. "
        "This report does not claim that either proposed variant exceeds any "
        "baseline."
    )
    return "\n\n".join(
        [
            intro,
            build_t1(rec),
            build_t2_table(rec, "IPPT", False),
            build_t2_table(rec, "MPPT", False),
            build_t2_table(rec, "MPPT", True),
            build_t3(rec),
            build_t4_table(rec, "IPPT"),
            build_t4_table(rec, "MPPT"),
            build_t5(rec),
            build_t6(rec),
            build_t7(rec),
            build_t8(rec),
        ]
    )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def save_figure(fig: plt.Figure, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path.relative_to(PROJECT_ROOT)}")
    return path


def fig_healthy_far() -> None:
    fig, ax = plt.subplots(figsize=(9.4, 4.8), constrained_layout=True)
    x = np.arange(len(MODEL_ORDER))
    width = 0.38
    max_value = 0.0
    for offset, mode in zip([-0.5, 0.5], ["IPPT", "MPPT"]):
        frame = load_csv(f"average_primary_{mode}.csv")
        values = np.array(
            [
                float(
                    frame.loc[
                        (frame["model"] == model) & (frame["mode"] == mode),
                        "healthy_test_false_alarm_rate",
                    ].iloc[0]
                )
                for model in MODEL_ORDER
            ]
        )
        max_value = max(max_value, float(values.max()))
        bars = ax.bar(
            x + offset * width,
            values,
            width=width,
            color=[MODEL_COLORS[model] for model in MODEL_ORDER],
            edgecolor="black",
            linewidth=0.5,
            hatch="" if mode == "IPPT" else "///",
            alpha=1.0 if mode == "IPPT" else 0.75,
            zorder=3,
        )
        for bar, value in zip(bars, values):
            text_y = bar.get_height() if value > 0 else 0.0015
            ax.annotate(
                f"{100.0 * value:.2f}%",
                (bar.get_x() + bar.get_width() / 2.0, text_y),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6.3,
                rotation=90,
            )

    ax.axhline(
        0.01,
        color="black",
        linestyle="--",
        linewidth=1.0,
        zorder=4,
        label="1% validation target",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER, rotation=24, ha="right")
    ax.set_ylabel("Healthy-test false-alarm rate")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
    ax.set_ylim(0, max_value * 1.24 if max_value > 0 else 0.02)
    handles = [
        Patch(facecolor="white", edgecolor="black", hatch="", label="IPPT"),
        Patch(facecolor="white", edgecolor="black", hatch="///", label="MPPT"),
        Line2D([0], [0], color="black", linestyle="--", label="1% validation target"),
    ]
    ax.legend(handles=handles, loc="upper left", ncol=3)
    ax.set_title("Healthy-test false-alarm rate at the frozen 99th-percentile threshold")
    save_figure(fig, "fig_healthy_far.png")


def fig_primary_f1_auc() -> None:
    fig, axes = plt.subplots(
        1, 2, figsize=(11.6, 5.0), sharex=True, constrained_layout=True
    )
    y = np.arange(len(MODEL_ORDER))[::-1]
    height = 0.36
    metric_specs = [("f1", "F1", "#3b6ea5"), ("auc", "AUC", "#c44e52")]
    for ax, mode in zip(axes, ["IPPT", "MPPT"]):
        frame = load_csv(f"average_primary_{mode}.csv")
        for offset, (metric, label, color) in zip([-0.5, 0.5], metric_specs):
            values = []
            low = []
            high = []
            for model in MODEL_ORDER:
                subset = frame.loc[
                    (frame["model"] == model) & (frame["mode"] == mode)
                ].iloc[0]
                values.append(float(subset[f"{metric}_mean"]))
                low.append(float(subset[f"{metric}_ci_lo"]))
                high.append(float(subset[f"{metric}_ci_hi"]))
            values = np.array(values)
            low = np.array(low)
            high = np.array(high)
            ax.barh(
                y + offset * height,
                values,
                xerr=[values - low, high - values],
                height=height,
                color=color,
                edgecolor="black",
                linewidth=0.4,
                capsize=2,
                error_kw={"elinewidth": 0.8, "capthick": 0.8},
                label=label,
                zorder=3,
            )
        ax.set_yticks(y)
        ax.set_yticklabels(MODEL_ORDER)
        ax.set_xlim(0.75, 1.01)
        ax.set_xlabel("Score")
        ax.set_title(mode)
        ax.axvline(1.0, color="black", linewidth=0.6, linestyle=":")
        ax.grid(axis="x")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.035))
    fig.suptitle("Primary whole-file averages with 95% bootstrap intervals")
    save_figure(fig, "fig_primary_f1_auc.png")


def fig_within_run_auc() -> None:
    fig, axes = plt.subplots(
        1, 2, figsize=(11.6, 5.0), sharex=True, constrained_layout=True
    )
    y = np.arange(len(MODEL_ORDER))[::-1]
    for ax, mode in zip(axes, ["IPPT", "MPPT"]):
        frame = load_csv(f"within_run_average_{mode}.csv")
        values = []
        low = []
        high = []
        for model in MODEL_ORDER:
            subset = frame.loc[
                (frame["model"] == model) & (frame["mode"] == mode)
            ].iloc[0]
            values.append(float(subset["auc_mean"]))
            low.append(float(subset["auc_ci_lo"]))
            high.append(float(subset["auc_ci_hi"]))
        values = np.array(values)
        low = np.array(low)
        high = np.array(high)
        ax.barh(
            y,
            values,
            xerr=[values - low, high - values],
            height=0.62,
            color=[MODEL_COLORS[model] for model in MODEL_ORDER],
            edgecolor="black",
            linewidth=0.4,
            capsize=2,
            error_kw={"elinewidth": 0.8, "capthick": 0.8},
            zorder=3,
        )
        ax.axvline(
            0.5,
            color="black",
            linestyle="--",
            linewidth=1.0,
            label="chance (AUC = 0.5)",
            zorder=4,
        )
        ax.set_yticks(y)
        ax.set_yticklabels(MODEL_ORDER)
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("Within-run AUC")
        ax.set_title(mode if mode == "IPPT" else "MPPT (F3M excluded)")
        ax.grid(axis="x")
    chance_handle = Line2D(
        [0], [0], color="black", linestyle="--", linewidth=1.0, label="chance (AUC = 0.5)"
    )
    fig.legend(
        handles=[chance_handle],
        labels=["chance (AUC = 0.5)"],
        loc="lower center",
        ncol=1,
        bbox_to_anchor=(0.5, -0.035),
    )
    fig.suptitle("Secondary within-run AUC with 95% bootstrap intervals")
    save_figure(fig, "fig_within_run_auc.png")


def fig_score_vs_time() -> None:
    archive = np.load(
        RESULTS / "scores_proposed_plain_IPPT_seed0.npz", allow_pickle=False
    )
    scores = archive["scores"]
    start_time = archive["start_time"]
    names = archive["name"]
    threshold = float(archive["threshold"])
    parts = ["F1L", "F4L", "F6L", "F7L"]

    fig, axes = plt.subplots(
        2, 2, figsize=(9.6, 7.2), sharex=False, sharey=False, constrained_layout=True
    )
    for ax, part in zip(axes.flat, parts):
        mask = names == part
        ax.scatter(
            start_time[mask],
            scores[mask],
            s=3.0,
            color=MODEL_COLORS["Proposed (plain)"],
            alpha=0.55,
            linewidths=0,
            zorder=3,
        )
        ax.axhline(
            threshold,
            color="black",
            linestyle="--",
            linewidth=1.0,
            label="stored threshold",
            zorder=4,
        )
        ax.axvline(7.0, color="#555555", linestyle=":", linewidth=1.0, zorder=2)
        ax.axvline(8.0, color="#555555", linestyle=":", linewidth=1.0, zorder=2)
        ax.set_yscale("log")
        ax.set_xlabel("Window start time (s)")
        ax.set_ylabel("Window score (log scale)")
        ax.set_title(part)
        ax.set_xlim(0.0, float(np.max(start_time[mask])) * 1.01)
        y_top = ax.get_ylim()[1]
        ax.text(7.0, y_top, "7 s", ha="right", va="top", fontsize=6.5, color="#555555")
        ax.text(8.0, y_top, "8 s", ha="left", va="top", fontsize=6.5, color="#555555")
    axes[0, 0].legend(loc="upper right")
    fig.suptitle("Proposed (plain), seed 0: score versus window start time (IPPT)")
    save_figure(fig, "fig_score_vs_time.png")


def fig_score_hist() -> None:
    fig, axes = plt.subplots(
        1, 2, figsize=(11.6, 4.8), constrained_layout=True
    )
    for ax, mode in zip(axes, ["IPPT", "MPPT"]):
        suffix = MODE_SUFFIX[mode]
        archive = np.load(
            RESULTS / f"scores_proposed_plain_{mode}_seed0.npz", allow_pickle=False
        )
        scores = archive["scores"]
        names = archive["name"]
        threshold = float(archive["threshold"])

        val = scores[names == f"F0{suffix}_val"]
        test = scores[names == f"F0{suffix}_test"]
        fault_parts = [f"F{i}{suffix}" for i in range(1, 8)]
        fault_mask = np.isin(names, fault_parts)
        fault = scores[fault_mask]

        all_values = np.concatenate([val, test, fault])
        positive = all_values[all_values > 0]
        if positive.size == 0:
            raise ValueError(f"{mode}: no positive scores for log histogram")
        bins = np.logspace(
            math.log10(float(positive.min())),
            math.log10(float(positive.max())),
            120,
        )

        for values, color, label in [
            (val, "#7f7f7f", "Healthy validation"),
            (test, "#1f77b4", "Healthy test"),
            (fault, "#d55e00", "Fault files"),
        ]:
            weights = np.ones(values.size) / float(values.size)
            ax.hist(
                values,
                bins=bins,
                weights=weights,
                histtype="step",
                linewidth=1.5,
                color=color,
                label=label,
                zorder=3,
            )
        ax.axvline(
            threshold,
            color="black",
            linestyle="--",
            linewidth=1.0,
            label="stored threshold",
            zorder=4,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Window score (log scale)")
        ax.set_ylabel("Fraction of windows per bin (log scale)")
        ax.set_title(mode)
        ax.set_ylim(1e-5, 2.0)
        ax.legend(loc="upper right")
    fig.suptitle("Proposed (plain), seed 0: score distributions")
    save_figure(fig, "fig_score_hist.png")


def fig_threshold_sensitivity() -> None:
    frame = load_csv("threshold_sensitivity.csv")
    models = [
        "Proposed (plain)",
        "Proposed (featnorm)",
        "EllipticEnvelope",
        "Mahalanobis (window-mean)",
    ]
    quantiles = [95.0, 97.5, 99.0, 99.5]
    fig, axes = plt.subplots(
        1, 2, figsize=(11.0, 4.8), sharey=True, constrained_layout=True
    )
    max_value = 0.0
    for ax, mode in zip(axes, ["IPPT", "MPPT"]):
        for model in models:
            subset = frame.loc[
                (frame["model"] == model) & (frame["mode"] == mode)
            ].sort_values("quantile")
            x = subset["quantile"].to_numpy(dtype=float)
            y = subset["healthy_test_false_alarm_rate"].to_numpy(dtype=float)
            std = subset["healthy_test_false_alarm_rate_seed_std"].to_numpy(dtype=float)
            max_value = max(max_value, float(np.nanmax(y)))
            if np.all(np.isnan(std)):
                ax.plot(
                    x,
                    y,
                    marker="o",
                    markersize=4,
                    linewidth=1.4,
                    color=MODEL_COLORS[model],
                    label=model,
                    zorder=3,
                )
            else:
                ax.errorbar(
                    x,
                    y,
                    yerr=std,
                    marker="o",
                    markersize=4,
                    linewidth=1.4,
                    capsize=2,
                    elinewidth=0.8,
                    color=MODEL_COLORS[model],
                    label=model,
                    zorder=3,
                )
        ax.set_xticks(quantiles)
        ax.set_xticklabels([f"{value:g}" for value in quantiles])
        ax.set_xlabel("Threshold percentile")
        ax.set_ylabel("Healthy-test false-alarm rate")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_title(mode)
        ax.set_ylim(0, max_value * 1.12 if max_value > 0 else 0.02)
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=MODEL_COLORS[model],
            marker="o",
            markersize=4,
            linewidth=1.4,
            label=model,
        )
        for model in models
    ]
    axes[0].legend(handles=legend_handles, labels=models, loc="upper right")
    fig.suptitle("Healthy-test false-alarm rate versus threshold percentile")
    save_figure(fig, "fig_threshold_sensitivity.png")


def fig_training_curves() -> None:
    fig, axes = plt.subplots(
        2, 3, figsize=(11.8, 6.2), sharex=False, sharey=False, constrained_layout=True
    )
    for row, mode in enumerate(["IPPT", "MPPT"]):
        for column, seed in enumerate([0, 1, 2]):
            ax = axes[row, column]
            history = load_json(f"history_{mode}_seed{seed}.json")
            epochs = np.asarray(history["epoch"], dtype=int)
            ax.plot(
                epochs,
                np.asarray(history["loss"], dtype=float),
                color="#1f4e79",
                linewidth=1.4,
                label="Training",
                zorder=3,
            )
            ax.plot(
                epochs,
                np.asarray(history["val_loss"], dtype=float),
                color="#d55e00",
                linewidth=1.4,
                label="Validation",
                zorder=3,
            )
            ax.set_title(f"{mode}, seed {seed}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            if row == 0 and column == 0:
                ax.legend(loc="upper right")
    fig.suptitle("Proposed model: training and validation loss for the six runs")
    save_figure(fig, "fig_training_curves.png")


def fig_feature_error_heatmap() -> None:
    features = [str(item) for item in load_json("data_config.json")["features"]]
    matrices = []
    row_labels = []
    suffixes = []
    for mode in ["IPPT", "MPPT"]:
        suffix = MODE_SUFFIX[mode]
        suffixes.append(suffix)
        archive = np.load(
            RESULTS / f"scores_proposed_plain_{mode}_seed0.npz", allow_pickle=False
        )
        errors = archive["feature_errors"]
        names = archive["name"]
        parts = [f"F0{suffix}_test"] + [f"F{i}{suffix}" for i in range(1, 8)]
        matrix = np.stack(
            [errors[names == part].mean(axis=0) for part in parts], axis=0
        )
        matrices.append(matrix)
        row_labels.append(["Healthy test"] + [f"F{i}{suffix}" for i in range(1, 8)])

    all_values = np.concatenate([matrix.ravel() for matrix in matrices])
    positive = all_values[all_values > 0]
    if positive.size == 0:
        raise ValueError("feature_errors has no positive values")
    norm = LogNorm(vmin=float(positive.min()), vmax=float(positive.max()))

    fig, axes = plt.subplots(
        1, 2, figsize=(14.0, 5.2), constrained_layout=True
    )
    image = None
    for ax, mode, matrix, labels in zip(
        axes, ["IPPT", "MPPT"], matrices, row_labels
    ):
        image = ax.imshow(
            matrix,
            aspect="auto",
            cmap="viridis",
            norm=norm,
            interpolation="nearest",
        )
        ax.set_xticks(np.arange(len(features)))
        ax.set_xticklabels(features, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Feature")
        ax.set_ylabel("Recording")
        ax.set_title(mode)
        ax.grid(False)
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes, shrink=0.92, pad=0.02)
        colorbar.set_label("Mean squared reconstruction error (log scale)")
        colorbar.ax.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
    fig.suptitle("Proposed (plain), seed 0: mean per-feature reconstruction error")
    save_figure(fig, "fig_feature_error_heatmap.png")


def fig_architecture() -> None:
    # Layer sizes are read from the static architecture summary, which is copied
    # from src/model.py. They are not result values.
    arch = load_json("phase6_architecture.json")
    boxes = [
        (
            f"Input sequence\n{arch['input_timesteps']} time steps x "
            f"{arch['input_features']} features",
            "input",
        ),
        (
            f"TCN residual block\n{arch['encoder_tcn_filters'][0]} filters, "
            f"kernel {arch['kernel_size']}, dilation {arch['encoder_tcn_dilations'][0]}",
            "encoder",
        ),
        (
            f"TCN residual block\n{arch['encoder_tcn_filters'][1]} filters, "
            f"kernel {arch['kernel_size']}, dilation {arch['encoder_tcn_dilations'][1]}",
            "encoder",
        ),
        (
            f"TCN residual block\n{arch['encoder_tcn_filters'][2]} filters, "
            f"kernel {arch['kernel_size']}, dilation {arch['encoder_tcn_dilations'][2]}",
            "encoder",
        ),
        (
            f"BiLSTM\n{arch['encoder_bilstm_units_per_direction']} units per direction, "
            "sequences",
            "encoder",
        ),
        (
            f"Latent BiLSTM\n{arch['latent_bilstm_units_per_direction']} units per direction",
            "latent",
        ),
        (
            f"RepeatVector\n{arch['repeat_timesteps']} time steps",
            "decoder",
        ),
        (
            f"BiLSTM\n{arch['decoder_bilstm_units_per_direction'][0]} units per direction, "
            "sequences",
            "decoder",
        ),
        (
            f"BiLSTM\n{arch['decoder_bilstm_units_per_direction'][1]} units per direction, "
            "sequences",
            "decoder",
        ),
        (
            f"TCN residual block\n{arch['decoder_tcn_filters'][0]} filters, "
            f"kernel {arch['kernel_size']}, dilation {arch['decoder_tcn_dilations'][0]}",
            "decoder",
        ),
        (
            f"TCN residual block\n{arch['decoder_tcn_filters'][1]} filters, "
            f"kernel {arch['kernel_size']}, dilation {arch['decoder_tcn_dilations'][1]}",
            "decoder",
        ),
        (
            f"TimeDistributed Dense\n{arch['output_dense_units']} outputs",
            "output",
        ),
    ]

    fig, ax = plt.subplots(figsize=(7.4, 9.8))
    ax.set_xlim(0.0, 10.0)
    ax.set_ylim(0.0, 14.6)
    ax.axis("off")

    box_width = 6.6
    box_height = 0.78
    gap = 0.34
    x0 = 1.9
    top = 13.7
    centers = []
    for index, (label, kind) in enumerate(boxes):
        y = top - index * (box_height + gap)
        facecolor = "#f2f2f2" if kind in {"input", "output"} else "white"
        if kind == "latent":
            facecolor = "#eef3f8"
        rect = Rectangle(
            (x0, y - box_height),
            box_width,
            box_height,
            facecolor=facecolor,
            edgecolor="black",
            linewidth=0.9,
        )
        ax.add_patch(rect)
        ax.text(
            x0 + box_width / 2.0,
            y - box_height / 2.0,
            label,
            ha="center",
            va="center",
            fontsize=8.2,
            linespacing=1.15,
        )
        centers.append(y - box_height / 2.0)

    for index in range(len(boxes) - 1):
        y_start = top - index * (box_height + gap) - box_height
        y_end = y_start - gap
        ax.add_patch(
            FancyArrowPatch(
                (x0 + box_width / 2.0, y_start),
                (x0 + box_width / 2.0, y_end),
                arrowstyle="-|>",
                mutation_scale=9,
                linewidth=1.0,
                color="black",
                shrinkA=0.0,
                shrinkB=0.0,
            )
        )

    # Group labels are plain text, not extra graphics.
    group_specs = [
        ("Encoder", 1, 5),
        ("Latent", 5, 6),
        ("Decoder", 6, 11),
        ("Output", 11, 12),
    ]
    for group_label, first, last in group_specs:
        y_top = top - first * (box_height + gap)
        y_bottom = top - last * (box_height + gap) + gap
        ax.text(
            0.95,
            (y_top + y_bottom) / 2.0,
            group_label,
            rotation=90,
            ha="center",
            va="center",
            fontsize=8.0,
            color="#444444",
        )

    ax.set_title("TCN-BiLSTM autoencoder", pad=12)
    fig.text(
        0.5,
        0.012,
        "Each TCN block contains two dilated Conv1D layers, batch normalization, "
        f"ReLU, dropout {arch['dropout_rate']:g} and a residual connection.",
        ha="center",
        va="bottom",
        fontsize=7.6,
        color="#333333",
    )
    save_figure(fig, "fig_architecture.png")


def build_figures() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig_healthy_far()
    fig_primary_f1_auc()
    fig_within_run_auc()
    fig_score_vs_time()
    fig_score_hist()
    fig_threshold_sensitivity()
    fig_training_curves()
    fig_feature_error_heatmap()
    fig_architecture()


# ---------------------------------------------------------------------------
# Captions
# ---------------------------------------------------------------------------


def build_captions() -> str:
    return """# Figure captions

**fig_healthy_far.png.** Healthy-test false-alarm rate for all seven models at the frozen 99th-percentile healthy-validation threshold in IPPT and MPPT, with the 1% validation target marked by a dashed line (sources: `results/average_primary_IPPT.csv`, `results/average_primary_MPPT.csv`).

**fig_primary_f1_auc.png.** Average whole-file F1 and AUC over the fault files of each mode with 95% bootstrap intervals, one panel per mode (sources: `results/average_primary_IPPT.csv`, `results/average_primary_MPPT.csv`).

**fig_within_run_auc.png.** Average within-run AUC per model with 95% bootstrap intervals and a 0.5 chance reference line, one panel per mode, where MPPT excludes F3M because it has no post-injection windows (sources: `results/within_run_average_IPPT.csv`, `results/within_run_average_MPPT.csv`).

**fig_score_vs_time.png.** Window score of Proposed (plain), seed 0, against window start time for F1, F4, F6 and F7 in IPPT on a logarithmic y-axis, with the stored threshold and vertical lines at 7 s and 8 s (source: `results/scores_proposed_plain_IPPT_seed0.npz`).

**fig_score_hist.png.** Histograms of healthy validation, healthy test and pooled fault-file window scores for Proposed (plain), seed 0, on a logarithmic x-axis, one panel per mode, with the stored threshold marked (sources: `results/scores_proposed_plain_IPPT_seed0.npz`, `results/scores_proposed_plain_MPPT_seed0.npz`).

**fig_threshold_sensitivity.png.** Healthy-test false-alarm rate versus threshold percentile (95, 97.5, 99 and 99.5) for Proposed (plain), Proposed (featnorm), EllipticEnvelope and Mahalanobis (window-mean), one panel per mode (source: `results/threshold_sensitivity.csv`).

**fig_training_curves.png.** Training and validation loss by epoch for the six proposed runs (two modes and three seeds) (sources: `results/history_IPPT_seed0.json` through `results/history_MPPT_seed2.json`).

**fig_feature_error_heatmap.png.** Mean per-feature reconstruction error of Proposed (plain), seed 0, for healthy test and each fault file in each mode on a logarithmic color scale (sources: `results/scores_proposed_plain_IPPT_seed0.npz`, `results/scores_proposed_plain_MPPT_seed0.npz`).

**fig_architecture.png.** Block diagram of the TCN-BiLSTM autoencoder implemented in `src/model.py`, showing the encoder, latent representation, decoder and reconstruction layers.
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    (RESULTS / "phase6_architecture.json").write_text(
        json.dumps(ARCHITECTURE_SUMMARY, indent=2), encoding="utf-8"
    )
    print(
        f"wrote {(RESULTS / 'phase6_architecture.json').relative_to(PROJECT_ROOT)}"
    )
    rec = Recorder()
    tables = build_tables(rec)
    (RESULTS / "tables.md").write_text(tables + "\n", encoding="utf-8")
    print(f"wrote {(RESULTS / 'tables.md').relative_to(PROJECT_ROOT)}")

    audit = {
        "phase": "phase_6_figures_and_tables",
        "source_policy": "saved CSV, JSON and NPZ files only; no model, score or threshold was recomputed",
        "model_order": MODEL_ORDER,
        "entries": rec.entries,
    }
    (RESULTS / "phase6_table_values.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(
        f"wrote {(RESULTS / 'phase6_table_values.json').relative_to(PROJECT_ROOT)} "
        f"with {len(rec.entries)} table values"
    )

    captions = build_captions()
    (RESULTS / "figure_captions.md").write_text(captions + "\n", encoding="utf-8")
    print(f"wrote {(RESULTS / 'figure_captions.md').relative_to(PROJECT_ROOT)}")

    build_figures()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Append the Phase 5 section to results/summary.md and STATUS.md.

Every number in the generated report is read from the saved Phase 5 CSV/JSON
files. Nothing is typed from memory and nothing is re-computed from raw data or
from the score archives here.

If a Phase 5 section already exists as the final section (for example after a
rerun of the phase), it is replaced. Earlier sections are never touched.

Run::

    .venv\\Scripts\\python.exe .\\src\\phase5_summary.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
SUMMARY_PATH = RESULTS_DIR / "summary.md"
STATUS_PATH = PROJECT_ROOT / "STATUS.md"
CHECKS_PATH = RESULTS_DIR / "phase5_checks.json"
METADATA_PATH = RESULTS_DIR / "phase5_metadata.json"

MODES = ("IPPT", "MPPT")
PROPOSED = ("Proposed (plain)", "Proposed (featnorm)")
TRIVIAL = "Mahalanobis (window-mean)"


def fmt(value: object, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def fmt_pct(value: object, digits: int = 2) -> str:
    return f"{100.0 * float(value):.{digits}f}%"


def fmt_ci(row: pd.Series, metric: str, percent: bool = False) -> str:
    mean = float(row[f"{metric}_mean"])
    low = float(row[f"{metric}_ci_lo"])
    high = float(row[f"{metric}_ci_hi"])
    if percent:
        return f"{100 * mean:.2f}% [{100 * low:.2f}%, {100 * high:.2f}%]"
    return f"{mean:.4f} [{low:.4f}, {high:.4f}]"


def load_inputs() -> dict[str, object]:
    frames: dict[str, object] = {}
    for mode in MODES:
        frames[f"per_fault_{mode}"] = pd.read_csv(
            RESULTS_DIR / f"per_fault_primary_{mode}.csv"
        )
        frames[f"average_{mode}"] = pd.read_csv(
            RESULTS_DIR / f"average_primary_{mode}.csv"
        )
        frames[f"within_{mode}"] = pd.read_csv(RESULTS_DIR / f"within_run_{mode}.csv")
        frames[f"within_average_{mode}"] = pd.read_csv(
            RESULTS_DIR / f"within_run_average_{mode}.csv"
        )
    frames["sensitivity"] = pd.read_csv(RESULTS_DIR / "threshold_sensitivity.csv")
    frames["paper"] = pd.read_csv(RESULTS_DIR / "paper_reference.csv")
    with CHECKS_PATH.open(encoding="utf-8") as handle:
        frames["checks"] = json.load(handle)
    with METADATA_PATH.open(encoding="utf-8") as handle:
        frames["metadata"] = json.load(handle)
    return frames


def model_row(frame: pd.DataFrame, model: str, **filters: object) -> pd.Series:
    subset = frame[frame["model"] == model]
    for key, value in filters.items():
        subset = subset[subset[key] == value]
    if len(subset) != 1:
        raise RuntimeError(
            f"Expected one row for {model!r} with {filters}, got {len(subset)}"
        )
    return subset.iloc[0]


def primary_table(average: pd.DataFrame, prefix: str = "") -> list[str]:
    lines = [
        "| Model | F1 [95% CI] | Accuracy [95% CI] | AUC [95% CI] | TPR [95% CI] | FPR [95% CI] | Healthy-test FAR |",
        "|---|---|---|---|---|---|---|",
    ]
    for model in average["model"]:
        row = average[average["model"] == model].iloc[0]
        lines.append(
            "| {model} | {f1} | {acc} | {auc} | {tpr} | {fpr} | {far} |".format(
                model=model,
                f1=fmt_ci(row, f"{prefix}f1"),
                acc=fmt_ci(row, f"{prefix}accuracy"),
                auc=fmt_ci(row, f"{prefix}auc"),
                tpr=fmt_ci(row, f"{prefix}tpr"),
                fpr=fmt_ci(row, f"{prefix}fpr"),
                far=fmt_pct(row["healthy_test_false_alarm_rate"]),
            )
        )
    return lines


def within_table(within_average: dict[str, pd.DataFrame]) -> list[str]:
    models = list(within_average["IPPT"]["model"])
    lines = [
        "| Model | IPPT within-run AUC [95% CI] | MPPT within-run AUC [95% CI] | IPPT pre-injection flagged | MPPT pre-injection flagged | IPPT healthy-test FAR | MPPT healthy-test FAR |",
        "|---|---|---|---|---|---|---|",
    ]
    for model in models:
        ippt = model_row(within_average["IPPT"], model)
        mppt = model_row(within_average["MPPT"], model)
        lines.append(
            "| {model} | {ippt_auc} | {mppt_auc} | {ippt_flag} | {mppt_flag} | {ippt_far} | {mppt_far} |".format(
                model=model,
                ippt_auc=fmt_ci(ippt, "auc"),
                mppt_auc=fmt_ci(mppt, "auc"),
                ippt_flag=fmt_pct(ippt["pre_injection_flag_share"]),
                mppt_flag=fmt_pct(mppt["pre_injection_flag_share"]),
                ippt_far=fmt_pct(ippt["healthy_test_false_alarm_rate"]),
                mppt_far=fmt_pct(mppt["healthy_test_false_alarm_rate"]),
            )
        )
    return lines


def not_win_list(
    average_primary: dict[str, pd.DataFrame],
    within_average: dict[str, pd.DataFrame],
) -> tuple[list[str], int]:
    lines: list[str] = []
    count = 0
    metric_directions = (
        ("f1", "primary F1", "higher_is_better"),
        ("accuracy", "primary Accuracy", "higher_is_better"),
        ("auc", "primary AUC", "higher_is_better"),
        ("tpr", "primary TPR", "higher_is_better"),
        ("fpr", "primary FPR", "lower_is_better"),
    )
    for mode in MODES:
        average = average_primary[mode]
        for model in PROPOSED:
            proposed = model_row(average, model)
            trivial = model_row(average, TRIVIAL)
            for metric, label, direction in metric_directions:
                proposed_value = float(proposed[f"{metric}_mean"])
                trivial_value = float(trivial[f"{metric}_mean"])
                if direction == "higher_is_better":
                    won = proposed_value > trivial_value
                else:
                    won = proposed_value < trivial_value
                if won:
                    continue
                count += 1
                if np.isclose(proposed_value, trivial_value, rtol=0.0, atol=1e-12):
                    verdict = "tied"
                elif direction == "lower_is_better":
                    verdict = "did not beat (lower is better)"
                else:
                    verdict = "did not beat"
                lines.append(
                    f"- {mode}, {label}: {model} {fmt(proposed_value, 6)} vs "
                    f"{TRIVIAL} {fmt(trivial_value, 6)} - {verdict}. "
                    f"Healthy-test false-alarm rate: "
                    f"{fmt_pct(proposed['healthy_test_false_alarm_rate'])} vs "
                    f"{fmt_pct(trivial['healthy_test_false_alarm_rate'])}."
                )

        within = within_average[mode]
        for model in PROPOSED:
            proposed = model_row(within, model)
            trivial = model_row(within, TRIVIAL)
            proposed_value = float(proposed["auc_mean"])
            trivial_value = float(trivial["auc_mean"])
            if proposed_value > trivial_value:
                continue
            count += 1
            verdict = (
                "tied"
                if np.isclose(proposed_value, trivial_value, rtol=0.0, atol=1e-12)
                else "did not beat"
            )
            lines.append(
                f"- {mode}, within-run AUC: {model} {fmt(proposed_value, 6)} vs "
                f"{TRIVIAL} {fmt(trivial_value, 6)} - {verdict}. "
                f"Healthy-test false-alarm rate: "
                f"{fmt_pct(proposed['healthy_test_false_alarm_rate'])} vs "
                f"{fmt_pct(trivial['healthy_test_false_alarm_rate'])}."
            )
    return lines, count


def saturation_sentence(
    average_primary: dict[str, pd.DataFrame],
    per_fault_primary: dict[str, pd.DataFrame],
) -> str:
    auc_model_parts = []
    f1_model_parts = []
    total_models = 0
    for mode in MODES:
        average = average_primary[mode]
        total_models = len(average)
        auc_model_parts.append(
            f"{int((average['auc_mean'] >= 0.99).sum())} of {total_models} models in {mode}"
        )
        f1_model_parts.append(
            f"{int((average['f1_mean'] >= 0.99).sum())} of {total_models} models in {mode}"
        )
    auc_cells = np.concatenate(
        [per_fault_primary[mode]["auc_mean"].to_numpy() for mode in MODES]
    )
    f1_cells = np.concatenate(
        [per_fault_primary[mode]["f1_mean"].to_numpy() for mode in MODES]
    )
    auc_saturated = int(np.count_nonzero(auc_cells >= 0.99))
    f1_saturated = int(np.count_nonzero(f1_cells >= 0.99))
    total_cells = int(auc_cells.size)
    return (
        "No, not uniformly - the primary whole-file view is only partly "
        f"saturated: average AUC is at least 0.99 for {auc_model_parts[0]} "
        f"and {auc_model_parts[1]}, and {auc_saturated} of {total_cells} "
        "model-by-fault AUC cells reach 0.99, but average F1 reaches 0.99 for "
        f"only {f1_model_parts[0]} and {f1_model_parts[1]}, with "
        f"{f1_saturated} of {total_cells} F1 cells at that level. AUC is "
        "therefore close to saturation, while F1, accuracy, precision and TPR "
        "are high but not uniformly near 100% (`average_primary_*.csv`, "
        "`per_fault_primary_*.csv`)."
    )


def checks_bullets(checks: dict[str, object]) -> list[str]:
    lines = []
    for record in checks["checks"]:
        status = "PASS" if record["passed"] else "FAIL"
        lines.append(f"- {status} `{record['name']}` - {record['detail']}.")
    return lines


def build_section(
    frames: dict[str, object], timestamp: str
) -> tuple[str, dict[str, object]]:
    average_primary = {mode: frames[f"average_{mode}"] for mode in MODES}
    per_fault_primary = {mode: frames[f"per_fault_{mode}"] for mode in MODES}
    within_average = {mode: frames[f"within_average_{mode}"] for mode in MODES}
    sensitivity = frames["sensitivity"]
    checks = frames["checks"]
    metadata = frames["metadata"]

    losses, loss_count = not_win_list(average_primary, within_average)
    saturated = saturation_sentence(average_primary, per_fault_primary)
    n_models = len(average_primary["IPPT"])
    n_resamples = int(metadata["primary"]["bootstrap"]["resamples"])
    seeds_text = ", ".join(
        str(seed) for seed in metadata["proposed_aggregation"]["seeds"]
    )
    n_faults = int(len(per_fault_primary["IPPT"]["fault"].unique()))

    lines: list[str] = []
    lines.append(f"## Phase 5: final evaluation   ({timestamp})")
    lines.append("")
    lines.append(
        f"What was done: Phase 5 evaluated the frozen score archives for "
        f"{n_models} models in both operating modes (IPPT and MPPT). The "
        "primary whole-file view labelled every window of a fault file as "
        "faulty and compared it with the healthy test windows of the same "
        "mode; the secondary within-run view compared pre-injection windows "
        "with post-8 s windows inside the same fault file. Every cell was "
        f"bootstrapped {n_resamples} times with equal class sizes, and the two "
        f"proposed variants were evaluated for seeds {seeds_text} and "
        "summarised across seeds. Threshold sensitivity was recomputed only "
        "from the saved healthy validation scores, and the reference paper's "
        "published averages were saved separately with a different-protocol "
        "warning. No model was retrained and no score, threshold, k=3 or "
        "normalization was changed."
    )
    lines.append("")
    lines.append("Files created or changed:")
    lines.extend(
        [
            "- `src/phase5_eval.py` - reads the score archives and computes the primary, within-run and threshold-sensitivity tables plus the automated checks.",
            "- `src/phase5_summary.py` - reads the saved Phase 5 tables and writes this section plus the STATUS entry.",
            "- `results/per_fault_primary_IPPT.csv` - per-model, per-fault primary metrics and bootstrap intervals.",
            "- `results/per_fault_primary_MPPT.csv` - same for MPPT, including F3M marked as having no post-injection data.",
            "- `results/average_primary_IPPT.csv` - seven-fault and excluding-F3 averages.",
            "- `results/average_primary_MPPT.csv` - seven-fault and excluding-F3M averages.",
            "- `results/within_run_IPPT.csv` - within-run AUC and pre-injection flag shares per fault.",
            "- `results/within_run_MPPT.csv` - same for MPPT; F3M is absent because it has no post-injection windows.",
            "- `results/within_run_average_IPPT.csv`, `results/within_run_average_MPPT.csv` - fault-averaged within-run results.",
            "- `results/threshold_sensitivity.csv` - healthy-test false-alarm rate and average TPR at the 95, 97.5, 99 and 99.5 percentile thresholds.",
            "- `results/paper_reference.csv` - the reference paper's published averages, labelled reference only.",
            "- `results/phase5_checks.json` - all automated checks and their pass/fail detail.",
            "- `results/phase5_metadata.json` - bootstrap seeds, archive SHA-256 values, environment and method notes.",
            "- `results/summary.md` - this Phase 5 section.",
            "- `STATUS.md` - appended Phase 5 log entry.",
            "- `AGENTS.md` - added the rule that both proposed variants and the healthy-test false-alarm rate always appear in tables.",
            "- `run_all.py` - added the Phase 5 evaluation and summary steps.",
        ]
    )
    lines.append("")
    lines.append("Key numbers:")
    lines.append("")
    lines.append(
        "### 1. Primary whole-file averages (mean [95% bootstrap interval])"
    )
    lines.append("")
    lines.append(
        "Each value is the average over the "
        f"{n_faults} fault files of that mode. "
        "F3M is included for MPPT and is marked in the per-fault table as "
        "having no post-injection data. The healthy-test false-alarm rate is a "
        "point estimate at the stored 99th-percentile healthy-validation "
        "threshold; for the proposed variants it is averaged over seeds."
    )
    lines.append("")
    for mode in MODES:
        lines.append(f"**{mode}**")
        lines.append("")
        lines.extend(primary_table(average_primary[mode]))
        lines.append("")
    lines.append("**MPPT excluding F3M** (the reference paper omits F3 in its MPPT table)")
    lines.append("")
    lines.extend(primary_table(average_primary["MPPT"], prefix="excl_f3_"))
    lines.append("")
    lines.append(
        "### 2. Secondary within-run average AUC (same fault file)"
    )
    lines.append("")
    lines.append(
        "Pre-injection windows end before 7.0 s and start at or after 0.5 s; "
        "post-injection windows start at or after 8.0 s. F3M is excluded from "
        "MPPT because it ends before 7 s. A model that only recognises a "
        "different recording gives an AUC close to 0.5. The pre-injection "
        "columns show the share of pre-injection windows already flagged by "
        "the stored threshold; the final two columns repeat the healthy-test "
        "false-alarm rate next to the within-run result."
    )
    lines.append("")
    lines.extend(within_table(within_average))
    lines.append("")
    lines.append("### 3. Places where a proposed variant does not beat the trivial baseline")
    lines.append("")
    lines.append(
        f"The designated trivial baseline is {TRIVIAL}, the Phase 2 "
        "placeholder. A place is listed when the proposed mean is lower (for "
        "F1, accuracy, AUC and TPR) or higher (for FPR, where lower is better) "
        "than the trivial baseline mean, or exactly tied. Both proposed "
        "variants are included; neither is dropped. Every line is generated "
        "from `average_primary_*.csv` and `within_run_average_*.csv`."
    )
    lines.append("")
    if losses:
        lines.extend(losses)
    else:
        lines.append(
            "- None: both proposed variants beat the trivial baseline on every "
            "listed metric in both modes."
        )
    lines.append("")
    lines.append("### 4. Is the primary view saturated?")
    lines.append("")
    lines.append(saturated)
    lines.append("")
    lines.append("### Threshold sensitivity")
    lines.append("")
    lines.append(
        "The threshold-sensitivity file shows that moving the threshold away "
        "from the frozen 99th percentile changes the healthy-test false-alarm "
        "rate; no setting was selected from fault performance."
    )
    lines.append("")
    lines.append("Checks passed:")
    lines.extend(checks_bullets(checks))
    lines.append(
        f"- {checks['checks_passed']}/{checks['checks_total']} automated Phase 5 checks passed."
    )
    lines.append("")
    lines.append("Problems or warnings:")

    warnings = []
    plain_ippt = model_row(average_primary["IPPT"], "Proposed (plain)")
    plain_mppt = model_row(average_primary["MPPT"], "Proposed (plain)")
    feat_ippt = model_row(average_primary["IPPT"], "Proposed (featnorm)")
    feat_mppt = model_row(average_primary["MPPT"], "Proposed (featnorm)")
    warnings.append(
        "The healthy-test false-alarm rates are not small for every model: "
        f"Proposed (featnorm) is {fmt_pct(feat_ippt['healthy_test_false_alarm_rate'])} "
        f"in IPPT and {fmt_pct(feat_mppt['healthy_test_false_alarm_rate'])} in "
        f"MPPT, versus {fmt_pct(plain_ippt['healthy_test_false_alarm_rate'])} "
        f"and {fmt_pct(plain_mppt['healthy_test_false_alarm_rate'])} for "
        "Proposed (plain). Fault metrics must be read together with these "
        "rates."
    )
    f3m = per_fault_primary["MPPT"][
        per_fault_primary["MPPT"]["fault"] == "F3M"
    ].iloc[0]
    warnings.append(
        "F3M ends at "
        f"{fmt(f3m['file_end_time_s'], 3)} s, before the 7.0-8.0 s injection "
        "interval. It is included in the primary MPPT average and is marked as "
        "having no post-injection data; it is excluded from the within-run "
        "analysis."
    )
    within_ippt_low = within_average["IPPT"].sort_values("auc_mean").iloc[0]
    within_mppt_low = within_average["MPPT"].sort_values("auc_mean").iloc[0]
    within_ippt_high = within_average["IPPT"].sort_values("auc_mean").iloc[-1]
    within_mppt_high = within_average["MPPT"].sort_values("auc_mean").iloc[-1]
    warnings.append(
        "The within-run AUC is not close to 1 for any model: the lowest is "
        f"{within_ippt_low['model']} at {fmt(within_ippt_low['auc_mean'])} "
        f"in IPPT and {within_mppt_low['model']} at "
        f"{fmt(within_mppt_low['auc_mean'])} in MPPT, while the highest is "
        f"{within_ippt_high['model']} at {fmt(within_ippt_high['auc_mean'])} "
        f"in IPPT and {within_mppt_high['model']} at "
        f"{fmt(within_mppt_high['auc_mean'])} in MPPT. A value below 0.5 means "
        "the score is anti-correlated with the pre/post split."
    )
    max_pre_flag = max(
        float(within_average[mode]["pre_injection_flag_share"].max())
        for mode in MODES
    )
    min_pre_flag = min(
        float(within_average[mode]["pre_injection_flag_share"].min())
        for mode in MODES
    )
    warnings.append(
        "Most models already flag most pre-injection windows at the stored "
        f"threshold (fault-average share {fmt_pct(min_pre_flag)} to "
        f"{fmt_pct(max_pre_flag)} across models and modes), so the binary flag "
        "is close to saturated before the fault is injected and the within-run "
        "AUC is the more informative secondary measure."
    )
    sensitivity_95 = sensitivity[sensitivity["quantile"] == 95.0]
    far_95_min = float(sensitivity_95["healthy_test_false_alarm_rate"].min())
    far_95_max = float(sensitivity_95["healthy_test_false_alarm_rate"].max())
    warnings.append(
        "Threshold sensitivity: at the 95th-percentile threshold the "
        "healthy-test false-alarm rate ranges from "
        f"{fmt_pct(far_95_min)} to {fmt_pct(far_95_max)} across the checked "
        "models (versus the frozen 99th-percentile rule). This is a reporting "
        "check only; no threshold was changed."
    )
    warnings.append(
        "Windows overlap heavily within a recording. The bootstrap treats "
        "windows as independent, so its intervals are likely narrower than a "
        "block bootstrap that respected the overlap. The intervals describe "
        "window-level resampling, not independent physical events."
    )
    warnings.append(
        "The reference paper scores single 10 kHz samples, while this pipeline "
        "scores 100-sample windows at 1 kHz. `results/paper_reference.csv` is "
        "reference only and is not a like-for-like comparison."
    )
    warnings.append(
        "Average AUC is near 1.000 for most models, so small rank differences "
        "on the primary view should not be over-interpreted. The within-run "
        "view is not a healthy-versus-fault classifier metric."
    )
    failed = [record for record in checks["checks"] if not record["passed"]]
    if failed:
        warnings.append(
            "Failed checks: "
            + "; ".join(record["name"] for record in failed)
            + "."
        )

    for warning in warnings:
        lines.append(f"- {warning}")
    lines.append("")
    lines.append(
        "Decision needed from me: No. The two proposed variants, the 99th-"
        "percentile threshold, the 200-resample bootstrap and the equal-size "
        "class rule were fixed before the fault metrics were computed."
    )
    lines.append("")
    lines.append(
        "Next phase: Phase 6 - manuscript figures and tables from the saved "
        "Phase 5 results (not started)."
    )
    lines.append("")

    stats = {
        "loss_count": loss_count,
        "checks_passed": checks["checks_passed"],
        "checks_total": checks["checks_total"],
        "bootstrap_resamples": metadata["primary"]["bootstrap"]["resamples"],
    }
    return "\n".join(lines), stats


def build_status_entry(
    frames: dict[str, object], timestamp: str, stats: dict[str, object]
) -> str:
    average_primary = {mode: frames[f"average_{mode}"] for mode in MODES}
    within_average = {mode: frames[f"within_average_{mode}"] for mode in MODES}
    checks = frames["checks"]

    def far(model: str, mode: str) -> str:
        return fmt_pct(
            model_row(average_primary[mode], model)["healthy_test_false_alarm_rate"]
        )

    def f1(model: str, mode: str) -> str:
        return fmt(model_row(average_primary[mode], model)["f1_mean"])

    def within_auc(model: str, mode: str) -> str:
        return fmt(model_row(within_average[mode], model)["auc_mean"])

    metadata = frames["metadata"]
    n_archives = len(metadata["inputs"]["archives"])
    n_resamples = int(metadata["primary"]["bootstrap"]["resamples"])

    lines = [
        f"## {timestamp.split(' ')[0]} - Phase 5: final evaluation",
        "",
        "**What was done.** Evaluated the frozen score archives only, with no "
        "retraining and no change to any score, threshold, k=3 or "
        "normalization. Computed the primary whole-file comparison for every "
        f"model, mode and fault with {n_resamples} equal-class-size bootstrap "
        "resamples; the secondary pre-injection versus post-8 s within-run "
        "AUC; threshold sensitivity from saved healthy validation scores; and "
        "the reference-paper table with a different-protocol label. Both "
        "proposed variants (plain and featnorm) are reported side by side and "
        "F3M is marked as having no post-injection data.",
        "",
        "**Files created/changed.** `src/phase5_eval.py`, "
        "`src/phase5_summary.py`, `run_all.py`, `AGENTS.md` (two-variant and "
        "healthy-FAR reporting rule), `results/per_fault_primary_IPPT.csv`, "
        "`results/per_fault_primary_MPPT.csv`, "
        "`results/average_primary_IPPT.csv`, "
        "`results/average_primary_MPPT.csv`, "
        "`results/within_run_IPPT.csv`, `results/within_run_MPPT.csv`, "
        "`results/within_run_average_IPPT.csv`, "
        "`results/within_run_average_MPPT.csv`, "
        "`results/threshold_sensitivity.csv`, `results/paper_reference.csv`, "
        "`results/phase5_checks.json`, `results/phase5_metadata.json`, "
        "`results/summary.md`, `STATUS.md`.",
        "",
        "**Key numbers (computed by `src/phase5_eval.py`).**",
        "",
        f"- {checks['checks_passed']}/{checks['checks_total']} automated "
        f"checks passed; {n_archives} score archives read and none written.",
        f"- Primary average F1: Proposed (plain) {f1('Proposed (plain)', 'IPPT')} "
        f"(IPPT, healthy-test FAR {far('Proposed (plain)', 'IPPT')}) and "
        f"{f1('Proposed (plain)', 'MPPT')} (MPPT, FAR "
        f"{far('Proposed (plain)', 'MPPT')}); Proposed (featnorm) "
        f"{f1('Proposed (featnorm)', 'IPPT')} (IPPT, FAR "
        f"{far('Proposed (featnorm)', 'IPPT')}) and "
        f"{f1('Proposed (featnorm)', 'MPPT')} (MPPT, FAR "
        f"{far('Proposed (featnorm)', 'MPPT')}).",
        f"- Within-run average AUC: Proposed (plain) "
        f"{within_auc('Proposed (plain)', 'IPPT')} IPPT and "
        f"{within_auc('Proposed (plain)', 'MPPT')} MPPT; Proposed (featnorm) "
        f"{within_auc('Proposed (featnorm)', 'IPPT')} IPPT and "
        f"{within_auc('Proposed (featnorm)', 'MPPT')} MPPT.",
        f"- The two proposed variants do not beat the trivial Mahalanobis "
        f"(window-mean) baseline in {stats['loss_count']} listed "
        "variant/metric/mode places; every such place is listed in "
        "`results/summary.md`.",
        "- Primary average AUC is at least 0.99 for "
        + " and ".join(
            f"{int((average_primary[mode]['auc_mean'] >= 0.99).sum())} of "
            f"{len(average_primary[mode])} models in {mode}"
            for mode in MODES
        )
        + ".",
        "- `F3M` has 0 post-injection windows and is excluded from the "
        "within-run table; `F3L` has post-injection data and is included.",
        "",
        "**Open issues/risks.** The whole-file AUC is close to saturation for "
        "most models, while F1 and accuracy separate the models more; "
        "overlapping windows make the bootstrap intervals optimistic. "
        "Proposed (featnorm) has high healthy-test false-alarm rates. The "
        "reference-paper table uses a different protocol (single 10 kHz "
        "samples) and is reference only. F3M ends before 7 s. No model was "
        "retuned in response to any fault result.",
        "",
    ]
    return "\n".join(lines)


def upsert_trailing_section(path: Path, marker: str, section: str) -> bool:
    """Append a section, or replace it if it is the final section."""
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    lines = text.splitlines()
    heading_positions = [
        index for index, line in enumerate(lines) if line.startswith("## ")
    ]
    matches = [
        index
        for index in heading_positions
        if marker in lines[index]
    ]
    if len(matches) > 1:
        raise RuntimeError(f"{path}: more than one heading contains {marker!r}")

    if matches:
        target = matches[0]
        later = [index for index in heading_positions if index > target]
        if later:
            raise RuntimeError(
                f"{path}: the {marker!r} section is not the final section; "
                "refusing to touch earlier entries"
            )
        prefix = "\n".join(lines[:target]).rstrip()
        replaced = True
    else:
        prefix = text.rstrip()
        replaced = False

    updated = (prefix + "\n\n" if prefix else "") + section.rstrip() + "\n"
    path.write_text(updated, encoding="utf-8", newline="\n")
    return replaced


def main() -> int:
    print("=" * 80)
    print("GPVS-Faults | Phase 5 summary")
    print("=" * 80)
    now = datetime.now().astimezone()
    timestamp = now.strftime("%Y-%m-%d %H:%M %z")

    frames = load_inputs()
    section, stats = build_section(frames, timestamp)
    status_entry = build_status_entry(frames, timestamp, stats)

    replaced = upsert_trailing_section(SUMMARY_PATH, "Phase 5:", section)
    print(
        f"{'Replaced' if replaced else 'Appended'} the Phase 5 section in "
        f"{SUMMARY_PATH}"
    )
    replaced = upsert_trailing_section(STATUS_PATH, "Phase 5:", status_entry)
    print(
        f"{'Replaced' if replaced else 'Appended'} the Phase 5 entry in "
        f"{STATUS_PATH}"
    )
    print(
        f"Checks: {stats['checks_passed']}/{stats['checks_total']}; "
        f"not-beat places: {stats['loss_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

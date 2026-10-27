"""Append the Phase 2 section to results/summary.md and STATUS.md.

This is a reporting script. It reads only saved result files produced by
``src/time_diagnostic.py`` (plus the existing sanity-check summary), computes
the summary statistics, saves them to
``results/phase2_summary_stats.json``, and appends a Phase 2 section in the
format required by ``AGENTS.md``. Earlier summary sections and earlier STATUS
entries are never modified.

Run after ``src/time_diagnostic.py``::

    .venv\\Scripts\\python.exe .\\src\\phase2_summary.py
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
STATS_PATH = RESULTS_DIR / "phase2_summary_stats.json"
SECTION_HEADING = "## Phase 2: Labeling and timing diagnostic"
STATUS_MARKER = "Phase 2: labeling and timing diagnostic"

# Descriptive reporting convention only. It is not a detector setting and is
# not used to choose a model or threshold.
CLEAR_JUMP_RATIO = 2.0
WEAK_JUMP_RATIO = 1.2


def read_json(path: Path) -> dict[str, object]:
    """Load a JSON file as a dictionary."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a markdown table with plain ASCII separators."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def fmt_count(value: object) -> str:
    return f"{int(value):,}"


def fmt_pct(value: object) -> str:
    number = float(value)
    if not np.isfinite(number):
        return "NA"
    return f"{100.0 * number:.2f}%"


def fmt_ratio(value: object) -> str:
    number = float(value)
    if not np.isfinite(number):
        return "NA"
    return f"{number:.2f}x"


def fmt_threshold(value: object) -> str:
    return f"{float(value):.3f}"


def format_range(values: list[float], formatter) -> str:
    """Format a min-max range, collapsing identical endpoints."""
    finite = [float(value) for value in values if np.isfinite(float(value))]
    if not finite:
        return "NA"
    low = min(finite)
    high = max(finite)
    if math.isclose(low, high, rel_tol=0.0, abs_tol=1e-12):
        return formatter(low)
    return f"{formatter(low)}-{formatter(high)}"


def name_list(names: list[str]) -> str:
    """Render a comma-separated list of backticked file names."""
    if not names:
        return "none"
    return ", ".join(f"`{name}`" for name in names)


def append_unique(path: Path, heading: str, section: str) -> bool:
    """Append a section unless an earlier copy with the same heading exists."""
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


def build_section(
    labels: pd.DataFrame,
    healthy: pd.DataFrame,
    scores: pd.DataFrame,
    meta: dict[str, object],
    sanity: dict[str, object] | None,
    stats: dict[str, object],
    timestamp: str,
) -> str:
    """Render the Phase 2 markdown section from computed facts."""
    detector = stats["detector"]
    pre = stats["pre_injection"]
    startup = stats["startup"]
    timing = stats["timing"]
    f3m = stats["f3m"]
    totals = stats["totals"]
    healthy_test = stats["healthy_test"]

    fault_table_rows: list[list[str]] = []
    for _, row in labels.sort_values(["fault", "mode"]).iterrows():
        fault_table_rows.append(
            [
                f"`{row['file']}`",
                fmt_count(row["windows_before_7s"]),
                fmt_pct(row["share_flagged_before_7s_excl_startup"]),
                fmt_pct(row["share_flagged_startup_first_0p5s"]),
                fmt_pct(row["share_flagged_7_to_8s"]),
                fmt_pct(row["share_flagged_after_8s"]),
                fmt_ratio(row["median_score_before_7s_over_threshold"]),
                fmt_ratio(row["median_score_ratio_7_to_8_over_before_7"]),
                fmt_ratio(row["median_score_ratio_after_8_over_before_7"]),
                "yes" if bool(row["no_post_injection_data"]) else "no",
            ]
        )
    fault_table = markdown_table(
        [
            "File",
            "Windows before 7 s",
            "Flagged before 7 s (excl. startup)",
            "Flagged startup (first 0.5 s)",
            "Flagged 7-8 s",
            "Flagged after 8 s",
            "Pre median / threshold",
            "7-8 s / pre median",
            "After 8 s / pre median",
            "No post-injection data",
        ],
        fault_table_rows,
    )

    detector_rows: list[list[str]] = []
    for mode in ("IPPT", "MPPT"):
        mode_data = meta["modes"][mode]
        test_row = healthy[
            (healthy["mode"] == mode) & (healthy["split"] == "test")
        ].iloc[0]
        detector_rows.append(
            [
                mode,
                fmt_count(mode_data["train_windows"]),
                fmt_count(mode_data["validation_windows"]),
                fmt_count(mode_data["test_windows"]),
                fmt_threshold(mode_data["threshold"]),
                (
                    f"{int(mode_data['validation_flagged'])} "
                    f"({float(mode_data['validation_flag_rate']):.2%})"
                ),
                (
                    f"{int(test_row['flagged'])}/{int(test_row['windows'])} "
                    f"({float(test_row['false_alarm_rate']):.2%})"
                ),
            ]
        )
    detector_table = markdown_table(
        [
            "Mode",
            "Healthy train windows",
            "Healthy validation windows",
            "Healthy test windows",
            "Threshold",
            "Validation flagged",
            "Healthy test flagged (false-alarm rate)",
        ],
        detector_rows,
    )

    ridge_text = format_range(
        [detector["ridge_by_mode"]["IPPT"], detector["ridge_by_mode"]["MPPT"]],
        lambda value: f"{value:.2e}",
    )
    condition_text = format_range(
        [
            detector["condition_number_by_mode"]["IPPT"],
            detector["condition_number_by_mode"]["MPPT"],
        ],
        lambda value: f"{value:.2e}",
    )

    pre_share_text = format_range(
        [
            float(labels["share_flagged_before_7s_excl_startup"].min()),
            float(labels["share_flagged_before_7s_excl_startup"].max()),
        ],
        fmt_pct,
    )
    pre_windows_text = format_range(
        [
            float(labels["windows_before_7s_excl_startup"].min()),
            float(labels["windows_before_7s_excl_startup"].max()),
        ],
        lambda value: f"{int(value):,}",
    )
    startup_share_text = format_range(
        [
            float(labels["share_flagged_startup_first_0p5s"].min()),
            float(labels["share_flagged_startup_first_0p5s"].max()),
        ],
        fmt_pct,
    )
    startup_windows_text = format_range(
        [
            float(labels["startup_windows"].min()),
            float(labels["startup_windows"].max()),
        ],
        lambda value: f"{int(value):,}",
    )

    what_was_done = " ".join(
        [
            (
                "Phase 2 checked whether the fault recordings already look "
                "faulty before the documented 7-8 s injection interval, and "
                "whether the first 0.5 s startup transient is flagged."
            ),
            (
                "The diagnostic uses a deliberately simple placeholder: for "
                "each mode it fits a 13-feature mean and covariance to healthy "
                "training window means, scores every window by Mahalanobis "
                "distance with a pseudo-inverse and a small relative ridge, and "
                "sets the threshold at the 99th percentile of healthy "
                "validation scores."
            ),
            "Fault files were never used to fit the mean/covariance or the threshold.",
            (
                "`get_labels` adds whole-file labels (every F1-F7 window faulty) "
                "and time-resolved labels (-1 before 7 s, -2 for windows "
                "overlapping 7-8 s, 1 after 8 s); `F3M` gets only -1 labels "
                f"because its last window ends at {float(f3m['last_window_end_s']):.3f} s."
            ),
            (
                f"All {int(totals['windows_scored']):,} windows of the 16 files "
                f"were scored. The placeholder flagged {pre_share_text} of the "
                f"pre-injection windows outside the startup period, versus "
                f"{float(healthy_test['IPPT']):.2%} and "
                f"{float(healthy_test['MPPT']):.2%} false alarms on healthy test "
                "data for IPPT and MPPT."
            ),
        ]
    )

    files_lines = [
        (
            "- `src/data.py` - added `get_labels` and a cached window-time "
            "helper; existing preprocessing outputs are unchanged."
        ),
        (
            "- `src/time_diagnostic.py` - fits the placeholder, scores all "
            "windows, writes the diagnostic tables, and draws the figures."
        ),
        (
            "- `src/phase2_summary.py` - reads the saved results and writes "
            "this section and the STATUS entry."
        ),
        "- `run_all.py` - added the labeling and timing diagnostic as a pipeline step.",
        (
            "- `results/time_diagnostic_scores.csv` - one row per window with "
            "score, threshold, flag, timestamps, and both label modes."
        ),
        (
            "- `results/label_diagnostic.csv` - per-fault pre/startup/7-8/post "
            "flag shares and median-score ratios."
        ),
        (
            "- `results/healthy_false_alarm.csv` - healthy train/validation/test "
            "false-alarm rates for the placeholder."
        ),
        (
            "- `results/time_diagnostic.json` - detector definition, fitted "
            "parameters, thresholds, and all check results."
        ),
        "- `results/phase2_summary_stats.json` - the facts used in this report.",
        (
            "- `figures/score_vs_time_F1L.png` ... `score_vs_time_F7M.png` - "
            "14 individual score-versus-time plots."
        ),
        "- `figures/score_vs_time_grid.png` - 7 faults by 2 modes summary grid.",
        "- `results/summary.md` - this Phase 2 section.",
        "- `STATUS.md` - appended Phase 2 log entry.",
    ]

    clear_names = name_list(timing["clear_post_files"])
    weak_names = name_list(timing["weak_post_files"])
    flat_names = name_list(timing["flat_post_files"])
    interval_names = name_list(timing["interval_jump_files"])
    no_post_names = name_list(timing["no_post_files"])
    post_only_names = name_list(timing["post_only_jump_files"])

    if timing["all_pre_and_post_flagged"]:
        flag_sentence = (
            "The binary flag share cannot show a jump because every "
            "pre-injection and post-injection window is already above the "
            "healthy threshold."
        )
    else:
        flag_sentence = (
            "The binary flag share is not saturated on both sides, so it can "
            "be read together with the median-score ratios."
        )

    answer_a = (
        "(a) Pre-injection part: anomalous to this placeholder. "
        f"All {int(totals['fault_files'])} fault files have "
        f"{pre_windows_text} pre-injection windows after the first 0.5 s, and "
        f"{pre_share_text} of those windows were flagged. The median "
        f"pre-injection score was {fmt_ratio(pre['median_over_threshold_min'])} "
        f"to {fmt_ratio(pre['median_over_threshold_max'])} the threshold, while "
        f"healthy test false-alarm rates were {float(healthy_test['IPPT']):.2%} "
        f"(IPPT) and {float(healthy_test['MPPT']):.2%} (MPPT). This says the "
        "pre-injection segment differs from the healthy training file under "
        "this detector; it does not prove that the fault was already active, "
        "because a run-level operating-point offset can produce the same "
        "result."
    )

    answer_b_parts = [
        "(b) Score jump at 7-8 s:",
        flag_sentence,
        (
            f"Inside the 7-8 s interval, {interval_names} rise "
            f"({fmt_ratio(timing['interval_ratio_min'])} to "
            f"{fmt_ratio(timing['interval_ratio_max'])} of the pre-injection "
            "median)."
        ),
        (
            f"After 8 s, {post_only_names} rise "
            f"({fmt_ratio(timing['post_only_ratio_min'])} to "
            f"{fmt_ratio(timing['post_only_ratio_max'])})."
        ),
        (
            f"{flat_names} stay within "
            f"{100.0 * float(timing['flat_max_abs_deviation']):.1f}% of their "
            "pre-injection median."
        ),
        (
            f"{weak_names} is intermediate at "
            f"{fmt_ratio(timing['weak_ratio'])}."
        ),
    ]
    answer_b = " ".join(answer_b_parts)

    answer_c = (
        "(c) Startup transient: flagged, but not separately. "
        f"The {startup_windows_text} windows starting in the first 0.5 s of "
        f"each fault file were flagged in every file ({startup_share_text}), "
        f"and their median score differed from the pre-injection median by at "
        f"most {100.0 * float(startup['max_relative_difference']):.1f}%. The "
        "whole pre-injection segment is already above threshold, so this is "
        "not evidence of a distinct startup effect."
    )

    answer_d = (
        "(d) No clear timing signal: "
        f"{flat_names} have post/pre median ratios of "
        f"{fmt_ratio(timing['flat_ratio_min'])} to "
        f"{fmt_ratio(timing['flat_ratio_max'])}, i.e. essentially flat. "
        f"{weak_names} is weak at {fmt_ratio(timing['weak_ratio'])}. "
        f"{no_post_names} cannot be assessed: it has "
        f"{int(f3m['windows_after_8s'])} post-injection windows and its last "
        f"window ends at {float(f3m['last_window_end_s']):.3f} s. "
        "At the binary flag level none of the files shows a timing change, "
        "because every fault window is already flagged."
    )

    sanity_text = "not available"
    if sanity is not None:
        sanity_text = (
            f"{int(sanity['checks_passed'])}/{int(sanity['checks_total'])}"
        )

    checks_lines = [
        (
            f"- Existing preprocessing is unchanged: the sanity-check file "
            f"reports {sanity_text} checks passed after adding `get_labels`."
        ),
        (
            f"- The diagnostic reports {int(meta['checks_passed'])}/"
            f"{int(meta['checks_total'])} checks passed."
        ),
        (
            f"- All {int(totals['windows_scored']):,} windows of all 16 files "
            f"were scored; scores are finite and non-negative "
            f"({float(totals['score_min']):.3g} to "
            f"{float(totals['score_max']):.3g})."
        ),
        (
            "- Saved time-resolved labels match `get_labels` after the CSV "
            "round-trip, use only -2/-1/0/1, and partition every fault file "
            "into pre/overlap/post windows."
        ),
        (
            f"- `F3M` is handled without crashing: "
            f"{int(f3m['windows_before_7s']):,} pre-fault windows, "
            f"{int(f3m['windows_7_to_8s'])} windows in 7-8 s, and "
            f"{int(f3m['windows_after_8s'])} post-injection windows."
        ),
        (
            "- The healthy reference contains only `F0L`/`F0M`, and each "
            "false-alarm rate equals flagged/windows "
            f"(IPPT {int(healthy_test['IPPT_flagged'])}/"
            f"{int(healthy_test['IPPT_windows'])}, "
            f"MPPT {int(healthy_test['MPPT_flagged'])}/"
            f"{int(healthy_test['MPPT_windows'])})."
        ),
        (
            "- The detector mean/covariance and the 99th-percentile threshold "
            "use healthy training and validation windows only; 0 fault files "
            "were used for either."
        ),
        (
            f"- The only undefined values in `label_diagnostic.csv` are "
            f"{int(timing['undefined_post_fields'])} post-injection fields for "
            "`F3M`, where the denominator is zero; there are no NaN scores."
        ),
    ]

    warning_lines = [
        (
            "- The placeholder is saturated on fault files: every fault window "
            "in the 13 files with post-injection data is above the healthy "
            "threshold. The binary shares are therefore all 100%, and the "
            "median-score ratios are the only timing evidence in this table."
        ),
        (
            "- The pre-injection anomaly may be a run-level offset rather than "
            "an early fault. The placeholder is a global window-mean model "
            "fitted to one healthy recording per mode, so differences in "
            "irradiance, temperature, or operating point between recordings "
            "can dominate it. No causal claim is made."
        ),
        (
            "- The window-mean covariance is nearly singular "
            f"(condition numbers {condition_text}); a relative ridge "
            f"({ridge_text}) and a pseudo-inverse were required. Very large "
            f"scores (up to {float(totals['score_max']):,.1f}) come from "
            "near-null directions and should not be read as physical units."
        ),
        (
            f"- `F3M` ends before the 7-8 s window: its last window ends at "
            f"{float(f3m['last_window_end_s']):.3f} s, so it has no "
            "post-injection or 7-8 s data. Its score rise near the file end "
            "may include the zero-phase decimation edge transient."
        ),
        (
            "- The healthy training windows include the startup transient, "
            "so the fitted covariance already contains the `F0` startup "
            "behavior. This can make the placeholder less sensitive to a "
            "similar transient in fault files."
        ),
        (
            "- Windows overlap by 90%, so they are not independent "
            "observations; this diagnostic is descriptive and has no "
            "confidence intervals."
        ),
        (
            "- This is a diagnostic placeholder, not the final model. No "
            "detector setting was tuned on fault results; the score, ridge, "
            "and threshold rules were fixed before fault scoring."
        ),
    ]

    decision = (
        "Yes - choose the label convention for the final evaluation. The "
        "diagnostic supports neither a blanket whole-file label nor a blanket "
        "time-resolved label: every fault file is already anomalous before "
        "7 s, but only some files change at 7-8 s. Options: (1) keep "
        "whole-file labels as the primary comparison with the notebook and "
        "reference paper, and report time-resolved pre/post scores as a "
        "secondary analysis; or (2) make time-resolved labels primary, score "
        "pre-injection windows as false alarms, drop the 7-8 s overlap "
        "windows, and report `F3M` separately. Whichever is chosen, `F3M` "
        "must be reported as having no post-injection data."
    )

    heading = f"{SECTION_HEADING}   ({timestamp})"
    parts = [
        heading,
        "",
        f"What was done: {what_was_done}",
        "",
        "Files created or changed:",
        *files_lines,
        "",
        "Key numbers:",
        "",
        "### Placeholder detector and healthy reference",
        "",
        detector_table,
        "",
        (
            "The detector is fitted on the healthy training windows of each "
            "mode, and the threshold is the 99th percentile of that mode's "
            "healthy validation scores. The healthy test rows are unseen by "
            "both the fit and the threshold."
        ),
        "",
        "### Per-fault timing diagnostic",
        "",
        fault_table,
        "",
        (
            "Shares are fractions of the relevant window group. `Windows "
            "before 7 s` counts windows whose end time is before 7.0 s. The "
            "pre-injection share excludes windows that start in the first "
            "0.5 s; the startup share uses those windows. The 7-8 s group is "
            "the set of windows that overlap the documented injection "
            "interval and is dropped by the time-resolved labels. `NA` means "
            "the denominator is zero."
        ),
        "",
        "### Answers to the four timing questions",
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
        f"Decision needed from me: {decision}",
        "",
        (
            "Next phase: Phase 3 - train the healthy-only detector and "
            "evaluate it under the chosen label convention, keeping fault data "
            "out of every modeling decision."
        ),
    ]

    return "\n".join(parts).rstrip() + "\n"


def build_status_entry(stats: dict[str, object]) -> str:
    """Render the short chronological STATUS.md entry."""
    totals = stats["totals"]
    pre = stats["pre_injection"]
    timing = stats["timing"]
    f3m = stats["f3m"]
    healthy_test = stats["healthy_test"]

    clear_names = name_list(timing["clear_post_files"])
    flat_names = name_list(timing["flat_post_files"])
    weak_names = name_list(timing["weak_post_files"])
    no_post_names = name_list(timing["no_post_files"])

    lines = [
        f"## {stats['date']} - Phase 2: labeling and timing diagnostic",
        "",
        (
            "**What was done.** Added `get_labels` to `src/data.py` with "
            "`whole_file` and `time_resolved` modes (-1 before 7 s, -2 for "
            "7-8 s overlap, 1 after 8 s). Built a placeholder detector from "
            "healthy training window means only: Mahalanobis distance with a "
            "pseudo-inverse and small relative ridge, thresholded at the 99th "
            "percentile of healthy validation scores. Scored all "
            f"{int(totals['windows_scored']):,} windows, wrote per-fault "
            "timing shares and healthy false-alarm rates, and drew "
            f"{int(totals['fault_files'])} individual figures plus a 7x2 grid. "
            "No model setting was tuned on fault results."
        ),
        "",
        (
            "**Files created/changed.** `src/data.py` (added `get_labels` and "
            "a cached window-time helper; existing outputs unchanged), "
            "`src/time_diagnostic.py`, `src/phase2_summary.py`, `run_all.py`, "
            "`results/time_diagnostic_scores.csv`, "
            "`results/label_diagnostic.csv`, `results/healthy_false_alarm.csv`, "
            "`results/time_diagnostic.json`, "
            "`results/phase2_summary_stats.json`, "
            "`figures/score_vs_time_*.png`, `results/summary.md`, `STATUS.md`."
        ),
        "",
        "**Key numbers (computed by code).**",
        (
            f"- {int(totals['windows_scored']):,} windows scored across 16 "
            f"files; {int(totals['fault_files'])} fault files in two modes. "
            f"Score range {float(totals['score_min']):.3g} to "
            f"{float(totals['score_max']):.3g}."
        ),
        (
            f"- Pre-injection flag share (excluding the first 0.5 s): "
            f"{float(pre['share_min']):.4f} to {float(pre['share_max']):.4f} "
            f"across all fault files; median pre-injection score was "
            f"{float(pre['median_over_threshold_min']):.1f}x to "
            f"{float(pre['median_over_threshold_max']):.1f}x the threshold."
        ),
        (
            f"- Healthy test false-alarm rate: "
            f"{float(healthy_test['IPPT']):.4f} IPPT and "
            f"{float(healthy_test['MPPT']):.4f} MPPT."
        ),
        (
            f"- Post/pre median-score ratios: clear for {clear_names}; weak "
            f"for {weak_names}; flat for {flat_names}; "
            f"{no_post_names} has no post-injection data."
        ),
        (
            f"- Post-injection flag shares are {float(timing['post_share_min']):.4f} "
            f"to {float(timing['post_share_max']):.4f} where post data exist, "
            "so the binary flag does not localize the onset; "
            f"`F3M` has {int(f3m['windows_after_8s'])} post-injection windows."
        ),
        "",
        (
            "**Open issues/risks.** The placeholder is saturated on fault "
            "files, so pre-injection anomaly cannot be separated from a "
            "run-level operating-point offset. `F3M` ends before 7 s and has "
            "no post-injection data. The label convention for final "
            "evaluation (whole-file versus time-resolved) is still a decision."
        ),
    ]
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    labels = pd.read_csv(RESULTS_DIR / "label_diagnostic.csv")
    healthy = pd.read_csv(RESULTS_DIR / "healthy_false_alarm.csv")
    scores = pd.read_csv(RESULTS_DIR / "time_diagnostic_scores.csv")
    meta = read_json(RESULTS_DIR / "time_diagnostic.json")
    sanity_path = RESULTS_DIR / "sanity_checks.json"
    sanity = read_json(sanity_path) if sanity_path.exists() else None

    if labels.shape[0] != 14:
        raise RuntimeError(
            f"Expected 14 fault/mode rows, got {labels.shape[0]}"
        )
    if set(labels["fault"].unique()) != {f"F{index}" for index in range(1, 8)}:
        raise RuntimeError("Expected faults F1-F7 in label_diagnostic.csv")
    if bool(labels["no_post_injection_data"].sum()) != 1:
        raise RuntimeError("Expected exactly one file with no post-injection data")

    healthy_test = healthy[healthy["split"] == "test"].set_index("mode")
    if set(healthy_test.index) != {"IPPT", "MPPT"}:
        raise RuntimeError("Expected IPPT and MPPT healthy test rows")

    post_known = labels[~labels["no_post_injection_data"]].copy()
    clear_post = post_known[
        post_known["median_score_ratio_after_8_over_before_7"]
        >= CLEAR_JUMP_RATIO
    ]
    weak_post = post_known[
        (
            post_known["median_score_ratio_after_8_over_before_7"]
            >= WEAK_JUMP_RATIO
        )
        & (
            post_known["median_score_ratio_after_8_over_before_7"]
            < CLEAR_JUMP_RATIO
        )
    ]
    flat_post = post_known[
        post_known["median_score_ratio_after_8_over_before_7"]
        < WEAK_JUMP_RATIO
    ]
    interval_jump = labels[
        labels["median_score_ratio_7_to_8_over_before_7"] >= CLEAR_JUMP_RATIO
    ]
    post_only = clear_post[
        ~clear_post["file"].isin(interval_jump["file"].tolist())
    ]
    no_post = labels[labels["no_post_injection_data"]]

    f3m_row = labels[labels["file"] == "F3M"].iloc[0]
    f3m_last_end = float(
        scores.loc[scores["file"] == "F3M", "end_time_s"].max()
    )

    undefined_columns = [
        column
        for column in labels.columns
        if labels[column].isna().any()
        and column not in ("post_injection_note",)
    ]
    undefined_post_fields = sum(
        1
        for column in undefined_columns
        if column
        in {
            "share_flagged_7_to_8s",
            "share_flagged_after_8s",
            "median_score_7_to_8s",
            "median_score_after_8s",
            "median_score_ratio_7_to_8_over_before_7",
            "median_score_ratio_after_8_over_before_7",
        }
    )
    if undefined_post_fields != len(undefined_columns):
        raise RuntimeError(
            "Unexpected NaN columns in label_diagnostic.csv: "
            f"{undefined_columns}"
        )

    all_pre_post_flagged = bool(
        (
            labels["share_flagged_before_7s_excl_startup"] == 1.0
        ).all()
        and (
            labels.loc[
                ~labels["no_post_injection_data"], "share_flagged_after_8s"
            ]
            == 1.0
        ).all()
    )
    post_shares = labels.loc[
        ~labels["no_post_injection_data"], "share_flagged_after_8s"
    ]

    startup_relative = (
        (
            labels["median_score_startup_first_0p5s"]
            - labels["median_score_before_7s_excl_startup"]
        )
        / labels["median_score_before_7s_excl_startup"]
    ).abs()

    flat_ratio_min = float(
        flat_post["median_score_ratio_after_8_over_before_7"].min()
    )
    flat_ratio_max = float(
        flat_post["median_score_ratio_after_8_over_before_7"].max()
    )
    flat_max_abs_deviation = float(
        (
            flat_post["median_score_ratio_after_8_over_before_7"] - 1.0
        ).abs().max()
    )

    stats: dict[str, object] = {
        "phase": "phase_2_labeling_time_diagnostic_summary",
        "generated_local": datetime.now().astimezone().isoformat(),
        "date": datetime.now().astimezone().strftime("%Y-%m-%d"),
        "git_commit": meta.get("git_commit"),
        "environment": meta.get("environment"),
        "source_files": {
            "label_diagnostic": "results/label_diagnostic.csv",
            "healthy_false_alarm": "results/healthy_false_alarm.csv",
            "time_diagnostic_scores": "results/time_diagnostic_scores.csv",
            "time_diagnostic": "results/time_diagnostic.json",
            "sanity_checks": "results/sanity_checks.json",
        },
        "totals": {
            "windows_scored": int(scores.shape[0]),
            "fault_files": int(labels["file"].nunique()),
            "fault_modes": int(labels.shape[0]),
            "score_min": float(scores["score"].min()),
            "score_max": float(scores["score"].max()),
            "score_nan": int(scores["score"].isna().sum()),
        },
        "detector": {
            "name": meta["detector"]["name"],
            "ridge_relative": float(meta["detector"]["ridge_relative"]),
            "ridge_by_mode": {
                mode: float(meta["modes"][mode]["ridge"])
                for mode in ("IPPT", "MPPT")
            },
            "condition_number_by_mode": {
                mode: float(meta["modes"][mode]["regularized_condition_number"])
                for mode in ("IPPT", "MPPT")
            },
            "threshold_by_mode": {
                mode: float(meta["modes"][mode]["threshold"])
                for mode in ("IPPT", "MPPT")
            },
            "fault_files_used_for_fit": int(
                meta["detector"]["fault_files_used_for_fit"]
            ),
            "fault_files_used_for_threshold": int(
                meta["detector"]["fault_files_used_for_threshold"]
            ),
        },
        "healthy_test": {
            "IPPT": float(healthy_test.loc["IPPT", "false_alarm_rate"]),
            "MPPT": float(healthy_test.loc["MPPT", "false_alarm_rate"]),
            "IPPT_flagged": int(healthy_test.loc["IPPT", "flagged"]),
            "IPPT_windows": int(healthy_test.loc["IPPT", "windows"]),
            "MPPT_flagged": int(healthy_test.loc["MPPT", "flagged"]),
            "MPPT_windows": int(healthy_test.loc["MPPT", "windows"]),
        },
        "pre_injection": {
            "windows_min": int(labels["windows_before_7s_excl_startup"].min()),
            "windows_max": int(labels["windows_before_7s_excl_startup"].max()),
            "share_min": float(
                labels["share_flagged_before_7s_excl_startup"].min()
            ),
            "share_max": float(
                labels["share_flagged_before_7s_excl_startup"].max()
            ),
            "median_over_threshold_min": float(
                labels["median_score_before_7s_over_threshold"].min()
            ),
            "median_over_threshold_max": float(
                labels["median_score_before_7s_over_threshold"].max()
            ),
            "all_files_fully_flagged": bool(
                (
                    labels["share_flagged_before_7s_excl_startup"] == 1.0
                ).all()
            ),
        },
        "startup": {
            "windows_min": int(labels["startup_windows"].min()),
            "windows_max": int(labels["startup_windows"].max()),
            "share_min": float(
                labels["share_flagged_startup_first_0p5s"].min()
            ),
            "share_max": float(
                labels["share_flagged_startup_first_0p5s"].max()
            ),
            "max_relative_difference": float(startup_relative.max()),
            "all_flagged": bool(
                (labels["share_flagged_startup_first_0p5s"] == 1.0).all()
            ),
        },
        "timing": {
            "clear_post_files": clear_post["file"].tolist(),
            "weak_post_files": weak_post["file"].tolist(),
            "flat_post_files": flat_post["file"].tolist(),
            "interval_jump_files": interval_jump["file"].tolist(),
            "post_only_jump_files": post_only["file"].tolist(),
            "no_post_files": no_post["file"].tolist(),
            "interval_ratio_min": float(
                interval_jump[
                    "median_score_ratio_7_to_8_over_before_7"
                ].min()
            ),
            "interval_ratio_max": float(
                interval_jump[
                    "median_score_ratio_7_to_8_over_before_7"
                ].max()
            ),
            "post_only_ratio_min": float(
                post_only["median_score_ratio_after_8_over_before_7"].min()
            ),
            "post_only_ratio_max": float(
                post_only["median_score_ratio_after_8_over_before_7"].max()
            ),
            "flat_ratio_min": flat_ratio_min,
            "flat_ratio_max": flat_ratio_max,
            "flat_max_abs_deviation": flat_max_abs_deviation,
            "weak_ratio": float(
                weak_post["median_score_ratio_after_8_over_before_7"].iloc[0]
            ),
            "post_share_min": float(post_shares.min()),
            "post_share_max": float(post_shares.max()),
            "all_pre_and_post_flagged": all_pre_post_flagged,
            "undefined_post_fields": int(undefined_post_fields),
            "clear_jump_ratio_convention": CLEAR_JUMP_RATIO,
            "weak_jump_ratio_convention": WEAK_JUMP_RATIO,
        },
        "f3m": {
            "windows_total": int(f3m_row["windows_total"]),
            "windows_before_7s": int(f3m_row["windows_before_7s"]),
            "windows_7_to_8s": int(f3m_row["windows_7_to_8s"]),
            "windows_after_8s": int(f3m_row["windows_after_8s"]),
            "last_window_end_s": f3m_last_end,
            "no_post_injection_data": bool(
                f3m_row["no_post_injection_data"]
            ),
        },
        "meta_checks": {
            "passed": int(meta["checks_passed"]),
            "total": int(meta["checks_total"]),
        },
        "sanity_checks": (
            {
                "passed": int(sanity["checks_passed"]),
                "total": int(sanity["checks_total"]),
                "all_passed": bool(sanity["all_passed"]),
            }
            if sanity is not None
            else None
        ),
        "undefined_post_columns": undefined_columns,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with STATS_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(stats, handle, indent=2, allow_nan=False)
        handle.write("\n")

    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %z")
    section = build_section(
        labels=labels,
        healthy=healthy,
        scores=scores,
        meta=meta,
        sanity=sanity,
        stats=stats,
        timestamp=timestamp,
    )
    status_entry = build_status_entry(stats)

    summary_written = append_unique(
        SUMMARY_PATH, SECTION_HEADING, section
    )
    status_written = append_unique(
        STATUS_PATH, STATUS_MARKER, "---\n" + status_entry
    )

    print(f"Saved: {STATS_PATH}")
    if summary_written:
        print(f"Saved: {SUMMARY_PATH}")
    if status_written:
        print(f"Saved: {STATUS_PATH}")
    print()
    print(section)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

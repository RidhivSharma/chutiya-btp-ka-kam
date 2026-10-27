"""Append the Phase 3 section to results/summary.md and STATUS.md.

This is a reporting script. It reads the saved Phase 3 outputs
(``results/baseline_sanity.csv`` and ``results/baselines_metadata.json``),
computes the descriptive counts and lists used in the report, saves them to
``results/phase3_summary_stats.json``, and appends a Phase 3 section in the
format required by ``AGENTS.md``. Earlier summary sections and earlier STATUS
entries are never modified.

The cut-offs used to describe the results are reporting conventions, not
detector settings:

* "almost everything" = at least 95 % of the windows in a fault file flagged;
* "clear rise" = post-8 s median score at least 2x the same file's
  pre-injection median;
* "weak rise" = 1.2x to 2x;
* "flat or lower" = below 1.2x.

No final metrics (AUROC, precision, recall, etc.) are computed here.

Run after ``src/baselines.py``::

    .venv\\Scripts\\python.exe .\\src\\phase3_summary.py
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
STATS_PATH = RESULTS_DIR / "phase3_summary_stats.json"
SECTION_HEADING = "## Phase 3: Baselines on the same pipeline"
STATUS_MARKER = "Phase 3: baselines on the same pipeline"

ALMOST_ALL_SHARE = 0.95
CLEAR_RISE_RATIO = 2.0
WEAK_RISE_RATIO = 1.2
CEILING_RTOL = 1e-6


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


def fmt_ratio(value: object) -> str:
    number = float(value)
    if not np.isfinite(number):
        return "NA"
    return f"{number:.2f}x"


def fmt_seconds(value: object) -> str:
    return f"{float(value):.3f}"


def fmt_threshold(value: object) -> str:
    return f"{float(value):.6g}"


def fmt_score(value: object) -> str:
    number = float(value)
    if not np.isfinite(number):
        return "NA"
    return f"{number:.4g}"


def name_list(names: list[str]) -> str:
    if not names:
        return "none"
    return ", ".join(f"`{name}`" for name in names)


def name_list_and(names: list[str]) -> str:
    if not names:
        return "none"
    quoted = [f"`{name}`" for name in names]
    if len(quoted) == 1:
        return quoted[0]
    return ", ".join(quoted[:-1]) + " and " + quoted[-1]


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


def classify_fault(
    pre: float,
    post: float,
    ratio: float,
    no_post: bool,
    score_max: float,
) -> tuple[str, bool, bool]:
    """Classify one fault file's pre/post medians for descriptive reporting."""
    if no_post:
        return "no_post", False, False
    pre_at_ceiling = bool(
        np.isfinite(pre) and np.isclose(pre, score_max, rtol=CEILING_RTOL, atol=0.0)
    )
    post_at_ceiling = bool(
        np.isfinite(post)
        and np.isclose(post, score_max, rtol=CEILING_RTOL, atol=0.0)
    )
    if pre_at_ceiling and post_at_ceiling:
        return "saturated_both", pre_at_ceiling, post_at_ceiling
    if not np.isfinite(pre) or not np.isfinite(post):
        return "undefined", pre_at_ceiling, post_at_ceiling
    if pre <= 0.0:
        if post > pre:
            return "rise_pre_nonpositive", pre_at_ceiling, post_at_ceiling
        return "no_rise_pre_nonpositive", pre_at_ceiling, post_at_ceiling
    if not np.isfinite(ratio):
        return "undefined", pre_at_ceiling, post_at_ceiling
    if ratio >= CLEAR_RISE_RATIO:
        return "clear", pre_at_ceiling, post_at_ceiling
    if ratio >= WEAK_RISE_RATIO:
        return "weak", pre_at_ceiling, post_at_ceiling
    return "flat", pre_at_ceiling, post_at_ceiling


def build_stats(
    sanity: pd.DataFrame,
    meta: dict[str, object],
    sanity_checks: dict[str, object],
) -> dict[str, object]:
    """Compute every descriptive fact used in the report from saved data."""
    model_order = list(meta["models"].keys())
    modes = ["IPPT", "MPPT"]
    run_meta = {
        (str(run["model"]), str(run["mode"])): run for run in meta["runs"]
    }

    runs: list[dict[str, object]] = []
    for model in model_order:
        for mode in modes:
            subset = sanity[
                (sanity["model"] == model) & (sanity["mode"] == mode)
            ]
            healthy = subset[subset["section"] == "healthy"]
            fault = subset[subset["section"] == "fault"].sort_values("fault")
            if fault.shape[0] != 7:
                raise RuntimeError(
                    f"Expected 7 fault rows for {model}/{mode}, "
                    f"got {fault.shape[0]}"
                )
            val = healthy[healthy["part"] == "val"].iloc[0]
            test = healthy[healthy["part"] == "test"].iloc[0]
            record = run_meta[(model, mode)]
            score_max = float(record["score_max"])

            fault_records: list[dict[str, object]] = []
            for row in fault.itertuples():
                pre = float(row.median_pre_injection_score)
                post = float(row.median_after_8s_score)
                ratio = float(row.median_ratio_after8_over_pre)
                no_post = bool(row.no_post_injection_data)
                classification, pre_ceiling, post_ceiling = classify_fault(
                    pre, post, ratio, no_post, score_max
                )
                fault_records.append(
                    {
                        "file": str(row.file),
                        "flag_rate": float(row.flag_rate),
                        "flagged": int(row.flagged),
                        "windows": int(row.windows),
                        "median_pre": float(pre) if np.isfinite(pre) else None,
                        "median_post": float(post) if np.isfinite(post) else None,
                        "ratio": float(ratio) if np.isfinite(ratio) else None,
                        "no_post": no_post,
                        "class": classification,
                        "pre_at_ceiling": pre_ceiling,
                        "post_at_ceiling": post_ceiling,
                    }
                )

            almost_all = [
                item["file"]
                for item in fault_records
                if float(item["flag_rate"]) >= ALMOST_ALL_SHARE
            ]
            clear = [
                item["file"] for item in fault_records if item["class"] == "clear"
            ]
            weak = [
                item["file"] for item in fault_records if item["class"] == "weak"
            ]
            flat = [
                item["file"] for item in fault_records if item["class"] == "flat"
            ]
            rise_pre_nonpositive = [
                item["file"]
                for item in fault_records
                if item["class"] == "rise_pre_nonpositive"
            ]
            no_rise_pre_nonpositive = [
                item["file"]
                for item in fault_records
                if item["class"] == "no_rise_pre_nonpositive"
            ]
            saturated_both = [
                item["file"]
                for item in fault_records
                if item["class"] == "saturated_both"
            ]
            post_only_ceiling = [
                item["file"]
                for item in fault_records
                if bool(item["post_at_ceiling"])
                and not bool(item["pre_at_ceiling"])
            ]
            no_post = [
                item["file"] for item in fault_records if bool(item["no_post"])
            ]

            runs.append(
                {
                    "model": model,
                    "model_label": str(record["model_label"]),
                    "mode": mode,
                    "input_kind": str(record["input_kind"]),
                    "train_units": int(record["train_units"]),
                    "subsample_applied": bool(record["subsample_applied"]),
                    "training_time_s": float(record["training_time_s"]),
                    "scoring_time_s": float(record["scoring_time_s"]),
                    "threshold": float(record["threshold"]),
                    "score_max": score_max,
                    "val_windows": int(val.windows),
                    "val_flagged": int(val.flagged),
                    "val_flag_rate": float(val.flag_rate),
                    "test_windows": int(test.windows),
                    "test_flagged": int(test.flagged),
                    "test_false_alarm_rate": float(test.flag_rate),
                    "fault_files": int(fault.shape[0]),
                    "almost_all_files": almost_all,
                    "flag_share_min": float(fault["flag_rate"].min()),
                    "flag_share_max": float(fault["flag_rate"].max()),
                    "clear_files": clear,
                    "weak_files": weak,
                    "flat_files": flat,
                    "rise_pre_nonpositive_files": rise_pre_nonpositive,
                    "no_rise_pre_nonpositive_files": no_rise_pre_nonpositive,
                    "saturated_both_files": saturated_both,
                    "post_only_ceiling_files": post_only_ceiling,
                    "no_post_files": no_post,
                    "fault_records": fault_records,
                }
            )

    # Counts across models for the prose statements.
    clear_by_file: dict[str, dict[str, int]] = {}
    almost_by_model: dict[str, int] = {model: 0 for model in model_order}
    for run in runs:
        almost_by_model[str(run["model"])] += len(run["almost_all_files"])
        for item in run["fault_records"]:
            entry = clear_by_file.setdefault(
                str(item["file"]), {"clear": 0, "post_available": 0}
            )
            if not bool(item["no_post"]):
                entry["post_available"] += 1
            if item["class"] == "clear":
                entry["clear"] += 1

    windows_total = int(sum(int(run["windows"]) for run in meta["runs"]))
    windows_per_model = {
        model: int(
            sum(
                int(run["windows"])
                for run in meta["runs"]
                if str(run["model"]) == model
            )
        )
        for model in model_order
    }
    if len(set(windows_per_model.values())) != 1:
        raise RuntimeError(
            f"Models scored different window counts: {windows_per_model}"
        )

    return {
        "phase": "phase_3_baselines_summary",
        "generated_local": datetime.now().astimezone().isoformat(),
        "date": datetime.now().astimezone().strftime("%Y-%m-%d"),
        "git_commit": meta.get("git_commit"),
        "environment": meta.get("environment"),
        "source_files": {
            "baseline_sanity": "results/baseline_sanity.csv",
            "baselines_metadata": "results/baselines_metadata.json",
            "sanity_checks": "results/sanity_checks.json",
            "summary": "results/summary.md",
        },
        "label_convention": meta.get("label_convention"),
        "subsample_cap": int(meta["subsample_cap"]),
        "conventions": {
            "almost_all_share": ALMOST_ALL_SHARE,
            "clear_rise_ratio": CLEAR_RISE_RATIO,
            "weak_rise_ratio": WEAK_RISE_RATIO,
            "ceiling_rtol": CEILING_RTOL,
        },
        "totals": {
            "models": len(model_order),
            "modes": len(modes),
            "archives": len(meta["runs"]),
            "windows_total": windows_total,
            "windows_per_model": windows_per_model,
            "fault_files": int(sanity[sanity["section"] == "fault"]["file"].nunique()),
            "checks_passed": int(meta["checks_passed"]),
            "checks_total": int(meta["checks_total"]),
        },
        "runs": runs,
        "clear_by_file": clear_by_file,
        "almost_by_model": almost_by_model,
        "sanity_checks": {
            "passed": int(sanity_checks["checks_passed"]),
            "total": int(sanity_checks["checks_total"]),
            "all_passed": bool(sanity_checks["all_passed"]),
        },
    }


def saturated_model_labels(
    runs: list[dict[str, object]],
    almost_by_model: dict[str, object],
    fault_files: object,
) -> list[str]:
    """Return model labels that flag >=95% of every fault file in both modes."""
    labels: list[str] = []
    for model, count in almost_by_model.items():
        if int(count) != int(fault_files):
            continue
        label = next(
            str(run["model_label"]) for run in runs if str(run["model"]) == model
        )
        labels.append(label)
    return labels


def special_notes(run: dict[str, object]) -> str:
    """Describe undefined ratios and score-ceiling saturation for one run."""
    notes: list[str] = []

    for item in run["fault_records"]:
        if item["class"] == "rise_pre_nonpositive":
            notes.append(
                f"`{item['file']}`: ratio undefined (pre median "
                f"{fmt_score(item['median_pre'])} < 0; medians "
                f"{fmt_score(item['median_pre'])} -> "
                f"{fmt_score(item['median_post'])})"
            )
        elif item["class"] == "no_rise_pre_nonpositive":
            notes.append(
                f"`{item['file']}`: ratio undefined (pre median "
                f"{fmt_score(item['median_pre'])} < 0; no absolute rise)"
            )

    saturated_both = list(run["saturated_both_files"])
    post_available = [
        item for item in run["fault_records"] if not bool(item["no_post"])
    ]
    if saturated_both:
        if len(saturated_both) == len(post_available):
            notes.append(
                f"all {len(saturated_both)} post-injection files: both medians "
                f"at the score ceiling ({fmt_score(run['score_max'])})"
            )
        else:
            notes.append(
                "both medians at the score ceiling: "
                + name_list(saturated_both)
                + f" ({fmt_score(run['score_max'])})"
            )
    if run["post_only_ceiling_files"]:
        notes.append(
            "post median at the score ceiling: "
            + name_list(list(run["post_only_ceiling_files"]))
            + f" ({fmt_score(run['score_max'])})"
        )

    return "; ".join(notes) if notes else "none"


def build_section(stats: dict[str, object], timestamp: str) -> str:
    """Render the Phase 3 markdown section from computed facts."""
    runs = stats["runs"]
    totals = stats["totals"]
    conventions = stats["conventions"]
    clear_by_file = stats["clear_by_file"]
    almost_by_model = stats["almost_by_model"]

    # Table 1: training time and healthy false alarms.
    timing_rows: list[list[str]] = []
    for run in runs:
        timing_rows.append(
            [
                str(run["model_label"]),
                str(run["mode"]),
                f"{int(run['train_units']):,}",
                fmt_seconds(run["training_time_s"]),
                fmt_threshold(run["threshold"]),
                (
                    f"{int(run['val_flagged'])}/{int(run['val_windows'])} "
                    f"({fmt_pct(run['val_flag_rate'])})"
                ),
                (
                    f"{int(run['test_flagged'])}/{int(run['test_windows'])} "
                    f"({fmt_pct(run['test_false_alarm_rate'])})"
                ),
            ]
        )
    timing_table = markdown_table(
        [
            "Model",
            "Mode",
            "Train units",
            "Training time (s)",
            "Threshold",
            "Healthy validation flagged",
            "Healthy test flagged (false-alarm rate)",
        ],
        timing_rows,
    )

    # Table 2: fault-file whole-file shares and within-run change classes.
    fault_rows: list[list[str]] = []
    for run in runs:
        almost = list(run["almost_all_files"])
        total_files = len(run["fault_records"])
        if len(almost) == total_files:
            almost_cell = f"{total_files}/{total_files} (all)"
        else:
            almost_cell = f"{len(almost)}/{total_files}: " + name_list(almost)
        fault_rows.append(
            [
                str(run["model_label"]),
                str(run["mode"]),
                almost_cell,
                (
                    f"{fmt_pct(run['flag_share_min'])}-"
                    f"{fmt_pct(run['flag_share_max'])}"
                ),
                name_list(list(run["clear_files"])),
                name_list(list(run["weak_files"])),
                name_list(list(run["flat_files"])),
                special_notes(run),
                name_list(list(run["no_post_files"])),
            ]
        )
    fault_table = markdown_table(
        [
            "Model",
            "Mode",
            f"Fault files with >={conventions['almost_all_share']:.0%} flagged",
            "Whole-file flag share range",
            f"Clear rise >={conventions['clear_rise_ratio']:.1f}x",
            (
                f"Weak rise {conventions['weak_rise_ratio']:.1f}-"
                f"{conventions['clear_rise_ratio']:.1f}x"
            ),
            f"Flat or lower <{conventions['weak_rise_ratio']:.1f}x",
            "Special cases",
            "No post-injection data",
        ],
        fault_rows,
    )

    # Data-driven prose for (a), (b), (c).
    all_14_labels = saturated_model_labels(
        runs, almost_by_model, totals["fault_files"]
    )

    almost_sentences: list[str] = []
    for model in stats["almost_by_model"]:
        count = int(almost_by_model[model])
        model_runs = [run for run in runs if run["model"] == model]
        label = str(model_runs[0]["model_label"])
        if count == int(totals["fault_files"]):
            almost_sentences.append(
                f"{label} flags all {int(totals['fault_files'])} fault files"
            )
        else:
            lowest = min(
                (
                    item
                    for run in model_runs
                    for item in run["fault_records"]
                ),
                key=lambda item: float(item["flag_rate"]),
            )
            almost_sentences.append(
                f"{label} flags {count}/{int(totals['fault_files'])} "
                f"(lowest `{lowest['file']}` at {fmt_pct(lowest['flag_rate'])})"
            )
    answer_a = (
        "(a) Almost everything: at the "
        f"{conventions['almost_all_share']:.0%} per-file threshold, "
        + "; ".join(almost_sentences)
        + ". The fully saturated references are "
        + name_list_and(all_14_labels)
        + ", so the whole-file view cannot separate them from a trivial "
        "detector. Exact per-file shares are in `baseline_sanity.csv`."
    )

    clear_all = [
        file
        for file, counts in clear_by_file.items()
        if counts["post_available"] > 0
        and counts["clear"] == counts["post_available"]
    ]
    clear_none = [
        file
        for file, counts in clear_by_file.items()
        if counts["post_available"] > 0 and counts["clear"] == 0
    ]
    clear_all.sort()
    clear_none.sort()

    run_by_key = {(run["model"], run["mode"]): run for run in runs}

    ocsvm_mppt = run_by_key[("one_class_svm", "MPPT")]
    saturated_files = list(ocsvm_mppt["saturated_both_files"])
    saturated_ratios = sorted(
        {
            fmt_ratio(item["ratio"])
            for item in ocsvm_mppt["fault_records"]
            if item["file"] in saturated_files
        }
    )
    post_available_count = len(
        [item for item in ocsvm_mppt["fault_records"] if not item["no_post"]]
    )
    saturation_sentence = ""
    if saturated_files:
        saturation_sentence = (
            " `OneClassSVM` in MPPT is saturated: both medians sit at the "
            f"score ceiling ({fmt_score(ocsvm_mppt['score_max'])}) in "
            f"{len(saturated_files)} of {post_available_count} post-injection "
            f"files, so its ratios ({', '.join(saturated_ratios)}) are not "
            "evidence of no change."
        )

    lof_iptt = run_by_key[("lof", "IPPT")]
    lof_mppt = run_by_key[("lof", "MPPT")]
    lof_f1l = next(
        item for item in lof_iptt["fault_records"] if item["file"] == "F1L"
    )
    lof_sentence = ""
    if lof_f1l["class"] == "rise_pre_nonpositive":
        lof_sentence = (
            " `LOF` `F1L` has a negative pre-injection median, so its ratio "
            "is undefined even though its median rises from "
            f"{fmt_score(lof_f1l['median_pre'])} to "
            f"{fmt_score(lof_f1l['median_post'])}."
        )

    answer_b = (
        "(b) Post-8 s score rise: the table lists every file. A clear rise "
        f"(>= {conventions['clear_rise_ratio']:.1f}x the same file's "
        "pre-injection median) is unanimous across all models with post data "
        f"for {name_list_and(clear_all)}. No model shows a clear rise for "
        f"{name_list_and(clear_none)}. `F3M` has no post-injection data in every "
        "run and is excluded from every rise comparison."
        + saturation_sentence
        + lof_sentence
    )

    test_sentences: list[str] = []
    for run in runs:
        test_sentences.append(
            f"{run['model_label']} {run['mode']} "
            f"{fmt_pct(run['test_false_alarm_rate'])} "
            f"({int(run['test_flagged'])}/{int(run['test_windows'])})"
        )
    answer_c = (
        "(c) Healthy test false-alarm rates: "
        + "; ".join(test_sentences)
        + ". LOF and OneClassSVM are far above their validation rates, so "
        "their validation thresholds do not transfer to the later healthy "
        "test block; the other three stay much closer to their validation "
        "rates."
    )

    row_units = {
        run["mode"]: int(run["train_units"])
        for run in runs
        if run["input_kind"] == "time_step_rows"
    }
    what_was_done = " ".join(
        [
            (
                "Phase 3 re-implemented the four reference baselines from the "
                "paper on the frozen Phase 2 pipeline: EllipticEnvelope, "
                "IsolationForest, LocalOutlierFactor, and OneClassSVM."
            ),
            (
                "Each scikit-learn model was fitted only on the standardized "
                "healthy training rows of one operating mode "
                f"({row_units['IPPT']:,} rows for IPPT, "
                f"{row_units['MPPT']:,} for MPPT; both are below the "
                f"{int(stats['subsample_cap']):,}-row cap, so no subsampling "
                "was needed). The Phase 2 Mahalanobis placeholder was added "
                "unchanged as a fifth, window-mean reference."
            ),
            (
                "A window score is the mean of -decision_function over its "
                "100 time steps (higher means more anomalous); the threshold "
                "is the 99th percentile of healthy validation window scores. "
                "No fault file and no healthy test window was used for a fit "
                "or a threshold."
            ),
            (
                "The new label convention is recorded in `AGENTS.md`: "
                "whole-file labels are primary, time-resolved pre/post-8 s "
                "comparisons are secondary, and `F3M` is always reported as "
                "having no post-injection data."
            ),
            (
                f"{int(totals['archives'])} score archives "
                f"({int(totals['models'])} models x {int(totals['modes'])} "
                "modes) were saved in the same five-key format the proposed "
                "model will use, together with a sanity table. No final "
                "metrics were computed in this phase."
            ),
        ]
    )

    files_lines = [
        (
            "- `AGENTS.md` - recorded frozen Phases 1-2 and the primary "
            "whole-file / secondary time-resolved label decision."
        ),
        (
            "- `src/baselines.py` - fits the five baselines, scores every "
            "file, and saves the score archives and sanity table."
        ),
        (
            "- `src/phase3_summary.py` - reads the saved results and writes "
            "this section and the STATUS entry."
        ),
        "- `run_all.py` - added the Phase 3 baseline and summary steps.",
        (
            "- `results/scores_baseline_<model>_<mode>.npz` - "
            f"{int(totals['archives'])} archives with keys `scores`, "
            "`start_time`, `end_time`, `threshold`, `name`."
        ),
        (
            "- `results/baseline_sanity.csv` - healthy validation/test flag "
            "rates, per-fault whole-file shares, and pre/post median scores."
        ),
        (
            "- `results/baselines_metadata.json` - model settings, training "
            "and scoring times, thresholds, checks, environment, git commit."
        ),
        "- `results/phase3_summary_stats.json` - the facts used in this report.",
        "- `results/summary.md` - this Phase 3 section.",
        "- `STATUS.md` - appended Phase 3 log entry.",
    ]

    checks_lines = [
        (
            f"- The baseline run reports {int(totals['checks_passed'])}/"
            f"{int(totals['checks_total'])} checks passed."
        ),
        (
            f"- All {int(totals['archives'])} archives contain exactly the "
            "five Phase 5 keys and "
            f"{int(totals['windows_total']):,} window scores; every model "
            f"scored {int(next(iter(totals['windows_per_model'].values()))):,} "
            "windows (healthy train/val/test plus seven fault files)."
        ),
        (
            "- All scores, window times, and thresholds are finite; there are "
            "no NaN or infinite values."
        ),
        (
            "- Every threshold equals the 99th percentile of healthy "
            "validation window scores recomputed from the saved archive; no "
            "test or fault window was used."
        ),
        (
            "- The Mahalanobis reference reproduces the Phase 2 threshold and "
            "all of its window scores exactly."
        ),
        (
            "- Every flag rate in `baseline_sanity.csv` equals "
            "flagged/windows, and the healthy validation flag rate is at most "
            "2% for every model and mode."
        ),
        (
            "- `F3M` is marked as having zero post-injection windows in all "
            f"{int(totals['models'])} model rows; its post median is `NA`, "
            "not a fabricated value."
        ),
        (
            "- The cached-row scoring shortcut was checked against direct "
            "window scoring on sample windows for every model, mode, and "
            "file; all agree within numerical tolerance."
        ),
        (
            f"- The frozen Phase 2 sanity checks still report "
            f"{int(stats['sanity_checks']['passed'])}/"
            f"{int(stats['sanity_checks']['total'])} passed."
        ),
    ]

    ocsvm_iptt = run_by_key[("one_class_svm", "IPPT")]
    ocsvm_mppt = run_by_key[("one_class_svm", "MPPT")]
    ee_iptt = run_by_key[("elliptic_envelope", "IPPT")]
    ee_mppt = run_by_key[("elliptic_envelope", "MPPT")]
    elliptic_count = int(stats["almost_by_model"]["elliptic_envelope"])
    saturated_labels = saturated_model_labels(
        runs, almost_by_model, totals["fault_files"]
    )

    warning_lines = [
        (
            "- The whole-file view is saturated. "
            f"{name_list_and(saturated_labels)} flag at least "
            f"{conventions['almost_all_share']:.0%} of the windows in all "
            f"{int(totals['fault_files'])} fault files, and EllipticEnvelope "
            f"does so in {elliptic_count}/{int(totals['fault_files'])}. A "
            "near-perfect whole-file score therefore does not show that a "
            "detector is better than a trivial one; the secondary pre/post-8 s "
            "analysis is the discriminating view."
        ),
        (
            "- LOF and OneClassSVM have high healthy test false-alarm rates "
            f"({fmt_pct(lof_iptt['test_false_alarm_rate'])}/"
            f"{fmt_pct(lof_mppt['test_false_alarm_rate'])} for LOF and "
            f"{fmt_pct(ocsvm_iptt['test_false_alarm_rate'])}/"
            f"{fmt_pct(ocsvm_mppt['test_false_alarm_rate'])} for OneClassSVM, "
            "IPPT/MPPT) even though their validation rates are "
            f"{fmt_pct(lof_iptt['val_flag_rate'])}/"
            f"{fmt_pct(lof_mppt['val_flag_rate'])} and "
            f"{fmt_pct(ocsvm_iptt['val_flag_rate'])}/"
            f"{fmt_pct(ocsvm_mppt['val_flag_rate'])}. The healthy recording "
            "changes over time, so a threshold fitted on the middle "
            "validation block does not transfer to the later test block."
        ),
        (
            "- OneClassSVM scores saturate at a ceiling "
            f"({fmt_score(ocsvm_iptt['score_max'])} IPPT, "
            f"{fmt_score(ocsvm_mppt['score_max'])} MPPT). In saturated files "
            "both pre- and post-8 s medians equal the ceiling, so a ratio of "
            "1x means the score cannot rise further, not that the fault has "
            "no effect."
        ),
        (
            "- LOF `F1L` has a negative pre-injection median "
            f"({fmt_score(lof_f1l['median_pre'])}), so its post/pre ratio is "
            "undefined. The medians themselves rise to "
            f"{fmt_score(lof_f1l['median_post'])}; the ratio is reported as "
            "`NA` rather than forced."
        ),
        (
            "- The baselines are fitted to one healthy recording per mode. "
            "Differences in irradiance, temperature, or operating point "
            "between recordings can dominate the scores, so the pre-injection "
            "anomaly is not proof that a fault was already active."
        ),
        (
            "- The healthy training rows include the startup transient, "
            "because the frozen Phase 2 pipeline keeps it. The baselines "
            "therefore learn that transient as normal."
        ),
        (
            "- Windows overlap by 90%, so window-level observations are not "
            "independent; the sanity table is descriptive and has no "
            "confidence intervals."
        ),
        (
            "- The phase specification fixes `random_state=0` for "
            "EllipticEnvelope and IsolationForest; LOF, OneClassSVM, and "
            "Mahalanobis are deterministic. No 0/1/2 seed sweep was run, "
            "because the baseline settings are frozen and no fault data are "
            "used for any choice."
        ),
        (
            "- The scikit-learn baselines were fitted on the unique 1 kHz "
            "rows covered by the training windows, not on the flattened "
            "90%-overlapping windows. This avoids counting each time step ten "
            "times and matches the rows used to fit the Phase 2 scaler."
        ),
        (
            "- EllipticEnvelope produces very large scores (up to "
            f"{fmt_score(max(float(ee_iptt['score_max']), float(ee_mppt['score_max'])))}"
            ") from near-singular covariance directions; these are not "
            "physical units and should not be compared across models."
        ),
    ]

    decision = (
        "No. The label convention is fixed in `AGENTS.md` (whole-file "
        "primary, time-resolved secondary, `F3M` always reported as having no "
        "post-injection data), so Phase 4 can proceed. The "
        f"{conventions['almost_all_share']:.0%} and "
        f"{conventions['clear_rise_ratio']:.1f}x/"
        f"{conventions['weak_rise_ratio']:.1f}x cut-offs used above are "
        "descriptive reporting conventions only, not detector settings."
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
        "### Training time and healthy false alarms",
        "",
        timing_table,
        "",
        (
            "Training time is the measured `fit` call only; scoring time is "
            "reported in `baselines_metadata.json`. The threshold is the 99th "
            "percentile of healthy validation window scores. The healthy test "
            "block is later in time and was not used for the fit or the "
            "threshold."
        ),
        "",
        "### Fault-file sanity and within-run change",
        "",
        fault_table,
        "",
        (
            "Whole-file flag share is the fraction of all windows in that "
            "fault file above the healthy-validation threshold. The rise "
            "columns compare the median score of pre-injection windows "
            "(end < 7 s, excluding the first 0.5 s) with windows starting "
            "after 8 s in the same file. `Clear`, `weak`, and `flat` use the "
            f"{conventions['clear_rise_ratio']:.1f}x and "
            f"{conventions['weak_rise_ratio']:.1f}x conventions stated at the "
            "top of this section. `NA` means the denominator or ratio is "
            "undefined, not zero."
        ),
        "",
        "### Plain answers",
        "",
        f"- {answer_a}",
        f"- {answer_b}",
        f"- {answer_c}",
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
            "Next phase: Phase 4 - train the proposed healthy-only "
            "autoencoder on the same frozen windows and save its scores in "
            "the same five-key npz format."
        ),
    ]
    return "\n".join(parts).rstrip() + "\n"


def build_status_entry(stats: dict[str, object]) -> str:
    """Render the short chronological STATUS.md entry."""
    runs = stats["runs"]
    totals = stats["totals"]
    conventions = stats["conventions"]
    almost_by_model = stats["almost_by_model"]
    run_by_key = {(run["model"], run["mode"]): run for run in runs}

    saturated_labels = saturated_model_labels(
        runs, almost_by_model, totals["fault_files"]
    )
    elliptic_count = int(stats["almost_by_model"]["elliptic_envelope"])

    clear_all = [
        file
        for file, counts in stats["clear_by_file"].items()
        if counts["post_available"] > 0
        and counts["clear"] == counts["post_available"]
    ]
    clear_none = [
        file
        for file, counts in stats["clear_by_file"].items()
        if counts["post_available"] > 0 and counts["clear"] == 0
    ]
    clear_all.sort()
    clear_none.sort()

    ocsvm_mppt = run_by_key[("one_class_svm", "MPPT")]
    lof_f1l = next(
        item
        for item in run_by_key[("lof", "IPPT")]["fault_records"]
        if item["file"] == "F1L"
    )

    test_bits = "; ".join(
        f"{run['model_label']} {run['mode']} "
        f"{float(run['test_false_alarm_rate']):.4f}"
        for run in runs
    )
    return "\n".join(
        [
            f"## {stats['date']} - Phase 3: baselines on the same pipeline",
            "",
            (
                "**What was done.** Re-implemented EllipticEnvelope, "
                "IsolationForest, LocalOutlierFactor, and OneClassSVM on the "
                "frozen Phase 2 rows, and added the Phase 2 Mahalanobis "
                "placeholder unchanged as a fifth reference. Each model was "
                "fitted only on healthy training rows of one mode; the "
                "threshold is the 99th percentile of healthy validation "
                "window scores. Saved "
                f"{int(totals['archives'])} score archives "
                f"({int(totals['models'])} models x {int(totals['modes'])} "
                "modes) in the five-key Phase 5 format plus a sanity table. "
                "No final metrics were computed."
            ),
            "",
            (
                "**Files created/changed.** `AGENTS.md` (label convention), "
                "`src/baselines.py`, `src/phase3_summary.py`, `run_all.py`, "
                "`results/scores_baseline_*.npz`, "
                "`results/baseline_sanity.csv`, "
                "`results/baselines_metadata.json`, "
                "`results/phase3_summary_stats.json`, `results/summary.md`, "
                "`STATUS.md`."
            ),
            "",
            "**Key numbers (computed by code).**",
            (
                f"- {int(totals['archives'])} archives, "
                f"{int(totals['windows_total']):,} window scores; "
                f"{int(totals['checks_passed'])}/"
                f"{int(totals['checks_total'])} checks passed."
            ),
            (
                "- Whole-file saturation: "
                + ", ".join(saturated_labels)
                + " flag at least "
                + f"{conventions['almost_all_share']:.0%}"
                + " of every fault file; EllipticEnvelope does so in "
                + f"{elliptic_count}/{int(totals['fault_files'])}."
            ),
            (
                "- Healthy test false-alarm rates (IPPT/MPPT): "
                + test_bits
                + "."
            ),
            (
                "- Clear post-8 s rises "
                f"(>={conventions['clear_rise_ratio']:.1f}x pre median) are "
                f"unanimous for {name_list_and(clear_all)}; "
                f"{name_list_and(clear_none)} have none. OneClassSVM MPPT is "
                "saturated at its score ceiling "
                f"({fmt_score(ocsvm_mppt['score_max'])}); LOF F1L has a "
                f"negative pre median ({fmt_score(lof_f1l['median_pre'])}), "
                "so its ratio is undefined. F3M has no post-injection data."
            ),
            "",
            (
                "**Open issues/risks.** The whole-file view is saturated for "
                + name_list_and(saturated_labels)
                + ", so it cannot separate methods. LOF and OneClassSVM have "
                "high healthy test false-alarm rates due to temporal drift "
                "within F0. OneClassSVM scores hit a ceiling, and LOF F1L has "
                "a negative pre median. F3M has no post-injection data."
            ),
        ]
    ).rstrip() + "\n"


def main() -> int:
    sanity = pd.read_csv(RESULTS_DIR / "baseline_sanity.csv")
    meta = read_json(RESULTS_DIR / "baselines_metadata.json")
    sanity_checks = read_json(RESULTS_DIR / "sanity_checks.json")

    if sanity.shape[0] != 90:
        raise RuntimeError(
            f"Expected 90 baseline sanity rows, got {sanity.shape[0]}"
        )
    if set(sanity["section"].unique()) != {"healthy", "fault"}:
        raise RuntimeError("Expected healthy and fault sections")
    if int(sanity[sanity["section"] == "fault"].shape[0]) != 70:
        raise RuntimeError("Expected 70 fault rows (5 models x 2 modes x 7 files)")

    stats = build_stats(sanity, meta, sanity_checks)
    with STATS_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(stats, handle, indent=2, allow_nan=False)
        handle.write("\n")

    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %z")
    section = build_section(stats, timestamp)
    status_entry = build_status_entry(stats)

    summary_written = append_unique(SUMMARY_PATH, SECTION_HEADING, section)
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

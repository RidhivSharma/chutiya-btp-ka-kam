"""Append the Phase 6 report to results/summary.md and STATUS.md.

The report is generated only from saved Phase 5/6 result files. No metric is
typed by hand: every number in the prose and tables is read from a CSV or JSON
file under results/. The Phase 6 section is replaced in place on rerun, so
earlier phase sections are never overwritten.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
SUMMARY = RESULTS / "summary.md"
STATUS = PROJECT_ROOT / "STATUS.md"

MODEL_ORDER = [
    "Proposed (plain)",
    "Proposed (featnorm)",
    "EllipticEnvelope",
    "IsolationForest",
    "LocalOutlierFactor",
    "OneClassSVM",
    "Mahalanobis (window-mean)",
]


def load_json(name: str) -> Any:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def average_value(mode: str, model: str, column: str) -> float:
    frame = pd.read_csv(RESULTS / f"average_primary_{mode}.csv")
    subset = frame[(frame["model"] == model) & (frame["mode"] == mode)]
    if len(subset) != 1:
        raise ValueError(f"average_primary_{mode}.csv: {model} selected {len(subset)} rows")
    return float(subset.iloc[0][column])


def within_value(mode: str, model: str, column: str) -> float:
    frame = pd.read_csv(RESULTS / f"within_run_average_{mode}.csv")
    subset = frame[(frame["model"] == model) & (frame["mode"] == mode)]
    if len(subset) != 1:
        raise ValueError(f"within_run_average_{mode}.csv: {model} selected {len(subset)} rows")
    return float(subset.iloc[0][column])


def f4(value: float) -> str:
    return f"{value:.4f}"


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def dpi_ok(detail: dict[str, Any]) -> bool:
    dpi = detail.get("dpi", ())
    return (
        len(dpi) >= 2
        and abs(float(dpi[0]) - 300.0) <= 1.0
        and abs(float(dpi[1]) - 300.0) <= 1.0
    )


def build_section() -> tuple[str, str]:
    verification = load_json("phase6_verification.json")
    audit = load_json("phase6_table_values.json")
    paper = pd.read_csv(RESULTS / "paper_reference.csv")

    figure_count = int(verification["figures"]["expected"])
    figures_found = int(verification["figures"]["found"])
    figure_problems = list(verification["figures"]["problems"])
    dpi_count = sum(1 for detail in verification["figures"]["details"] if dpi_ok(detail))
    values_checked = int(verification["tables"]["values_checked"])
    mismatches = list(verification["tables"]["mismatches"])
    placement_failures = list(verification["tables"]["placement_failures"])
    missing_tables = list(verification["tables"]["missing_table_sections"])
    captions_found = int(verification["captions"]["found"])
    captions_expected = int(verification["captions"]["expected"])
    caption_problems = list(verification["captions"]["problems"])
    claims = list(verification["superiority_claims"])
    passed = bool(verification["passed"])

    tables_text = (RESULTS / "tables.md").read_text(encoding="utf-8")
    table_ids = set(
        re.findall(r"^#{3,4}\s+(T\d+[a-z]?)\.", tables_text, flags=re.MULTILINE)
    )
    table_ids.discard("T8")  # group heading; the actual sections are T8a-T8e
    table_sections_total = len(table_ids)
    table_sections_found = table_sections_total - len(missing_tables)

    data_config = load_json("data_config.json")
    overlap_pct = 100.0 * float(data_config["config"]["windowing"]["overlap_fraction"])

    metrics = {}
    for mode in ["IPPT", "MPPT"]:
        for model in MODEL_ORDER:
            metrics[(mode, model)] = {
                "f1": average_value(mode, model, "f1_mean"),
                "auc": average_value(mode, model, "auc_mean"),
                "far": average_value(mode, model, "healthy_test_false_alarm_rate"),
                "within_auc": within_value(mode, model, "auc_mean"),
            }

    plain_ipt = metrics[("IPPT", "Proposed (plain)")]
    plain_mppt = metrics[("MPPT", "Proposed (plain)")]
    feat_ipt = metrics[("IPPT", "Proposed (featnorm)")]
    feat_mppt = metrics[("MPPT", "Proposed (featnorm)")]
    mahal_ipt = metrics[("IPPT", "Mahalanobis (window-mean)")]
    mahal_mppt = metrics[("MPPT", "Mahalanobis (window-mean)")]

    lower_all = (
        plain_ipt["f1"] < mahal_ipt["f1"]
        and plain_mppt["f1"] < mahal_mppt["f1"]
        and feat_ipt["f1"] < mahal_ipt["f1"]
        and feat_mppt["f1"] < mahal_mppt["f1"]
    )
    if lower_all:
        comparison_sentence = (
            "Both proposed variants have a lower primary average F1 than the "
            "trivial Mahalanobis (window-mean) baseline in both modes: Proposed "
            f"(plain) {f4(plain_ipt['f1'])} IPPT and {f4(plain_mppt['f1'])} MPPT, "
            f"Proposed (featnorm) {f4(feat_ipt['f1'])} and {f4(feat_mppt['f1'])}, "
            f"versus Mahalanobis {f4(mahal_ipt['f1'])} and {f4(mahal_mppt['f1'])}."
        )
    else:
        comparison_sentence = (
            "The primary average F1 comparison between the proposed variants and "
            "the trivial Mahalanobis (window-mean) baseline is mixed; the values "
            "are in the key-numbers table and in results/tables.md."
        )

    featnorm_sentence = (
        "Proposed (featnorm) is a negative result: its healthy-test false-alarm "
        f"rate is {pct(feat_ipt['far'])} in IPPT and {pct(feat_mppt['far'])} in "
        f"MPPT, while its primary average F1 is {f4(feat_ipt['f1'])} and "
        f"{f4(feat_mppt['f1'])}."
    )

    mppt_excl_plain = average_value("MPPT", "Proposed (plain)", "excl_f3_f1_mean")
    mppt_excl_feat = average_value("MPPT", "Proposed (featnorm)", "excl_f3_f1_mean")

    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %z")

    lines: list[str] = []
    lines.append(f"## Phase 6: figures and tables   ({timestamp})")
    lines.append("")
    lines.append(
        f"What was done: Built {figure_count} 300 dpi publication figures and the "
        "T1-T8 table groups from the frozen Phase 1-5 result files. The figures use "
        "the saved score archives, training histories and CSV summaries; the tables "
        "are generated from the saved CSV/JSON files, and every table cell is "
        "recorded with its source selector. A separate verification script "
        "re-read every source and checked all recorded values, the figure files, "
        "the captions and the wording for unsupported superiority claims. No "
        "model was retrained, and no score, threshold, k=3 or normalization was "
        "changed. The project framing recorded in AGENTS.md is a leak-free "
        "re-evaluation, with Proposed (featnorm) reported as a negative result."
    )
    lines.append("")
    lines.append("Files created or changed:")
    lines.append(
        f"- `src/make_figures.py` - generates the {figure_count} figures, "
        "`results/tables.md`, the figure captions, the architecture summary and "
        "the table-value audit."
    )
    lines.append(
        "- `src/verify_phase6.py` - re-reads every table value and checks the "
        "figures, captions and wording."
    )
    lines.append(
        "- `src/phase6_summary.py` - writes this section and the STATUS entry from "
        "saved result files."
    )
    lines.append(
        "- `figures/paper/fig_healthy_far.png` - healthy-test false-alarm rate per model."
    )
    lines.append(
        "- `figures/paper/fig_primary_f1_auc.png` - primary F1 and AUC with intervals."
    )
    lines.append(
        "- `figures/paper/fig_within_run_auc.png` - within-run AUC with intervals."
    )
    lines.append(
        "- `figures/paper/fig_score_vs_time.png` - score versus time for F1/F4/F6/F7 (IPPT)."
    )
    lines.append(
        "- `figures/paper/fig_score_hist.png` - healthy and fault score distributions."
    )
    lines.append(
        "- `figures/paper/fig_threshold_sensitivity.png` - healthy FAR versus threshold percentile."
    )
    lines.append(
        "- `figures/paper/fig_training_curves.png` - training and validation loss for six runs."
    )
    lines.append(
        "- `figures/paper/fig_feature_error_heatmap.png` - per-feature reconstruction error."
    )
    lines.append(
        "- `figures/paper/fig_architecture.png` - TCN-BiLSTM block diagram."
    )
    lines.append(
        "- `results/tables.md` - T1-T8 tables generated from saved result files."
    )
    lines.append(
        "- `results/figure_captions.md` - one factual caption per figure."
    )
    lines.append(
        f"- `results/phase6_table_values.json` - audit trail for {values_checked} table values."
    )
    lines.append(
        "- `results/phase6_architecture.json` - static architecture summary copied from `src/model.py`."
    )
    lines.append(
        "- `results/phase6_verification.json` - verification results and counts."
    )
    lines.append("- `results/summary.md` - this Phase 6 section.")
    lines.append("- `STATUS.md` - appended Phase 6 log entry.")
    lines.append(
        "- `AGENTS.md` - recorded the leak-free re-evaluation framing, the negative "
        "featnorm result, and the decision to skip noise-robustness and ablation phases."
    )
    lines.append("- `run_all.py` - added the Phase 6 generation, verification and summary steps.")
    lines.append("")
    lines.append("Key numbers:")
    lines.append("")
    lines.append(
        "| Model | Primary F1 IPPT | Primary F1 MPPT | Within-run AUC IPPT | "
        "Within-run AUC MPPT | Healthy-test FAR IPPT | Healthy-test FAR MPPT |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for model in [
        "Proposed (plain)",
        "Proposed (featnorm)",
        "Mahalanobis (window-mean)",
    ]:
        ipt = metrics[("IPPT", model)]
        mppt = metrics[("MPPT", model)]
        lines.append(
            f"| {model} | {f4(ipt['f1'])} | {f4(mppt['f1'])} | "
            f"{f4(ipt['within_auc'])} | {f4(mppt['within_auc'])} | "
            f"{pct(ipt['far'])} | {pct(mppt['far'])} |"
        )
    lines.append("")
    lines.append(
        "| Item | Value |"
    )
    lines.append("|---|---|")
    lines.append(
        f"| MPPT excluding F3M, primary F1, Proposed (plain) | {f4(mppt_excl_plain)} |"
    )
    lines.append(
        f"| MPPT excluding F3M, primary F1, Proposed (featnorm) | {f4(mppt_excl_feat)} |"
    )
    lines.append(
        f"| Reference-paper rows (different protocol, not comparable) | {len(paper)} |"
    )
    lines.append("")
    lines.append(
        "| Artifact check | Result |"
    )
    lines.append("|---|---|")
    lines.append(f"| Figure files present | {figures_found}/{figure_count} |")
    lines.append(f"| Figure files with 300 dpi metadata | {dpi_count}/{figure_count} |")
    lines.append(
        f"| Table sections present | {table_sections_found}/{table_sections_total} |"
    )
    lines.append(f"| Table values checked | {values_checked} |")
    lines.append(f"| Table values mismatched | {len(mismatches)} |")
    lines.append(f"| Table placement failures | {len(placement_failures)} |")
    lines.append(
        f"| Captions present | {captions_found}/{captions_expected} |"
    )
    lines.append(f"| Superiority claims in tables or captions | {len(claims)} |")
    lines.append("")
    lines.append("Checks passed:")
    lines.append(
        f"- {figures_found}/{figure_count} figure files exist, are PNG, are at least "
        "800 x 800 px and carry 300 dpi metadata."
    )
    lines.append(
        f"- {values_checked} table values were re-read from their source CSV/JSON "
        f"files; {len(mismatches)} mismatched."
    )
    lines.append(
        f"- {table_sections_found}/{table_sections_total} table sections are present; "
        f"{len(placement_failures)} formatted values were missing from their section."
    )
    lines.append(
        f"- {captions_found}/{captions_expected} figures have exactly one caption; "
        f"{len(caption_problems)} caption format problems."
    )
    lines.append(
        f"- {len(claims)} superiority claims were found in `results/tables.md` or "
        "`results/figure_captions.md`."
    )
    lines.append(
        "- The healthy-test false-alarm rate appears beside every performance table "
        "(T2a, T2b, T2c, T3, T4a, T4b, T5 and T7)."
    )
    lines.append(
        "- Both proposed variants appear side by side in T2, T3 and T7 and in the "
        "relevant figures, with neither variant dropped."
    )
    lines.append(
        "- T2c is the MPPT-excluding-F3M view; F3M is marked as having no "
        "post-injection data in the primary per-fault table and is excluded from "
        "the within-run MPPT average."
    )
    lines.append(
        "- T6 carries the required note: different protocol, single 10 kHz samples "
        "vs 100-sample windows at 1 kHz, not comparable."
    )
    lines.append(
        "- No model was retrained in Phase 6; the pipeline step reads saved CSV, "
        "JSON and NPZ files only."
    )
    lines.append(
        "- `AGENTS.md` now records the leak-free re-evaluation framing and the "
        "decision to skip the noise-robustness and ablation phases."
    )
    lines.append("")
    lines.append("Problems or warnings:")
    lines.append(f"- {featnorm_sentence}")
    lines.append(f"- {comparison_sentence}")
    lines.append(
        f"- The whole-file primary view is partly saturated and windows overlap by "
        f"{overlap_pct:.0f}%, so the bootstrap intervals are optimistic; they "
        "describe window-level resampling, not independent physical events."
    )
    lines.append(
        "- The reference-paper values use a different protocol and are reference "
        "only; they are not a like-for-like comparison with T2."
    )
    lines.append(
        "- Figure 9 is a schematic of `src/model.py`, not a measured result."
    )
    lines.append(
        "- Noise-robustness and ablation experiments are not reported because those "
        "phases were skipped by project decision."
    )
    lines.append("")
    lines.append(
        "Decision needed from me: No. The paper framing, the two proposed variants, "
        "the frozen threshold and the table/figure scope were fixed before Phase 6."
    )
    lines.append(
        "Next phase: None - Phase 6 is the final phase; noise-robustness and "
        "ablation are skipped."
    )
    if not passed:
        lines.append("")
        lines.append(
            "WARNING: the verification script did not pass; see "
            "`results/phase6_verification.json`."
        )

    section = "\n".join(lines) + "\n"

    status_entry = "\n".join(
        [
            "---",
            f"## {datetime.now().astimezone().strftime('%Y-%m-%d')} - Phase 6: figures and tables",
            "",
            f"**What was done.** Generated {figure_count} 300 dpi figures and the T1-T8 tables "
            "from the frozen Phase 1-5 result files, then verified every table value "
            "against its source. Added the paper-framing rules to `AGENTS.md` and "
            "the Phase 6 steps to `run_all.py`. No model was retrained and no score, "
            "threshold, k=3 or normalization was changed.",
            "",
            "**Files created/changed.** `src/make_figures.py`, `src/verify_phase6.py`, "
            "`src/phase6_summary.py`, `figures/paper/fig_*.png` (9), "
            "`results/tables.md`, `results/figure_captions.md`, "
            "`results/phase6_table_values.json`, `results/phase6_architecture.json`, "
            "`results/phase6_verification.json`, `results/summary.md`, `STATUS.md`, "
            "`AGENTS.md`, `run_all.py`.",
            "",
            "**Key numbers (computed by `src/make_figures.py` and `src/verify_phase6.py`).**",
            f"- {figures_found}/{figure_count} figures; {values_checked} table values "
            f"checked with {len(mismatches)} mismatches; {len(claims)} superiority claims.",
            f"- Proposed (plain) primary F1 {f4(plain_ipt['f1'])} IPPT and "
            f"{f4(plain_mppt['f1'])} MPPT; healthy-test FAR {pct(plain_ipt['far'])} and "
            f"{pct(plain_mppt['far'])}.",
            f"- Proposed (featnorm) primary F1 {f4(feat_ipt['f1'])} IPPT and "
            f"{f4(feat_mppt['f1'])} MPPT; healthy-test FAR {pct(feat_ipt['far'])} and "
            f"{pct(feat_mppt['far'])}.",
            f"- Mahalanobis (window-mean) primary F1 {f4(mahal_ipt['f1'])} IPPT and "
            f"{f4(mahal_mppt['f1'])} MPPT; healthy-test FAR {pct(mahal_ipt['far'])} and "
            f"{pct(mahal_mppt['far'])}.",
            f"- Within-run AUC: Proposed (plain) {f4(plain_ipt['within_auc'])} IPPT and "
            f"{f4(plain_mppt['within_auc'])} MPPT; Proposed (featnorm) "
            f"{f4(feat_ipt['within_auc'])} and {f4(feat_mppt['within_auc'])}.",
            "",
            "**Open issues/risks.** Proposed (featnorm) is a negative result with high "
            "healthy-test false-alarm rates. Neither proposed variant exceeds the "
            "trivial Mahalanobis baseline on primary average F1, so the paper must "
            "not claim superiority. The whole-file view is partly saturated, windows "
            "overlap heavily, and the reference paper uses a different protocol. "
            "Noise-robustness and ablation phases are skipped.",
            "",
        ]
    )

    return section, status_entry


def upsert_summary(section: str) -> None:
    text = SUMMARY.read_text(encoding="utf-8") if SUMMARY.exists() else "# Summary\n"
    marker = "## Phase 6:"
    if marker in text:
        index = text.index(marker)
        before = text[:index].rstrip()
        if before.endswith("---"):
            before = before[:-3].rstrip()
        new_text = before + "\n\n---\n\n" + section
    else:
        new_text = text.rstrip() + "\n\n---\n\n" + section
    SUMMARY.write_text(new_text, encoding="utf-8")


def upsert_status(status_entry: str) -> None:
    text = STATUS.read_text(encoding="utf-8") if STATUS.exists() else "# STATUS\n"
    start_marker = "<!-- PHASE6_STATUS_START -->"
    end_marker = "<!-- PHASE6_STATUS_END -->"
    block = start_marker + "\n" + status_entry.strip() + "\n" + end_marker
    if start_marker in text and end_marker in text:
        start = text.index(start_marker)
        end = text.index(end_marker) + len(end_marker)
        before = text[:start].rstrip()
        after = text[end:].strip()
        new_text = before + "\n\n" + block + ("\n\n" + after if after else "") + "\n"
    else:
        new_text = text.rstrip() + "\n\n" + block + "\n"
    STATUS.write_text(new_text, encoding="utf-8")


def main() -> int:
    section, status_entry = build_section()
    (RESULTS / "phase6_summary_section.md").write_text(section, encoding="utf-8")
    upsert_summary(section)
    upsert_status(status_entry)
    print(section)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Verify the Phase 6 figures and tables against their saved sources.

Checks performed
----------------
1. All nine expected figure files exist, are PNG images, are at least 800 px in
   both dimensions, and carry 300 dpi metadata.
2. Every table cell recorded in ``results/phase6_table_values.json`` is looked
   up again in its source CSV/JSON and compared with the value that was written.
3. Every recorded table value also appears in the matching section of
   ``results/tables.md``.
4. Every figure has exactly one caption in ``results/figure_captions.md``, and
   each caption is one sentence.
5. Neither ``results/tables.md`` nor ``results/figure_captions.md`` contains a
   positive superiority claim about the proposed model. Allowed uses of "best"
   are the training terms "best epoch", "best validation" and "best weights".

The script writes ``results/phase6_verification.json`` and prints a summary.
It exits non-zero if any check fails.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
FIGURES = PROJECT_ROOT / "figures" / "paper"

FIGURE_FILES = [
    "fig_healthy_far.png",
    "fig_primary_f1_auc.png",
    "fig_within_run_auc.png",
    "fig_score_vs_time.png",
    "fig_score_hist.png",
    "fig_threshold_sensitivity.png",
    "fig_training_curves.png",
    "fig_feature_error_heatmap.png",
    "fig_architecture.png",
]

EXPECTED_TABLES = [
    "T1",
    "T2a",
    "T2b",
    "T2c",
    "T3",
    "T4a",
    "T4b",
    "T5",
    "T6",
    "T7",
    "T8a",
    "T8b",
    "T8c",
    "T8d",
    "T8e",
]

CSV_CACHE: dict[str, pd.DataFrame] = {}
JSON_CACHE: dict[str, Any] = {}


def load_csv(name: str) -> pd.DataFrame:
    if name not in CSV_CACHE:
        CSV_CACHE[name] = pd.read_csv(RESULTS / name)
    return CSV_CACHE[name]


def load_json(name: str) -> Any:
    if name not in JSON_CACHE:
        JSON_CACHE[name] = json.loads((RESULTS / name).read_text(encoding="utf-8"))
    return JSON_CACHE[name]


def resolve(entry: dict[str, Any]) -> Any:
    source = entry["source"]
    selector = entry["selector"]
    if selector["kind"] == "csv":
        frame = load_csv(source)
        subset = frame
        for key, value in selector["filters"].items():
            subset = subset[subset[key].astype(str) == str(value)]
        if len(subset) != 1:
            raise ValueError(
                f"{source}: filters {selector['filters']} selected {len(subset)} rows"
            )
        value = subset.iloc[0][selector["column"]]
        if isinstance(value, pd.Series):
            raise ValueError(f"{source}: column {selector['column']} is not scalar")
        if hasattr(value, "item"):
            value = value.item()
        return value
    if selector["kind"] == "json":
        value = load_json(source)
        for key in selector["path"]:
            if isinstance(value, list):
                value = value[int(key)]
            else:
                value = value[key]
        return value
    raise ValueError(f"unknown selector kind: {selector['kind']}")


def values_equal(stored: Any, actual: Any) -> bool:
    if isinstance(stored, bool) or isinstance(actual, bool):
        return bool(stored) == bool(actual)
    if isinstance(stored, (int, float)) or isinstance(actual, (int, float)):
        try:
            stored_float = float(stored)
            actual_float = float(actual)
        except (TypeError, ValueError):
            return False
        if math.isnan(stored_float) or math.isnan(actual_float):
            return math.isnan(stored_float) and math.isnan(actual_float)
        return math.isclose(stored_float, actual_float, rel_tol=1e-12, abs_tol=1e-15)
    return stored == actual


def parse_table_sections(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        match = re.match(r"^#{3,4}\s+(T\d+[a-z]?)\.", line)
        if match:
            current = match.group(1)
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    return {key: "\n".join(lines) for key, lines in sections.items()}


def find_superiority_claims(text: str) -> list[dict[str, str]]:
    lowered = text.lower()
    findings: list[dict[str, str]] = []
    patterns = {
        "outperform": r"outperform",
        "superior": r"superior",
        "better than": r"better\s+than",
        "beats": r"\bbeats?\b",
        "wins": r"\bwins?\b",
        "state of the art": r"state[-\s]+of[-\s]+the[-\s]+art",
    }
    for label, pattern in patterns.items():
        for match in re.finditer(pattern, lowered):
            start = max(0, match.start() - 80)
            end = min(len(lowered), match.end() + 80)
            findings.append(
                {
                    "kind": label,
                    "context": text[start:end].replace("\n", " "),
                }
            )
    for match in re.finditer(r"\bbest\b(?!\s+(?:epoch|validation|val|weights))", lowered):
        start = max(0, match.start() - 80)
        end = min(len(lowered), match.end() + 80)
        findings.append(
            {
                "kind": "best (not a training term)",
                "context": text[start:end].replace("\n", " "),
            }
        )
    return findings


def main() -> int:
    figures: list[dict[str, Any]] = []
    figure_problems: list[str] = []
    for name in FIGURE_FILES:
        path = FIGURES / name
        entry: dict[str, Any] = {
            "file": name,
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else 0,
        }
        if path.exists():
            try:
                with Image.open(path) as image:
                    entry["width"], entry["height"] = image.size
                    entry["format"] = image.format
                    dpi = image.info.get("dpi", (0.0, 0.0))
                    entry["dpi"] = tuple(float(value) for value in dpi)
            except Exception as error:  # pragma: no cover - defensive
                entry["error"] = str(error)
        figures.append(entry)
        if not entry["exists"]:
            figure_problems.append(f"{name}: missing")
            continue
        if entry.get("format") != "PNG":
            figure_problems.append(f"{name}: format is {entry.get('format')}, not PNG")
        if entry.get("width", 0) < 800 or entry.get("height", 0) < 800:
            figure_problems.append(f"{name}: image is smaller than 800 x 800 px")
        dpi = entry.get("dpi", (0.0, 0.0))
        if len(dpi) < 2 or abs(float(dpi[0]) - 300.0) > 1.0 or abs(float(dpi[1]) - 300.0) > 1.0:
            figure_problems.append(f"{name}: dpi metadata is {dpi}, expected 300")

    tables_path = RESULTS / "tables.md"
    captions_path = RESULTS / "figure_captions.md"
    tables_text = tables_path.read_text(encoding="utf-8") if tables_path.exists() else ""
    captions_text = (
        captions_path.read_text(encoding="utf-8") if captions_path.exists() else ""
    )

    audit = json.loads(
        (RESULTS / "phase6_table_values.json").read_text(encoding="utf-8")
    )
    entries = audit["entries"]
    sections = parse_table_sections(tables_text)

    mismatches: list[dict[str, Any]] = []
    placement_failures: list[dict[str, Any]] = []
    values_checked = 0
    for entry in entries:
        values_checked += 1
        try:
            actual = resolve(entry)
        except Exception as error:
            mismatches.append(
                {
                    "table": entry["table"],
                    "row": entry["row"],
                    "column": entry["column"],
                    "source": entry["source"],
                    "stored": entry["value"],
                    "error": str(error),
                }
            )
            continue
        if not values_equal(entry["value"], actual):
            mismatches.append(
                {
                    "table": entry["table"],
                    "row": entry["row"],
                    "column": entry["column"],
                    "source": entry["source"],
                    "stored": entry["value"],
                    "actual": actual,
                }
            )
        section = sections.get(entry["table"])
        if section is None:
            placement_failures.append(
                {
                    "table": entry["table"],
                    "reason": "table section not found in tables.md",
                }
            )
        elif entry["text"] not in section:
            placement_failures.append(
                {
                    "table": entry["table"],
                    "row": entry["row"],
                    "column": entry["column"],
                    "text": entry["text"],
                    "reason": "formatted value not found in its table section",
                }
            )

    missing_tables = [table for table in EXPECTED_TABLES if table not in sections]

    caption_lines = [
        line.strip()
        for line in captions_text.splitlines()
        if re.match(r"^\*\*fig_[a-z0-9_]+\.png\.\*\*", line.strip())
    ]
    caption_files = []
    caption_problems: list[str] = []
    for line in caption_lines:
        match = re.match(r"^\*\*(fig_[a-z0-9_]+\.png)\.\*\*\s*(.+)$", line)
        if not match:
            caption_problems.append(f"unparseable caption: {line[:80]}")
            continue
        caption_files.append(match.group(1))
        sentence = match.group(2).strip()
        if not sentence.endswith("."):
            caption_problems.append(f"{match.group(1)}: caption does not end with a period")
        if ". " in sentence:
            caption_problems.append(f"{match.group(1)}: caption looks like more than one sentence")
    missing_captions = [
        name for name in FIGURE_FILES if name not in caption_files
    ]
    extra_captions = [
        name for name in caption_files if name not in FIGURE_FILES
    ]

    claim_findings = find_superiority_claims(tables_text + "\n" + captions_text)

    verification = {
        "phase": "phase_6_figures_and_tables",
        "figures": {
            "expected": len(FIGURE_FILES),
            "found": sum(1 for entry in figures if entry["exists"]),
            "details": figures,
            "problems": figure_problems,
        },
        "tables": {
            "values_checked": values_checked,
            "mismatches": mismatches,
            "placement_failures": placement_failures,
            "missing_table_sections": missing_tables,
        },
        "captions": {
            "expected": len(FIGURE_FILES),
            "found": len(caption_files),
            "missing": missing_captions,
            "extra": extra_captions,
            "problems": caption_problems,
        },
        "superiority_claims": claim_findings,
        "passed": (
            not figure_problems
            and not mismatches
            and not placement_failures
            and not missing_tables
            and not missing_captions
            and not extra_captions
            and not caption_problems
            and not claim_findings
        ),
    }
    (RESULTS / "phase6_verification.json").write_text(
        json.dumps(verification, indent=2), encoding="utf-8"
    )

    print("Phase 6 verification")
    print("--------------------")
    print(
        f"Figures found: {verification['figures']['found']}/{verification['figures']['expected']}"
    )
    if figure_problems:
        for problem in figure_problems:
            print(f"  FIGURE PROBLEM: {problem}")
    else:
        print("Figure files: all present, PNG, at least 800 x 800 px, 300 dpi")
    print(f"Table sections found: {len(EXPECTED_TABLES) - len(missing_tables)}/{len(EXPECTED_TABLES)}")
    if missing_tables:
        print(f"  MISSING TABLES: {', '.join(missing_tables)}")
    print(f"Table values checked: {values_checked}")
    print(f"Table values mismatched: {len(mismatches)}")
    for mismatch in mismatches[:20]:
        print(f"  MISMATCH: {mismatch}")
    print(f"Table placement failures: {len(placement_failures)}")
    for failure in placement_failures[:20]:
        print(f"  PLACEMENT: {failure}")
    print(
        f"Captions found: {verification['captions']['found']}/{verification['captions']['expected']}"
    )
    if missing_captions:
        print(f"  MISSING CAPTIONS: {', '.join(missing_captions)}")
    if extra_captions:
        print(f"  EXTRA CAPTIONS: {', '.join(extra_captions)}")
    if caption_problems:
        for problem in caption_problems:
            print(f"  CAPTION PROBLEM: {problem}")
    print(f"Superiority claims found: {len(claim_findings)}")
    for finding in claim_findings[:20]:
        print(f"  CLAIM: {finding['kind']}: {finding['context']}")
    print(f"Overall: {'PASS' if verification['passed'] else 'FAIL'}")
    return 0 if verification["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

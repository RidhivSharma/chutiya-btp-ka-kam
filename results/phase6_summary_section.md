## Phase 6: figures and tables   (2026-10-28 01:32 +0530)

What was done: Built 9 300 dpi publication figures and the T1-T8 table groups from the frozen Phase 1-5 result files. The figures use the saved score archives, training histories and CSV summaries; the tables are generated from the saved CSV/JSON files, and every table cell is recorded with its source selector. A separate verification script re-read every source and checked all recorded values, the figure files, the captions and the wording for unsupported superiority claims. No model was retrained, and no score, threshold, k=3 or normalization was changed. The project framing recorded in AGENTS.md is a leak-free re-evaluation, with Proposed (featnorm) reported as a negative result.

Files created or changed:
- `src/make_figures.py` - generates the 9 figures, `results/tables.md`, the figure captions, the architecture summary and the table-value audit.
- `src/verify_phase6.py` - re-reads every table value and checks the figures, captions and wording.
- `src/phase6_summary.py` - writes this section and the STATUS entry from saved result files.
- `figures/paper/fig_healthy_far.png` - healthy-test false-alarm rate per model.
- `figures/paper/fig_primary_f1_auc.png` - primary F1 and AUC with intervals.
- `figures/paper/fig_within_run_auc.png` - within-run AUC with intervals.
- `figures/paper/fig_score_vs_time.png` - score versus time for F1/F4/F6/F7 (IPPT).
- `figures/paper/fig_score_hist.png` - healthy and fault score distributions.
- `figures/paper/fig_threshold_sensitivity.png` - healthy FAR versus threshold percentile.
- `figures/paper/fig_training_curves.png` - training and validation loss for six runs.
- `figures/paper/fig_feature_error_heatmap.png` - per-feature reconstruction error.
- `figures/paper/fig_architecture.png` - TCN-BiLSTM block diagram.
- `results/tables.md` - T1-T8 tables generated from saved result files.
- `results/figure_captions.md` - one factual caption per figure.
- `results/phase6_table_values.json` - audit trail for 943 table values.
- `results/phase6_architecture.json` - static architecture summary copied from `src/model.py`.
- `results/phase6_verification.json` - verification results and counts.
- `results/summary.md` - this Phase 6 section.
- `STATUS.md` - appended Phase 6 log entry.
- `AGENTS.md` - recorded the leak-free re-evaluation framing, the negative featnorm result, and the decision to skip noise-robustness and ablation phases.
- `run_all.py` - added the Phase 6 generation, verification and summary steps.

Key numbers:

| Model | Primary F1 IPPT | Primary F1 MPPT | Within-run AUC IPPT | Within-run AUC MPPT | Healthy-test FAR IPPT | Healthy-test FAR MPPT |
|---|---|---|---|---|---|---|
| Proposed (plain) | 0.9869 | 0.9891 | 0.7507 | 0.7606 | 0.00% | 2.17% |
| Proposed (featnorm) | 0.9472 | 0.8758 | 0.7676 | 0.7420 | 11.24% | 28.74% |
| Mahalanobis (window-mean) | 0.9962 | 0.9963 | 0.7194 | 0.7060 | 0.75% | 0.77% |

| Item | Value |
|---|---|
| MPPT excluding F3M, primary F1, Proposed (plain) | 0.9891 |
| MPPT excluding F3M, primary F1, Proposed (featnorm) | 0.8759 |
| Reference-paper rows (different protocol, not comparable) | 8 |

| Artifact check | Result |
|---|---|
| Figure files present | 9/9 |
| Figure files with 300 dpi metadata | 9/9 |
| Table sections present | 15/15 |
| Table values checked | 943 |
| Table values mismatched | 0 |
| Table placement failures | 0 |
| Captions present | 9/9 |
| Superiority claims in tables or captions | 0 |

Checks passed:
- 9/9 figure files exist, are PNG, are at least 800 x 800 px and carry 300 dpi metadata.
- 943 table values were re-read from their source CSV/JSON files; 0 mismatched.
- 15/15 table sections are present; 0 formatted values were missing from their section.
- 9/9 figures have exactly one caption; 0 caption format problems.
- 0 superiority claims were found in `results/tables.md` or `results/figure_captions.md`.
- The healthy-test false-alarm rate appears beside every performance table (T2a, T2b, T2c, T3, T4a, T4b, T5 and T7).
- Both proposed variants appear side by side in T2, T3 and T7 and in the relevant figures, with neither variant dropped.
- T2c is the MPPT-excluding-F3M view; F3M is marked as having no post-injection data in the primary per-fault table and is excluded from the within-run MPPT average.
- T6 carries the required note: different protocol, single 10 kHz samples vs 100-sample windows at 1 kHz, not comparable.
- No model was retrained in Phase 6; the pipeline step reads saved CSV, JSON and NPZ files only.
- `AGENTS.md` now records the leak-free re-evaluation framing and the decision to skip the noise-robustness and ablation phases.

Problems or warnings:
- Proposed (featnorm) is a negative result: its healthy-test false-alarm rate is 11.24% in IPPT and 28.74% in MPPT, while its primary average F1 is 0.9472 and 0.8758.
- Both proposed variants have a lower primary average F1 than the trivial Mahalanobis (window-mean) baseline in both modes: Proposed (plain) 0.9869 IPPT and 0.9891 MPPT, Proposed (featnorm) 0.9472 and 0.8758, versus Mahalanobis 0.9962 and 0.9963.
- The whole-file primary view is partly saturated and windows overlap by 90%, so the bootstrap intervals are optimistic; they describe window-level resampling, not independent physical events.
- The reference-paper values use a different protocol and are reference only; they are not a like-for-like comparison with T2.
- Figure 9 is a schematic of `src/model.py`, not a measured result.
- Noise-robustness and ablation experiments are not reported because those phases were skipped by project decision.

Decision needed from me: No. The paper framing, the two proposed variants, the frozen threshold and the table/figure scope were fixed before Phase 6.
Next phase: None - Phase 6 is the final phase; noise-robustness and ablation are skipped.

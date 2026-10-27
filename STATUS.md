# STATUS

Project status log. Entries are appended at the end, newest last.
Each entry records what was done, files touched, key numbers computed by code,
and open issues. Do not overwrite earlier entries.

---
## 2026-10-26 - Phase 1: project setup and raw-data verification

**What was done.** Created the permanent project rules, virtual environment,
directory structure, and a code-only Phase 1 check. Extracted `CSV_Files.zip`
into `data/raw/CSV_Files/` and loaded every CSV in full with pandas. Verified
file presence, exact column order, missing/infinite values, duplicate rows, and
strictly increasing `Time`. No cleaning, modeling, or fault analysis was done.

**Files created/changed.** `AGENTS.md`, `STATUS.md`, `requirements.txt`,
`setup_venv.ps1`, `run_all.py`, `src/phase1_data_check.py`,
`results/data_inventory.csv`, `results/data_check.json`, `figures/.gitkeep`,
`models/.gitkeep`, `.gitignore` (added `data/raw/`, `.venv/`, cache ignores).
The original `Untitled0.ipynb` was not modified.

**Key numbers (computed by `src/phase1_data_check.py`).**

- 16 of 16 expected CSVs present and loaded; all 14 columns in exact expected
  order; Time strictly increasing in every file.
- 2,163,480 rows total: 1,093,412 IPPT rows and 1,070,068 MPPT rows.
- 284,729 healthy rows (`F0L` + `F0M`) and 1,878,751 fault-file rows.
- 0 missing values, 0 infinite values, 0 duplicate rows.
- Nominal 10 kHz sampling; measured median rates are 10,000.31 Hz for most
  files and 10,000.34 Hz for `F6*`/`F7*` (floating-point timestamp precision).
- Per-file sizes: 69,967 to 144,015 rows; durations 6.996 s to 14.400 s.
- `F3M` is the shortest file (69,967 rows, 6.996 s); `F3L` is 103,497 rows
  (10.349 s). Other files are 12.9-14.4 s.
- Archive SHA-256:
  `88cd20c848fee86752870cf9b198eab45568c31355685328dd75aba982bf1a63`.
- Environment verified in `.venv`: Python 3.13.1, numpy 2.2.3, pandas 2.2.3,
  scikit-learn 1.6.1, tensorflow 2.20.0, matplotlib 3.10.8,
  seaborn 0.13.2, scipy 1.15.2.

**Open issues.** (1) `F3M`/`F3L` are much shorter than the other files, so
windowing and per-file evaluation must handle short recordings and report
coverage rather than silently dropping data. (2) The healthy-only
train/validation split has not been defined yet; it must be chronological or
blocked to avoid leakage from adjacent 10 kHz samples. (3) The notebook used
1 kHz resampling and 100-sample windows; that preprocessing choice is not yet
ratified. (4) TensorFlow oneDNN non-determinism must be addressed before
training. (5) No model or fault evaluation exists yet.

---
## 2026-10-26 - Phase 2: preprocessing, healthy split, and scaling

**What was done.** Replaced the notebook's preprocessing with a clean,
leakage-free `src/data.py`. All 16 CSVs are loaded in their documented column
order and converted to float64. The notebook's `df.iloc[::10]` was replaced by
`scipy.signal.decimate(q=10, ftype="iir", zero_phase=True)`, which low-pass
filters at 400 Hz before taking every 10th sample; `Time[::10]` is kept aligned
with the decimated rows. Windows are 100 samples with stride 10 and store
their first/last timestamps. `F0L`/`F0M` windows are split chronologically
60/20/20 per mode with 10-window gaps between train/val/test. Kept the
notebook's separate IPPT and MPPT `StandardScaler`s, but each is now fitted
only on the unique raw decimated rows covered by that mode's healthy training
windows; the scalers are applied to all 16 files by mode. No fault file is
read for any fit. `src/sanity_checks.py` rebuilds the data and asserts the
contract. Both scripts are now steps in `run_all.py`.

**Files created/changed.** `src/data.py`, `src/sanity_checks.py`,
`src/__init__.py`, `run_all.py`, `results/data_summary.csv`,
`results/split_summary.csv`, `results/split_indices.csv`,
`results/data_config.json`, `results/sanity_checks.json`. The notebook and raw
CSVs were not modified.

**Key numbers (computed by `src/data.py` and `src/sanity_checks.py`).**

- 21,488 windows over the 16 files. Decimated rows per file range from 6,997
  (`F3M`) to 14,402; decimated rate is 1 kHz (median-based estimates
  1000.03-1000.91 Hz).
- Healthy IPPT (`F0L`): 1,428 windows total; train 856, validation 285,
  test 267; 20 gap windows excluded.
- Healthy MPPT (`F0M`): 1,401 windows total; train 840, validation 280,
  test 261; 20 gap windows excluded.
- Actual fractions (train/val/test): IPPT 0.5994/0.1996/0.1870 and MPPT
  0.5996/0.1999/0.1863. Test is below 0.20 because the two 10-window gaps
  are removed from the total.
- Scaler fit rows: 8,650 IPPT rows `[0, 8650)` and 8,490 MPPT rows
  `[0, 8490)`. Fitted means/scales match these rows exactly
  (`n_samples_seen` confirms) and share 0 raw rows with validation/test.
- 0 NaN and 0 infinite values in windows, timestamps, and decimated rows.
- All 30 sanity checks passed. Train/val/test window indices and raw sample
  sets are pairwise disjoint; splits are chronological; every healthy window
  is assigned to exactly one of train/val/test/gap.

**Open issues.** (1) Each 10-window gap drops ten candidate windows between
two parts; because stride 10 divides the 100-sample window, the effective
unused raw block at a boundary is 10 samples (10 ms), and no split windows
share samples. (2) `filtfilt` edge transients in the decimation are not
trimmed, so the first and last window of each file contain a short transient.
(3) The per-mode scaler decision was kept from the notebook; a pooled
IPPT+MPPT scaler was not used. (4) `F2L` has a median-based rate estimate of
1000.91 Hz due to timestamp jitter; q=10 is still used as specified.
(5) Windows within a split still overlap by 90 %, so observations within a
split are not independent; this is intended for window-level detection but
must be considered in statistical tests. (6) No model or fault evaluation
exists yet. (7) TensorFlow oneDNN non-determinism still needs a decision
before training.

---
## 2026-10-26 - Phase 1 summary report (clean data pipeline)

**What was done.** Re-ran `src/sanity_checks.py` and created the missing
`results/summary.md` Phase 1 section without changing any existing Phase 1
code. Added `src/phase1_summary.py`, a reporting script that reads the saved
Phase 1/preprocessing outputs, rebuilds the dataset once to compute
scaled-training and leakage statistics, and appends the Phase 1 section in the
format required by `AGENTS.md`. The same statistics were saved to
`results/phase1_summary_stats.json`.

**Files created/changed.** `results/summary.md` (new Phase 1 report),
`results/phase1_summary_stats.json` (new supporting statistics),
`src/phase1_summary.py` (new reporting script), `STATUS.md` (this entry).
`results/data_summary.csv`, `results/split_summary.csv`, and
`results/sanity_checks.json` were rewritten by the required sanity-check rerun
with the same 30/30 result. No existing Phase 1 code or raw data was modified.

**Key numbers (computed by `src/sanity_checks.py` and `src/phase1_summary.py`).**

- 16 files, 2,163,480 raw rows, 216,356 rows after 1 kHz decimation, and
  21,488 windows.
- Healthy IPPT (`F0L`): 1,428 windows; train 856, validation 285, test 267.
  Healthy MPPT (`F0M`): 1,401 windows; train 840, validation 280, test 261.
  Each mode excludes 20 gap windows (two 10-window gaps).
- Scaler fit rows: 8,650 IPPT (`F0L`) and 8,490 MPPT (`F0M`); `n_samples_seen`
  matches both. No fault file was used in fitting, and train/validation/test
  share 0 window indices and 0 raw samples.
- 30 of 30 sanity checks passed; 0 NaN and 0 infinite values.
- Unique scaled healthy training rows have pooled mean 0.000000 and std
  1.000000 in both modes. The overlapping training windows have pooled std
  0.937302 (IPPT) and 0.947445 (MPPT); the first 10 windows contain a `Vabc`
  startup transient (first-window minimum -22.3 IPPT and -20.8 MPPT), which
  inflates the pooled std. Excluding those windows, `Vabc` std is 0.037 (IPPT)
  and 0.035 (MPPT).
- `F3M` is still the shortest file at 6.996 s and 690 windows; `F2L` still has
  a decimated rate estimate of 1000.91 Hz versus a median of 1000.03 Hz.

**Open issues/risks.** The first 10 training windows contain a startup
transient that makes the pooled window std below 1; decide whether to keep or
trim them before modeling. `F3M` needs explicit coverage reporting; `F2L`
timestamp jitter remains; filtfilt edge transients are kept; windows within a
split are 90% overlapped; test shares are 18.70% (IPPT) and 18.63% (MPPT)
because of the gaps. No model or fault evaluation exists yet.

---
## 2026-10-26 - Phase 2: labeling and timing diagnostic

**What was done.** Added `get_labels` to `src/data.py` with `whole_file` and `time_resolved` modes (-1 before 7 s, -2 for 7-8 s overlap, 1 after 8 s). Built a placeholder detector from healthy training window means only: Mahalanobis distance with a pseudo-inverse and small relative ridge, thresholded at the 99th percentile of healthy validation scores. Scored all 21,488 windows, wrote per-fault timing shares and healthy false-alarm rates, and drew 14 individual figures plus a 7x2 grid. No model setting was tuned on fault results.

**Files created/changed.** `src/data.py` (added `get_labels` and a cached window-time helper; existing outputs unchanged), `src/time_diagnostic.py`, `src/phase2_summary.py`, `run_all.py`, `results/time_diagnostic_scores.csv`, `results/label_diagnostic.csv`, `results/healthy_false_alarm.csv`, `results/time_diagnostic.json`, `results/phase2_summary_stats.json`, `figures/score_vs_time_*.png`, `results/summary.md`, `STATUS.md`.

**Key numbers (computed by code).**
- 21,488 windows scored across 16 files; 14 fault files in two modes. Score range 1.34 to 1.08e+04.
- Pre-injection flag share (excluding the first 0.5 s): 1.0000 to 1.0000 across all fault files; median pre-injection score was 4.1x to 532.3x the threshold.
- Healthy test false-alarm rate: 0.0075 IPPT and 0.0077 MPPT.
- Post/pre median-score ratios: clear for `F1L`, `F1M`, `F2L`, `F3L`, `F5L`, `F5M`; weak for `F2M`; flat for `F4L`, `F4M`, `F6L`, `F6M`, `F7L`, `F7M`; `F3M` has no post-injection data.
- Post-injection flag shares are 1.0000 to 1.0000 where post data exist, so the binary flag does not localize the onset; `F3M` has 0 post-injection windows.

**Open issues/risks.** The placeholder is saturated on fault files, so pre-injection anomaly cannot be separated from a run-level operating-point offset. `F3M` ends before 7 s and has no post-injection data. The label convention for final evaluation (whole-file versus time-resolved) is still a decision.

---
## 2026-10-27 - Phase 3: baselines on the same pipeline

**What was done.** Re-implemented EllipticEnvelope, IsolationForest, LocalOutlierFactor, and OneClassSVM on the frozen Phase 2 rows, and added the Phase 2 Mahalanobis placeholder unchanged as a fifth reference. Each model was fitted only on healthy training rows of one mode; the threshold is the 99th percentile of healthy validation window scores. Saved 10 score archives (5 models x 2 modes) in the five-key Phase 5 format plus a sanity table. No final metrics were computed.

**Files created/changed.** `AGENTS.md` (label convention), `src/baselines.py`, `src/phase3_summary.py`, `run_all.py`, `results/scores_baseline_*.npz`, `results/baseline_sanity.csv`, `results/baselines_metadata.json`, `results/phase3_summary_stats.json`, `results/summary.md`, `STATUS.md`.

**Key numbers (computed by code).**
- 10 archives, 107,240 window scores; 13/13 checks passed.
- Whole-file saturation: LocalOutlierFactor, Mahalanobis (window-mean) flag at least 95% of every fault file; EllipticEnvelope does so in 13/14.
- Healthy test false-alarm rates (IPPT/MPPT): EllipticEnvelope IPPT 0.0187; EllipticEnvelope MPPT 0.0192; IsolationForest IPPT 0.0000; IsolationForest MPPT 0.0115; LocalOutlierFactor IPPT 0.1311; LocalOutlierFactor MPPT 0.2107; OneClassSVM IPPT 0.1086; OneClassSVM MPPT 0.2835; Mahalanobis (window-mean) IPPT 0.0075; Mahalanobis (window-mean) MPPT 0.0077.
- Clear post-8 s rises (>=2.0x pre median) are unanimous for `F3L` and `F5L`; `F4L`, `F4M`, `F6L`, `F6M` and `F7M` have none. OneClassSVM MPPT is saturated at its score ceiling (93.79); LOF F1L has a negative pre median (-0.1278), so its ratio is undefined. F3M has no post-injection data.

**Open issues/risks.** The whole-file view is saturated for `LocalOutlierFactor` and `Mahalanobis (window-mean)`, so it cannot separate methods. LOF and OneClassSVM have high healthy test false-alarm rates due to temporal drift within F0. OneClassSVM scores hit a ceiling, and LOF F1L has a negative pre median. F3M has no post-injection data. A reporting-only bug in the first draft (the saturated-model list compared per-mode counts with the 14-file total and printed `none`) was corrected before the final section was written; no model fit, score, threshold, or saved result changed.

## 2026-10-27 - Phase 4: proposed TCN-BiLSTM autoencoder

**What was done.** Copied the notebook's TCN-BiLSTM autoencoder unchanged into `src/model.py` and trained it healthy-only for MPPT and IPPT with seeds 0, 1 and 2. Each run used only the healthy training windows for fitting and healthy validation windows for early stopping. Two window scores (`plain` and fixed k=3 `featnorm`) were saved for every file, with thresholds at the 99th percentile of healthy validation scores. No fault file was used for training, normalization, thresholds, or model selection.

**Files created/changed.** `src/model.py`, `src/train.py`, `src/phase4_summary.py`, `run_all.py`, `models/proposed_tcn_bilstm_<mode>_seed<k>.keras` (6), `results/history_<mode>_seed<k>.json` (6), `figures/training_curve_<mode>_seed<k>.png` (6), `results/scores_proposed_<score>_<mode>_seed<k>.npz` (12), `results/proposed_sanity.csv`, `results/proposed_training.csv`, `results/proposed_run_<mode>_seed<k>.json` (6), `results/proposed_metadata.json`, `results/phase4_summary_stats.json`, `logs/train.log`, `results/summary.md`, `STATUS.md`. Frozen `src/data.py`, `src/baselines.py`, and `Untitled0.ipynb` were not modified.

**Key numbers (computed by `src/train.py` and `src/phase4_summary.py`).**
- 6 runs, 12 archives, 128,688 saved window scores; 23/23 checks passed.
- Epochs run 92-100; best epochs 80-100; total training time 91.6 min on `/CPU:0`.
- Healthy test false-alarm rates: plain MPPT test 2.17% (1.92%-2.30%) vs validation 1.07%; plain IPPT test 0.00% (0.00%-0.00%) vs validation 1.05%; featnorm MPPT test 28.74% (16.86%-41.38%) vs validation 1.07%; featnorm IPPT test 11.24% (8.61%-15.36%) vs validation 1.05%.
- F4/F6/F7 rise classes: plain `F4L` flat 3/3; plain `F4M` flat 3/3; plain `F6L` weak 3/3; plain `F6M` flat 3/3; plain `F7L` flat 3/3; plain `F7M` flat 3/3; featnorm `F4L` flat 3/3; featnorm `F4M` flat 3/3; featnorm `F6L` weak 3/3; featnorm `F6M` flat 3/3; featnorm `F7L` weak 3/3; featnorm `F7M` flat 3/3.
- Seed consistency: largest threshold CV 8.13%; rise-class agreement 23/26; largest single-file seed ratio spread 4.29x.

**Open issues/risks.** No GPU was available. Mean whole-file fault flag shares are 86.28%-100.00% (lowest single-seed share 82.69%), so the whole-file view is close to saturation; the pre/post-8 s comparison is the discriminating sanity view. `F3M` has no post-injection data. Windows overlap by 90%, and the healthy validation-to-test drift affects false-alarm transfer. No final metrics were computed.

## 2026-10-28 - Phase 5: final evaluation

**What was done.** Evaluated the frozen score archives only, with no retraining and no change to any score, threshold, k=3 or normalization. Computed the primary whole-file comparison for every model, mode and fault with 200 equal-class-size bootstrap resamples; the secondary pre-injection versus post-8 s within-run AUC; threshold sensitivity from saved healthy validation scores; and the reference-paper table with a different-protocol label. Both proposed variants (plain and featnorm) are reported side by side and F3M is marked as having no post-injection data.

**Files created/changed.** `src/phase5_eval.py`, `src/phase5_summary.py`, `run_all.py`, `AGENTS.md` (two-variant and healthy-FAR reporting rule), `results/per_fault_primary_IPPT.csv`, `results/per_fault_primary_MPPT.csv`, `results/average_primary_IPPT.csv`, `results/average_primary_MPPT.csv`, `results/within_run_IPPT.csv`, `results/within_run_MPPT.csv`, `results/within_run_average_IPPT.csv`, `results/within_run_average_MPPT.csv`, `results/threshold_sensitivity.csv`, `results/paper_reference.csv`, `results/phase5_checks.json`, `results/phase5_metadata.json`, `results/summary.md`, `STATUS.md`.

**Key numbers (computed by `src/phase5_eval.py`).**

- 21/21 automated checks passed; 22 score archives read and none written.
- Primary average F1: Proposed (plain) 0.9869 (IPPT, healthy-test FAR 0.00%) and 0.9891 (MPPT, FAR 2.17%); Proposed (featnorm) 0.9472 (IPPT, FAR 11.24%) and 0.8758 (MPPT, FAR 28.74%).
- Within-run average AUC: Proposed (plain) 0.7507 IPPT and 0.7606 MPPT; Proposed (featnorm) 0.7676 IPPT and 0.7420 MPPT.
- The two proposed variants do not beat the trivial Mahalanobis (window-mean) baseline in 19 listed variant/metric/mode places; every such place is listed in `results/summary.md`.
- Primary average AUC is at least 0.99 for 6 of 7 models in IPPT and 5 of 7 models in MPPT.
- `F3M` has 0 post-injection windows and is excluded from the within-run table; `F3L` has post-injection data and is included.

**Open issues/risks.** The whole-file AUC is close to saturation for most models, while F1 and accuracy separate the models more; overlapping windows make the bootstrap intervals optimistic. Proposed (featnorm) has high healthy-test false-alarm rates. The reference-paper table uses a different protocol (single 10 kHz samples) and is reference only. F3M ends before 7 s. No model was retuned in response to any fault result.

<!-- PHASE6_STATUS_START -->
---
## 2026-10-28 - Phase 6: figures and tables

**What was done.** Generated 9 300 dpi figures and the T1-T8 tables from the frozen Phase 1-5 result files, then verified every table value against its source. Added the paper-framing rules to `AGENTS.md` and the Phase 6 steps to `run_all.py`. No model was retrained and no score, threshold, k=3 or normalization was changed.

**Files created/changed.** `src/make_figures.py`, `src/verify_phase6.py`, `src/phase6_summary.py`, `figures/paper/fig_*.png` (9), `results/tables.md`, `results/figure_captions.md`, `results/phase6_table_values.json`, `results/phase6_architecture.json`, `results/phase6_verification.json`, `results/summary.md`, `STATUS.md`, `AGENTS.md`, `run_all.py`.

**Key numbers (computed by `src/make_figures.py` and `src/verify_phase6.py`).**
- 9/9 figures; 943 table values checked with 0 mismatches; 0 superiority claims.
- Proposed (plain) primary F1 0.9869 IPPT and 0.9891 MPPT; healthy-test FAR 0.00% and 2.17%.
- Proposed (featnorm) primary F1 0.9472 IPPT and 0.8758 MPPT; healthy-test FAR 11.24% and 28.74%.
- Mahalanobis (window-mean) primary F1 0.9962 IPPT and 0.9963 MPPT; healthy-test FAR 0.75% and 0.77%.
- Within-run AUC: Proposed (plain) 0.7507 IPPT and 0.7606 MPPT; Proposed (featnorm) 0.7676 and 0.7420.

**Open issues/risks.** Proposed (featnorm) is a negative result with high healthy-test false-alarm rates. Neither proposed variant exceeds the trivial Mahalanobis baseline on primary average F1, so the paper must not claim superiority. The whole-file view is partly saturated, windows overlap heavily, and the reference paper uses a different protocol. Noise-robustness and ablation phases are skipped.
<!-- PHASE6_STATUS_END -->

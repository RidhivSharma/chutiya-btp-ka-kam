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

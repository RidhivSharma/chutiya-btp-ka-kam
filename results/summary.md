## Phase 1: Clean data pipeline   (2026-10-26 14:09 +0530)

What was done: Phase 1 turned the 16 raw GPVS-Faults CSV recordings into clean, non-overlapping train/validation/test sets for the healthy panels. Each 10 kHz recording was low-pass filtered and reduced to 1 kHz, then cut into 100-sample (100 ms) windows with a 10-sample step. The healthy IPPT (F0L) and MPPT (F0M) recordings were split chronologically 60/20/20, with 10 windows dropped at each boundary so that no window or raw sample is shared between parts. One standardizer per operating mode was fitted only on healthy training rows and then applied to every file of that mode. All 30 automated checks passed, including no missing values, no split overlap, and no fault data used in fitting.

Files created or changed:
- `src/phase1_data_check.py` - extracts and validates the 16 raw CSVs.
- `src/data.py` - decimates, windows, splits, and standardizes the data.
- `src/sanity_checks.py` - rebuilds the dataset and runs the automated checks.
- `src/phase1_summary.py` - computes the extra statistics and writes this summary.
- `run_all.py` - runs the Phase 1 and preprocessing steps in order.
- `results/data_inventory.csv` - per-file raw-data inventory.
- `results/data_check.json` - raw-data integrity summary.
- `results/data_summary.csv` - per-file row and window counts.
- `results/split_summary.csv` - split counts and time ranges.
- `results/split_indices.csv` - one row per healthy window with its split label.
- `results/data_config.json` - preprocessing configuration and scaler metadata.
- `results/sanity_checks.json` - machine-readable results of all checks.
- `results/phase1_summary_stats.json` - scaled-data and leakage statistics.
- `results/summary.md` - this report.
- `STATUS.md` - appended Phase 1 log entry.

Key numbers:

### Raw rows and windows per file

| File | Mode | Condition | Raw rows | 1,000 Hz rows | Windows | Measured rate (Hz) |
|---|---|---|---|---|---|---|
| `F0L` | IPPT | healthy | 143,715 | 14,372 | 1,428 | 1000.03 |
| `F0M` | MPPT | healthy | 141,014 | 14,102 | 1,401 | 1000.03 |
| `F1L` | IPPT | fault | 129,013 | 12,902 | 1,281 | 1000.03 |
| `F1M` | MPPT | fault | 139,014 | 13,902 | 1,381 | 1000.03 |
| `F2L` | IPPT | fault | 142,128 | 14,213 | 1,412 | 1000.91 |
| `F2M` | MPPT | fault | 144,015 | 14,402 | 1,431 | 1000.03 |
| `F3L` | IPPT | fault | 103,497 | 10,350 | 1,026 | 1000.03 |
| `F3M` | MPPT | fault | 69,967 | 6,997 | 690 | 1000.03 |
| `F4L` | IPPT | fault | 144,014 | 14,402 | 1,431 | 1000.03 |
| `F4M` | MPPT | fault | 144,014 | 14,402 | 1,431 | 1000.03 |
| `F5L` | IPPT | fault | 143,015 | 14,302 | 1,421 | 1000.03 |
| `F5M` | MPPT | fault | 144,014 | 14,402 | 1,431 | 1000.03 |
| `F6L` | IPPT | fault | 144,015 | 14,402 | 1,431 | 1000.03 |
| `F6M` | MPPT | fault | 144,015 | 14,402 | 1,431 | 1000.03 |
| `F7L` | IPPT | fault | 144,015 | 14,402 | 1,431 | 1000.03 |
| `F7M` | MPPT | fault | 144,015 | 14,402 | 1,431 | 1000.03 |
| **Total** |  |  | **2,163,480** | **216,356** | **21,488** |  |

`Raw rows` are the rows in each original CSV. `1,000 Hz rows` are the rows after anti-alias filtering and decimation, and `Windows` are the full 100-sample windows cut from those rows with stride 10.

### Downsampling

| Setting | Value |
|---|---|
| Function | `scipy.signal.decimate` |
| Filter | order-8 Chebyshev type I, 0.05 dB ripple |
| Decimation factor q | 10 |
| Zero-phase | yes (`zero_phase=True`, forward-backward `filtfilt`) |
| Low-pass cutoff | 400 Hz |
| Output sampling rate | 1,000 Hz |
| Time alignment | `Time[::10]` |

### Healthy split (F0L and F0M only)

| Mode | Split | Windows | Share of healthy | Start (s) | End (s) | Gap before (windows) |
|---|---|---|---|---|---|---|
| IPPT | train | 856 | 59.94% | 0.000 | 8.648 | 0 |
| IPPT | val | 285 | 19.96% | 8.659 | 11.598 | 10 |
| IPPT | test | 267 | 18.70% | 11.609 | 14.368 | 10 |
| MPPT | train | 840 | 59.96% | 0.000 | 8.488 | 0 |
| MPPT | val | 280 | 19.99% | 8.499 | 11.388 | 10 |
| MPPT | test | 261 | 18.63% | 11.399 | 14.098 | 10 |

Each boundary drops 10 windows. Because the stride divides the window length, that removes only 10 raw samples (about 11.0 ms between the last kept timestamp and the first timestamp of the next part), and those gap windows belong to no split. The IPPT and MPPT boundaries behave the same way.

### Leakage checks

| Check | IPPT | MPPT |
|---|---|---|
| Scaler fit source | `F0L` only | `F0M` only |
| Scaler fit rows | 8,650 | 8,490 |
| `n_samples_seen` | 8,650 | 8,490 |
| Fault files used in fit | 0 | 0 |
| train/val window-index overlap | 0 | 0 |
| train/test window-index overlap | 0 | 0 |
| val/test window-index overlap | 0 | 0 |
| train/val shared raw samples | 0 | 0 |
| train/test shared raw samples | 0 | 0 |
| val/test shared raw samples | 0 | 0 |
| Gap windows excluded from every split | 20 | 20 |
| Max abs difference: scaler mean vs F0 train mean | 0 | 0 |
| Max abs difference: scaler mean vs pooled fault mean | 6.71 | 13.3 |

### Scaled healthy training data

| Mode | Scope | Windows or rows | Pooled mean | Pooled std | Max abs feature mean | Feature std range |
|---|---|---|---|---|---|---|
| IPPT | Unique fit rows | 8,650 | 0.000000 | 1.000000 | 0.000000 | 1.000000 to 1.000000 |
| MPPT | Unique fit rows | 8,490 | 0.000000 | 1.000000 | 0.000000 | 1.000000 to 1.000000 |
| IPPT | Training windows | 856 | 0.001411 | 0.937302 | 0.042200 | 0.394726 to 1.002721 |
| MPPT | Training windows | 840 | 0.004102 | 0.947445 | 0.049299 | 0.432696 to 1.003639 |

The unique fit rows are the non-overlapping raw rows covered by the healthy training windows; the training windows are the actual model inputs, which overlap by design. Both are pooled across all 13 features. The unique fit rows have per-feature mean 0 and std 1 by construction. The training windows have a pooled std below 1 (0.937302 for IPPT and 0.947445 for MPPT) because the first few windows contain a startup transient, especially in `Vabc`; see the warnings. This is a property of the window distribution, not a scaler error.

Checks passed:
- All 30 automated sanity checks passed.
- No missing or infinite values: 0 NaN and 0 infinite values in windows, timestamps, and decimated rows.
- The IPPT scaler was fitted on 8,650 raw rows from `F0L` only, and the MPPT scaler on 8,490 raw rows from `F0M` only; `n_samples_seen` matches both row counts.
- No fault file was used in fitting: 0 fault files for IPPT and 0 for MPPT.
- Train/validation/test window indices are pairwise disjoint (0 train/val, 0 train/test, 0 val/test overlaps for IPPT; the same zeros hold for MPPT).
- Train/validation/test raw sample ranges are pairwise disjoint (0 train/val, 0 train/test, 0 val/test shared samples for IPPT; the same zeros hold for MPPT).
- The splits are chronological, the 10-window gaps belong to no split (20 IPPT and 20 MPPT gap windows), and every healthy window is assigned exactly once.
- The scaled healthy training rows have pooled mean and std 0.000000 and 1.000000 for IPPT, and 0.000000 and 1.000000 for MPPT; the target is 0 and 1.
- All 16 files loaded with the expected columns in order, and `Time` is strictly increasing in every file.
- The raw CSVs contain 0 missing values, 0 infinite values, and 0 duplicate rows.

Problems or warnings:
- `F3M` is the shortest recording: 6.996 s, 69,967 raw rows, and 690 windows. Per-file evaluation must report coverage for this file rather than silently dropping it.
- `F2L` has a slightly different decimated rate estimate: 1000.91 Hz versus a median of 1000.03 Hz across files (0.091% from 1,000 Hz). The native timestamps are nominal 10,000 Hz; the difference comes from timestamp jitter, and q=10 is still applied as specified.
- Startup transient: the first 10 training windows are outliers. `Vabc` reaches -22.3 standardized units in the first IPPT window and -20.8 in the first MPPT window. Excluding those windows, the pooled `Vabc` std drops from 0.395 to 0.037 for IPPT and from 0.433 to 0.035 for MPPT. The scaler is correct for the fit rows, but the first windows mix a startup transient into the training distribution.
- The zero-phase decimation filter has a short edge transient at the start and end of every file. It is kept, so the first and last window of each file may contain a small transient.
- Windows overlap by 90% within each split, so windows are not independent observations. This is intended for window-level detection but must be considered in statistical tests.
- The test parts are below 20% (18.70% IPPT and 18.63% MPPT) because the two 10-window gaps are excluded from the total.

Decision needed from me: Yes - confirm that these Phase 1 choices are frozen for modeling: 1,000 Hz anti-aliased data, 100-sample windows with stride 10, one scaler per operating mode, and a chronological 60/20/20 healthy split with two 10-window gaps (the test share is 18.7% IPPT and 18.6% MPPT). If not, say which choice should be revisited.

Next phase: Phase 2 - train and validate a healthy-only anomaly detector on these frozen windows, keeping fault files unseen until final evaluation.

## Phase 2: Labeling and timing diagnostic   (2026-10-26 14:30 +0530)

What was done: Phase 2 checked whether the fault recordings already look faulty before the documented 7-8 s injection interval, and whether the first 0.5 s startup transient is flagged. The diagnostic uses a deliberately simple placeholder: for each mode it fits a 13-feature mean and covariance to healthy training window means, scores every window by Mahalanobis distance with a pseudo-inverse and a small relative ridge, and sets the threshold at the 99th percentile of healthy validation scores. Fault files were never used to fit the mean/covariance or the threshold. `get_labels` adds whole-file labels (every F1-F7 window faulty) and time-resolved labels (-1 before 7 s, -2 for windows overlapping 7-8 s, 1 after 8 s); `F3M` gets only -1 labels because its last window ends at 6.988 s. All 21,488 windows of the 16 files were scored. The placeholder flagged 100.00% of the pre-injection windows outside the startup period, versus 0.75% and 0.77% false alarms on healthy test data for IPPT and MPPT.

Files created or changed:
- `src/data.py` - added `get_labels` and a cached window-time helper; existing preprocessing outputs are unchanged.
- `src/time_diagnostic.py` - fits the placeholder, scores all windows, writes the diagnostic tables, and draws the figures.
- `src/phase2_summary.py` - reads the saved results and writes this section and the STATUS entry.
- `run_all.py` - added the labeling and timing diagnostic as a pipeline step.
- `results/time_diagnostic_scores.csv` - one row per window with score, threshold, flag, timestamps, and both label modes.
- `results/label_diagnostic.csv` - per-fault pre/startup/7-8/post flag shares and median-score ratios.
- `results/healthy_false_alarm.csv` - healthy train/validation/test false-alarm rates for the placeholder.
- `results/time_diagnostic.json` - detector definition, fitted parameters, thresholds, and all check results.
- `results/phase2_summary_stats.json` - the facts used in this report.
- `figures/score_vs_time_F1L.png` ... `score_vs_time_F7M.png` - 14 individual score-versus-time plots.
- `figures/score_vs_time_grid.png` - 7 faults by 2 modes summary grid.
- `results/summary.md` - this Phase 2 section.
- `STATUS.md` - appended Phase 2 log entry.

Key numbers:

### Placeholder detector and healthy reference

| Mode | Healthy train windows | Healthy validation windows | Healthy test windows | Threshold | Validation flagged | Healthy test flagged (false-alarm rate) |
|---|---|---|---|---|---|---|
| IPPT | 856 | 285 | 267 | 5.301 | 3 (1.05%) | 2/267 (0.75%) |
| MPPT | 840 | 280 | 261 | 6.619 | 3 (1.07%) | 2/261 (0.77%) |

The detector is fitted on the healthy training windows of each mode, and the threshold is the 99th percentile of that mode's healthy validation scores. The healthy test rows are unseen by both the fit and the threshold.

### Per-fault timing diagnostic

| File | Windows before 7 s | Flagged before 7 s (excl. startup) | Flagged startup (first 0.5 s) | Flagged 7-8 s | Flagged after 8 s | Pre median / threshold | 7-8 s / pre median | After 8 s / pre median | No post-injection data |
|---|---|---|---|---|---|---|---|---|---|
| `F1L` | 691 | 100.00% | 100.00% | 100.00% | 100.00% | 15.14x | 0.99x | 2.38x | no |
| `F1M` | 691 | 100.00% | 100.00% | 100.00% | 100.00% | 32.17x | 1.00x | 37.73x | no |
| `F2L` | 691 | 100.00% | 100.00% | 100.00% | 100.00% | 8.02x | 0.98x | 10.00x | no |
| `F2M` | 691 | 100.00% | 100.00% | 100.00% | 100.00% | 74.75x | 1.00x | 1.20x | no |
| `F3L` | 691 | 100.00% | 100.00% | 100.00% | 100.00% | 7.78x | 1.01x | 12.57x | no |
| `F3M` | 690 | 100.00% | 100.00% | NA | NA | 8.83x | NA | NA | yes |
| `F4L` | 691 | 100.00% | 100.00% | 100.00% | 100.00% | 532.30x | 1.00x | 1.00x | no |
| `F4M` | 691 | 100.00% | 100.00% | 100.00% | 100.00% | 38.66x | 1.00x | 1.00x | no |
| `F5L` | 691 | 100.00% | 100.00% | 100.00% | 100.00% | 4.11x | 260.12x | 37.27x | no |
| `F5M` | 691 | 100.00% | 100.00% | 100.00% | 100.00% | 53.13x | 9.72x | 22.91x | no |
| `F6L` | 691 | 100.00% | 100.00% | 100.00% | 100.00% | 4.86x | 0.99x | 1.02x | no |
| `F6M` | 691 | 100.00% | 100.00% | 100.00% | 100.00% | 28.52x | 1.00x | 1.00x | no |
| `F7L` | 691 | 100.00% | 100.00% | 100.00% | 100.00% | 8.37x | 0.98x | 0.99x | no |
| `F7M` | 691 | 100.00% | 100.00% | 100.00% | 100.00% | 47.99x | 1.00x | 1.00x | no |

Shares are fractions of the relevant window group. `Windows before 7 s` counts windows whose end time is before 7.0 s. The pre-injection share excludes windows that start in the first 0.5 s; the startup share uses those windows. The 7-8 s group is the set of windows that overlap the documented injection interval and is dropped by the time-resolved labels. `NA` means the denominator is zero.

### Answers to the four timing questions

- (a) Pre-injection part: anomalous to this placeholder. All 14 fault files have 640-641 pre-injection windows after the first 0.5 s, and 100.00% of those windows were flagged. The median pre-injection score was 4.11x to 532.30x the threshold, while healthy test false-alarm rates were 0.75% (IPPT) and 0.77% (MPPT). This says the pre-injection segment differs from the healthy training file under this detector; it does not prove that the fault was already active, because a run-level operating-point offset can produce the same result.
- (b) Score jump at 7-8 s: The binary flag share cannot show a jump because every pre-injection and post-injection window is already above the healthy threshold. Inside the 7-8 s interval, `F5L`, `F5M` rise (9.72x to 260.12x of the pre-injection median). After 8 s, `F1L`, `F1M`, `F2L`, `F3L` rise (2.38x to 37.73x). `F4L`, `F4M`, `F6L`, `F6M`, `F7L`, `F7M` stay within 1.5% of their pre-injection median. `F2M` is intermediate at 1.20x.
- (c) Startup transient: flagged, but not separately. The 50-51 windows starting in the first 0.5 s of each fault file were flagged in every file (100.00%), and their median score differed from the pre-injection median by at most 11.3%. The whole pre-injection segment is already above threshold, so this is not evidence of a distinct startup effect.
- (d) No clear timing signal: `F4L`, `F4M`, `F6L`, `F6M`, `F7L`, `F7M` have post/pre median ratios of 0.99x to 1.02x, i.e. essentially flat. `F2M` is weak at 1.20x. `F3M` cannot be assessed: it has 0 post-injection windows and its last window ends at 6.988 s. At the binary flag level none of the files shows a timing change, because every fault window is already flagged.

Checks passed:
- Existing preprocessing is unchanged: the sanity-check file reports 30/30 checks passed after adding `get_labels`.
- The diagnostic reports 9/9 checks passed.
- All 21,488 windows of all 16 files were scored; scores are finite and non-negative (1.34 to 1.08e+04).
- Saved time-resolved labels match `get_labels` after the CSV round-trip, use only -2/-1/0/1, and partition every fault file into pre/overlap/post windows.
- `F3M` is handled without crashing: 690 pre-fault windows, 0 windows in 7-8 s, and 0 post-injection windows.
- The healthy reference contains only `F0L`/`F0M`, and each false-alarm rate equals flagged/windows (IPPT 2/267, MPPT 2/261).
- The detector mean/covariance and the 99th-percentile threshold use healthy training and validation windows only; 0 fault files were used for either.
- The only undefined values in `label_diagnostic.csv` are 6 post-injection fields for `F3M`, where the denominator is zero; there are no NaN scores.

Problems or warnings:
- The placeholder is saturated on fault files: every fault window in the 13 files with post-injection data is above the healthy threshold. The binary shares are therefore all 100%, and the median-score ratios are the only timing evidence in this table.
- The pre-injection anomaly may be a run-level offset rather than an early fault. The placeholder is a global window-mean model fitted to one healthy recording per mode, so differences in irradiance, temperature, or operating point between recordings can dominate it. No causal claim is made.
- The window-mean covariance is nearly singular (condition numbers 7.00e+06-7.68e+06); a relative ridge (1.81e-07-2.27e-07) and a pseudo-inverse were required. Very large scores (up to 10,822.0) come from near-null directions and should not be read as physical units.
- `F3M` ends before the 7-8 s window: its last window ends at 6.988 s, so it has no post-injection or 7-8 s data. Its score rise near the file end may include the zero-phase decimation edge transient.
- The healthy training windows include the startup transient, so the fitted covariance already contains the `F0` startup behavior. This can make the placeholder less sensitive to a similar transient in fault files.
- Windows overlap by 90%, so they are not independent observations; this diagnostic is descriptive and has no confidence intervals.
- This is a diagnostic placeholder, not the final model. No detector setting was tuned on fault results; the score, ridge, and threshold rules were fixed before fault scoring.

Decision needed from me: Yes - choose the label convention for the final evaluation. The diagnostic supports neither a blanket whole-file label nor a blanket time-resolved label: every fault file is already anomalous before 7 s, but only some files change at 7-8 s. Options: (1) keep whole-file labels as the primary comparison with the notebook and reference paper, and report time-resolved pre/post scores as a secondary analysis; or (2) make time-resolved labels primary, score pre-injection windows as false alarms, drop the 7-8 s overlap windows, and report `F3M` separately. Whichever is chosen, `F3M` must be reported as having no post-injection data.

Next phase: Phase 3 - train the healthy-only detector and evaluate it under the chosen label convention, keeping fault data out of every modeling decision.

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

## Phase 3: Baselines on the same pipeline   (2026-10-27 11:57 +0530)

What was done: Phase 3 re-implemented the four reference baselines from the paper on the frozen Phase 2 pipeline: EllipticEnvelope, IsolationForest, LocalOutlierFactor, and OneClassSVM. Each scikit-learn model was fitted only on the standardized healthy training rows of one operating mode (8,650 rows for IPPT, 8,490 for MPPT; both are below the 20,000-row cap, so no subsampling was needed). The Phase 2 Mahalanobis placeholder was added unchanged as a fifth, window-mean reference. A window score is the mean of -decision_function over its 100 time steps (higher means more anomalous); the threshold is the 99th percentile of healthy validation window scores. No fault file and no healthy test window was used for a fit or a threshold. The new label convention is recorded in `AGENTS.md`: whole-file labels are primary, time-resolved pre/post-8 s comparisons are secondary, and `F3M` is always reported as having no post-injection data. 10 score archives (5 models x 2 modes) were saved in the same five-key format the proposed model will use, together with a sanity table. No final metrics were computed in this phase.

Files created or changed:
- `AGENTS.md` - recorded frozen Phases 1-2 and the primary whole-file / secondary time-resolved label decision.
- `src/baselines.py` - fits the five baselines, scores every file, and saves the score archives and sanity table.
- `src/phase3_summary.py` - reads the saved results and writes this section and the STATUS entry.
- `run_all.py` - added the Phase 3 baseline and summary steps.
- `results/scores_baseline_<model>_<mode>.npz` - 10 archives with keys `scores`, `start_time`, `end_time`, `threshold`, `name`.
- `results/baseline_sanity.csv` - healthy validation/test flag rates, per-fault whole-file shares, and pre/post median scores.
- `results/baselines_metadata.json` - model settings, training and scoring times, thresholds, checks, environment, git commit.
- `results/phase3_summary_stats.json` - the facts used in this report.
- `results/summary.md` - this Phase 3 section.
- `STATUS.md` - appended Phase 3 log entry.

Key numbers:

### Training time and healthy false alarms

| Model | Mode | Train units | Training time (s) | Threshold | Healthy validation flagged | Healthy test flagged (false-alarm rate) |
|---|---|---|---|---|---|---|
| EllipticEnvelope | IPPT | 8,650 | 0.899 | 2.76234 | 3/285 (1.05%) | 5/267 (1.87%) |
| EllipticEnvelope | MPPT | 8,490 | 0.849 | 4.80809 | 3/280 (1.07%) | 5/261 (1.92%) |
| IsolationForest | IPPT | 8,650 | 0.259 | 0.000577689 | 3/285 (1.05%) | 0/267 (0.00%) |
| IsolationForest | MPPT | 8,490 | 0.141 | -0.00284774 | 3/280 (1.07%) | 3/261 (1.15%) |
| LocalOutlierFactor | IPPT | 8,650 | 0.841 | -0.240832 | 3/285 (1.05%) | 35/267 (13.11%) |
| LocalOutlierFactor | MPPT | 8,490 | 0.899 | -0.244838 | 3/280 (1.07%) | 55/261 (21.07%) |
| OneClassSVM | IPPT | 8,650 | 0.581 | 0.304288 | 3/285 (1.05%) | 29/267 (10.86%) |
| OneClassSVM | MPPT | 8,490 | 0.357 | -1.20806 | 3/280 (1.07%) | 74/261 (28.35%) |
| Mahalanobis (window-mean) | IPPT | 856 | 0.010 | 5.30072 | 3/285 (1.05%) | 2/267 (0.75%) |
| Mahalanobis (window-mean) | MPPT | 840 | 0.001 | 6.6192 | 3/280 (1.07%) | 2/261 (0.77%) |

Training time is the measured `fit` call only; scoring time is reported in `baselines_metadata.json`. The threshold is the 99th percentile of healthy validation window scores. The healthy test block is later in time and was not used for the fit or the threshold.

### Fault-file sanity and within-run change

| Model | Mode | Fault files with >=95% flagged | Whole-file flag share range | Clear rise >=2.0x | Weak rise 1.2-2.0x | Flat or lower <1.2x | Special cases | No post-injection data |
|---|---|---|---|---|---|---|---|---|
| EllipticEnvelope | IPPT | 6/7: `F1L`, `F2L`, `F3L`, `F4L`, `F6L`, `F7L` | 85.71%-100.00% | `F1L`, `F2L`, `F3L`, `F5L` | `F6L` | `F4L`, `F7L` | none | none |
| EllipticEnvelope | MPPT | 7/7 (all) | 100.00%-100.00% | `F1M`, `F2M`, `F5M` | none | `F4M`, `F6M`, `F7M` | none | `F3M` |
| IsolationForest | IPPT | 4/7: `F1L`, `F2L`, `F4L`, `F7L` | 78.47%-100.00% | `F3L`, `F5L` | `F1L`, `F2L`, `F6L` | `F4L`, `F7L` | none | none |
| IsolationForest | MPPT | 6/7: `F1M`, `F2M`, `F4M`, `F5M`, `F6M`, `F7M` | 47.83%-100.00% | `F1M`, `F2M`, `F5M` | none | `F4M`, `F6M`, `F7M` | none | `F3M` |
| LocalOutlierFactor | IPPT | 7/7 (all) | 99.16%-100.00% | `F2L`, `F3L`, `F5L`, `F7L` | `F6L` | `F4L` | `F1L`: ratio undefined (pre median -0.1278 < 0; medians -0.1278 -> 0.4163) | none |
| LocalOutlierFactor | MPPT | 7/7 (all) | 98.41%-100.00% | `F1M`, `F5M` | none | `F2M`, `F4M`, `F6M`, `F7M` | none | `F3M` |
| OneClassSVM | IPPT | 4/7: `F1L`, `F2L`, `F4L`, `F7L` | 76.21%-100.00% | `F3L`, `F5L` | `F2L`, `F6L` | `F1L`, `F7L` | both medians at the score ceiling: `F4L` (85.79); post median at the score ceiling: `F3L`, `F5L` (85.79) | none |
| OneClassSVM | MPPT | 6/7: `F1M`, `F2M`, `F4M`, `F5M`, `F6M`, `F7M` | 60.87%-100.00% | none | none | none | all 6 post-injection files: both medians at the score ceiling (93.79) | `F3M` |
| Mahalanobis (window-mean) | IPPT | 7/7 (all) | 100.00%-100.00% | `F1L`, `F2L`, `F3L`, `F5L` | none | `F4L`, `F6L`, `F7L` | none | none |
| Mahalanobis (window-mean) | MPPT | 7/7 (all) | 100.00%-100.00% | `F1M`, `F5M` | `F2M` | `F4M`, `F6M`, `F7M` | none | `F3M` |

Whole-file flag share is the fraction of all windows in that fault file above the healthy-validation threshold. The rise columns compare the median score of pre-injection windows (end < 7 s, excluding the first 0.5 s) with windows starting after 8 s in the same file. `Clear`, `weak`, and `flat` use the 2.0x and 1.2x conventions stated at the top of this section. `NA` means the denominator or ratio is undefined, not zero.

### Plain answers

- (a) Almost everything: at the 95% per-file threshold, EllipticEnvelope flags 13/14 (lowest `F5L` at 85.71%); IsolationForest flags 10/14 (lowest `F3M` at 47.83%); LocalOutlierFactor flags all 14 fault files; OneClassSVM flags 10/14 (lowest `F3M` at 60.87%); Mahalanobis (window-mean) flags all 14 fault files. The fully saturated references are `LocalOutlierFactor` and `Mahalanobis (window-mean)`, so the whole-file view cannot separate them from a trivial detector. Exact per-file shares are in `baseline_sanity.csv`.
- (b) Post-8 s score rise: the table lists every file. A clear rise (>= 2.0x the same file's pre-injection median) is unanimous across all models with post data for `F3L` and `F5L`. No model shows a clear rise for `F4L`, `F4M`, `F6L`, `F6M` and `F7M`. `F3M` has no post-injection data in every run and is excluded from every rise comparison. `OneClassSVM` in MPPT is saturated: both medians sit at the score ceiling (93.79) in 6 of 6 post-injection files, so its ratios (1.00x) are not evidence of no change. `LOF` `F1L` has a negative pre-injection median, so its ratio is undefined even though its median rises from -0.1278 to 0.4163.
- (c) Healthy test false-alarm rates: EllipticEnvelope IPPT 1.87% (5/267); EllipticEnvelope MPPT 1.92% (5/261); IsolationForest IPPT 0.00% (0/267); IsolationForest MPPT 1.15% (3/261); LocalOutlierFactor IPPT 13.11% (35/267); LocalOutlierFactor MPPT 21.07% (55/261); OneClassSVM IPPT 10.86% (29/267); OneClassSVM MPPT 28.35% (74/261); Mahalanobis (window-mean) IPPT 0.75% (2/267); Mahalanobis (window-mean) MPPT 0.77% (2/261). LOF and OneClassSVM are far above their validation rates, so their validation thresholds do not transfer to the later healthy test block; the other three stay much closer to their validation rates.

Checks passed:
- The baseline run reports 13/13 checks passed.
- All 10 archives contain exactly the five Phase 5 keys and 107,240 window scores; every model scored 21,448 windows (healthy train/val/test plus seven fault files).
- All scores, window times, and thresholds are finite; there are no NaN or infinite values.
- Every threshold equals the 99th percentile of healthy validation window scores recomputed from the saved archive; no test or fault window was used.
- The Mahalanobis reference reproduces the Phase 2 threshold and all of its window scores exactly.
- Every flag rate in `baseline_sanity.csv` equals flagged/windows, and the healthy validation flag rate is at most 2% for every model and mode.
- `F3M` is marked as having zero post-injection windows in all 5 model rows; its post median is `NA`, not a fabricated value.
- The cached-row scoring shortcut was checked against direct window scoring on sample windows for every model, mode, and file; all agree within numerical tolerance.
- The frozen Phase 2 sanity checks still report 30/30 passed.

Problems or warnings:
- The whole-file view is saturated. `LocalOutlierFactor` and `Mahalanobis (window-mean)` flag at least 95% of the windows in all 14 fault files, and EllipticEnvelope does so in 13/14. A near-perfect whole-file score therefore does not show that a detector is better than a trivial one; the secondary pre/post-8 s analysis is the discriminating view.
- LOF and OneClassSVM have high healthy test false-alarm rates (13.11%/21.07% for LOF and 10.86%/28.35% for OneClassSVM, IPPT/MPPT) even though their validation rates are 1.05%/1.07% and 1.05%/1.07%. The healthy recording changes over time, so a threshold fitted on the middle validation block does not transfer to the later test block.
- OneClassSVM scores saturate at a ceiling (85.79 IPPT, 93.79 MPPT). In saturated files both pre- and post-8 s medians equal the ceiling, so a ratio of 1x means the score cannot rise further, not that the fault has no effect.
- LOF `F1L` has a negative pre-injection median (-0.1278), so its post/pre ratio is undefined. The medians themselves rise to 0.4163; the ratio is reported as `NA` rather than forced.
- The baselines are fitted to one healthy recording per mode. Differences in irradiance, temperature, or operating point between recordings can dominate the scores, so the pre-injection anomaly is not proof that a fault was already active.
- The healthy training rows include the startup transient, because the frozen Phase 2 pipeline keeps it. The baselines therefore learn that transient as normal.
- Windows overlap by 90%, so window-level observations are not independent; the sanity table is descriptive and has no confidence intervals.
- The phase specification fixes `random_state=0` for EllipticEnvelope and IsolationForest; LOF, OneClassSVM, and Mahalanobis are deterministic. No 0/1/2 seed sweep was run, because the baseline settings are frozen and no fault data are used for any choice.
- The scikit-learn baselines were fitted on the unique 1 kHz rows covered by the training windows, not on the flattened 90%-overlapping windows. This avoids counting each time step ten times and matches the rows used to fit the Phase 2 scaler.
- EllipticEnvelope produces very large scores (up to 2.883e+06) from near-singular covariance directions; these are not physical units and should not be compared across models.

Decision needed from me: No. The label convention is fixed in `AGENTS.md` (whole-file primary, time-resolved secondary, `F3M` always reported as having no post-injection data), so Phase 4 can proceed. The 95% and 2.0x/1.2x cut-offs used above are descriptive reporting conventions only, not detector settings.

Next phase: Phase 4 - train the proposed healthy-only autoencoder on the same frozen windows and save its scores in the same five-key npz format.

## Phase 4: Proposed healthy-only TCN-BiLSTM autoencoder   (2026-10-27 17:42)

What was done: The proposed TCN-BiLSTM autoencoder was copied unchanged from `Untitled0.ipynb` and trained separately for MPPT and IPPT with seeds 0, 1 and 2. Each run used only the healthy training windows of its mode for fitting and the healthy validation windows for early stopping; no fault file influenced training, normalization, threshold selection, or any model decision. Two scores were saved for every window of every file: the plain mean squared reconstruction error and a fixed k=3 feature-normalized score whose per-feature mean and standard deviation come only from healthy validation windows. The threshold of each score is the 99th percentile of that score on healthy validation windows; the fixed k=3 was set before any fault scoring and was not tuned. The training script is resumable through the saved model file and writes one progress line per epoch to `logs/train.log`. Only sanity outputs were produced; no final classification metrics were computed.

Files created or changed:
- `src/model.py` - the notebook's `tcn_residual_block` and `build_tcn_bilstm_autoencoder` copied without architecture changes.
- `src/train.py` - resumable healthy-only training, two-score scoring, archive saving, sanity/training tables, and checks.
- `src/phase4_summary.py` - reads saved results and writes this section plus the STATUS entry.
- `run_all.py` - added the Phase 4 training and summary steps.
- `models/proposed_tcn_bilstm_<mode>_seed<k>.keras` - 6 trained models.
- `results/history_<mode>_seed<k>.json` and `figures/training_curve_<mode>_seed<k>.png` - one history and training curve per run.
- `results/scores_proposed_<score>_<mode>_seed<k>.npz` - 12 archives with the five baseline keys plus `feature_errors` of shape `(n_windows, 13)`.
- `results/proposed_sanity.csv` - healthy validation/test flag rates, per-fault whole-file shares, and pre/post-8 s medians for both scores and all seeds.
- `results/proposed_training.csv` - epochs run, best epoch, and best/final validation loss per mode and seed.
- `results/proposed_run_<mode>_seed<k>.json` - per-run configuration, thresholds, normalization statistics, timing, and check results.
- `results/proposed_metadata.json` - configuration, thresholds, normalization statistics, check results, environment, git commit.
- `results/phase4_summary_stats.json` - the facts used in this report.
- `results/summary.md` - this Phase 4 section.
- `STATUS.md` - appended Phase 4 log entry.
- `logs/train.log` - per-epoch progress and resumability log.

Key numbers:

### Training runs

| Mode | Seed | Epochs run | Best epoch | Best val loss | Final val loss | Stopped early | Training time (min) |
|---|---|---|---|---|---|---|---|
| MPPT | 0 | 100 | 100 | 0.131720 | 0.131720 | no | 11.0 |
| MPPT | 1 | 92 | 80 | 0.126161 | 0.126189 | yes | 8.9 |
| MPPT | 2 | 100 | 100 | 0.122294 | 0.122294 | no | 10.8 |
| IPPT | 0 | 100 | 100 | 0.128270 | 0.128270 | no | 18.5 |
| IPPT | 1 | 100 | 100 | 0.130297 | 0.130297 | no | 19.7 |
| IPPT | 2 | 100 | 100 | 0.140197 | 0.140197 | no | 22.6 |

Training time is the measured `fit` call only. The best epoch is the `argmin` of the saved validation-loss history; the model file contains the restored best weights.

### Healthy false alarms

| Score | Mode | Validation FAR mean (range) | Test FAR mean (range) | Test false alarms by seed |
|---|---|---|---|---|
| plain | MPPT | 1.07% (1.07%-1.07%) | 2.17% (1.92%-2.30%) | s0: 5/261; s1: 6/261; s2: 6/261 |
| plain | IPPT | 1.05% (1.05%-1.05%) | 0.00% (0.00%-0.00%) | s0: 0/267; s1: 0/267; s2: 0/267 |
| featnorm | MPPT | 1.07% (1.07%-1.07%) | 28.74% (16.86%-41.38%) | s0: 108/261; s1: 73/261; s2: 44/261 |
| featnorm | IPPT | 1.05% (1.05%-1.05%) | 11.24% (8.61%-15.36%) | s0: 26/267; s1: 23/267; s2: 41/267 |

A false alarm is a healthy window with score > threshold. Validation windows set the threshold; test windows are later in time and were never used for fitting or thresholding.

### F4, F6 and F7: post-8 s rise versus baselines

| Score | Fault file | Mode | Post/pre ratio seed 0 / 1 / 2 | Mean (range) | Rise class across seeds | Baselines with clear rise |
|---|---|---|---|---|---|---|
| plain | F4L | IPPT | 1.00x / 1.00x / 1.00x | 1.00x (1.00x-1.00x) | flat 3/3 | 0/5 |
| plain | F4M | MPPT | 1.01x / 1.01x / 1.01x | 1.01x (1.01x-1.01x) | flat 3/3 | 0/5 |
| plain | F6L | IPPT | 1.43x / 1.42x / 1.38x | 1.41x (1.38x-1.43x) | weak 3/3 | 0/5 |
| plain | F6M | MPPT | 1.00x / 1.00x / 1.00x | 1.00x (1.00x-1.00x) | flat 3/3 | 0/5 |
| plain | F7L | IPPT | 1.04x / 1.03x / 0.96x | 1.01x (0.96x-1.04x) | flat 3/3 | 1/5 (LocalOutlierFactor) |
| plain | F7M | MPPT | 1.01x / 1.00x / 1.00x | 1.00x (1.00x-1.01x) | flat 3/3 | 0/5 |
| featnorm | F4L | IPPT | 1.00x / 1.00x / 1.00x | 1.00x (1.00x-1.00x) | flat 3/3 | 0/5 |
| featnorm | F4M | MPPT | 1.04x / 0.99x / 1.01x | 1.01x (0.99x-1.04x) | flat 3/3 | 0/5 |
| featnorm | F6L | IPPT | 1.67x / 1.74x / 1.75x | 1.72x (1.67x-1.75x) | weak 3/3 | 0/5 |
| featnorm | F6M | MPPT | 1.17x / 0.90x / 0.97x | 1.01x (0.90x-1.17x) | flat 3/3 | 0/5 |
| featnorm | F7L | IPPT | 1.72x / 1.39x / 1.36x | 1.49x (1.36x-1.72x) | weak 3/3 | 1/5 (LocalOutlierFactor) |
| featnorm | F7M | MPPT | 1.01x / 0.94x / 0.97x | 0.97x (0.94x-1.01x) | flat 3/3 | 0/5 |

The proposed ratio is the median score of windows starting after 8 s divided by the median score of pre-injection windows ending before 7 s, excluding the first 0.5 s. Baseline counts come from `baseline_sanity.csv`; the denominator is the number of baseline models with post-injection data for that file.

### Seed consistency

| Score | Mode | Validation threshold mean (range) | Threshold CV | Max per-file flag-share range across seeds | Max per-file ratio spread across seeds (max/min) | Same rise class across seeds |
|---|---|---|---|---|---|---|
| plain | MPPT | 0.391791 (0.385935-0.399639) | 1.47% | 0.14 pp | 1.14x | 6/6 |
| plain | IPPT | 0.443281 (0.414091-0.474312) | 5.55% | 8.09 pp | 1.23x | 6/7 |
| featnorm | MPPT | 4.23948 (4.02601-4.52394) | 4.94% | 0.00 pp | 4.29x | 5/6 |
| featnorm | IPPT | 4.49965 (4.01023-4.88885) | 8.13% | 0.00 pp | 1.82x | 6/7 |

### Plain answers

- (a) Healthy test false-alarm rates versus the validation rates (mean and range across the three seeds): plain MPPT: validation 1.07% (1.07%-1.07%), test 2.17% (1.92%-2.30%); plain IPPT: validation 1.05% (1.05%-1.05%), test 0.00% (0.00%-0.00%); featnorm MPPT: validation 1.07% (1.07%-1.07%), test 28.74% (16.86%-41.38%); featnorm IPPT: validation 1.05% (1.05%-1.05%), test 11.24% (8.61%-15.36%). The test rate is above the validation rate for plain MPPT, featnorm MPPT, featnorm IPPT.
- (b) Post-8 s rise for F4, F6 and F7: plain clear-rise files: none; featnorm clear-rise files: none. Baseline clear-rise counts on the same six files are `F4L` 0/5, `F4M` 0/5, `F6L` 0/5, `F6M` 0/5, `F7L` 1/5, `F7M` 0/5 (clear rise is >= 2.0x the same file's pre-injection median). No proposed score shows a clear rise on any target file. `F3M` has no post-injection data and is excluded from this comparison.
- (c) Seed consistency: the largest threshold coefficient of variation is 8.13%; the largest per-file flag-share range is 8.09 percentage points; the largest spread between the smallest and largest seed post/pre ratio is 4.29x (`featnorm` `MPPT` `F5M`, 77.23x-330.97x). The three seeds give the same clear/weak/flat rise class in 23/26 score/mode/file comparisons with post-injection data.
- (d) Early stopping and convergence: epochs run ranged 92-100 and best epochs ranged 80-100. No run stopped before the very-early cut-off (<= 20 epochs); every run improved on its first-epoch validation loss; no run peaked at epoch 1. Total measured training time was 91.6 minutes on /CPU:0.

Checks passed:
- The proposed-model run reports 23/23 checks passed.
- All 12 archives contain exactly the five baseline keys plus `feature_errors`, and all 128,688 saved window scores are finite.
- Every threshold equals the 99th percentile of healthy validation scores recomputed from the saved archive; no healthy test or fault window was used.
- Training and validation arrays are exactly the frozen healthy F0L/F0M splits (SHA-256 verified), and no fault file contributed a window.
- The feature-error mean/std used for `featnorm` match the healthy validation windows exactly when recomputed from each saved model.
- Batch scoring agrees with one-window-at-a-time scoring on sample windows for every run, mode, and file.
- `plain` equals the mean of the saved feature errors, and `featnorm` equals the fixed k=3 formula recomputed from the saved feature errors.
- Every flag rate in `proposed_sanity.csv` equals flagged/windows; `F3M` is marked as having zero post-injection windows for every score and seed.
- The frozen Phase 1-3 modules `src/data.py` and `src/baselines.py` were not edited; `Untitled0.ipynb` was not modified.

Problems or warnings:
- No TensorFlow GPU was found; all training used `/CPU:0`. Measured training time was 91.6 minutes for six runs.
- Windows overlap by 90%, so window-level observations are not independent; the sanity table is descriptive and has no confidence intervals or significance tests.
- The frozen preprocessing keeps the `filtfilt` edge transient and the startup transient in the first windows, so those windows are learned as normal healthy behavior.
- Mean whole-file fault flag shares per score/mode/file range from 86.28% to 100.00%; the lowest single-seed share is 82.69%. The whole-file view is close to saturation but not uniform.
- Post/pre ratios in the tens, hundreds, or thousands occur when the post-injection score is far outside the healthy range; they are descriptive magnitudes, not calibrated effect sizes, and the exact ratio varies with seed.
- The healthy recording drifts over time; a threshold fitted on the validation block can transfer imperfectly to the later test block. The test false-alarm rates above are the relevant measurement, not a guarantee.
- `F3M` ends before 7 s and has no post-injection data. It is always reported as such and is excluded from the post-8 s comparison; no denominator was invented for it.
- Comparisons with `baseline_sanity.csv` are descriptive. The baselines use the same frozen windows and threshold rule but different model classes; no statistical test was run.
- No AUROC, precision, recall, F1, or other final metric was computed in this phase. The saved archives are the input to the Phase 5 evaluation.

Decision needed from me: No. The architecture, fixed k=3, validation normalization, and threshold rule are frozen by this phase; Phase 5 can consume the saved archives without retuning anything after seeing fault sanity.

Next phase: Phase 5 - final whole-file and time-resolved evaluation of the saved proposed-model score archives.

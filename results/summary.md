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

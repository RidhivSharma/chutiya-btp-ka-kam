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

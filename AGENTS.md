# AGENTS.md - Permanent project rules

Read this file at the start of **every phase**, before touching code or data.
These rules are permanent unless the project owner explicitly changes them in writing.

## Project

Convert the exploratory notebook `Untitled0.ipynb` into a clean, reproducible,
conference-quality experiment on the GPVS-Faults photovoltaic dataset.

The original notebook is a historical artifact and must remain untouched.

## 1. Dataset contract

- Dataset: **GPVS-Faults**, distributed as `CSV_Files.zip`.
- Extract the archive into `data/raw/` (the CSVs live in `data/raw/CSV_Files/`).
- There are **16 CSV files**: `F0L` ... `F7L` and `F0M` ... `F7M`.
- `F0L` and `F0M` are **healthy**.
- `F1` ... `F7` are **faults**.
- Suffix `L` = **IPPT** operating mode.
- Suffix `M` = **MPPT** operating mode.
- Every file has **13 features**:
  `Ipv, Vpv, Vdc, ia, ib, ic, va, vb, vc, Iabc, If, Vabc, Vf`
  plus the `Time` column.
- Native sampling is **10 kHz**. `Time` is in seconds, not a sample index.

## 2. Healthy-only training (non-negotiable)

The model is trained **ONLY on healthy data (F0)**.

Fault files (`F1` ... `F7`) are **NEVER** used for:

- training or fine-tuning,
- fitting scalers, encoders, or normalization statistics,
- choosing thresholds or decision rules,
- hyperparameter choices, early stopping, or model selection,
- feature selection, calibration, or any other modeling decision.

Fault files are used **only for final evaluation**. No peeking at fault metrics
while developing a model. If a bug is fixed after fault results have been seen,
document the bug and rerun the entire pipeline from scratch; do not tune the
method in response to fault performance.

Every fitted transform must be fitted on healthy training data only. Healthy
validation data may be used for legitimate healthy-only model selection
(thresholds, epochs, etc.); fault data may not.

Evaluation labels may be used for scoring only, never as detector inputs.

## 3. No hard-coded result numbers

- Every number that appears in a table, figure, or claim must be computed by
  code and saved to `results/` as CSV or JSON.
- Never transcribe, round, or hard-code a result anywhere.
- Figures must be generated from saved data by code, not drawn from literals.
- Save run metadata with results: seeds, package versions, configuration, and
  the git commit hash when available.
- `STATUS.md` may quote numbers only when they were produced by the code.

## 4. Reproducibility and seeds

- Set seeds everywhere: Python `random`, NumPy, and TensorFlow.
- Default seeds are **0, 1, 2**. Run all seeds unless a phase explicitly has
  no stochastic component.
- Use `tf.keras.utils.set_random_seed(seed)` and deterministic TensorFlow
  operations where feasible.
- Record the seed with every result row.
- Report mean and standard deviation across seeds where applicable.
- Avoid unseeded shuffling, sampling, or initialization.

## 5. Source layout and notebook

- **Do not delete or modify `Untitled0.ipynb`.**
- All new code lives in `src/` as `.py` files.
- `run_all.py` in the project root reruns the entire pipeline from raw data to
  final tables and figures.
- Use `pathlib` and paths relative to the project root. Do not use absolute
  paths that only work on one machine.
- Keep the pipeline runnable from a clean checkout after the dataset archive is
  placed in the project root.

## 6. Windows compatibility

- The project must run on Windows. Use Python and PowerShell commands only.
- No bash-only commands (`rm -rf`, `export`, heredocs, `chmod`, etc.).
- Avoid Unicode status symbols in console output; Windows terminals may not
  encode them. Plain ASCII is preferred in scripts.
- Use `pathlib.Path`, not hard-coded `/` or `\` path strings.

## 7. Honest reporting

- Report bad, null, or surprising results honestly.
- Do not tune anything to improve fault-detection numbers.
- Do not hide failed seeds, degenerate baselines, or results that contradict
  the hypothesis.
- Clearly separate descriptive dataset statistics from model evaluation.

## 8. Phase workflow

At the end of every phase:

1. Run the relevant code and save all outputs.
2. Append a short entry to `STATUS.md` with:
   - date and phase name,
   - what was done,
   - files created or changed,
   - key numbers produced by code,
   - open issues or risks.
3. Never overwrite or delete earlier status entries.

## 9. Environment

- Python **3.13** virtual environment in `.venv/`.
- Create and populate it with `powershell -ExecutionPolicy Bypass -File .\setup_venv.ps1`.
- Pinned dependencies are in `requirements.txt`.
- Run project code with `.venv\Scripts\python.exe` (or after activating the venv).
- Main command: `.venv\Scripts\python.exe .\run_all.py`.

## 10. Directory conventions

- `src/` - all pipeline code.
- `results/` - computed CSV/JSON outputs (source of truth for numbers).
- `figures/` - generated figures (PDF/PNG).
- `models/` - trained model artifacts and checkpoints.
- `data/raw/` - extracted raw CSVs; never edit raw data.
- `STATUS.md` - chronological phase log.

## 11. Phase summary report (`results/summary.md`)

- After finishing **every** phase, update `results/summary.md`. Never overwrite
  earlier sections; append a new section for the phase. Each section must use
  this format:

    ```
    ## Phase N: <title>   (date and time)
    What was done: 3 to 6 plain-language sentences for a non-ML reader.
    Files created or changed: list with one-line purpose each.
    Key numbers: a simple markdown table with the important values, all read from saved csv/json files or printed by code (never typed from memory).
    Checks passed: bullet list (e.g. no NaN, scaler fitted on train only, no overlap between splits).
    Problems or warnings: anything unexpected, or "none".
    Decision needed from me: yes/no, and what.
    Next phase: one line.
    ```

- Also append the same short entry to `STATUS.md`.
- The summary must be created even if the phase had problems. If something
  failed, say exactly what failed.
- In later phases, if a prompt tells you to write `results/summary.md`, add a
  new section to the existing file instead of replacing it.

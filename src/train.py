"""Phase 4: train the proposed healthy-only TCN-BiLSTM autoencoder.

The model is the notebook architecture unchanged (``src/model.py``). For each
operating mode (MPPT, IPPT) and each seed in 0, 1, 2 it is trained only on the
healthy *training* windows of that mode and validated only on the healthy
*validation* windows. Adam with learning rate 1e-3, batch size 32, at most 100
epochs, EarlyStopping on ``val_loss`` (patience 12, restore best weights) and
ReduceLROnPlateau (factor 0.5, patience 5, min_lr 1e-6).

Every mode/seed run is independent and resumable: a finished model file is the
completion marker, and progress is appended to ``logs/train.log``. Fault files
are never used for training, validation, early stopping, threshold selection,
or feature-error normalization.

Two window scores are saved for every file:

* ``plain``: mean squared reconstruction error over the 100 time steps and the
  13 features;
* ``featnorm``: per-feature mean squared error over time, standardized with the
  mean and standard deviation of that feature's error over the healthy
  *validation* windows of the same mode and seed. Negative z-scores are
  clipped to zero and the window score is the mean of the three largest
  z-scores. ``k = 3`` is fixed and is never tuned.

The threshold of each score is the 99th percentile of that score over healthy
validation windows. The saved archives have the five baseline keys plus
``feature_errors`` of shape ``(n_windows, 13)``.

Outputs
-------
``models/proposed_tcn_bilstm_<mode>_seed<k>.keras``
``results/history_<mode>_seed<k>.json``
``figures/training_curve_<mode>_seed<k>.png``
``results/scores_proposed_plain_<mode>_seed<k>.npz``
``results/scores_proposed_featnorm_<mode>_seed<k>.npz``
``results/proposed_sanity.csv``
``results/proposed_training.csv``
``results/proposed_metadata.json``
``logs/train.log``

Run::

    .venv\\Scripts\\python.exe .\\src\\train.py
"""

from __future__ import annotations

import os

# Keep CPU math deterministic and avoid oneDNN non-determinism. These variables
# must be set before TensorFlow is imported. The benchmark showed no measurable
# speed difference between oneDNN on and off for this model.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402
from tensorflow.keras.callbacks import (  # noqa: E402
    Callback,
    EarlyStopping,
    ReduceLROnPlateau,
)
from tensorflow.keras.optimizers import Adam  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (  # noqa: E402
    FAULT_END_S,
    FAULT_START_S,
    FEATURES,
    HEALTHY_STEMS,
    MODES,
    RESULTS_DIR,
    WINDOW_SIZE,
    build_dataset,
    environment_info,
    git_commit,
)
from src.model import build_tcn_bilstm_autoencoder  # noqa: E402
from src.time_diagnostic import STARTUP_END_S, THRESHOLD_QUANTILE  # noqa: E402

MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "figures"
LOGS_DIR = PROJECT_ROOT / "logs"
LOG_PATH = LOGS_DIR / "train.log"
SANITY_PATH = RESULTS_DIR / "proposed_sanity.csv"
TRAINING_PATH = RESULTS_DIR / "proposed_training.csv"
METADATA_PATH = RESULTS_DIR / "proposed_metadata.json"

SEEDS = (0, 1, 2)
MODE_ORDER = (("M", "MPPT"), ("L", "IPPT"))
MAX_EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 12
REDUCE_LR_FACTOR = 0.5
REDUCE_LR_PATIENCE = 5
MIN_LR = 1e-6
SCORE_KINDS = ("plain", "featnorm")
TOP_K = 3
STD_FLOOR = 1e-8
SCORING_BATCH_SIZE = 256
FAULT_STEMS_BY_MODE = {
    "L": tuple(f"F{i}L" for i in range(1, 8)),
    "M": tuple(f"F{i}M" for i in range(1, 8)),
}
MODEL_LABELS = {
    "plain": "TCN-BiLSTM (plain)",
    "featnorm": "TCN-BiLSTM (featnorm)",
}
MODEL_KEYS = {
    "plain": "proposed_plain",
    "featnorm": "proposed_featnorm",
}
REQUIRED_KEYS = {
    "scores",
    "start_time",
    "end_time",
    "threshold",
    "name",
    "feature_errors",
}

CHECK_RESULTS: list[dict[str, object]] = []
DEVICE = "/CPU:0"


def check(name: str, passed: bool, detail: str) -> bool:
    """Record and print one named assertion."""
    passed = bool(passed)
    CHECK_RESULTS.append({"name": name, "passed": passed, "detail": detail})
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}", flush=True)
    return passed


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_line(message: str) -> None:
    """Print and append one line to logs/train.log."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{stamp} {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")


def json_safe(value):
    """Convert numpy scalars/arrays to plain JSON-safe Python values."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(json_safe(payload), handle, indent=2, allow_nan=False)
        handle.write("\n")


def read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def set_seeds(seed: int) -> None:
    """Set Python, NumPy, and TensorFlow seeds."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.keras.utils.set_random_seed(seed)


def configure_device() -> str:
    """Enable GPU memory growth if a GPU exists and return the device string."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
        device = "/GPU:0"
    else:
        device = "/CPU:0"
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Physical GPUs found: {gpus if gpus else 'none'}")
    print(f"Training and scoring device: {device}")
    return device


def score_windows(
    model: tf.keras.Model,
    windows: np.ndarray,
    batch_size: int = SCORING_BATCH_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (plain score, (n_windows, 13) feature errors) for windows."""
    windows = np.asarray(windows, dtype=np.float32)
    if windows.ndim != 3 or windows.shape[1:] != (WINDOW_SIZE, len(FEATURES)):
        raise ValueError(f"Unexpected window shape: {windows.shape}")

    reconstructed = model.predict(windows, batch_size=batch_size, verbose=0)
    reconstructed = np.asarray(reconstructed, dtype=np.float64)
    if reconstructed.shape != windows.shape:
        raise RuntimeError(
            f"Reconstruction shape {reconstructed.shape} != input {windows.shape}"
        )

    squared = np.square(windows.astype(np.float64) - reconstructed)
    feature_errors = squared.mean(axis=1)
    plain_scores = feature_errors.mean(axis=1)
    return plain_scores, feature_errors


def featnorm_scores(
    feature_errors: np.ndarray,
    reference_mean: np.ndarray,
    reference_std: np.ndarray,
) -> np.ndarray:
    """Fixed k=3 feature-normalized score; negative z-scores are clipped."""
    feature_errors = np.asarray(feature_errors, dtype=np.float64)
    if feature_errors.ndim != 2 or feature_errors.shape[1] != len(FEATURES):
        raise ValueError(f"Unexpected feature-error shape: {feature_errors.shape}")
    if feature_errors.shape[1] < TOP_K:
        raise ValueError("Not enough features for the fixed top-k score")

    z_scores = (feature_errors - reference_mean) / reference_std
    z_scores = np.maximum(z_scores, 0.0)
    top = np.sort(z_scores, axis=1)[:, -TOP_K:]
    return top.mean(axis=1)


def healthy_training_arrays(
    data: dict[str, object], suffix: str
) -> dict[str, object]:
    """Build train/val arrays only from the healthy file of one mode.

    The function asserts that the arrays are exactly the frozen healthy split
    and that no fault file contributes a single window.
    """
    healthy_stem = HEALTHY_STEMS[suffix]
    if healthy_stem not in ("F0L", "F0M"):
        raise RuntimeError(f"Unexpected healthy file: {healthy_stem}")

    splits = data["splits"][suffix]
    train_indices = np.asarray(splits["train"], dtype=np.int64)
    val_indices = np.asarray(splits["val"], dtype=np.int64)
    if np.intersect1d(train_indices, val_indices).size:
        raise RuntimeError("Healthy train and validation indices overlap")

    healthy_windows = np.asarray(data["windows"][healthy_stem])
    train_x = np.asarray(healthy_windows[train_indices], dtype=np.float32)
    val_x = np.asarray(healthy_windows[val_indices], dtype=np.float32)

    if not np.array_equal(train_x, healthy_windows[train_indices]):
        raise RuntimeError("Training array is not the healthy train slice")
    if not np.array_equal(val_x, healthy_windows[val_indices]):
        raise RuntimeError("Validation array is not the healthy val slice")
    if train_x.shape[1:] != (WINDOW_SIZE, len(FEATURES)):
        raise RuntimeError(f"Unexpected training shape: {train_x.shape}")
    if val_x.shape[1:] != (WINDOW_SIZE, len(FEATURES)):
        raise RuntimeError(f"Unexpected validation shape: {val_x.shape}")
    if not np.isfinite(train_x).all() or not np.isfinite(val_x).all():
        raise RuntimeError("Non-finite value in healthy train/validation arrays")

    source_files = {healthy_stem}
    if not all(name.startswith("F0") for name in source_files):
        raise RuntimeError("A non-healthy file entered the training arrays")

    return {
        "train_x": train_x,
        "val_x": val_x,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "source_file": healthy_stem,
        "train_sha256": hashlib.sha256(train_x.tobytes()).hexdigest(),
        "val_sha256": hashlib.sha256(val_x.tobytes()).hexdigest(),
    }


class EpochProgressLogger(Callback):
    """Write one readable progress line per epoch."""

    def __init__(
        self,
        mode: str,
        seed: int,
        max_epochs: int,
        run_start: float,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.seed = seed
        self.max_epochs = max_epochs
        self.run_start = run_start
        self.epoch_start = 0.0

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        try:
            learning_rate = float(
                tf.keras.backend.get_value(self.model.optimizer.learning_rate)
            )
        except (AttributeError, TypeError, ValueError):
            learning_rate = float("nan")
        elapsed = time.perf_counter() - self.epoch_start
        total = time.perf_counter() - self.run_start
        log_line(
            f"[{self.mode}][seed={self.seed}] "
            f"epoch {epoch + 1}/{self.max_epochs} "
            f"loss={float(logs.get('loss', float('nan'))):.6f} "
            f"val_loss={float(logs.get('val_loss', float('nan'))):.6f} "
            f"lr={learning_rate:.3g} "
            f"epoch_s={elapsed:.2f} total_s={total:.1f}"
        )


def plot_training_curve(
    history: dict[str, list[float]],
    mode: str,
    seed: int,
    best_epoch: int,
    path: Path,
) -> None:
    """Save train/validation loss and learning-rate curves."""
    epochs = np.arange(1, len(history["loss"]) + 1)
    figure, axes = plt.subplots(2, 1, figsize=(8.0, 7.0), sharex=True)

    axes[0].plot(epochs, history["loss"], label="train loss", color="#1f77b4")
    if "val_loss" in history:
        axes[0].plot(
            epochs,
            history["val_loss"],
            label="validation loss",
            color="#d62728",
        )
    axes[0].axvline(
        best_epoch,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label=f"best epoch ({best_epoch})",
    )
    axes[0].set_ylabel("MSE loss")
    axes[0].set_title(f"TCN-BiLSTM autoencoder - {mode} - seed {seed}")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    if "learning_rate" in history:
        axes[1].plot(
            epochs,
            history["learning_rate"],
            color="#2ca02c",
        )
        axes[1].set_ylabel("learning rate")
        axes[1].set_yscale("log")
    else:
        axes[1].text(0.5, 0.5, "learning rate not logged", ha="center")
    axes[1].set_xlabel("epoch")
    axes[1].grid(True, alpha=0.3)

    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def model_paths(mode: str, seed: int) -> dict[str, Path]:
    return {
        "model": MODELS_DIR / f"proposed_tcn_bilstm_{mode}_seed{seed}.keras",
        "history": RESULTS_DIR / f"history_{mode}_seed{seed}.json",
        "figure": FIGURES_DIR / f"training_curve_{mode}_seed{seed}.png",
        "run_meta": RESULTS_DIR / f"proposed_run_{mode}_seed{seed}.json",
        "plain": RESULTS_DIR / f"scores_proposed_plain_{mode}_seed{seed}.npz",
        "featnorm": RESULTS_DIR
        / f"scores_proposed_featnorm_{mode}_seed{seed}.npz",
    }


def train_run(
    data: dict[str, object],
    suffix: str,
    seed: int,
    max_epochs: int,
) -> dict[str, object]:
    """Train one mode/seed from scratch and save history, figure, and model."""
    mode = MODES[suffix]
    paths = model_paths(mode, seed)
    arrays = healthy_training_arrays(data, suffix)
    train_x = arrays["train_x"]
    val_x = arrays["val_x"]

    log_line(
        f"[{mode}][seed={seed}] START training "
        f"train_windows={train_x.shape[0]} val_windows={val_x.shape[0]} "
        f"source={arrays['source_file']} max_epochs={max_epochs} "
        f"batch_size={BATCH_SIZE} lr={LEARNING_RATE}"
    )

    set_seeds(seed)
    run_start = time.perf_counter()
    with tf.device(DEVICE):
        model = build_tcn_bilstm_autoencoder(
            timesteps=WINDOW_SIZE,
            n_features=len(FEATURES),
        )
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss="mse",
        )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=0,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=MIN_LR,
            verbose=0,
        ),
        EpochProgressLogger(mode, seed, max_epochs, run_start),
    ]

    fit_start = time.perf_counter()
    with tf.device(DEVICE):
        fit_history = model.fit(
            train_x,
            train_x,
            validation_data=(val_x, val_x),
            epochs=max_epochs,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            shuffle=True,
            verbose=0,
        )
    training_time = time.perf_counter() - fit_start

    history = {
        key: [float(value) for value in values]
        for key, values in fit_history.history.items()
    }
    if "loss" not in history or "val_loss" not in history:
        raise RuntimeError("Training history is missing loss or val_loss")
    if not np.isfinite(history["loss"]).all():
        raise RuntimeError("Non-finite training loss")
    if not np.isfinite(history["val_loss"]).all():
        raise RuntimeError("Non-finite validation loss")

    epochs_run = len(history["loss"])
    best_index = int(np.argmin(history["val_loss"]))
    best_epoch = best_index + 1
    best_val_loss = float(history["val_loss"][best_index])
    final_val_loss = float(history["val_loss"][-1])
    initial_val_loss = float(history["val_loss"][0])
    history["epoch"] = list(range(1, epochs_run + 1))

    write_json(paths["history"], history)
    plot_training_curve(
        history,
        mode,
        seed,
        best_epoch,
        paths["figure"],
    )

    # Feature-error reference statistics are computed from healthy validation
    # windows only, before the completion marker (the model file) is written.
    _, val_feature_errors = score_windows(model, val_x)
    reference_mean = val_feature_errors.mean(axis=0)
    reference_std = np.maximum(
        val_feature_errors.std(axis=0, ddof=0),
        STD_FLOOR,
    )
    if not np.isfinite(reference_mean).all() or not np.isfinite(
        reference_std
    ).all():
        raise RuntimeError("Non-finite feature-error reference statistics")

    run_meta = {
        "mode": mode,
        "mode_suffix": suffix,
        "seed": seed,
        "max_epochs": max_epochs,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "early_stopping": {
            "monitor": "val_loss",
            "patience": EARLY_STOPPING_PATIENCE,
            "restore_best_weights": True,
        },
        "reduce_lr": {
            "monitor": "val_loss",
            "factor": REDUCE_LR_FACTOR,
            "patience": REDUCE_LR_PATIENCE,
            "min_lr": MIN_LR,
        },
        "epochs_run": int(epochs_run),
        "best_epoch": int(best_epoch),
        "best_val_loss": best_val_loss,
        "final_val_loss": final_val_loss,
        "initial_val_loss": initial_val_loss,
        "stopped_early": bool(epochs_run < max_epochs),
        "improved_over_initial": bool(best_val_loss < initial_val_loss),
        "training_time_s": float(training_time),
        "train_windows": int(train_x.shape[0]),
        "val_windows": int(val_x.shape[0]),
        "train_source_file": str(arrays["source_file"]),
        "val_source_file": str(arrays["source_file"]),
        "train_source_indices": f"splits[{suffix!r}]['train']",
        "val_source_indices": f"splits[{suffix!r}]['val']",
        "train_sha256": str(arrays["train_sha256"]),
        "val_sha256": str(arrays["val_sha256"]),
        "feature_error_reference": {
            "source": "healthy validation windows of this mode and seed",
            "mean": reference_mean.tolist(),
            "std": reference_std.tolist(),
            "std_ddof": 0,
            "std_floor": STD_FLOOR,
        },
        "feature_error_reference_raw_std_min": float(
            val_feature_errors.std(axis=0, ddof=0).min()
        ),
        "model_params": int(model.count_params()),
        "device": DEVICE,
        "tensorflow_version": tf.__version__,
        "model_path": str(paths["model"].relative_to(PROJECT_ROOT)),
        "history_path": str(paths["history"].relative_to(PROJECT_ROOT)),
        "figure_path": str(paths["figure"].relative_to(PROJECT_ROOT)),
        "generated_utc": now_iso(),
    }
    write_json(paths["run_meta"], run_meta)

    # Save the model last: its existence means the whole run is complete.
    with tf.device(DEVICE):
        model.save(paths["model"])

    log_line(
        f"[{mode}][seed={seed}] DONE training "
        f"epochs={epochs_run} best_epoch={best_epoch} "
        f"best_val_loss={best_val_loss:.6f} "
        f"final_val_loss={final_val_loss:.6f} "
        f"time_s={training_time:.1f}"
    )
    return run_meta


def load_run(
    suffix: str,
    seed: int,
) -> tuple[tf.keras.Model, dict[str, object]]:
    """Load a finished model and its run metadata."""
    mode = MODES[suffix]
    paths = model_paths(mode, seed)
    if not paths["run_meta"].exists():
        raise FileNotFoundError(
            f"Model exists but run metadata is missing: {paths['run_meta']}"
        )
    if not paths["history"].exists():
        raise FileNotFoundError(
            f"Model exists but history is missing: {paths['history']}"
        )
    run_meta = read_json(paths["run_meta"])
    with tf.device(DEVICE):
        model = tf.keras.models.load_model(paths["model"], compile=False)
    return model, run_meta


def make_group(
    name: str,
    file: str,
    part: str,
    indices: np.ndarray,
    scores: np.ndarray,
    feature_errors: np.ndarray,
    times: np.ndarray,
) -> dict[str, object]:
    return {
        "name": name,
        "file": file,
        "part": part,
        "indices": np.asarray(indices, dtype=np.int64),
        "scores": np.asarray(scores, dtype=np.float64),
        "feature_errors": np.asarray(feature_errors, dtype=np.float64),
        "start_time": np.asarray(times[:, 0], dtype=np.float64),
        "end_time": np.asarray(times[:, 1], dtype=np.float64),
    }


def score_all_groups(
    data: dict[str, object],
    suffix: str,
    model: tf.keras.Model,
) -> list[dict[str, object]]:
    """Score healthy train/val/test parts and all seven fault files."""
    healthy_stem = HEALTHY_STEMS[suffix]
    splits = data["splits"][suffix]
    groups: list[dict[str, object]] = []

    healthy_windows = np.asarray(data["windows"][healthy_stem])
    plain, feature_errors = score_windows(model, healthy_windows)
    for part in ("train", "val", "test"):
        indices = np.asarray(splits[part], dtype=np.int64)
        groups.append(
            make_group(
                f"{healthy_stem}_{part}",
                healthy_stem,
                part,
                indices,
                plain[indices],
                feature_errors[indices],
                data["times"][healthy_stem][indices],
            )
        )

    for fault in FAULT_STEMS_BY_MODE[suffix]:
        fault_windows = np.asarray(data["windows"][fault])
        plain, feature_errors = score_windows(model, fault_windows)
        indices = np.arange(fault_windows.shape[0], dtype=np.int64)
        groups.append(
            make_group(
                fault,
                fault,
                "whole_file",
                indices,
                plain,
                feature_errors,
                data["times"][fault],
            )
        )

    return groups


def check_batch_vs_single(
    groups: list[dict[str, object]],
    data: dict[str, object],
    model: tf.keras.Model,
) -> dict[str, object]:
    """Compare batch scores with one-window-at-a-time scoring on samples."""
    max_plain_diff = 0.0
    max_feature_diff = 0.0
    problems: list[str] = []
    comparisons = 0

    for group in groups:
        source = np.asarray(data["windows"][group["file"]])
        indices = np.asarray(group["indices"], dtype=np.int64)
        count = int(indices.size)
        samples = sorted({0, count // 2, count - 1})
        for local_index in samples:
            source_index = int(indices[local_index])
            single_plain, single_feature = score_windows(
                model,
                source[source_index : source_index + 1],
                batch_size=1,
            )
            batch_plain = float(group["scores"][local_index])
            batch_feature = np.asarray(
                group["feature_errors"][local_index], dtype=np.float64
            )
            single_plain_value = float(single_plain[0])
            single_feature_value = np.asarray(single_feature[0], dtype=np.float64)

            plain_scale = max(abs(batch_plain), abs(single_plain_value), 1e-12)
            feature_scale = np.maximum(
                np.maximum(np.abs(batch_feature), np.abs(single_feature_value)),
                1e-12,
            )
            max_plain_diff = max(
                max_plain_diff,
                abs(batch_plain - single_plain_value) / plain_scale,
            )
            max_feature_diff = max(
                max_feature_diff,
                float(
                    np.max(
                        np.abs(batch_feature - single_feature_value)
                        / feature_scale
                    )
                ),
            )
            comparisons += 1

            if not np.isclose(
                batch_plain,
                single_plain_value,
                rtol=2e-5,
                atol=1e-9,
            ):
                problems.append(
                    f"{group['name']}[{local_index}] plain "
                    f"batch={batch_plain:.12g} single={single_plain_value:.12g}"
                )
            if not np.allclose(
                batch_feature,
                single_feature_value,
                rtol=2e-5,
                atol=1e-9,
            ):
                problems.append(
                    f"{group['name']}[{local_index}] feature errors differ"
                )

    return {
        "comparisons": comparisons,
        "max_plain_relative_difference": float(max_plain_diff),
        "max_feature_relative_difference": float(max_feature_diff),
        "problems": problems,
    }


def save_archive(
    path: Path,
    scores: np.ndarray,
    start_time: np.ndarray,
    end_time: np.ndarray,
    threshold: float,
    names: np.ndarray,
    feature_errors: np.ndarray,
) -> None:
    """Save the five baseline keys plus the (n_windows, 13) feature errors."""
    np.savez_compressed(
        path,
        scores=np.asarray(scores, dtype=np.float64),
        start_time=np.asarray(start_time, dtype=np.float64),
        end_time=np.asarray(end_time, dtype=np.float64),
        threshold=np.float64(threshold),
        name=np.asarray(names, dtype="U32"),
        feature_errors=np.asarray(feature_errors, dtype=np.float64),
    )


def score_and_save_run(
    data: dict[str, object],
    suffix: str,
    seed: int,
    model: tf.keras.Model,
    run_meta: dict[str, object],
) -> dict[str, object]:
    """Compute both scores, set validation thresholds, and save archives."""
    mode = MODES[suffix]
    healthy_stem = HEALTHY_STEMS[suffix]
    paths = model_paths(mode, seed)
    reference = run_meta["feature_error_reference"]
    reference_mean = np.asarray(reference["mean"], dtype=np.float64)
    reference_std = np.asarray(reference["std"], dtype=np.float64)

    scoring_start = time.perf_counter()
    groups = score_all_groups(data, suffix, model)
    batch_check = check_batch_vs_single(groups, data, model)

    val_group = next(
        group for group in groups if group["name"] == f"{healthy_stem}_val"
    )
    val_plain = np.asarray(val_group["scores"], dtype=np.float64)
    val_feature_errors = np.asarray(
        val_group["feature_errors"], dtype=np.float64
    )
    val_featnorm = featnorm_scores(
        val_feature_errors,
        reference_mean,
        reference_std,
    )
    thresholds = {
        "plain": float(np.quantile(val_plain, THRESHOLD_QUANTILE)),
        "featnorm": float(np.quantile(val_featnorm, THRESHOLD_QUANTILE)),
    }

    # The stored normalization statistics must be exactly what the saved model
    # produces on the healthy validation windows. This is a source check, not a
    # tuning step.
    recomputed_mean = val_feature_errors.mean(axis=0)
    recomputed_std = np.maximum(
        val_feature_errors.std(axis=0, ddof=0),
        STD_FLOOR,
    )
    reference_ok = bool(
        np.allclose(reference_mean, recomputed_mean, rtol=1e-12, atol=1e-14)
        and np.allclose(reference_std, recomputed_std, rtol=1e-12, atol=1e-14)
    )
    reference_max_diff = float(
        max(
            np.max(np.abs(reference_mean - recomputed_mean)),
            np.max(np.abs(reference_std - recomputed_std)),
        )
    )

    feature_errors_all = np.concatenate(
        [np.asarray(group["feature_errors"]) for group in groups], axis=0
    )
    start_time_all = np.concatenate(
        [np.asarray(group["start_time"]) for group in groups]
    )
    end_time_all = np.concatenate(
        [np.asarray(group["end_time"]) for group in groups]
    )
    names_all = np.concatenate(
        [
            np.full(
                np.asarray(group["scores"]).size,
                str(group["name"]),
                dtype="U32",
            )
            for group in groups
        ]
    )

    archive_paths: dict[str, str] = {}
    score_minmax: dict[str, dict[str, float]] = {}
    for score_kind in SCORE_KINDS:
        if score_kind == "plain":
            scores = np.concatenate(
                [np.asarray(group["scores"]) for group in groups]
            )
        else:
            scores = np.concatenate(
                [
                    featnorm_scores(
                        np.asarray(group["feature_errors"]),
                        reference_mean,
                        reference_std,
                    )
                    for group in groups
                ]
            )

        if scores.shape != names_all.shape:
            raise RuntimeError(
                f"{mode}/{seed}/{score_kind}: score/name length mismatch"
            )
        if not np.isfinite(scores).all():
            raise RuntimeError(f"{mode}/{seed}/{score_kind}: non-finite scores")

        path = paths[score_kind]
        save_archive(
            path,
            scores,
            start_time_all,
            end_time_all,
            thresholds[score_kind],
            names_all,
            feature_errors_all,
        )
        archive_paths[score_kind] = str(path.relative_to(PROJECT_ROOT))
        score_minmax[score_kind] = {
            "min": float(scores.min()),
            "max": float(scores.max()),
        }

    scoring_time = time.perf_counter() - scoring_start
    log_line(
        f"[{mode}][seed={seed}] SCORED windows={names_all.size} "
        f"plain_threshold={thresholds['plain']:.6g} "
        f"featnorm_threshold={thresholds['featnorm']:.6g} "
        f"time_s={scoring_time:.1f} "
        f"batch_vs_single={'PASS' if not batch_check['problems'] else 'FAIL'}"
    )

    return {
        "mode": mode,
        "mode_suffix": suffix,
        "seed": seed,
        "healthy_file": healthy_stem,
        "windows": int(names_all.size),
        "thresholds": thresholds,
        "score_minmax": score_minmax,
        "scoring_time_s": float(scoring_time),
        "batch_vs_single": batch_check,
        "reference_recomputed_ok": reference_ok,
        "reference_max_abs_difference": reference_max_diff,
        "archives": archive_paths,
        "feature_error_reference": {
            "mean": reference_mean.tolist(),
            "std": reference_std.tolist(),
        },
        "training_time_s": float(run_meta["training_time_s"]),
        "train_windows": int(run_meta["train_windows"]),
        "val_windows": int(run_meta["val_windows"]),
        "max_epochs": int(run_meta["max_epochs"]),
        "epochs_run": int(run_meta["epochs_run"]),
        "best_epoch": int(run_meta["best_epoch"]),
        "best_val_loss": float(run_meta["best_val_loss"]),
        "final_val_loss": float(run_meta["final_val_loss"]),
        "initial_val_loss": float(run_meta["initial_val_loss"]),
        "stopped_early": bool(run_meta["stopped_early"]),
        "improved_over_initial": bool(run_meta["improved_over_initial"]),
        "model_params": int(run_meta["model_params"]),
        "model_path": str(run_meta["model_path"]),
        "history_path": str(run_meta["history_path"]),
        "figure_path": str(run_meta["figure_path"]),
    }


def expected_group_counts(
    data: dict[str, object], suffix: str
) -> dict[str, int]:
    healthy_stem = HEALTHY_STEMS[suffix]
    counts = {
        f"{healthy_stem}_{part}": int(
            np.asarray(data["splits"][suffix][part]).size
        )
        for part in ("train", "val", "test")
    }
    for fault in FAULT_STEMS_BY_MODE[suffix]:
        counts[fault] = int(data["windows"][fault].shape[0])
    return counts


def build_training_frame(records: list[dict[str, object]]) -> pd.DataFrame:
    """One row per mode/seed with epochs run and best validation loss."""
    rows = []
    for record in records:
        rows.append(
            {
                "mode": record["mode"],
                "seed": int(record["seed"]),
                "epochs_run": int(record["epochs_run"]),
                "max_epochs": int(record["max_epochs"]),
                "best_epoch": int(record["best_epoch"]),
                "best_val_loss": float(record["best_val_loss"]),
                "final_val_loss": float(record["final_val_loss"]),
                "initial_val_loss": float(record["initial_val_loss"]),
                "stopped_early": bool(record["stopped_early"]),
                "improved_over_initial": bool(
                    record["improved_over_initial"]
                ),
                "train_windows": int(record["train_windows"]),
                "val_windows": int(record["val_windows"]),
                "model_params": int(record["model_params"]),
                "training_time_s": float(record["training_time_s"]),
                "scoring_time_s": float(record["scoring_time_s"]),
                "device": DEVICE,
                "model_path": str(record["model_path"]),
            }
        )
    return pd.DataFrame(rows)


def safe_median(values: np.ndarray) -> float:
    return float(np.median(values)) if values.size > 0 else float("nan")


def share(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def build_sanity_frame(records: list[dict[str, object]]) -> pd.DataFrame:
    """Healthy flag rates and per-fault whole-file/pre/post sanity numbers."""
    rows: list[dict[str, object]] = []
    for record in records:
        mode = str(record["mode"])
        seed = int(record["seed"])
        healthy_stem = str(record["healthy_file"])
        for score_kind in SCORE_KINDS:
            path = PROJECT_ROOT / str(record["archives"][score_kind])
            with np.load(path, allow_pickle=False) as archive:
                scores = np.asarray(archive["scores"], dtype=np.float64)
                start_time = np.asarray(archive["start_time"], dtype=np.float64)
                end_time = np.asarray(archive["end_time"], dtype=np.float64)
                threshold = float(np.asarray(archive["threshold"]))
                names = np.asarray(archive["name"]).astype(str)

            base = {
                "model": MODEL_KEYS[score_kind],
                "model_label": MODEL_LABELS[score_kind],
                "mode": mode,
                "seed": seed,
                "threshold": threshold,
                "input_kind": "window_sequences",
                "train_units": int(record["train_windows"]),
                "subsample_applied": False,
                "training_time_s": float(record["training_time_s"]),
                "scoring_time_s": float(record["scoring_time_s"]),
            }

            for part in ("val", "test"):
                mask = names == f"{healthy_stem}_{part}"
                group_scores = scores[mask]
                flagged = int(np.sum(group_scores > threshold))
                rows.append(
                    {
                        **base,
                        "section": "healthy",
                        "file": healthy_stem,
                        "part": part,
                        "fault": "",
                        "windows": int(group_scores.size),
                        "flagged": flagged,
                        "flag_rate": share(
                            flagged, int(group_scores.size)
                        ),
                        "median_score": safe_median(group_scores),
                        "pre_injection_windows": np.nan,
                        "median_pre_injection_score": np.nan,
                        "post_injection_windows": np.nan,
                        "median_after_8s_score": np.nan,
                        "median_ratio_after8_over_pre": np.nan,
                        "no_post_injection_data": False,
                    }
                )

            for fault in FAULT_STEMS_BY_MODE[str(record["mode_suffix"])]:
                mask = names == fault
                group_scores = scores[mask]
                group_start = start_time[mask]
                group_end = end_time[mask]
                flagged = int(np.sum(group_scores > threshold))

                pre_mask = (group_end < FAULT_START_S) & (
                    group_start >= STARTUP_END_S
                )
                post_mask = group_start > FAULT_END_S
                median_pre = safe_median(group_scores[pre_mask])
                median_post = safe_median(group_scores[post_mask])
                ratio = (
                    float(median_post / median_pre)
                    if np.isfinite(median_pre)
                    and np.isfinite(median_post)
                    and median_pre > 0.0
                    else float("nan")
                )
                no_post = int(post_mask.sum()) == 0

                rows.append(
                    {
                        **base,
                        "section": "fault",
                        "file": fault,
                        "part": "whole_file",
                        "fault": fault[:2],
                        "windows": int(group_scores.size),
                        "flagged": flagged,
                        "flag_rate": share(
                            flagged, int(group_scores.size)
                        ),
                        "median_score": safe_median(group_scores),
                        "pre_injection_windows": int(pre_mask.sum()),
                        "median_pre_injection_score": median_pre,
                        "post_injection_windows": int(post_mask.sum()),
                        "median_after_8s_score": median_post,
                        "median_ratio_after8_over_pre": ratio,
                        "no_post_injection_data": bool(no_post),
                    }
                )

    frame = pd.DataFrame(rows)
    column_order = [
        "model",
        "model_label",
        "mode",
        "seed",
        "section",
        "file",
        "part",
        "fault",
        "windows",
        "flagged",
        "flag_rate",
        "median_score",
        "pre_injection_windows",
        "median_pre_injection_score",
        "post_injection_windows",
        "median_after_8s_score",
        "median_ratio_after8_over_pre",
        "no_post_injection_data",
        "threshold",
        "input_kind",
        "train_units",
        "subsample_applied",
        "training_time_s",
        "scoring_time_s",
    ]
    return frame[column_order]


def validate_archives(
    data: dict[str, object], records: list[dict[str, object]]
) -> None:
    """Check keys, finiteness, group counts, thresholds, and score formulas."""
    keys_ok = True
    lengths_ok = True
    finite_ok = True
    feature_ok = True
    groups_ok = True
    thresholds_ok = True
    plain_formula_ok = True
    featnorm_formula_ok = True
    total_windows = 0

    for record in records:
        suffix = str(record["mode_suffix"])
        seed = int(record["seed"])
        healthy_stem = str(record["healthy_file"])
        expected_counts = expected_group_counts(data, suffix)
        reference = record["feature_error_reference"]
        reference_mean = np.asarray(reference["mean"], dtype=np.float64)
        reference_std = np.asarray(reference["std"], dtype=np.float64)

        for score_kind in SCORE_KINDS:
            path = PROJECT_ROOT / str(record["archives"][score_kind])
            with np.load(path, allow_pickle=False) as archive:
                keys = set(archive.files)
                scores = np.asarray(archive["scores"], dtype=np.float64)
                start_time = np.asarray(
                    archive["start_time"], dtype=np.float64
                )
                end_time = np.asarray(archive["end_time"], dtype=np.float64)
                threshold = float(np.asarray(archive["threshold"]))
                names = np.asarray(archive["name"]).astype(str)
                feature_errors = np.asarray(
                    archive["feature_errors"], dtype=np.float64
                )

            keys_ok = keys_ok and keys == REQUIRED_KEYS
            lengths_ok = lengths_ok and (
                scores.shape
                == start_time.shape
                == end_time.shape
                == names.shape
                == (feature_errors.shape[0],)
            )
            finite_ok = finite_ok and bool(
                np.isfinite(scores).all()
                and np.isfinite(start_time).all()
                and np.isfinite(end_time).all()
                and np.isfinite(threshold)
            )
            feature_ok = feature_ok and bool(
                feature_errors.shape == (scores.size, len(FEATURES))
                and np.isfinite(feature_errors).all()
                and (feature_errors >= 0.0).all()
            )
            observed_counts = {
                name: int(np.sum(names == name)) for name in np.unique(names)
            }
            groups_ok = groups_ok and observed_counts == expected_counts

            val_mask = names == f"{healthy_stem}_val"
            val_scores = scores[val_mask]
            recomputed = float(np.quantile(val_scores, THRESHOLD_QUANTILE))
            thresholds_ok = thresholds_ok and bool(
                np.isclose(threshold, recomputed, rtol=1e-12, atol=1e-12)
            )

            if score_kind == "plain":
                plain_formula_ok = plain_formula_ok and bool(
                    np.allclose(
                        scores,
                        feature_errors.mean(axis=1),
                        rtol=1e-12,
                        atol=1e-12,
                    )
                )
            else:
                recomputed_scores = featnorm_scores(
                    feature_errors,
                    reference_mean,
                    reference_std,
                )
                featnorm_formula_ok = featnorm_formula_ok and bool(
                    np.allclose(
                        scores,
                        recomputed_scores,
                        rtol=1e-12,
                        atol=1e-12,
                    )
                )

            total_windows += int(scores.size)

    check(
        "every proposed archive has exactly the five baseline keys plus feature_errors",
        keys_ok,
        "keys = scores, start_time, end_time, threshold, name, feature_errors",
    )
    check(
        "score/time/name/feature-error arrays have matching lengths",
        lengths_ok,
        f"{len(records) * len(SCORE_KINDS)} archives checked",
    )
    check(
        "all scores, times, thresholds, and feature errors are finite",
        finite_ok,
        f"{total_windows:,} window scores across "
        f"{len(records) * len(SCORE_KINDS)} archives",
    )
    check(
        "feature_errors has shape (n_windows, 13) and is non-negative",
        feature_ok,
        f"{len(FEATURES)} features per window",
    )
    check(
        "every archive has the expected nine groups and window counts",
        groups_ok,
        "healthy train/val/test and seven fault files per mode",
    )
    check(
        "every threshold equals the saved healthy-validation p99",
        thresholds_ok,
        "no healthy test or fault window used for a threshold",
    )
    check(
        "plain scores equal the mean of the saved feature errors",
        plain_formula_ok,
        "mean squared error over time and 13 features",
    )
    check(
        "featnorm scores equal the fixed k=3 formula from saved feature errors",
        featnorm_formula_ok,
        "validation-only mean/std, negative z clipped, top-3 mean",
    )


def validate_training_source(
    data: dict[str, object], records: list[dict[str, object]]
) -> None:
    """Assert that every training/validation array is a healthy F0 slice."""
    source_ok = True
    no_fault_ok = True
    disjoint_ok = True
    details: list[str] = []

    for record in records:
        suffix = str(record["mode_suffix"])
        healthy_stem = HEALTHY_STEMS[suffix]
        paths = model_paths(str(record["mode"]), int(record["seed"]))
        run_meta = read_json(paths["run_meta"])
        arrays = healthy_training_arrays(data, suffix)

        source_ok = source_ok and bool(
            run_meta["train_source_file"] == healthy_stem
            and run_meta["val_source_file"] == healthy_stem
            and run_meta["train_sha256"] == arrays["train_sha256"]
            and run_meta["val_sha256"] == arrays["val_sha256"]
        )
        no_fault_ok = no_fault_ok and bool(
            str(run_meta["train_source_file"]).startswith("F0")
            and str(run_meta["val_source_file"]).startswith("F0")
        )
        disjoint_ok = disjoint_ok and bool(
            np.intersect1d(arrays["train_indices"], arrays["val_indices"]).size
            == 0
        )
        details.append(
            f"{record['mode']}/seed{record['seed']} "
            f"{run_meta['train_source_file']}"
        )

    check(
        "training and validation arrays match the frozen healthy splits exactly",
        source_ok,
        ", ".join(sorted(set(details))) + " (SHA-256 verified)",
    )
    check(
        "no fault file contributed any training or validation window",
        no_fault_ok,
        "all source files are F0L/F0M",
    )
    check(
        "healthy train and validation window indices are disjoint",
        disjoint_ok,
        "train/validation intersection is empty",
    )


def validate_reference_stats(records: list[dict[str, object]]) -> None:
    """Check that normalization statistics come from the healthy validation."""
    all_ok = True
    max_diff = 0.0
    for record in records:
        all_ok = all_ok and bool(record["reference_recomputed_ok"])
        max_diff = max(
            max_diff,
            float(record["reference_max_abs_difference"]),
        )
    check(
        "feature-error mean/std match the healthy validation windows exactly",
        all_ok,
        f"recomputed from the saved model for all runs; "
        f"max abs difference={max_diff:.3g}",
    )


def validate_batch_checks(records: list[dict[str, object]]) -> None:
    problems: list[str] = []
    comparisons = 0
    max_plain = 0.0
    max_feature = 0.0
    for record in records:
        batch = record["batch_vs_single"]
        comparisons += int(batch["comparisons"])
        max_plain = max(
            max_plain, float(batch["max_plain_relative_difference"])
        )
        max_feature = max(
            max_feature, float(batch["max_feature_relative_difference"])
        )
        problems.extend(str(item) for item in batch["problems"])
    check(
        "batch scoring equals one-window-at-a-time scoring on sample windows",
        not problems,
        f"{comparisons} windows compared across all runs; "
        f"max relative diff plain={max_plain:.3g}, "
        f"feature={max_feature:.3g}",
    )
    if problems:
        raise RuntimeError(
            "Batch/single scoring mismatch: " + "; ".join(problems[:5])
        )


def validate_sanity_frame(frame: pd.DataFrame, n_runs: int) -> None:
    expected_rows = n_runs * len(SCORE_KINDS) * (2 + 7)
    check(
        "sanity table has one row per score/run/group",
        frame.shape[0] == expected_rows,
        f"{frame.shape[0]} rows, expected {expected_rows} "
        f"for {n_runs} runs x {len(SCORE_KINDS)} scores x 9 groups",
    )

    rate_ok = True
    for row in frame.itertuples():
        expected = (
            float(row.flagged / row.windows) if row.windows > 0 else float("nan")
        )
        if not np.isclose(float(row.flag_rate), expected, rtol=0.0, atol=1e-12):
            rate_ok = False
    check(
        "every flag rate equals flagged/windows",
        rate_ok,
        "healthy and fault rows",
    )

    f3m = frame[frame["file"] == "F3M"]
    f3m_ok = bool(
        (f3m["post_injection_windows"] == 0).all()
        and f3m["no_post_injection_data"].all()
        and f3m["median_after_8s_score"].isna().all()
    )
    check(
        "F3M is marked as having no post-injection data",
        f3m_ok,
        f"{f3m.shape[0]} score/seed rows; post windows = 0",
    )
    only_f3m = bool(
        set(frame.loc[frame["no_post_injection_data"], "file"].unique())
        <= {"F3M"}
    )
    check(
        "no file other than F3M is marked as lacking post-injection data",
        only_f3m,
        "all other fault files have post-8 s windows",
    )


def validate_training_frame(frame: pd.DataFrame, n_runs: int) -> None:
    check(
        "training table has one row per completed mode/seed run",
        frame.shape[0] == n_runs,
        f"{frame.shape[0]} rows, expected {n_runs}",
    )
    epochs_ok = bool(
        (frame["epochs_run"] >= 1).all()
        and (frame["best_epoch"] >= 1).all()
        and (frame["best_epoch"] <= frame["epochs_run"]).all()
        and (frame["epochs_run"] <= frame["max_epochs"]).all()
    )
    check(
        "epochs run and best epoch are internally consistent",
        epochs_ok,
        "1 <= best_epoch <= epochs_run <= max_epochs",
    )
    loss_ok = bool(
        np.isfinite(frame["best_val_loss"]).all()
        and np.isfinite(frame["final_val_loss"]).all()
    )
    check(
        "best and final validation losses are finite",
        loss_ok,
        "no NaN or infinite val_loss",
    )


def build_metadata(
    records: list[dict[str, object]],
    training: pd.DataFrame,
    sanity: pd.DataFrame,
) -> dict[str, object]:
    return {
        "phase": "phase_4_proposed_tcn_bilstm",
        "generated_utc": now_iso(),
        "git_commit": git_commit(),
        "environment": environment_info(),
        "tensorflow": tf.__version__,
        "keras": tf.keras.__version__,
        "device": DEVICE,
        "gpus_found": [
            device.name for device in tf.config.list_physical_devices("GPU")
        ],
        "seeds": list(SEEDS),
        "modes": [mode for _, mode in MODE_ORDER],
        "architecture": (
            "copied unchanged from Untitled0.ipynb: TCN blocks 32/32/64 "
            "dilations 1/2/4, BiLSTM 64, latent BiLSTM 32, RepeatVector, "
            "BiLSTM 32 and 64, TCN 64/32, TimeDistributed Dense"
        ),
        "training": {
            "max_epochs": MAX_EPOCHS,
            "batch_size": BATCH_SIZE,
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "early_stopping": {
                "monitor": "val_loss",
                "patience": EARLY_STOPPING_PATIENCE,
                "restore_best_weights": True,
            },
            "reduce_lr": {
                "monitor": "val_loss",
                "factor": REDUCE_LR_FACTOR,
                "patience": REDUCE_LR_PATIENCE,
                "min_lr": MIN_LR,
            },
            "training_data": (
                "healthy training windows only (F0L for IPPT, F0M for MPPT)"
            ),
            "validation_data": (
                "healthy validation windows only (F0L for IPPT, F0M for MPPT)"
            ),
            "fault_files_used": "none",
            "determinism": {
                "tf_enable_onednn_opts": os.environ.get(
                    "TF_ENABLE_ONEDNN_OPTS"
                ),
                "op_determinism": True,
                "seed_function": "tf.keras.utils.set_random_seed",
            },
        },
        "scoring": {
            "plain": "mean squared error over time and 13 features",
            "featnorm": (
                "per-feature MSE over time; z = (e - val_mean) / "
                "max(val_std, 1e-8); negative z clipped to 0; "
                "window score = mean of the 3 largest z values"
            ),
            "top_k": TOP_K,
            "top_k_note": "fixed before any fault scoring and never tuned",
            "std_floor": STD_FLOOR,
            "feature_reference": "healthy validation windows only",
            "threshold_rule": (
                "99th percentile of each score on healthy validation windows"
            ),
            "threshold_quantile": THRESHOLD_QUANTILE,
            "flag_rule": "score > threshold",
            "scoring_batch_size": SCORING_BATCH_SIZE,
        },
        "runs": records,
        "training_csv": str(TRAINING_PATH.relative_to(PROJECT_ROOT)),
        "sanity_csv": str(SANITY_PATH.relative_to(PROJECT_ROOT)),
        "training_rows": int(training.shape[0]),
        "sanity_rows": int(sanity.shape[0]),
        "checks_passed": sum(bool(result["passed"]) for result in CHECK_RESULTS),
        "checks_total": len(CHECK_RESULTS),
        "checks": CHECK_RESULTS,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=MAX_EPOCHS,
        help="Maximum epochs per run (default: 100).",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["MPPT", "IPPT"],
        default=[mode for _, mode in MODE_ORDER],
        help="Operating modes to run (default: MPPT IPPT).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(SEEDS),
        help="Seeds to run (default: 0 1 2).",
    )
    return parser.parse_args()


def main() -> int:
    global DEVICE
    args = parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("GPVS-Faults | Phase 4 proposed TCN-BiLSTM autoencoder")
    print("=" * 80)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Max epochs   : {args.max_epochs}")
    print(f"Modes        : {args.modes}")
    print(f"Seeds        : {args.seeds}")

    DEVICE = configure_device()
    tf.config.experimental.enable_op_determinism()
    log_line(
        f"RUN START max_epochs={args.max_epochs} modes={args.modes} "
        f"seeds={args.seeds} device={DEVICE} tf={tf.__version__}"
    )

    data = build_dataset(verbose=False)
    print("Dataset built from the frozen Phase 2 pipeline.")

    # Pre-training assertions: the arrays used for fitting are healthy only.
    for suffix, mode in MODE_ORDER:
        arrays = healthy_training_arrays(data, suffix)
        check(
            f"{mode} training arrays are exactly healthy train windows",
            bool(
                arrays["source_file"] == HEALTHY_STEMS[suffix]
                and arrays["train_x"].shape[0] == len(arrays["train_indices"])
                and arrays["val_x"].shape[0] == len(arrays["val_indices"])
            ),
            f"{arrays['source_file']}: {arrays['train_x'].shape[0]} train, "
            f"{arrays['val_x'].shape[0]} val windows",
        )

    records: list[dict[str, object]] = []
    failures: list[str] = []
    for suffix, mode in MODE_ORDER:
        if mode not in args.modes:
            continue
        for seed in args.seeds:
            paths = model_paths(mode, seed)
            try:
                if paths["model"].exists():
                    log_line(
                        f"[{mode}][seed={seed}] SKIP training "
                        f"(model exists: {paths['model'].name})"
                    )
                    model, run_meta = load_run(suffix, seed)
                else:
                    run_meta = train_run(
                        data,
                        suffix,
                        seed,
                        int(args.max_epochs),
                    )
                    model, loaded_meta = load_run(suffix, seed)
                    if int(loaded_meta["seed"]) != seed:
                        raise RuntimeError("Loaded model seed mismatch")
                record = score_and_save_run(
                    data,
                    suffix,
                    seed,
                    model,
                    run_meta,
                )
                records.append(record)
            except Exception as exc:  # keep going so other runs are not lost
                message = f"{mode}/seed{seed}: {type(exc).__name__}: {exc}"
                failures.append(message)
                log_line(f"[{mode}][seed={seed}] FAILED {message}")
                print(f"[FAIL] {message}", flush=True)

    if failures:
        check(
            "every mode/seed run completed",
            False,
            "; ".join(failures),
        )
    else:
        check(
            "every mode/seed run completed",
            True,
            f"{len(records)} runs",
        )

    if not records:
        raise RuntimeError("No proposed-model run completed")

    print("\nArchive checks:")
    validate_archives(data, records)
    print("\nTraining-source checks:")
    validate_training_source(data, records)
    print("\nFeature-normalization source checks:")
    validate_reference_stats(records)
    print("\nBatch vs single-window scoring checks:")
    validate_batch_checks(records)

    print("\nBuilding training and sanity tables from saved archives:")
    training = build_training_frame(records)
    sanity = build_sanity_frame(records)
    validate_training_frame(training, len(records))
    validate_sanity_frame(sanity, len(records))
    training.to_csv(TRAINING_PATH, index=False)
    sanity.to_csv(SANITY_PATH, index=False)
    print(f"  saved {TRAINING_PATH}")
    print(f"  saved {SANITY_PATH}")

    metadata = build_metadata(records, training, sanity)
    write_json(METADATA_PATH, metadata)
    print(f"  saved {METADATA_PATH}")

    print("\nTraining summary:")
    print(
        training[
            [
                "mode",
                "seed",
                "epochs_run",
                "best_epoch",
                "best_val_loss",
                "final_val_loss",
                "stopped_early",
                "improved_over_initial",
                "training_time_s",
            ]
        ].to_string(index=False)
    )

    print("\nHealthy flag rates:")
    healthy = sanity[sanity["section"] == "healthy"]
    print(
        healthy[
            [
                "model_label",
                "mode",
                "seed",
                "part",
                "windows",
                "flagged",
                "flag_rate",
            ]
        ].to_string(index=False)
    )

    all_passed = all(bool(result["passed"]) for result in CHECK_RESULTS)
    print(
        f"\nRESULT: {'PASS' if all_passed else 'FAIL'} - "
        f"{sum(bool(result['passed']) for result in CHECK_RESULTS)}"
        f"/{len(CHECK_RESULTS)} checks passed."
    )
    log_line(
        f"RUN END checks={sum(bool(r['passed']) for r in CHECK_RESULTS)}"
        f"/{len(CHECK_RESULTS)} status={'PASS' if all_passed else 'FAIL'}"
    )
    if not all_passed:
        failed = [
            str(result["name"])
            for result in CHECK_RESULTS
            if not result["passed"]
        ]
        raise AssertionError(f"Phase 4 check(s) failed: {failed}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

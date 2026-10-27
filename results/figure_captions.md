# Figure captions

**fig_healthy_far.png.** Healthy-test false-alarm rate for all seven models at the frozen 99th-percentile healthy-validation threshold in IPPT and MPPT, with the 1% validation target marked by a dashed line (sources: `results/average_primary_IPPT.csv`, `results/average_primary_MPPT.csv`).

**fig_primary_f1_auc.png.** Average whole-file F1 and AUC over the fault files of each mode with 95% bootstrap intervals, one panel per mode (sources: `results/average_primary_IPPT.csv`, `results/average_primary_MPPT.csv`).

**fig_within_run_auc.png.** Average within-run AUC per model with 95% bootstrap intervals and a 0.5 chance reference line, one panel per mode, where MPPT excludes F3M because it has no post-injection windows (sources: `results/within_run_average_IPPT.csv`, `results/within_run_average_MPPT.csv`).

**fig_score_vs_time.png.** Window score of Proposed (plain), seed 0, against window start time for F1, F4, F6 and F7 in IPPT on a logarithmic y-axis, with the stored threshold and vertical lines at 7 s and 8 s (source: `results/scores_proposed_plain_IPPT_seed0.npz`).

**fig_score_hist.png.** Histograms of healthy validation, healthy test and pooled fault-file window scores for Proposed (plain), seed 0, on a logarithmic x-axis, one panel per mode, with the stored threshold marked (sources: `results/scores_proposed_plain_IPPT_seed0.npz`, `results/scores_proposed_plain_MPPT_seed0.npz`).

**fig_threshold_sensitivity.png.** Healthy-test false-alarm rate versus threshold percentile (95, 97.5, 99 and 99.5) for Proposed (plain), Proposed (featnorm), EllipticEnvelope and Mahalanobis (window-mean), one panel per mode (source: `results/threshold_sensitivity.csv`).

**fig_training_curves.png.** Training and validation loss by epoch for the six proposed runs (two modes and three seeds) (sources: `results/history_IPPT_seed0.json` through `results/history_MPPT_seed2.json`).

**fig_feature_error_heatmap.png.** Mean per-feature reconstruction error of Proposed (plain), seed 0, for healthy test and each fault file in each mode on a logarithmic color scale (sources: `results/scores_proposed_plain_IPPT_seed0.npz`, `results/scores_proposed_plain_MPPT_seed0.npz`).

**fig_architecture.png.** Block diagram of the TCN-BiLSTM autoencoder implemented in `src/model.py`, showing the encoder, latent representation, decoder and reconstruction layers.


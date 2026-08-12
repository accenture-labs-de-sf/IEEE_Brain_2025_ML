"""Preprocessing to match EEGPT's expected input.

Rationale: keep inputs as close as possible to what EEGPT was pretrained with, so the model
sees in-distribution data (PhysioNet MI is itself one of EEGPT's *pretraining* datasets).

Two deliberately separated stages:

* **Raw-level (shared)** — average reference, ~0–38 Hz band-pass (low-pass), resample to
  **250 Hz**. Applied via the ``physionet_mi`` loaders using ``EEGPT_RAW_KWARGS``.
* **Normalization (model input ONLY)** — per-window z-score (``normalize_epochs``).

⚠️ Do **not** z-score before ERD / band-power analyses: z-scoring removes the amplitude
information that ERD is measured from. Normalization is for *feeding the model*, not for the
empirical neurophysiology reference.

**Sampling-rate correction:** the braindecode EEGPT checkpoint expects **250 Hz** (not the
256 Hz stated in the paper), verified from its `config.json`. Model *input formatting* (the
fixed 62-channel montage, 4 s / 1000-sample windows, channel interpolation) lives in
``eegxai.models.eegpt``.
"""

from __future__ import annotations

import numpy as np

# ── EEGPT input contract (verified from the braindecode checkpoint config) ───
EEGPT_SFREQ = 250.0          # resample target (checkpoint rate; our data 160 Hz -> upsample)
EEGPT_N_TIMES = 1000         # samples per window (= 4 s at 250 Hz)
EEGPT_WINDOW_SECONDS = 4.0   # model input window length
EEGPT_LOW_HZ: float | None = None   # no explicit high-pass ("0-38")
EEGPT_HIGH_HZ = 38.0         # low-pass
EEGPT_REFERENCE = "average"  # global average reference

# kwargs to reproduce EEGPT's raw-level preprocessing through the physionet_mi loaders:
#   load_subject_raw(subj, runs, **EEGPT_RAW_KWARGS)
EEGPT_RAW_KWARGS = dict(
    reference=EEGPT_REFERENCE,
    l_freq=EEGPT_LOW_HZ,
    h_freq=EEGPT_HIGH_HZ,
    resample_sfreq=EEGPT_SFREQ,
)


def normalize_epochs(X: np.ndarray, *, axis: int = -1, eps: float = 1e-7) -> np.ndarray:
    """Per-window z-score (default: per epoch, per channel, over time).

    For **model input only** — never before ERD/band-power analysis. EEGPT z-scores per
    token; per-epoch-per-channel is a close standing-in until we mirror the exact
    braindecode transform.
    """
    mean = X.mean(axis=axis, keepdims=True)
    std = X.std(axis=axis, keepdims=True)
    return (X - mean) / (std + eps)

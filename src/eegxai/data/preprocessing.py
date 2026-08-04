"""Preprocessing to match EEGPT's expected input.

Rationale: keep inputs as close as possible to what EEGPT was pretrained with, so the model
sees in-distribution data (PhysioNet MI is itself one of EEGPT's *pretraining* datasets).

Two deliberately separated stages:

* **Raw-level (shared)** — average reference, ~0–38 Hz band-pass (low-pass), resample to
  256 Hz. Applied via the ``physionet_mi`` loaders using ``EEGPT_RAW_KWARGS``.
* **Normalization (model input ONLY)** — per-window z-score (``normalize_epochs``).

⚠️ Do **not** z-score before ERD / band-power analyses: z-scoring removes the amplitude
information that ERD is measured from. Normalization is for *feeding the model*, not for the
empirical neurophysiology reference.

Verify against braindecode's EEGPT transform when wiring up the model — exact tokenization /
scaling live there. Values below reflect the EEGPT paper's described recipe.
"""

from __future__ import annotations

import numpy as np

# ── EEGPT recipe (paper) ─────────────────────────────────────────────────────
EEGPT_SFREQ = 256.0          # resample target (our data is 160 Hz -> upsample)
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

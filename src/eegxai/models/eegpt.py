"""EEGPT loading + embedding extraction, with the verified input formatting baked in.

Input contract (verified from the braindecode checkpoint `config.json`, not the paper):
  * sampling rate **250 Hz** (not 256), window **1000 samples = 4 s**;
  * a **fixed 62-channel montage** fed in the model's own order;
  * channels the data lacks (PhysioNet MI is missing PO5/PO6) are **interpolated**;
  * channel-name matching is **case-insensitive** (EEGPT uses FP1/FPZ/…; MNE uses Fp1/Fpz/…).

Load with `n_chans=62, chan_proj_type="none", return_encoder_output=True`; the encoder returns
`(batch, n_patches, n_tokens, embed_dim)`, pooled here to a `(batch, embed_dim)` embedding.

NOTE: the µV scaling and mean-pooling below are **provisional** — the smoke test showed high
trial-to-trial embedding correlation, so these are the first knobs to revisit if the
decodability check underperforms.
"""

from __future__ import annotations

import json
from typing import Sequence

import numpy as np
import mne

from eegxai.data.preprocessing import EEGPT_N_TIMES

EEGPT_HF_REPO = "braindecode/eegpt-pretrained"
_MONTAGE = "standard_1005"


def load_eegpt(repo: str = EEGPT_HF_REPO, *, device: str = "cpu"):
    """Load the pretrained EEGPT encoder (eval mode) returning encoder features."""
    from braindecode.models import EEGPT

    model = EEGPT.from_pretrained(
        repo, n_chans=62, chan_proj_type="none", return_encoder_output=True
    )
    return model.to(device).eval()


def eegpt_channel_names(repo: str = EEGPT_HF_REPO) -> list[str]:
    """The model's expected 62-channel order (from the checkpoint config)."""
    from huggingface_hub import hf_hub_download

    cfg = json.load(open(hf_hub_download(repo, "config.json")))
    return [c["ch_name"] for c in cfg["chs_info"]]


def align_raw_to_eegpt(
    raw: mne.io.BaseRaw,
    channel_names: Sequence[str],
    *,
    verbose: str = "ERROR",
) -> tuple[mne.io.BaseRaw, list[str]]:
    """Return a copy of ``raw`` reordered to EEGPT's exact channel set.

    Case-insensitive match; any expected channel missing from the data is added as a bad
    channel and **interpolated** from neighbours (needs a montage). Returns
    ``(aligned_raw, interpolated_channels)``.
    """
    raw = raw.copy()
    upper = {c.upper(): c for c in raw.ch_names}
    missing = [c for c in channel_names if c not in upper]
    if missing:
        zeros = mne.io.RawArray(
            np.zeros((len(missing), raw.n_times)),
            mne.create_info(list(missing), raw.info["sfreq"], "eeg"), verbose=verbose,
        )
        raw.add_channels([zeros], force_update_info=True)
        raw.set_montage(_MONTAGE, on_missing="ignore", verbose=verbose)
        raw.info["bads"] = list(missing)
        raw.interpolate_bads(reset_bads=True, verbose=verbose)
        upper = {c.upper(): c for c in raw.ch_names}
    raw.pick([upper[c] for c in channel_names])
    return raw, missing


def epochs_to_input(
    epochs: mne.Epochs,
    *,
    n_times: int = EEGPT_N_TIMES,
    scale_to_uv: bool = True,
) -> np.ndarray:
    """MNE Epochs (already channel-aligned) -> ``(n_trials, 62, n_times)`` float32 model input."""
    X = epochs.get_data(copy=False)[..., :n_times]
    if scale_to_uv:
        X = X * 1e6  # provisional scaling; revisit if decodability underperforms
    return np.ascontiguousarray(X, dtype=np.float32)


def extract_embeddings(
    model,
    X: np.ndarray,
    *,
    pool: str = "mean",
    device: str = "cpu",
    batch_size: int = 64,
) -> np.ndarray:
    """``X`` = ``(n, 62, n_times)`` -> ``(n, embed_dim)`` embeddings.

    ``pool``: "mean" over (patches, tokens) [provisional] or "flatten".
    """
    import torch

    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.as_tensor(X[i:i + batch_size], dtype=torch.float32, device=device)
            enc = model(xb)  # (b, n_patches, n_tokens, embed_dim)
            e = enc.mean(dim=(1, 2)) if pool == "mean" else enc.flatten(1)
            outs.append(e.cpu().numpy())
    return np.concatenate(outs)

"""Canonical ERD/ERDS via MNE multitaper time-frequency + percent baseline.

Implements the MNE-Python ERDS-maps recipe (Pfurtscheller & Lopes da Silva, 1999):
per-epoch **multitaper** time-frequency decomposition, baseline-corrected as **percent
change** relative to a pre-cue reference window. By convention here (matching the MNE
example): **percent < 0 = ERD** (desynchronisation), **> 0 = ERS**.

Read-outs, both derived from the *same* TFR so they stay consistent:
  * :func:`class_tfrs`      -> ``{class: AverageTFR}`` (all channels) for ERDS maps.
  * :func:`band_topography` -> per-channel percent change in a band + task window (topomap).

Parameters follow the MNE ERDS example: freqs 2-35 Hz, ``n_cycles = freqs``, baseline
``(-1, 0)`` s, ``mode="percent"``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import mne

MU_BAND = (8.0, 13.0)
BETA_BAND = (13.0, 30.0)
DEFAULT_FREQS = np.arange(2.0, 36.0)   # 2-35 Hz, 1-Hz steps
BASELINE = (-1.0, 0.0)                 # pre-cue reference window (percent-change reference)


def make_epochs(
    raw: mne.io.BaseRaw,
    *,
    classes: Sequence[str] = ("T1", "T2"),
    tmin: float = -1.5,
    tmax: float = 4.0,
) -> mne.Epochs:
    """Epoch a raw for ERDS.

    ``tmin`` must precede the baseline start (default baseline is -1 s), and **no** epoch
    baseline is applied here — baseline correction is done on the TFR as percent change.
    """
    events, ev_id = mne.events_from_annotations(raw, verbose="ERROR")
    keep = {k: ev_id[k] for k in classes if k in ev_id}
    return mne.Epochs(raw, events, keep, tmin=tmin, tmax=tmax, baseline=None,
                      picks="eeg", preload=True, verbose="ERROR")


def class_tfrs(
    epochs: mne.Epochs,
    *,
    freqs: np.ndarray = DEFAULT_FREQS,
    baseline: tuple[float, float] = BASELINE,
    decim: int = 3,
) -> dict[str, "mne.time_frequency.AverageTFR"]:
    """Multitaper TFR per class, percent-baseline corrected. Returns ``{class: AverageTFR}``."""
    out: dict[str, "mne.time_frequency.AverageTFR"] = {}
    for cls in epochs.event_id:
        tfr = epochs[cls].compute_tfr(
            method="multitaper", freqs=freqs, n_cycles=freqs, use_fft=True,
            return_itc=False, decim=decim, average=True, verbose="ERROR",
        )
        tfr.apply_baseline(baseline, mode="percent")
        out[cls] = tfr
    return out


def band_topography(
    tfr: "mne.time_frequency.AverageTFR",
    band: tuple[float, float] = MU_BAND,
    *,
    tmin: float = 0.5,
    tmax: float = 3.5,
) -> np.ndarray:
    """Average a percent-baseline ``AverageTFR`` over a band + task window -> ``(n_channels,)``.

    Negative = ERD (percent power drop vs baseline).
    """
    f, t = tfr.freqs, tfr.times
    fmask = (f >= band[0]) & (f <= band[1])
    tmask = (t >= tmin) & (t <= tmax)
    return tfr.data[:, fmask][:, :, tmask].mean(axis=(1, 2))

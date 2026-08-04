"""Event-related (de)synchronization — the neural ground truth for faithfulness checks.

Mu/beta ERD over sensorimotor cortex is the most reproducible motor-imagery signature; we
use it as the reference our model representations should align with. This module computes
band-limited power change (task vs. baseline) per channel and per class.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import mne


def band_erd_from_raw(
    raw: mne.io.BaseRaw,
    band: tuple[float, float] = (8.0, 13.0),
    *,
    classes: Sequence[str] = ("T1", "T2"),
    base_win: tuple[float, float] = (-0.5, 0.0),
    task_win: tuple[float, float] = (0.5, 3.5),
    tmin: float = -0.5,
    tmax: float = 4.0,
) -> tuple[dict[str, np.ndarray], mne.Info]:
    """Return ``(erd, info)`` where ``erd[class]`` is per-channel ERD in dB.

    Negative dB = desynchronization (power drop vs. baseline). Requires a montage on
    ``raw`` for downstream topographic plotting.
    """
    rb = raw.copy().filter(band[0], band[1], picks="eeg", verbose="ERROR")
    events, ev_id = mne.events_from_annotations(rb, verbose="ERROR")
    keep = {k: ev_id[k] for k in classes if k in ev_id}
    ep = mne.Epochs(rb, events, keep, tmin=tmin, tmax=tmax, baseline=None,
                    picks="eeg", preload=True, verbose="ERROR")
    t = ep.times
    bmask = (t >= base_win[0]) & (t < base_win[1])
    tmask = (t >= task_win[0]) & (t < task_win[1])

    erd: dict[str, np.ndarray] = {}
    for cls in keep:
        power = ep[cls].get_data(copy=False) ** 2          # trials x ch x time
        base = power[:, :, bmask].mean(axis=(0, 2))
        task = power[:, :, tmask].mean(axis=(0, 2))
        erd[cls] = 10.0 * np.log10(task / base)            # dB; negative = ERD
    info = ep.info.copy()
    del ep, rb
    return erd, info

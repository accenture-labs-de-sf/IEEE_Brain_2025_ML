"""ERDS time-frequency map plots (channels x classes grid)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt

_CLASS_LABEL = {"T1": "imagine LEFT", "T2": "imagine RIGHT"}


def plot_erds_maps(
    maps: dict[tuple[str, str], np.ndarray],
    freqs: np.ndarray,
    times: np.ndarray,
    channels: Sequence[str],
    classes: Sequence[str],
    out_path: str | Path,
    *,
    vmax: float | None = None,
    sig: dict[tuple[str, str], np.ndarray] | None = None,
) -> Path:
    """Grid of ERDS maps: rows = channels, cols = classes.

    ``maps[(class, channel)]`` is a 2D ``(freq x time)`` percent-change array. Convention:
    **red = ERD** (negative), blue = ERS (positive), matching the MNE ERDS example.
    ``sig[(class, channel)]`` (optional) is a boolean mask of significant points; where given,
    significant regions are outlined and non-significant areas are dimmed.
    """
    if vmax is None:
        vmax = max(float(np.abs(maps[(c, ch)]).max()) for c in classes for ch in channels)
    nrow, ncol = len(channels), len(classes)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.6 * nrow), squeeze=False)
    im = None
    for r, ch in enumerate(channels):
        for c, cls in enumerate(classes):
            ax = axes[r][c]
            im = ax.imshow(
                maps[(cls, ch)], aspect="auto", origin="lower",
                extent=[times[0], times[-1], freqs[0], freqs[-1]],
                cmap="RdBu", vmin=-vmax, vmax=vmax,
            )
            if sig is not None and (cls, ch) in sig:
                mask = sig[(cls, ch)]
                ax.contour(times, freqs, mask, levels=[0.5], colors="k", linewidths=1.2)
                ax.contourf(times, freqs, ~mask, levels=[0.5, 1.5],
                            colors="white", alpha=0.55)  # dim non-significant
            ax.axvline(0.0, color="k", lw=0.8, ls="--")  # cue onset
            if r == 0:
                ax.set_title(_CLASS_LABEL.get(cls, cls))
            ax.set_ylabel(f"{ch}\nfreq (Hz)" if c == 0 else "")
            ax.set_xlabel("time (s)" if r == nrow - 1 else "")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.03)
    cbar.set_label("% change vs baseline  ·  red = ERD (desync)")
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path

"""Topographic plots for ERD / channel-wise scalar maps."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
import mne

_CLASS_LABEL = {"T1": "imagine LEFT", "T2": "imagine RIGHT"}


def plot_erd_grid(
    rows: Sequence[tuple[str, dict[str, np.ndarray]]],
    classes: Sequence[str],
    info: mne.Info,
    band: tuple[float, float],
    out_path: str | Path,
    *,
    cmap: str = "RdBu",
    unit_label: str = "% change vs baseline  ·  red = ERD",
) -> Path:
    """Grid of ERD topomaps: one row per (label, erd_dict), one column per class.

    Shared symmetric color scale across the whole grid so rows are comparable (e.g. per
    subject + a group mean). Default convention (percent, MNE ERDS): **red = ERD**.
    """
    vmax = max(float(np.abs(erd[c]).max()) for _lbl, erd in rows for c in classes)
    nrow, ncol = len(rows), len(classes)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.4 * nrow), squeeze=False)
    im = None
    for r, (label, erd) in enumerate(rows):
        for c, cls in enumerate(classes):
            ax = axes[r][c]
            im, _ = mne.viz.plot_topomap(erd[cls], info, axes=ax, show=False,
                                         cmap=cmap, vlim=(-vmax, vmax), contours=4)
            title = _CLASS_LABEL.get(cls, cls)
            ax.set_title(f"{label} — {title}", fontsize=10)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.04)
    cbar.set_label(f"{band[0]:.0f}–{band[1]:.0f} Hz  ·  {unit_label}")
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path

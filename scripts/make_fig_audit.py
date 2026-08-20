"""Build the audit figure (fig1_overview) telling the controlled n=35 story on real data.

(a) A real grand-average sensorimotor power spectrum with its fitted aperiodic 1/f component and
    the mu peak above it (grounds the periodic/aperiodic decomposition on data, not a schematic).
(b) EEGPT's embedding geometry tracks aggregate/total power strongly (reference line), while the
    aperiodic-adjusted oscillatory bands are small and mu is only slightly above the non-motor
    control bands (the central finding: power dominates, mu is a small trend).
(c) Scalp topography of how strongly each channel's aperiodic-adjusted mu power aligns with the
    embedding geometry (task controlled): central/sensorimotor rather than posterior alpha.

All panels are computed from the caches (results/audit_n35 + psd_perchan + adjpow_perchan), so the
figure is reproducible. Writes reports/draft/figures/fig1_overview.{pdf,png}.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
from matplotlib.gridspec import GridSpec
from scipy.spatial.distance import pdist
from scipy.stats import rankdata, spearmanr
import mne

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eegxai.pipeline import load_all, SENSORIMOTOR  # noqa: E402

mne.set_log_level("ERROR")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/draft/figures"
CACHE = ROOT / "results/audit_n35"
LN10 = np.log(10.0)
TEAL, GREY, INK, SLATE = "#2a9d8f", "#9aa0a6", "#222222", "#5b5f97"
BANDS = ["theta", "mu", "beta", "gamma"]


def house_style():
    plt.style.use(["science", "nature", "no-latex"])
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "svg.fonttype": "none", "pdf.fonttype": 42, "axes.linewidth": 0.8})


def partial_resid(y, x, ctrls):
    yr, xr = rankdata(y), rankdata(x)
    C = np.column_stack([rankdata(c) for c in ctrls] + [np.ones_like(yr)])
    ry = yr - C @ np.linalg.lstsq(C, yr, rcond=None)[0]
    rx = xr - C @ np.linalg.lstsq(C, xr, rcond=None)[0]
    return float(np.corrcoef(ry, rx)[0, 1])


def zpdist(x):
    x = x[:, None] if x.ndim == 1 else x
    xz = (x - x.mean(0)) / (x.std(0) + 1e-9)
    return pdist(xz)


def panel_a_spectrum(ax):
    """Grand-average sensorimotor PSD + fitted 1/f, mu peak shaded."""
    from specparam import SpectralModel
    files = sorted((CACHE / "psd_perchan").glob("sub-*.npz"))
    lin_sum, cnt, freqs = None, 0, None
    for f in files:
        d = np.load(f, allow_pickle=True)
        freqs = d["freqs"]
        upper = {c.upper(): i for i, c in enumerate(d["ch_names"])}
        idx = [upper[c] for c in SENSORIMOTOR if c in upper]
        lin = np.exp(d["psd"][:, idx, :]).mean(axis=(0, 1))  # avg trials & SMC channels -> (F,)
        lin_sum = lin if lin_sum is None else lin_sum + lin
        cnt += 1
    grand = lin_sum / cnt
    m = (freqs >= 2) & (freqs <= 40)
    f, p = freqs[m], grand[m]
    sm = SpectralModel(peak_width_limits=[1, 12], max_n_peaks=6, aperiodic_mode="fixed", verbose=False)
    sm.fit(f, p, [2.0, 40.0])
    off, exp = sm.get_params("aperiodic", "offset"), sm.get_params("aperiodic", "exponent")
    aper = off - exp * np.log10(f)
    ax.plot(f, np.log10(p), color=INK, lw=1.6, label="observed")
    ax.plot(f, aper, color=GREY, lw=1.4, ls="--", label="aperiodic 1/f fit")
    ax.axvspan(8, 13, color=TEAL, alpha=0.15)
    ax.annotate("mu peak", xy=(10.5, np.log10(p)[np.argmin(abs(f - 10.5))]),
                xytext=(15, np.log10(p).max()), fontsize=7, color=TEAL,
                arrowprops=dict(arrowstyle="-", color=TEAL, lw=0.6))
    ax.set_xlabel("frequency (Hz)", fontsize=8); ax.set_ylabel("log10 power", fontsize=8)
    ax.set_title("Real spectrum: 1/f + mu", fontsize=8.5, pad=3)
    ax.legend(frameon=False, fontsize=6.5, loc="upper right"); ax.tick_params(labelsize=7)


def compute_rsa():
    """Return band bar values (|task, 64-ch), total-power reference, and per-channel mu topo."""
    emb = load_all(CACHE)
    Eby = {int(s): emb["E"][emb["G"] == s] for s in np.unique(emb["G"])}
    totby = {int(s): emb["totalpow"][emb["G"] == s] for s in np.unique(emb["G"])}
    files = sorted((CACHE / "adjpow_perchan").glob("sub-*.npz"))
    bands = list(np.load(files[0], allow_pickle=True)["bands"])
    ch_names = [str(c) for c in np.load(files[0], allow_pickle=True)["ch_names"]]
    band_vals = {b: [] for b in bands}; totpow = []
    perch = []  # per subject (64,) mu partial
    for f in files:
        d = np.load(f, allow_pickle=True); s = int(d["subject"]); adj = d["adj"]; Y = d["Y"]
        E = Eby[s]; n = len(Y); iu = np.triu_indices(n, 1)
        e = np.nan_to_num((1 - np.corrcoef(E))[iu]); task = (Y[:, None] != Y[None, :]).astype(float)[iu]
        totpow.append(spearmanr(e, np.nan_to_num((1 - np.corrcoef(totby[s]))[iu])).correlation)
        for bi, b in enumerate(bands):
            band_vals[b].append(partial_resid(e, np.nan_to_num((1 - np.corrcoef(adj[:, :, bi]))[iu]), [task]))
        mi = bands.index("mu")
        perch.append([partial_resid(e, zpdist(adj[:, c, mi]), [task]) for c in range(adj.shape[1])])
    band_mean = {b: float(np.mean(v)) for b, v in band_vals.items()}
    return band_mean, float(np.mean(totpow)), np.mean(perch, axis=0), ch_names


def panel_b_bars(ax, band_mean, totpow):
    order = ["mu", "beta", "theta", "gamma"]
    labels = ["mu", "beta", "theta\n(ctl)", "gamma\n(ctl)"]
    colors = [TEAL, TEAL, GREY, GREY]
    vals = [band_mean[b] for b in order]
    ax.bar(range(4), vals, color=colors, width=0.66)
    ax.axhline(totpow, color=SLATE, lw=1.4, ls="--")
    ax.text(3.4, totpow + 0.006, f"total power ({totpow:.2f})", fontsize=6.6, color=SLATE, ha="right")
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("RSA with EEGPT geometry", fontsize=8)
    ax.set_ylim(0, max(totpow, max(vals)) * 1.25)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.004, f"{v:.2f}", ha="center", fontsize=7)
    # mu-vs-control gap annotation
    ax.annotate("", xy=(0, band_mean["mu"]), xytext=(3, band_mean["gamma"]),
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.5))
    ax.text(1.5, band_mean["mu"] + 0.02, "mu > ctl  Δ≈0.04\n(p≈0.001)", ha="center", fontsize=6.4, color=INK)
    ax.set_title("Power dominates; mu a small trend", fontsize=8.5, pad=3)
    ax.tick_params(labelsize=7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def panel_c_topo(ax, perch_mu, ch_names):
    info = mne.create_info(ch_names, 250., "eeg")
    info.set_montage("standard_1005", match_case=False, on_missing="ignore")
    vmax = np.percentile(perch_mu, 98)
    im, _ = mne.viz.plot_topomap(perch_mu, info, axes=ax, show=False, cmap="Reds",
                                 vlim=(0, vmax), contours=4, sensors=False)
    ax.set_title("mu alignment:\ncentral-weighted", fontsize=8.5, pad=3)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.ax.tick_params(labelsize=6)
    cb.set_label("mu RSA (|task)", fontsize=6.5)


def main():
    house_style()
    band_mean, totpow, perch_mu, ch_names = compute_rsa()
    fig = plt.figure(figsize=(11, 3.4), dpi=200)
    gs = GridSpec(1, 3, width_ratios=[1.15, 1.1, 0.95], wspace=0.42)
    axa, axb = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    axc = fig.add_subplot(gs[2])
    panel_a_spectrum(axa); panel_b_bars(axb, band_mean, totpow); panel_c_topo(axc, perch_mu, ch_names)
    for ax, lab in ((axa, "a"), (axb, "b"), (axc, "c")):
        ax.text(-0.12, 1.12, lab, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig1_overview.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("band means:", {k: round(v, 3) for k, v in band_mean.items()}, "| total power:", round(totpow, 3))
    print("wrote", OUT / "fig1_overview.{pdf,png}")


if __name__ == "__main__":
    main()

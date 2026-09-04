"""Build the audit figure (fig1_overview) telling the controlled n=35 story on real data.

(a) A real grand-average sensorimotor power spectrum with its fitted aperiodic 1/f component and
    the mu peak above it (grounds the periodic/aperiodic decomposition on data, not a schematic).
(b) EEGPT's embedding geometry tracks aggregate/total power strongly (reference line), while the
    aperiodic-adjusted oscillatory bands are small and mu is only slightly above the non-motor
    control bands (the central finding: power dominates, mu is a small trend).
(c) Central vs occipital mu, the control that separates sensorimotor mu from posterior alpha
    (both 8-13 Hz). Mirrors Test 2 of rsa_spatial_control.py: a multivariate pattern RDM over each
    ROI's channels. A per-channel magnitude map is deliberately not used here, since single-channel
    mu magnitude is dominated by alpha variance and peaks posteriorly regardless of the effect.

All panels are computed from the caches (results/audit_n35 + psd_perchan + adjpow_perchan), so the
figure is reproducible. Writes reports/draft/figures/fig1_overview.{pdf,png}.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
from matplotlib.gridspec import GridSpec
from scipy.stats import rankdata, spearmanr
import mne

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eegxai.pipeline import load_all, SENSORIMOTOR  # noqa: E402
from rsa_spatial_control import CENTRAL, OCCIPITAL  # noqa: E402

mne.set_log_level("ERROR")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/draft/figures"
CACHE = ROOT / "results/audit_n35"
LN10 = np.log(10.0)
TEAL, GREY, INK, SLATE = "#2a9d8f", "#9aa0a6", "#222222", "#5b5f97"
BANDS = ["theta", "mu", "beta", "gamma"]

# Caption for the standalone upload. The submission form takes the figure as a separate PDF, so
# the version that ships alone has to carry its own caption; keep it in step with the caption in
# reports/draft/abstract_1page.md, which serves the in-document embed.
CAPTION = (
    "Figure 1. Auditing EEGPT on motor imagery (PhysioNet, n = 35 subjects, 2891 cued 4 s trials). (a)  "
    "Grand-average sensorimotor spectrum with its fitted aperiodic 1/f component and the mu peak above it. (b)  "
    "The embedding geometry tracks aggregate total power (dashed, r \u2248 0.24) far more than any  "
    "aperiodic-adjusted band; mu sits only \u2248 0.04 above the non-motor controls. (c) Mu and posterior  "
    "alpha share the 8-13 Hz band, so the trend is tested against location: alignment is much stronger over  "
    "central than occipital electrodes and survives partialling out the other region (dots are subjects).  "
    "Electrode regions are channel space, not source localization."
)



def house_style():
    plt.style.use(["science", "nature", "no-latex"])
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "svg.fonttype": "none", "pdf.fonttype": 42, "axes.linewidth": 0.8,
                         "xtick.top": False, "ytick.right": False})


def partial_resid(y, x, ctrls):
    yr, xr = rankdata(y), rankdata(x)
    C = np.column_stack([rankdata(c) for c in ctrls] + [np.ones_like(yr)])
    ry = yr - C @ np.linalg.lstsq(C, yr, rcond=None)[0]
    rx = xr - C @ np.linalg.lstsq(C, xr, rcond=None)[0]
    return float(np.corrcoef(ry, rx)[0, 1])


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
    """Return band bar values (|task, 64-ch), the total-power reference, and the central-vs-
    occipital mu contrast. The ROI contrast mirrors Test 2 of rsa_spatial_control.py exactly:
    a multivariate pattern RDM over the ROI's channels, not a per-channel magnitude map."""
    emb = load_all(CACHE)
    Eby = {int(s): emb["E"][emb["G"] == s] for s in np.unique(emb["G"])}
    totby = {int(s): emb["totalpow"][emb["G"] == s] for s in np.unique(emb["G"])}
    files = sorted((CACHE / "adjpow_perchan").glob("sub-*.npz"))
    bands = list(np.load(files[0], allow_pickle=True)["bands"])
    ch_names = [str(c) for c in np.load(files[0], allow_pickle=True)["ch_names"]]
    band_vals = {b: [] for b in bands}; totpow = []
    roi = {k: [] for k in ("cen", "occ", "cen_pp", "occ_pp")}
    for f in files:
        d = np.load(f, allow_pickle=True); s = int(d["subject"]); adj = d["adj"]; Y = d["Y"]
        E = Eby[s]; n = len(Y); iu = np.triu_indices(n, 1)
        e = np.nan_to_num((1 - np.corrcoef(E))[iu]); task = (Y[:, None] != Y[None, :]).astype(float)[iu]
        totpow.append(spearmanr(e, np.nan_to_num((1 - np.corrcoef(totby[s]))[iu])).correlation)
        for bi, b in enumerate(bands):
            band_vals[b].append(partial_resid(e, np.nan_to_num((1 - np.corrcoef(adj[:, :, bi]))[iu]), [task]))
        mi = bands.index("mu")
        upper = {c.upper(): i for i, c in enumerate(ch_names)}
        cen_idx = [upper[c] for c in CENTRAL if c in upper]
        occ_idx = [upper[c] for c in OCCIPITAL if c in upper]
        cen = np.nan_to_num((1 - np.corrcoef(adj[:, cen_idx, mi]))[iu])
        occ = np.nan_to_num((1 - np.corrcoef(adj[:, occ_idx, mi]))[iu])
        roi["cen"].append(partial_resid(e, cen, [task]))
        roi["occ"].append(partial_resid(e, occ, [task]))
        roi["cen_pp"].append(partial_resid(e, cen, [task, occ]))
        roi["occ_pp"].append(partial_resid(e, occ, [task, cen]))
    band_mean = {b: float(np.mean(v)) for b, v in band_vals.items()}
    return band_mean, float(np.mean(totpow)), {k: np.array(v) for k, v in roi.items()}, ch_names


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
        ax.text(i, v + 0.006, f"{v:.2f}", ha="center", fontsize=7)
    # mu-vs-control gap note, placed in the clear band between the bars and the total-power line
    ax.text(1.5, 0.195, "mu > control:  Δr ≈ 0.04  (p ≈ 0.001)", ha="center",
            va="center", fontsize=6.6, color=INK)
    ax.set_title("Power dominates; mu a small trend", fontsize=8.5, pad=3)
    ax.tick_params(labelsize=7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def _roi_head(ax, ch_names, roi, color):
    """Small head schematic with the ROI's channels marked, so the reader sees which sites."""
    info = mne.create_info(list(ch_names), 250., "eeg")
    info.set_montage("standard_1005", match_case=False, on_missing="ignore")
    upper = {c.upper(): i for i, c in enumerate(ch_names)}
    mask = np.zeros(len(ch_names), bool)
    for c in roi:
        if c in upper:
            mask[upper[c]] = True
    mne.viz.plot_topomap(np.zeros(len(ch_names)), info, axes=ax, show=False, cmap="Greys",
                         vlim=(0, 1), contours=0, sensors=False, mask=mask,
                         mask_params=dict(marker="o", markerfacecolor=color, markeredgecolor="none",
                                          markersize=2.6))


def panel_c_roi(ax, roi, ch_names):
    """Central vs occipital mu: the control that separates sensorimotor mu from posterior alpha."""
    cen, occ = roi["cen"], roi["occ"]
    ax.bar([0, 1], [cen.mean(), occ.mean()], color=[TEAL, GREY], width=0.40, zorder=2)
    jit = np.random.RandomState(0).uniform(-0.10, 0.10, len(cen))
    for x, v in ((0, cen), (1, occ)):               # individual subjects, to show the spread honestly
        ax.scatter(x + jit, v, s=4, color="#6f767d", alpha=0.55, linewidths=0, zorder=3)
    for x, v in ((0, cen.mean()), (1, occ.mean())):
        ax.text(x + 0.30, v, f"{v:.3f}", ha="left", va="center", fontsize=7.5, zorder=4)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["central\n(sensorimotor)", "occipital\n(alpha)"], fontsize=7.5)
    ax.set_ylabel("mu RSA (|task)", fontsize=8)
    ax.set_xlim(-0.62, 1.62); ax.set_ylim(-0.09, 0.35)
    ax.set_yticks([-0.05, 0.0, 0.05, 0.10, 0.15, 0.20])
    ax.axhline(0, color=INK, lw=0.6)
    ax.text(0.5, 1.0, f"controlling the other region: {roi['cen_pp'].mean():.3f} vs {roi['occ_pp'].mean():.3f}",
            transform=ax.transAxes, ha="center", va="top", fontsize=6.8, color=INK)
    ax.set_title("The small trend is mu, not alpha", fontsize=8.5, pad=13)
    ax.tick_params(labelsize=7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    # a head over each bar, marking which electrodes the region is made of
    for xc, roi_ch, col in ((0.28, CENTRAL, TEAL), (0.72, OCCIPITAL, GREY)):
        _roi_head(ax.inset_axes([xc - 0.14, 0.62, 0.28, 0.28]), ch_names, roi_ch, col)


def main():
    house_style()
    band_mean, totpow, roi, ch_names = compute_rsa()
    fig = plt.figure(figsize=(11.4, 3.5), dpi=200)
    gs = GridSpec(1, 3, width_ratios=[1.1, 1.05, 1.05], wspace=0.38)
    axa, axb = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    axc = fig.add_subplot(gs[2])
    panel_a_spectrum(axa); panel_b_bars(axb, band_mean, totpow); panel_c_roi(axc, roi, ch_names)
    for ax, lab in ((axa, "a"), (axb, "b"), (axc, "c")):
        ax.text(-0.12, 1.12, lab, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig1_overview.{ext}", dpi=200, bbox_inches="tight")
    # Captioned variant for the standalone upload; the tight bbox grows to include the text.
    fig.text(0.02, -0.02, "\n".join(textwrap.wrap(" ".join(CAPTION.split()), 218)), fontsize=7,
             color=INK, ha="left", va="top", linespacing=1.5)
    fig.savefig(OUT / "fig1_overview_captioned.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("band means:", {k: round(v, 3) for k, v in band_mean.items()}, "| total power:", round(totpow, 3))
    print("wrote", OUT / "fig1_overview.{pdf,png}", "+ fig1_overview_captioned.pdf")


if __name__ == "__main__":
    main()

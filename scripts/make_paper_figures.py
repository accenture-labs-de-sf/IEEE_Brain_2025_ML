"""Build the paper figures from validated run assets.

House style is SciencePlots (nature) with a sans-serif override, exported as vector PDF
plus PNG (the PNG feeds the Markdown/PDF preview; the PDF is the editable master).

Figure 1 (overview): a left-to-right narrative. (a) the real sensorimotor rhythm, native
mu ERD topography and C3 time-frequency map with significance contour; (b) the frozen
EEGPT encoder; (c) the embedding dissimilarity matrix (RSA), with the rest/imagery block
diagonal that is the central object of the audit; (d) a concept panel posing the
periodic (rhythm) versus aperiodic (1/f, aggregate power) question.

Figure 2 (geometry, interim): RDM heatmaps over the partial-correlation collapse. To be
rebuilt in the Figure 1 visual language.

Data sources (git-ignored results, regenerable):
  results/exploration/mi_erds_20260806T235629Z/erds.npz   native ERD topo + TF maps
  results/exploration/improved_features.npz               embeddings, labels, band power
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401  (registers styles)
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import mne

mne.set_log_level("ERROR")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/draft/figures"
OUT.mkdir(parents=True, exist_ok=True)
ERDS = ROOT / "results/exploration/mi_erds_20260806T235629Z/erds.npz"
FEATS = ROOT / "results/exploration/improved_features.npz"

TEAL, SLATE, INK, GREY = "#2a9d8f", "#5b5f97", "#222222", "#9aa0a6"


def house_style():
    plt.style.use(["science", "nature", "no-latex"])
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "svg.fonttype": "none", "pdf.fonttype": 42, "axes.linewidth": 0.8,
    })


def _save(fig, stem: str):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", stem + ".{pdf,png}")


def build_fig1_overview():
    er = np.load(ERDS, allow_pickle=True)
    ft = np.load(FEATS)
    chn = [str(c) for c in er["ch_names"]]; freqs = er["freqs"]; times = er["times"]
    info = mne.create_info(chn, 250., "eeg")
    info.set_montage("standard_1005", match_case=False, on_missing="ignore")

    fig = plt.figure(figsize=(12.4, 5.2), dpi=200)
    bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, 1); bg.set_ylim(0, 1); bg.axis("off")

    def arrow(x0, x1, y=0.56, color=GREY):
        bg.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                     mutation_scale=15, lw=1.6, color=color, shrinkA=0, shrinkB=0))

    def stage(x, letter, title):
        bg.text(x, 0.95, letter, fontsize=13, fontweight="bold", color=INK, ha="left")
        bg.text(x + 0.025, 0.953, title, fontsize=9.5, color=INK, ha="left", va="center")

    # a: rhythm
    stage(0.028, "a", "A real sensorimotor rhythm")
    axt = fig.add_axes([0.03, 0.44, 0.135, 0.40])
    v = np.abs(er["group_mu_T2"]).max()
    mne.viz.plot_topomap(er["group_mu_T2"], info, axes=axt, show=False, cmap="RdBu",
                         vlim=(-v, v), contours=4, sensors=False)
    axt.set_title("mu ERD\n(imagine right)", fontsize=8, pad=1)
    axtf = fig.add_axes([0.185, 0.30, 0.125, 0.50])
    M = er["map_T2_C3"]; sig = er["sig_T2_C3"]; ex = [times[0], times[-1], freqs[0], freqs[-1]]
    mx = np.nanpercentile(np.abs(M), 98)
    axtf.imshow(M, extent=ex, origin="lower", aspect="auto", cmap="RdBu", vmin=-mx, vmax=mx)
    axtf.contour(sig, levels=[0.5], extent=ex, origin="lower", colors="k", linewidths=0.5)
    axtf.axvline(0, color="k", lw=0.5, ls=":")
    axtf.set_xlabel("time (s)", fontsize=7.5); axtf.set_ylabel("Hz", fontsize=7.5)
    axtf.set_title("C3 time-frequency", fontsize=8, pad=1); axtf.tick_params(labelsize=6.5)

    # b: encoder
    stage(0.35, "b", "EEGPT encoder")
    arrow(0.315, 0.375)
    bx, by, bw, bh = 0.375, 0.50, 0.075, 0.16
    for dx in (0.011, 0.0055, 0.0):
        bg.add_patch(FancyBboxPatch((bx + dx, by - dx), bw, bh, boxstyle="round,pad=0.004",
                     fc="white", ec=SLATE, lw=1.2, mutation_aspect=0.55))
    bg.text(bx + 0.0055 + bw / 2, by + bh / 2, "EEGPT\nfrozen", ha="center", va="center",
            fontsize=8, color=SLATE)
    bg.text(bx + 0.0055 + bw / 2, by - 0.03, "dual objective:\nalign + reconstruct",
            ha="center", va="top", fontsize=6.3, color=GREY, style="italic")
    bg.text(bx + 0.0055 + bw / 2, by + bh + 0.055, "512-d embedding\nper trial",
            ha="center", va="bottom", fontsize=6.6, color=SLATE)

    # c: embedding RDM (centerpiece)
    stage(0.50, "c", "Representational structure")
    arrow(0.465, 0.508)
    E, Y, G = ft["E"], ft["Y"], ft["G"]; m = G == 6
    Es = E[m]; ys = Y[m]; order = np.argsort(ys, kind="stable"); Es = Es[order]; ys = ys[order]
    D = 1 - np.corrcoef(Es); nrest = int((ys == 0).sum()); n = len(ys)
    axr = fig.add_axes([0.515, 0.235, 0.215, 0.56])
    im = axr.imshow(D, cmap="viridis", interpolation="nearest")
    axr.axhline(nrest - 0.5, color="w", lw=1.1); axr.axvline(nrest - 0.5, color="w", lw=1.1)
    axr.set_xticks([nrest / 2, nrest + (n - nrest) / 2])
    axr.set_yticks([nrest / 2, nrest + (n - nrest) / 2])
    axr.set_xticklabels(["rest", "imagery"], fontsize=7.5)
    axr.set_yticklabels(["rest", "imagery"], fontsize=7.5, rotation=90, va="center")
    axr.set_title("embedding dissimilarity (1 - r)", fontsize=8, pad=2)
    axr.annotate("similar within\ncondition\n(block diagonal)", xy=(nrest * 0.5, nrest * 0.5),
                 xytext=(n * 0.62, n * 0.16), fontsize=6.8, color="w", ha="left", va="center",
                 arrowprops=dict(arrowstyle="-|>", color="w", lw=0.9))
    cax = fig.add_axes([0.735, 0.30, 0.008, 0.42]); cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=6); cb.set_label("dissimilarity", fontsize=7)

    # d: rhythm or 1/f
    stage(0.775, "d", "Rhythm or 1/f?")
    arrow(0.745, 0.788)
    axs = fig.add_axes([0.80, 0.30, 0.185, 0.48])
    f = np.linspace(2, 40, 400); aper = 1.0 / (f ** 1.5)
    mu = 0.9 * aper.max() * np.exp(-((f - 10) / 2.2) ** 2)
    beta = 0.5 * aper.max() * np.exp(-((f - 21) / 3.0) ** 2)
    base = np.log10(aper) + 3; full = np.log10(aper + mu + beta) + 3
    axs.plot(f, base, color=GREY, lw=1.4, ls="--"); axs.plot(f, full, color=INK, lw=1.6)
    axs.fill_between(f, base, full, where=((f > 7) & (f < 30)), color=TEAL, alpha=0.35)
    axs.annotate("aperiodic (1/f)\naggregate power", (31, base[np.argmin(abs(f - 31))]),
                 xytext=(24, 0.9), fontsize=6.6, color=GREY,
                 arrowprops=dict(arrowstyle="-", color=GREY, lw=0.6))
    axs.annotate("periodic\nmu / beta", (10, full[np.argmin(abs(f - 10))]), xytext=(11.5, 2.35),
                 fontsize=6.6, color=TEAL, arrowprops=dict(arrowstyle="-", color=TEAL, lw=0.6))
    axs.set_xlabel("frequency (Hz)", fontsize=7.5); axs.set_ylabel("log power", fontsize=7.5)
    axs.set_yticks([]); axs.tick_params(labelsize=6.5)
    axs.set_title("which does the structure track?", fontsize=8, pad=2)
    axs.text(0.97, 0.97, "schematic", transform=axs.transAxes, ha="right", va="top",
             fontsize=6.2, color=GREY, style="italic")

    # key quantitative results (folded in from Figure 2 so the panel is self-contained)
    from matplotlib.lines import Line2D
    bg.add_line(Line2D([0.35, 0.985], [0.185, 0.185], color="0.8", lw=0.8))
    bg.text(0.35, 0.205, "Key results", fontsize=7.5, fontweight="bold", color=INK, ha="left")
    bg.text(0.415, 0.125, "Decodes imagery vs rest\n73% within (CSP 76%)  ·  58%* cross-subject",
            ha="center", va="center", fontsize=6.7, color=SLATE)
    bg.text(0.622, 0.125, "Geometry tracks total power\nSpearman r = 0.31",
            ha="center", va="center", fontsize=6.7, color=INK)
    bg.text(0.885, 0.145, "After removing total power", ha="center", va="center",
            fontsize=6.7, color=INK)
    bg.text(0.885, 0.115, "mu .09**   beta .12***", ha="center", va="center",
            fontsize=6.7, color=TEAL)
    bg.text(0.885, 0.095, "control band  n.s.", ha="center", va="center",
            fontsize=6.7, color=GREY)

    _save(fig, "fig1_overview")


def build_fig2_geometry():
    from scipy.stats import spearmanr
    ft = np.load(FEATS)
    E, Y, G = ft["E"], ft["Y"], ft["G"]
    muP, betaP = ft["N_mu(8-13)"], ft["N_beta(13-30)"]
    names = ['FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz', 'C2',
             'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'Fp1', 'Fpz', 'Fp2',
             'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6',
             'F8', 'FT7', 'FT8', 'T7', 'T8', 'T9', 'T10', 'TP7', 'TP8', 'P7', 'P5', 'P3', 'P1',
             'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'Iz']
    info = mne.create_info(names, 250., "eeg")
    info.set_montage("standard_1005", match_case=False, on_missing="ignore")

    def triu(D):
        i = np.triu_indices(D.shape[0], 1); return D[i]

    s = G == 6
    order = np.argsort(Y[s], kind="stable")
    Es = E[s][order]; BPs = np.concatenate([muP[s], betaP[s]], 1)[order]
    ys = Y[s][order]; nrest = int((ys == 0).sum()); n = len(ys)
    Demb = 1 - np.corrcoef(Es); Dbp = 1 - np.corrcoef(BPs)
    Dtask = (ys[:, None] != ys[None, :]).astype(float)

    LT, DK = "#9ecae1", "#3182bd"
    fig = plt.figure(figsize=(12.4, 6.9), dpi=200)
    bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, 1); bg.set_ylim(0, 1); bg.axis("off")

    def stage(x, y, letter, title):
        bg.text(x, y, letter, fontsize=12, fontweight="bold", color=INK, ha="left")
        bg.text(x + 0.022, y + 0.003, title, fontsize=9.5, color=INK, ha="left", va="center")

    # Row A: RSA logic
    stage(0.03, 0.955, "a", "How the geometries are compared (RSA), one illustrative subject")

    def rdm_panel(pos, D, title, cmap="viridis", blocks=True):
        ax = fig.add_axes(pos); ax.imshow(D, cmap=cmap, interpolation="nearest")
        if blocks:
            ax.axhline(nrest - .5, color="w", lw=.9); ax.axvline(nrest - .5, color="w", lw=.9)
        ax.set_xticks([nrest / 2, nrest + (n - nrest) / 2])
        ax.set_yticks([nrest / 2, nrest + (n - nrest) / 2])
        ax.set_xticklabels(["rest", "img"], fontsize=6.5)
        ax.set_yticklabels(["rest", "img"], fontsize=6.5, rotation=90, va="center")
        ax.set_title(title, fontsize=7.8, pad=2)

    rdm_panel([0.045, 0.60, 0.15, 0.30], Demb, "EEGPT embeddings")
    rdm_panel([0.215, 0.60, 0.15, 0.30], Dbp, "mu/beta band power")
    rdm_panel([0.385, 0.60, 0.15, 0.30], Dtask, "task (rest/imagery)", cmap="cividis")
    bg.add_patch(FancyArrowPatch((0.55, 0.75), (0.60, 0.75), arrowstyle="-|>",
                 mutation_scale=15, lw=1.6, color=GREY))
    bg.text(0.575, 0.785, "correlate\noff-diagonal", fontsize=6.6, color=GREY, ha="center")
    axsc = fig.add_axes([0.635, 0.60, 0.24, 0.30])
    xe, xb = triu(Demb), triu(Dbp)
    idx = np.random.RandomState(0).choice(len(xe), min(4000, len(xe)), replace=False)
    axsc.scatter(xb[idx], xe[idx], s=2, alpha=0.12, color=SLATE, edgecolors="none", rasterized=True)
    bcoef, acoef = np.polyfit(xb, xe, 1); xx = np.linspace(xb.min(), xb.max(), 50)
    axsc.plot(xx, bcoef * xx + acoef, color=TEAL, lw=1.6)
    r = spearmanr(xb, xe).correlation
    axsc.text(0.05, 0.92, f"Spearman r = {r:.2f}", transform=axsc.transAxes, fontsize=7.5,
              color=INK, va="top")
    axsc.set_xlabel("band-power dissimilarity", fontsize=7.5)
    axsc.set_ylabel("embedding\ndissimilarity", fontsize=7.5)
    axsc.tick_params(labelsize=6.5); axsc.set_title("second-order correlation", fontsize=7.8, pad=2)

    # Row B: the finding
    stage(0.03, 0.44, "b", "What survives the total-power control (group, n = 20)")
    bands = ["mu\n(8-13)", "beta\n(13-30)", "control\n(2-7)"]
    task_only = [0.183, 0.228, 0.169]; task_pow = [0.085, 0.117, -0.054]; sig = ["**", "***", "n.s."]
    axb = fig.add_axes([0.075, 0.09, 0.44, 0.30]); x = np.arange(3); w = 0.38
    axb.bar(x - w / 2, task_only, w, color=LT, label="task removed")
    axb.bar(x + w / 2, task_pow, w, color=DK, label="task + total power removed")
    axb.axhline(0, color="0.4", lw=0.8)
    axb.set_xticks(x); axb.set_xticklabels(bands, fontsize=8)
    axb.set_ylabel("partial Spearman r\n(embedding vs band power)", fontsize=8)
    axb.set_ylim(-0.1, 0.28); axb.legend(frameon=False, fontsize=7.5, loc="upper right")
    for xi, v, sg in zip(x + w / 2, task_pow, sig):
        off = 0.012 if v >= 0 else -0.028
        axb.text(xi, v + off, sg, ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    axb.annotate("control band\nvanishes", xy=(2 + w / 2, -0.054), xytext=(1.35, -0.085),
                 fontsize=6.8, color=INK, arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8))

    # Row C: spatial pattern of the band change (honest: broad, not focal)
    stage(0.56, 0.44, "c", "The band change is broad, not focal")

    def topo(pos, band, lbl):
        diff = band[Y == 1].mean(0) - band[Y == 0].mean(0)
        ax = fig.add_axes(pos); v = np.abs(diff).max()
        mne.viz.plot_topomap(diff, info, axes=ax, show=False, cmap="RdBu", vlim=(-v, v),
                             contours=3, sensors=False)
        ax.set_title(lbl, fontsize=7.8, pad=1)

    topo([0.60, 0.10, 0.16, 0.26], muP, "mu (imagery - rest)")
    topo([0.78, 0.10, 0.16, 0.26], betaP, "beta (imagery - rest)")
    bg.text(0.77, 0.075, "log-power change; red = desynchronization", fontsize=6.3,
            color=GREY, ha="center")

    _save(fig, "fig2_geometry")


def main():
    house_style()
    build_fig1_overview()
    build_fig2_geometry()


if __name__ == "__main__":
    main()

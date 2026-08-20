"""Decisive specificity control: per-channel aperiodic-adjusted oscillatory power vs EEGPT geometry.

Three tests, all on aperiodic-adjusted band power (periodic component only), so the periodic/
aperiodic separation is isolated from spatial averaging:

1. Per-channel band comparison (64-channel construction, matching the raw band-power double
   control): does the embedding track mu/beta oscillatory geometry more than theta/gamma controls?
2. Central vs occipital mu: 8-13 Hz mu overlaps posterior alpha; if the embedding tracks
   central (sensorimotor) mu above occipital mu, it is the sensorimotor rhythm, not volume-
   conducted alpha.
3. Permutation null for the headline per-channel mu statistic (shuffle trial order of the
   reference within subject), so the effect is shown above chance.

Reads results/<out>/adjpow_perchan/ (from scripts/specparam_perchannel.py) and the embeddings in
results/<out>/. Method: Donoghue et al. 2020; Donoghue, Dominguez & Voytek 2020; Nili et al. 2014.

Run: python scripts/rsa_spatial_control.py --out results/audit_n35
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr, ttest_1samp, ttest_rel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eegxai.pipeline import load_all  # noqa: E402

CENTRAL = ["FC3", "FC1", "FCZ", "FC2", "FC4", "C3", "C1", "CZ", "C2", "C4",
           "CP3", "CP1", "CPZ", "CP2", "CP4"]
OCCIPITAL = ["O1", "OZ", "O2", "PO7", "PO3", "POZ", "PO4", "PO8", "P1", "PZ", "P2"]


def partial_resid(y, x, controls):
    yr, xr = rankdata(y), rankdata(x)
    C = np.column_stack([rankdata(c) for c in controls] + [np.ones_like(yr)])
    ry = yr - C @ np.linalg.lstsq(C, yr, rcond=None)[0]
    rx = xr - C @ np.linalg.lstsq(C, xr, rcond=None)[0]
    return float(np.corrcoef(ry, rx)[0, 1])


def corr_rdm(M, iu):
    D = 1 - np.corrcoef(M)
    return np.nan_to_num(D[iu])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/audit_n35"))
    ap.add_argument("--nperm", type=int, default=1000)
    args = ap.parse_args()

    emb_all = load_all(args.out)
    E_by = {int(s): emb_all["E"][emb_all["G"] == s] for s in np.unique(emb_all["G"])}

    adj_dir = args.out / "adjpow_perchan"
    files = sorted(adj_dir.glob("sub-*.npz"))
    bands = list(np.load(files[0], allow_pickle=True)["bands"])
    rng = np.random.RandomState(0)
    lines = [f"SPATIAL / FREQUENCY SPECIFICITY CONTROL (aperiodic-adjusted power, n={len(files)})", ""]

    # ---- Test 1: per-channel band comparison (64-channel construction) ----
    per_band = {b: [] for b in bands}
    mu_stats, mu_perm_group = [], []
    cen_stats, occ_stats, cen_pp, occ_pp = [], [], [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        s = int(d["subject"]); adj = d["adj"]; Y = d["Y"]
        E = E_by[s]
        if len(Y) != len(E):
            print(f"  WARN {f.name}: trial mismatch {len(Y)} vs {len(E)}, skipping"); continue
        n = len(Y); iu = np.triu_indices(n, 1)
        emb = corr_rdm(E, iu)
        task = (Y[:, None] != Y[None, :]).astype(float)[iu]
        upper = {c.upper(): i for i, c in enumerate(d["ch_names"])}
        for bi, b in enumerate(bands):
            ref = corr_rdm(adj[:, :, bi], iu)
            per_band[b].append(partial_resid(emb, ref, [task]))
        # central vs occipital mu
        mi = bands.index("mu")
        cen_idx = [upper[c] for c in CENTRAL if c in upper]
        occ_idx = [upper[c] for c in OCCIPITAL if c in upper]
        cen = corr_rdm(adj[:, cen_idx, mi], iu)
        occ = corr_rdm(adj[:, occ_idx, mi], iu)
        cen_stats.append(partial_resid(emb, cen, [task]))
        occ_stats.append(partial_resid(emb, occ, [task]))
        cen_pp.append(partial_resid(emb, cen, [task, occ]))   # central mu controlling occipital
        occ_pp.append(partial_resid(emb, occ, [task, cen]))   # occipital mu controlling central
        # permutation null for whole-head mu (control task), shuffle reference trial order
        mu_ref_sq = 1 - np.corrcoef(adj[:, :, mi])
        obs = partial_resid(emb, np.nan_to_num(mu_ref_sq[iu]), [task])
        mu_stats.append(obs)
        null = []
        for _ in range(args.nperm):
            p = rng.permutation(n)
            null.append(partial_resid(emb, np.nan_to_num(mu_ref_sq[np.ix_(p, p)][iu]), [task]))
        mu_perm_group.append((obs, np.array(null)))

    lines.append("Test 1 - per-channel aperiodic-adjusted band power (|task), all 64 channels:")
    lines.append(f"  {'band':8} {'mean r':>8} {'p':>10} {'subj>0':>8}")
    for b in bands:
        v = np.array(per_band[b])
        tag = " (ctl)" if b in ("theta", "gamma") else ""
        lines.append(f"  {b:8}{tag:6} {v.mean():>8.3f} {ttest_1samp(v,0).pvalue:>10.1e} {np.mean(v>0):>8.2f}")

    lines.append("  paired motor-vs-control (per-subject, is the gap reliable):")
    for motor in ("mu", "beta"):
        for ctl in ("theta", "gamma"):
            a, c = np.array(per_band[motor]), np.array(per_band[ctl])
            lines.append(f"    {motor} - {ctl}: {(a-c).mean():+.3f}  paired p={ttest_rel(a,c).pvalue:.2g}"
                         f"  subj>0 {np.mean((a-c)>0):.2f}")

    lines.append("")
    lines.append("Test 2 - central (sensorimotor) vs occipital mu (mu-vs-alpha):")
    lines.append(f"  central mu |task            {np.mean(cen_stats):>7.3f}  (p={ttest_1samp(cen_stats,0).pvalue:.1e})")
    lines.append(f"  occipital mu |task          {np.mean(occ_stats):>7.3f}  (p={ttest_1samp(occ_stats,0).pvalue:.1e})")
    lines.append(f"  central mu | task+occipital {np.mean(cen_pp):>7.3f}  (p={ttest_1samp(cen_pp,0).pvalue:.1e})")
    lines.append(f"  occipital mu | task+central {np.mean(occ_pp):>7.3f}  (p={ttest_1samp(occ_pp,0).pvalue:.1e})")

    # Test 3: group-level permutation null (mean over subjects of the per-subject null draws)
    obs_group = np.mean([o for o, _ in mu_perm_group])
    null_group = np.mean(np.stack([nl for _, nl in mu_perm_group], axis=0), axis=0)
    p_perm = float((np.sum(null_group >= obs_group) + 1) / (len(null_group) + 1))
    lines.append("")
    lines.append(f"Test 3 - permutation null for whole-head mu (|task), {args.nperm} perms:")
    lines.append(f"  observed group mean {obs_group:.3f}; null mean {null_group.mean():.3f} "
                 f"(sd {null_group.std():.3f}); p = {p_perm:.3g}")

    report = "\n".join(lines)
    print(report)
    (args.out / "spatial_control.txt").write_text(report + "\n")
    print(f"\nsaved: {args.out / 'spatial_control.txt'}")


if __name__ == "__main__":
    main()

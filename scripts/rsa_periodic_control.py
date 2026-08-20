"""Oscillatory frequency-specificity control, using aperiodic-adjusted band power.

Band power conflates periodic and aperiodic activity, so a raw-band-power "control band" test is not
a clean specificity check (Donoghue, Dominguez & Voytek 2020, eNeuro). The parameterized fix is to
subtract the fitted 1/f aperiodic component from the log spectrum and integrate the residual within
each band, giving a continuous oscillatory (aperiodic-adjusted) power per band (Donoghue et al.
2020, Nat Neurosci; Gerster et al. 2022). We then ask, via RSA with control RDMs and partial
correlation (Nili et al. 2014), whether EEGPT's embedding geometry tracks aperiodic-adjusted
oscillatory power in the motor bands (mu, beta) more than in non-motor control bands (theta, gamma),
controlling for task and for the aperiodic component itself.

Cache-only: uses the per-trial PSDs in the feature cache and the aperiodic params from
specparam_params.npz (run scripts/rsa_specparam.py first). Fast, re-runnable.

Run: python scripts/rsa_periodic_control.py --out results/audit_n35
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import rankdata, spearmanr, ttest_1samp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eegxai.pipeline import load_all  # noqa: E402

LN10 = np.log(10.0)
BANDS = {"theta 4-7 (ctl)": (4, 7), "mu 8-13": (8, 13),
         "beta 13-30": (13, 30), "gamma 30-45 (ctl)": (30, 45)}


def partial_resid(y, x, controls):
    yr, xr = rankdata(y), rankdata(x)
    C = np.column_stack([rankdata(c) for c in controls] + [np.ones_like(yr)])
    ry = yr - C @ np.linalg.lstsq(C, yr, rcond=None)[0]
    rx = xr - C @ np.linalg.lstsq(C, xr, rcond=None)[0]
    return float(np.corrcoef(ry, rx)[0, 1])


def zpdist(X):
    X = np.atleast_2d(X.T).T if X.ndim == 1 else X
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-9)
    return pdist(Xz)


def adjusted_band_power(psd_log_natural, freqs, offset, exponent):
    """Residual (log10 PSD minus fitted 1/f) integrated per band -> (n_trials, n_bands)."""
    mask = (freqs >= 1) & (freqs <= 45)
    f = freqs[mask]
    log10_psd = psd_log_natural[:, mask] / LN10                      # (n, F) in log10 power
    aper = offset[:, None] - exponent[:, None] * np.log10(f)[None, :]  # (n, F) fixed-mode 1/f
    resid = log10_psd - aper                                          # oscillatory residual
    out = {}
    for name, (lo, hi) in BANDS.items():
        b = (f >= lo) & (f < hi)
        out[name] = resid[:, b].mean(axis=1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/audit_n35"))
    args = ap.parse_args()

    d = load_all(args.out)
    E, Y, G = d["E"], d["Y"], d["G"]
    p = np.load(args.out / "specparam_params.npz")
    off, exp = p["offset"], p["exponent"]
    bandpow = adjusted_band_power(d["psd_smc"], d["psd_freqs"], off, exp)

    subs = np.unique(G)
    lines = [f"OSCILLATORY FREQUENCY-SPECIFICITY CONTROL  (n={len(subs)} subjects, {len(Y)} trials)",
             "aperiodic-adjusted band power (residual above the fitted 1/f); RSA vs EEGPT geometry",
             "controlling task and the aperiodic component. Motor bands vs non-motor control bands.",
             "",
             f"{'band':20} {'raw':>7} {'|task':>8} {'|task+aper':>11} {'p(|task+aper)':>14} {'subj>0':>8}"]

    for name in BANDS:
        vals_raw, vals_t, vals_ta = [], [], []
        for s in subs:
            m = G == s
            n = int(m.sum())
            iu = np.triu_indices(n, 1)
            emb = (1 - np.corrcoef(E[m]))[iu]
            task = (Y[m][:, None] != Y[m][None, :]).astype(float)[iu]
            aper_rdm = zpdist(np.column_stack([off[m], exp[m]]))
            band_rdm = zpdist(bandpow[name][m])
            vals_raw.append(spearmanr(emb, band_rdm).correlation)
            vals_t.append(partial_resid(emb, band_rdm, [task]))
            vals_ta.append(partial_resid(emb, band_rdm, [task, aper_rdm]))
        ta = np.array(vals_ta)
        lines.append(f"{name:20} {np.nanmean(vals_raw):>7.3f} {np.nanmean(vals_t):>8.3f} "
                     f"{ta.mean():>11.3f} {ttest_1samp(ta,0).pvalue:>14.1e} "
                     f"{np.mean(ta>0):>8.2f}")

    lines += ["",
              "Reading: motor bands (mu, beta) vs non-motor control bands (theta, gamma). If mu/beta",
              "exceed both control bands in the |task+aper column, the oscillatory structure EEGPT",
              "tracks is frequency-specific to the sensorimotor rhythm. If they are comparable, the",
              "tracking is oscillatory but not band-specific. (Spatial specificity is a separate",
              "question needing per-channel spectra.) Method: Donoghue et al. 2020 Nat Neurosci;",
              "Donoghue, Dominguez & Voytek 2020 eNeuro; Gerster et al. 2022; Nili et al. 2014."]
    report = "\n".join(lines)
    print(report)
    (args.out / "periodic_control.txt").write_text(report + "\n")
    print(f"\nsaved: {args.out / 'periodic_control.txt'}")


if __name__ == "__main__":
    main()

"""Periodic vs aperiodic RSA: does EEGPT's geometry track the 1/f or the oscillation?

Fits specparam (Donoghue et al. 2020; Gerster et al. 2022) to each trial's sensorimotor-averaged
power spectrum, splitting it into aperiodic parameters (offset, 1/f exponent) and periodic peaks
(mu and beta peak power). It then asks, per subject, whether the EEGPT embedding dissimilarity
geometry tracks the aperiodic features or the periodic features, and whether any periodic
correspondence survives controlling for the aperiodic component. This turns the crude total-power
control into the principled periodic/aperiodic decomposition.

Fitted parameters are cached to specparam_params.npz so the (slow) fitting runs once; the RSA
itself is instant and re-runnable from the cache. Uses the per-trial PSDs already in the feature
cache, so no re-extraction.

Run: python scripts/rsa_specparam.py --out results/audit_n35
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import rankdata, spearmanr, ttest_1samp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eegxai.pipeline import load_all  # noqa: E402


def partial_resid(y, x, controls):
    """Spearman partial correlation of y and x controlling for a list of control vectors."""
    yr, xr = rankdata(y), rankdata(x)
    C = np.column_stack([rankdata(c) for c in controls] + [np.ones_like(yr)])
    ry = yr - C @ np.linalg.lstsq(C, yr, rcond=None)[0]
    rx = xr - C @ np.linalg.lstsq(C, xr, rcond=None)[0]
    return float(np.corrcoef(ry, rx)[0, 1])


def fit_params(freqs, psd_log):
    """Fit specparam to each (log) PSD row; return offset, exponent, mu peak pw, beta peak pw."""
    from specparam import SpectralModel
    mask = freqs >= 1
    f = freqs[mask]
    off, exp, mupw, betapw = [], [], [], []
    sm = SpectralModel(peak_width_limits=[1, 12], max_n_peaks=6, min_peak_height=0.05,
                       aperiodic_mode="fixed", verbose=False)
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, row in enumerate(psd_log):
            sm.fit(f, np.exp(row)[mask], [1.0, 45.0])
            off.append(sm.get_params("aperiodic", "offset"))
            exp.append(sm.get_params("aperiodic", "exponent"))
            pk = np.atleast_2d(sm.get_params("peak"))

            def band(lo, hi):
                if pk.size == 0:
                    return 0.0
                inb = pk[(pk[:, 0] >= lo) & (pk[:, 0] < hi)]
                return float(inb[:, 1].max()) if len(inb) else 0.0
            mupw.append(band(8, 13)); betapw.append(band(13, 30))
            if (i + 1) % 500 == 0:
                print(f"  fit {i+1}/{len(psd_log)} ({time.time()-t0:.0f}s)", flush=True)
    return (np.array(off), np.array(exp), np.array(mupw), np.array(betapw))


def zpdist(X):
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-9)
    return pdist(Xz)  # condensed order matches np.triu_indices(n, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/audit_n35"))
    ap.add_argument("--refit", action="store_true", help="ignore cached params and refit")
    args = ap.parse_args()

    d = load_all(args.out)
    E, Y, G = d["E"], d["Y"], d["G"]
    cache = args.out / "specparam_params.npz"
    if cache.exists() and not args.refit:
        p = np.load(cache)
        off, exp, mupw, betapw = p["offset"], p["exponent"], p["mu_pw"], p["beta_pw"]
        print(f"[specparam] loaded cached params from {cache}")
    else:
        print(f"[specparam] fitting {len(Y)} spectra ...")
        off, exp, mupw, betapw = fit_params(d["psd_freqs"], d["psd_smc"])
        np.savez(cache, offset=off, exponent=exp, mu_pw=mupw, beta_pw=betapw)
        print(f"[specparam] saved params -> {cache}")

    subs = np.unique(G)
    peaks_found = float(np.mean((mupw > 0) | (betapw > 0)))
    lines = [f"PERIODIC vs APERIODIC RSA  (n={len(subs)} subjects, {len(Y)} trials)",
             f"trials with a detected mu or beta peak: {100*peaks_found:.0f}%", ""]
    lines.append(f"{'candidate':22} {'raw':>7} {'|task':>8} {'|task+aper':>11} {'p(|task+aper)':>14}")

    rows = {}
    for name in ("aperiodic (off+exp)", "periodic (mu+beta pk)", "offset", "exponent",
                 "mu peak", "beta peak"):
        rows[name] = ([], [], [])

    for s in subs:
        m = G == s
        n = int(m.sum())
        iu = np.triu_indices(n, 1)
        emb = (1 - np.corrcoef(E[m]))[iu]
        task = (Y[m][:, None] != Y[m][None, :]).astype(float)[iu]
        aper = zpdist(np.column_stack([off[m], exp[m]]))
        cands = {
            "aperiodic (off+exp)": aper,
            "periodic (mu+beta pk)": zpdist(np.column_stack([mupw[m], betapw[m]])),
            "offset": zpdist(off[m][:, None]),
            "exponent": zpdist(exp[m][:, None]),
            "mu peak": zpdist(mupw[m][:, None]),
            "beta peak": zpdist(betapw[m][:, None]),
        }
        for name, c in cands.items():
            raw = spearmanr(emb, c).correlation
            pt = partial_resid(emb, c, [task])
            # control for task and (for non-aperiodic candidates) the aperiodic RDM
            ctrls = [task] if name.startswith("aperiodic") else [task, aper]
            pta = partial_resid(emb, c, ctrls)
            rows[name][0].append(raw); rows[name][1].append(pt); rows[name][2].append(pta)

    for name, (raw, pt, pta) in rows.items():
        pta = np.array(pta)
        lines.append(f"{name:22} {np.mean(raw):>7.3f} {np.mean(pt):>8.3f} "
                     f"{pta.mean():>11.3f} {ttest_1samp(pta,0).pvalue:>14.1e}")

    lines += ["", "Reading: 'raw' is embedding~candidate; '|task' controls task; '|task+aper'",
              "additionally removes the aperiodic component (for periodic/peak candidates), i.e.",
              "does oscillatory structure survive once 1/f is accounted for."]
    report = "\n".join(lines)
    print("\n" + report)
    (args.out / "specparam_rsa.txt").write_text(report + "\n")
    print(f"\nsaved: {args.out / 'specparam_rsa.txt'}")


if __name__ == "__main__":
    main()

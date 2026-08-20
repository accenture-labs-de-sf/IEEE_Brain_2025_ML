"""Fit specparam per channel per trial and cache aperiodic-adjusted band power (fit-once).

For each subject, fits the aperiodic 1/f to every channel x trial spectrum (SpectralGroupModel),
then computes aperiodic-adjusted oscillatory power per band per channel = mean over the band of
(log10 PSD minus the fitted 1/f). This is the parameterized, per-channel oscillatory measure that
holds the spatial construction fixed against the raw band-power double control (Donoghue et al.
2020; Donoghue, Dominguez & Voytek 2020; Gerster et al. 2022).

Reads results/<out>/psd_perchan/sub-XXX.npz; writes results/<out>/adjpow_perchan/sub-XXX.npz with
adj (n_trials, 64, n_bands), band names, channel names, labels. Fitting is the slow part and runs
once; the RSA on top is instant. Resumable.

Run: python scripts/specparam_perchannel.py --out results/audit_n35
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

LN10 = np.log(10.0)
BANDS = {"theta": (4, 7), "mu": (8, 13), "beta": (13, 30), "gamma": (30, 45)}


def adjusted_power(psd_log, freqs):
    """psd_log: (n, 64, F) natural-log PSD. Returns adj (n, 64, n_bands)."""
    from specparam import SpectralGroupModel
    mask = (freqs >= 1) & (freqs <= 45)
    f = freqs[mask]
    n, nch, _ = psd_log.shape
    lin = np.exp(psd_log[:, :, mask]).reshape(n * nch, -1)       # linear power, (n*64, F)
    fg = SpectralGroupModel(peak_width_limits=[1, 12], max_n_peaks=6, min_peak_height=0.05,
                            aperiodic_mode="fixed", verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fg.fit(f, lin, [1.0, 45.0])
        aps = fg.get_params("aperiodic")                        # (n*64, 2): offset, exponent
    off = aps[:, 0][:, None]; exp = aps[:, 1][:, None]
    log10_psd = (psd_log[:, :, mask].reshape(n * nch, -1)) / LN10
    resid = log10_psd - (off - exp * np.log10(f)[None, :])       # oscillatory residual
    resid = resid.reshape(n, nch, -1)
    fmask = f
    adj = np.stack([resid[:, :, (fmask >= lo) & (fmask < hi)].mean(axis=2)
                    for (lo, hi) in BANDS.values()], axis=-1)    # (n, 64, n_bands)
    return adj.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/audit_n35"))
    args = ap.parse_args()
    src = args.out / "psd_perchan"
    dst = args.out / "adjpow_perchan"; dst.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("sub-*.npz"))
    pending = [f for f in files if not (dst / f.name).exists()]
    print(f"[adjpow] {len(pending)} subjects to fit -> {dst}")
    for f in pending:
        t = time.time()
        d = np.load(f, allow_pickle=True)
        adj = adjusted_power(d["psd"], d["freqs"])
        np.savez(dst / f.name, adj=adj, bands=np.array(list(BANDS)),
                 ch_names=d["ch_names"], Y=d["Y"], subject=d["subject"])
        print(f"[adjpow] {f.name}: {adj.shape} ({time.time()-t:.0f}s)", flush=True)
    print("[adjpow] done")


if __name__ == "__main__":
    main()

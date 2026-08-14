"""Band-resolved RSA with conservative artifact cleaning.

Per subject: load broadband (0-38 Hz, average reference, 250 Hz), clean with Autoreject
(Jas et al. 2017; repairs/rejects bad epochs, keeps the signal), then feed the SAME clean trials
to both EEGPT (embeddings) and band-power extraction. RSA is computed per band (mu, beta, and a
low control band), partialling out the task label. If mu/beta exceed the control band, the
correspondence is band-specific; if not, EEGPT's geometry tracks band-power broadly (aggregate
structure) rather than the sensorimotor rhythm.

Features are cached incrementally (resume-able). See docs/pivot-analysis.md.
Run: python scripts/rsa_bandresolved.py --subjects 1 2 ... 20
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import mne
from scipy.signal import butter, filtfilt
from scipy.stats import spearmanr, ttest_1samp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eegxai.data.physionet_mi import load_subject_raw, IMAGERY_LEFT_RIGHT  # noqa: E402
from eegxai.data.preprocessing import EEGPT_RAW_KWARGS  # noqa: E402
from eegxai.analysis.rsa import rdm, task_rdm, partial_spearman  # noqa: E402
from eegxai import io  # noqa: E402

BANDS = {"mu(8-13)": (8, 13), "beta(13-30)": (13, 30), "control(2-7)": (2, 7)}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 21)))
    p.add_argument("--out-base", type=Path, default=Path("results/exploration"))
    return p.parse_args()


def bandpow(X, lo, hi, fs=250):
    b, a = butter(4, [lo / (fs / 2), hi / (fs / 2)], "band")
    return np.log(np.var(filtfilt(b, a, X, axis=-1), axis=-1) + 1e-12)


def align_epochs(ep, names):
    """Reorder epochs to EEGPT's channel set, interpolating any the data lacks (PO5/PO6)."""
    ep = ep.copy()
    upper = {c.upper(): c for c in ep.ch_names}
    missing = [c for c in names if c not in upper]
    if missing:
        arr = np.zeros((len(ep), len(missing), len(ep.times)))
        add = mne.EpochsArray(arr, mne.create_info(list(missing), ep.info["sfreq"], "eeg"),
                              ep.events, tmin=ep.tmin, verbose="ERROR")
        ep.add_channels([add], force_update_info=True)
        ep.set_montage("standard_1005", on_missing="ignore", verbose="ERROR")
        ep.info["bads"] = list(missing)
        ep.interpolate_bads(reset_bads=True, verbose="ERROR")
        upper = {c.upper(): c for c in ep.ch_names}
    return ep.pick([upper[c] for c in names])


def main():
    args = parse_args()
    from autoreject import AutoReject
    from eegxai.models.eegpt import (load_eegpt, eegpt_channel_names,
                                     epochs_to_input, extract_embeddings)
    run_dir = io.new_run_dir(args.out_base, "rsa_bandresolved", config={"subjects": args.subjects})
    cache = run_dir / "features.npz"
    lines = []
    def log(m): print(m, flush=True); lines.append(m)

    model = load_eegpt(); chs = eegpt_channel_names()
    Es, Ys, Gs = [], [], []
    Ns = {b: [] for b in BANDS}
    for s in args.subjects:
        t = time.time()
        raw = load_subject_raw(s, IMAGERY_LEFT_RIGHT, **EEGPT_RAW_KWARGS)
        ev, eid = mne.events_from_annotations(raw, verbose="ERROR")
        keep = {k: eid[k] for k in ("T0", "T1", "T2") if k in eid}
        ep = mne.Epochs(raw, ev, keep, tmin=0., tmax=4., baseline=None, picks="eeg",
                        preload=True, verbose="ERROR")
        ep = AutoReject(n_interpolate=[1, 4], n_jobs=1, random_state=0, verbose=False).fit_transform(ep)
        y = (ep.events[:, 2] != eid["T0"]).astype(int)
        Xd = ep.get_data(copy=False)[..., :1000]
        for b, (lo, hi) in BANDS.items():
            Ns[b].append(bandpow(Xd, lo, hi))
        Es.append(extract_embeddings(model, epochs_to_input(align_epochs(ep, chs))))
        Ys.append(y); Gs.append(np.full(len(y), s))
        np.savez(cache, E=np.concatenate(Es), Y=np.concatenate(Ys), G=np.concatenate(Gs),
                 **{f"N_{b}": np.concatenate(Ns[b]) for b in BANDS})
        log(f"  subject {s}: {len(y)} clean trials ({time.time()-t:.0f}s)")
    E, Y, G = np.concatenate(Es), np.concatenate(Ys), np.concatenate(Gs)
    N = {b: np.concatenate(Ns[b]) for b in BANDS}

    log(f"\nBAND-RESOLVED RSA (within-subject partial | task, n={len(np.unique(G))})")
    log(f"  {'band':14} {'model~neural':>13} {'partial|task':>13}")
    for b in BANDS:
        raws, parts = [], []
        for s in np.unique(G):
            m = G == s
            rm, rn, rt = rdm(E[m]), rdm(N[b][m]), task_rdm(Y[m])
            r_mn = spearmanr(rm, rn).correlation
            r_mt = spearmanr(rm, rt).correlation
            r_nt = spearmanr(rn, rt).correlation
            raws.append(r_mn); parts.append(partial_spearman(r_mn, r_mt, r_nt))
        parts = np.array(parts)
        log(f"  {b:14} {np.mean(raws):>13.3f} {parts.mean():>13.3f}  (p={ttest_1samp(parts,0).pvalue:.1e})")
    (run_dir / "results.txt").write_text("\n".join(lines) + "\n")
    print(f"\nsaved: {run_dir}/results.txt")


if __name__ == "__main__":
    main()

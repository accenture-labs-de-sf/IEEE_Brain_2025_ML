"""Pivot analysis — imagery-vs-rest decodability + RSA on EEGPT embeddings.

Runs, across subjects:
  1. classical ceiling  (CSP+LDA / band-power+LDA; within & cross-subject)
  2. EEGPT embeddings    (extracted once, cached; resume-able)
  3. decode embeddings   (within & cross-subject, logistic regression + permutation)
  4. RSA                 (embedding geometry vs real mu/beta structure, controlling for task)

Why this pivot: left/right ERD/decoding is a weak contrast here (~55% ceiling), and running our
ERD on the model's *reconstruction* is invalid (its per-patch normalization breaks percent
baseline). So we moved to imagery-vs-rest (strong ERD) and to decode + RSA readouts that don't
need a baseline. See docs/pivot-analysis.md.

Run:  python scripts/pivot_analysis.py --subjects 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np  # noqa: E402
import mne  # noqa: E402
from scipy.signal import butter, filtfilt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eegxai.data.physionet_mi import load_subject_raw, IMAGERY_LEFT_RIGHT  # noqa: E402
from eegxai.data.preprocessing import EEGPT_RAW_KWARGS  # noqa: E402
from eegxai.analysis import decoding as D, rsa as RSA  # noqa: E402
from eegxai import io  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 21)))
    p.add_argument("--out-base", type=Path, default=Path("results/exploration"))
    p.add_argument("--n-perm", type=int, default=100)
    return p.parse_args()


def imagery_rest_epochs(subject, **raw_kwargs):
    """Return (epochs, y) with y = 1 for imagery (T1/T2), 0 for rest (T0)."""
    raw = load_subject_raw(subject, IMAGERY_LEFT_RIGHT, **raw_kwargs)
    ev, eid = mne.events_from_annotations(raw, verbose="ERROR")
    keep = {k: eid[k] for k in ("T0", "T1", "T2") if k in eid}
    ep = mne.Epochs(raw, ev, keep, tmin=0.0, tmax=4.0, baseline=None, picks="eeg",
                    preload=True, verbose="ERROR")
    return ep, (ep.events[:, 2] != eid["T0"]).astype(int)


def bandpow(X, lo, hi, fs=250):
    b, a = butter(4, [lo / (fs / 2), hi / (fs / 2)], "band")
    return np.log(np.var(filtfilt(b, a, X, axis=-1), axis=-1) + 1e-12)


def main():
    args = parse_args()
    run_dir = io.new_run_dir(args.out_base, "pivot", config={"subjects": args.subjects})
    emb_cache = run_dir / "embeddings.npz"
    lines = []
    def log(m): print(m, flush=True); lines.append(m)

    # ── 1. classical ceiling (raw epochs, native channels, 8-30 Hz) ──────────
    Xs, ys, gs = [], [], []
    for s in args.subjects:
        ep, y = imagery_rest_epochs(s, reference="average", l_freq=8., h_freq=30.)
        Xs.append(ep.get_data(copy=False)); ys.append(y); gs.append(np.full(len(y), s))
    X, Y, G = np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs)
    log(f"n={len(args.subjects)} subjects, {len(Y)} trials, balance={np.bincount(Y)}\n")
    log("1) CLASSICAL CEILING (chance 50%)")
    log("   CSP+LDA        within {:.1f}%±{:.1f}".format(*(v*100 for v in D.within_subject(D.csp_lda, X, Y, G))))
    log("   band-power+LDA within {:.1f}%±{:.1f}".format(*(v*100 for v in D.within_subject(D.bandpower_lda, X, Y, G))))
    sc, _, _ = D.cross_subject(D.bandpower_lda, X, Y, G)  # CSP cross is too slow -> band-power
    log(f"   band-power+LDA cross  {sc*100:.1f}%  (CSP cross omitted: ~hours)")

    # ── 2. EEGPT embeddings (aligned 62ch/250Hz), cached ─────────────────────
    from eegxai.models.eegpt import (load_eegpt, eegpt_channel_names,
                                     align_raw_to_eegpt, epochs_to_input, extract_embeddings)
    model = load_eegpt(); chs = eegpt_channel_names()
    Es = []
    for s in args.subjects:
        t = time.time()
        raw = load_subject_raw(s, IMAGERY_LEFT_RIGHT, **EEGPT_RAW_KWARGS)
        raw, _ = align_raw_to_eegpt(raw, chs)
        ev, eid = mne.events_from_annotations(raw, verbose="ERROR")
        keep = {k: eid[k] for k in ("T0", "T1", "T2") if k in eid}
        ep = mne.Epochs(raw, ev, keep, tmin=0.0, tmax=4.0, baseline=None, preload=True, verbose="ERROR")
        Es.append(extract_embeddings(model, epochs_to_input(ep)))
        np.savez(emb_cache, E=np.concatenate(Es), G=G[:sum(len(e) for e in Es)])  # incremental
        log(f"   [emb] subject {s} ({time.time()-t:.0f}s)")
    E = np.concatenate(Es)

    # ── 3. decode embeddings ─────────────────────────────────────────────────
    log("\n2) DECODE EEGPT EMBEDDINGS (chance 50%)")
    log("   within {:.1f}%±{:.1f}".format(*(v*100 for v in D.within_subject(D.logreg, E, Y, G))))
    sc, null, p = D.cross_subject(D.logreg, E, Y, G, n_permutations=args.n_perm)
    log(f"   cross  {sc*100:.1f}% | perm-null {null*100:.1f}% | p={p:.3f}")

    # ── 4. RSA (embedding geometry vs mu/beta neural structure) ──────────────
    neural = []
    for s in args.subjects:
        ep, _ = imagery_rest_epochs(s, **EEGPT_RAW_KWARGS)
        Xd = ep.get_data(copy=False)[..., :1000]
        neural.append(np.concatenate([bandpow(Xd, 8, 13), bandpow(Xd, 13, 30)], axis=1))
    neural = np.concatenate(neural)
    log("\n3) RSA (within-subject Spearman, mean ± sd, p vs 0)")
    for k, (mean, sd, pv) in RSA.summarize(RSA.within_subject_rsa(E, neural, Y, G)).items():
        log(f"   {k:20} {mean:+.3f} ± {sd:.3f}  (p={pv:.1e})")

    (run_dir / "results.txt").write_text("\n".join(lines) + "\n")
    print(f"\nsaved: {run_dir}/results.txt")


if __name__ == "__main__":
    main()

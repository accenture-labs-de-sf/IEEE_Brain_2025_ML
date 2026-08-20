"""Recompute per-channel power spectra for the audit subjects (for the spatial specificity control).

The feature cache stored only a sensorimotor-averaged spectrum, which cannot separate the
sensorimotor mu rhythm from posterior alpha or rule out that averaging over central channels
favored mu. This recomputes the full per-channel PSD from the same cleaned trials, so we can run
the aperiodic-adjusted oscillatory-power RSA per channel (holding the spatial construction fixed
against the band-power double control) and contrast central vs occipital.

Deterministic cleaning (Autoreject, random_state=0) reproduces the exact trials used for the
cached embeddings, so trial order aligns. Resumable and batched like the extraction. Saves one
`psd_perchan/sub-XXX.npz` per subject: log PSD (n, 64, F), freqs, channel names, labels.

Run: python scripts/recompute_psd_perchannel.py --subjects 1-35 --out results/audit_n35
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import mne
from scipy.signal import welch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eegxai.data.physionet_mi import load_subject_raw, IMAGERY_LEFT_RIGHT  # noqa: E402
from eegxai.data.preprocessing import EEGPT_RAW_KWARGS  # noqa: E402
from eegxai.pipeline import BAD_SUBJECTS, FS, N_TIMES  # noqa: E402

mne.set_log_level("ERROR")


def parse_subjects(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-"); out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return [s for s in out if s not in BAD_SUBJECTS]


def recompute_subject(subject, out_dir: Path):
    from autoreject import AutoReject
    raw = load_subject_raw(subject, IMAGERY_LEFT_RIGHT, **EEGPT_RAW_KWARGS)
    ev, eid = mne.events_from_annotations(raw, verbose="ERROR")
    keep = {k: eid[k] for k in ("T0", "T1", "T2") if k in eid}
    ep = mne.Epochs(raw, ev, keep, tmin=0., tmax=4., baseline=None, picks="eeg",
                    preload=True, verbose="ERROR")
    ep = AutoReject(n_interpolate=[1, 4], n_jobs=1, random_state=0, verbose=False).fit_transform(ep)
    y = (ep.events[:, 2] != eid["T0"]).astype(int)
    Xd = ep.get_data(copy=False)[..., :N_TIMES]              # (n, 64, 1000)
    freqs, psd = welch(Xd, fs=FS, nperseg=500, axis=-1)      # (n, 64, F)
    keepf = freqs <= 45
    np.savez(out_dir / f"sub-{subject:03d}.npz",
             psd=np.log(psd[..., keepf] + 1e-20).astype(np.float32),
             freqs=freqs[keepf].astype(np.float32),
             ch_names=np.array(ep.ch_names), Y=y.astype(np.int8),
             subject=np.array(subject))
    return len(y)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", default="1-35")
    ap.add_argument("--out", type=Path, default=Path("results/audit_n35"))
    args = ap.parse_args()
    out_dir = args.out / "psd_perchan"; out_dir.mkdir(parents=True, exist_ok=True)
    subjects = parse_subjects(args.subjects)
    pending = [s for s in subjects if not (out_dir / f"sub-{s:03d}.npz").exists()]
    print(f"[psd] {len(pending)} subjects to recompute -> {out_dir}")
    ok, failed = 0, []
    for s in pending:
        t = time.time()
        try:
            n = recompute_subject(s, out_dir)
            ok += 1
            print(f"[psd] subject {s}: {n} trials ({time.time()-t:.0f}s)", flush=True)
        except Exception as e:
            failed.append(s)
            print(f"[psd] subject {s} FAILED: {e!r}", flush=True)
    print(f"[psd] done: {ok} ok, {len(failed)} failed")


if __name__ == "__main__":
    main()

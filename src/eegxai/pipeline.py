"""Resumable, batched feature extraction for the EEGPT audit.

Design goals (kept deliberately lean, see docs/pivot-analysis.md and reports/draft/README.md):

* **Streamed, not hoarded.** One subject at a time: load raw, clean with Autoreject, extract the
  compact features from the *same* clean trials, write them, and drop the raw arrays. Raw EEG is
  never kept in long-term storage; only per-subject feature files (a few MB each) persist.
* **Resumable.** Each subject writes ``out_dir/sub-XX.npz``. A subject that already has a file is
  skipped, so a run can stop and continue, and a crash loses at most the in-flight subject.
* **Batched with a pause.** ``run_extraction_batch`` processes at most ``batch_size`` pending
  subjects per call, then returns. Call it again to do the next batch. ``summarize`` prints
  progress plus a quick decode/RSA snapshot on whatever is done so far, so you can watch the
  result take shape rather than waiting for all subjects.

Features saved per subject (from the cleaned imagery-vs-rest trials):
  E            (n, 512)   EEGPT mean-pooled embeddings
  Y            (n,)       0 = rest, 1 = imagery
  bp_<band>    (n, 64)    log band power per channel, for several bands incl. controls
  totalpow     (n, 64)    log broadband variance per channel (total-power proxy)
  psd_smc      (n, F)     log PSD averaged over sensorimotor channels (for specparam + the figure)
  psd_freqs    (F,)       PSD frequency axis

Typical use (one batch of 5 at a time):
    from eegxai.pipeline import run_extraction_batch, summarize
    run_extraction_batch(range(1, 36), "results/audit_n35", batch_size=5)
    summarize("results/audit_n35")
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import mne
from scipy.signal import butter, filtfilt, welch

from eegxai.data.physionet_mi import load_subject_raw, IMAGERY_LEFT_RIGHT
from eegxai.data.preprocessing import EEGPT_RAW_KWARGS

mne.set_log_level("ERROR")

# Known-malformed PhysioNet MI subjects (non-standard sampling / annotations); excluded by default.
BAD_SUBJECTS = frozenset({88, 89, 92, 100})

BANDS = {"mu": (8, 13), "beta": (13, 30), "ctl_low": (2, 7), "ctl_high": (30, 40)}
SENSORIMOTOR = ["FC3", "FC1", "FCZ", "FC2", "FC4", "C3", "C1", "CZ", "C2", "C4",
                "CP3", "CP1", "CPZ", "CP2", "CP4"]
FS = 250
N_TIMES = 1000


def _bandpow(X, lo, hi, fs=FS):
    b, a = butter(4, [lo / (fs / 2), hi / (fs / 2)], "band")
    return np.log(np.var(filtfilt(b, a, X, axis=-1), axis=-1) + 1e-12)


def _align_epochs(ep, names):
    """Reorder to EEGPT's channel set, interpolating any the data lacks (PO5/PO6)."""
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


def _extract_subject(subject, model, chs):
    """Return a dict of compact features for one subject (raw is not retained)."""
    from autoreject import AutoReject
    from eegxai.models.eegpt import epochs_to_input, extract_embeddings

    raw = load_subject_raw(subject, IMAGERY_LEFT_RIGHT, **EEGPT_RAW_KWARGS)
    ev, eid = mne.events_from_annotations(raw, verbose="ERROR")
    keep = {k: eid[k] for k in ("T0", "T1", "T2") if k in eid}
    ep = mne.Epochs(raw, ev, keep, tmin=0., tmax=4., baseline=None, picks="eeg",
                    preload=True, verbose="ERROR")
    ep = AutoReject(n_interpolate=[1, 4], n_jobs=1, random_state=0, verbose=False).fit_transform(ep)
    y = (ep.events[:, 2] != eid["T0"]).astype(int)
    Xd = ep.get_data(copy=False)[..., :N_TIMES]

    out = {"Y": y, "subject": np.array(subject)}
    for name, (lo, hi) in BANDS.items():
        out[f"bp_{name}"] = _bandpow(Xd, lo, hi)
    out["totalpow"] = np.log(np.var(Xd, axis=-1) + 1e-12)

    upper = {c.upper(): c for c in ep.ch_names}
    smc = [upper[c] for c in SENSORIMOTOR if c in upper]
    smc_idx = [ep.ch_names.index(c) for c in smc]
    freqs, psd = welch(Xd[:, smc_idx, :], fs=FS, nperseg=500, axis=-1)
    keepf = freqs <= 45
    out["psd_smc"] = np.log(psd[..., keepf].mean(axis=1) + 1e-20)  # (n, F) mean over SMC channels
    out["psd_freqs"] = freqs[keepf]

    out["E"] = extract_embeddings(model, epochs_to_input(_align_epochs(ep, chs)))
    return out


def _pending(subjects, out_dir: Path):
    subs = [s for s in subjects if s not in BAD_SUBJECTS]
    return [s for s in subs if not (out_dir / f"sub-{s:03d}.npz").exists()]


def run_extraction_batch(subjects, out_dir, *, batch_size=5):
    """Process up to ``batch_size`` not-yet-done subjects, writing one npz each. Returns a report."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    subjects = list(subjects)
    pending = _pending(subjects, out_dir)
    batch = pending[:batch_size]
    if not batch:
        print(f"[extract] nothing pending; {len(subjects)} requested, all present in {out_dir}")
        return {"processed": [], "failed": [], "remaining": 0}

    from eegxai.models.eegpt import load_eegpt, eegpt_channel_names
    print(f"[extract] loading EEGPT; batch = {batch}")
    model = load_eegpt(); chs = eegpt_channel_names()

    processed, failed = [], []
    for s in batch:
        t = time.time()
        try:
            feats = _extract_subject(s, model, chs)
            np.savez(out_dir / f"sub-{s:03d}.npz", **feats)
            processed.append(s)
            print(f"[extract] subject {s}: {len(feats['Y'])} clean trials "
                  f"({time.time() - t:.0f}s)  -> sub-{s:03d}.npz")
        except Exception as e:  # keep going; a bad subject should not sink the batch
            failed.append({"subject": s, "error": repr(e)})
            print(f"[extract] subject {s} FAILED after {time.time() - t:.0f}s: {e!r}")

    remaining = len(_pending(subjects, out_dir))
    if failed:
        (out_dir / "failures.json").write_text(json.dumps(failed, indent=2))
    print(f"[extract] batch done: {len(processed)} ok, {len(failed)} failed, {remaining} remaining")
    return {"processed": processed, "failed": failed, "remaining": remaining}


def load_all(out_dir):
    """Concatenate every per-subject npz into stacked arrays keyed like the per-subject files."""
    out_dir = Path(out_dir)
    files = sorted(out_dir.glob("sub-*.npz"))
    if not files:
        raise FileNotFoundError(f"no sub-*.npz in {out_dir}")
    perkey, G = {}, []
    freqs = None
    for f in files:
        d = np.load(f)
        n = len(d["Y"])
        G.append(np.full(n, int(d["subject"])))
        freqs = d["psd_freqs"]
        for k in d.files:
            if k in ("subject", "psd_freqs"):
                continue
            perkey.setdefault(k, []).append(d[k])
    out = {k: np.concatenate(v) for k, v in perkey.items()}
    out["G"] = np.concatenate(G)
    out["psd_freqs"] = freqs
    return out


def summarize(out_dir):
    """Print progress and a quick decode + RSA snapshot on the subjects done so far."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import spearmanr, ttest_1samp
    from eegxai.analysis.rsa import rdm, task_rdm, partial_spearman

    d = load_all(out_dir)
    E, Y, G = d["E"], d["Y"], d["G"]
    subs = np.unique(G)
    print(f"\n[summary] subjects done: {len(subs)}  trials: {len(Y)}  "
          f"(imagery {int((Y==1).sum())}, rest {int((Y==0).sum())})")

    accs = []
    for s in subs:
        m = G == s
        if len(np.unique(Y[m])) < 2 or m.sum() < 10:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        accs.append(cross_val_score(clf, E[m], Y[m], cv=cv).mean())
    print(f"[summary] within-subject decode (logreg, mean over subjects): "
          f"{100*np.mean(accs):.1f}%  (chance 50)")

    tot = d["totalpow"]
    print(f"[summary] RSA partial correlation snapshot (embedding vs band power), n={len(subs)};"
          f" single-control columns, the definitive task+power double-control lives in the analysis step")
    print(f"  {'band':10} {'| task':>8} {'| power':>10}")
    for band in ("mu", "beta", "ctl_low"):
        bp = d[f"bp_{band}"]
        p_task, p_pow = [], []
        for s in subs:
            m = G == s
            if m.sum() < 10:
                continue
            rm, rn, rt, rp = rdm(E[m]), rdm(bp[m]), task_rdm(Y[m]), rdm(tot[m])
            r_mn = spearmanr(rm, rn).correlation
            p_task.append(partial_spearman(r_mn, spearmanr(rm, rt).correlation,
                                            spearmanr(rn, rt).correlation))
            # control for both task and total power via residualizing on the power RDM too
            r_mp = spearmanr(rm, rp).correlation
            r_np = spearmanr(rn, rp).correlation
            p_pow.append(partial_spearman(r_mn, r_mp, r_np))
        pv = ttest_1samp(p_pow, 0).pvalue
        print(f"  {band:10} {np.mean(p_task):>8.3f} {np.mean(p_pow):>10.3f}  (p={pv:.1e})")

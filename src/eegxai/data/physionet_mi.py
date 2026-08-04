"""Memory-conscious ingestion for PhysioNet Motor Imagery (EEGMMIDB).

Designed for laptop / free-Colab RAM budgets: **never hold more than one subject in
memory at a time**. The core entry point is :func:`iter_subject_epochs`, a generator
that loads one subject, yields its epochs as ``float32``, and releases the raw before
moving to the next. Callers should consume each subject (e.g. extract embeddings, which
are tiny) and let the yielded array go out of scope, so peak memory stays ~one subject.

Why this matters
----------------
EEGMMIDB is 109 subjects x up to 14 runs. Loaded eagerly as float64 that is several GB
of RAM — enough to OOM a Colab instance. The generator pattern below keeps peak memory
at roughly one subject's epochs (tens of MB) no matter how many subjects you iterate.

Tips for constrained environments
----------------------------------
- Point MNE's cache at persistent storage on Colab to avoid re-downloading each session:
  ``mne.set_config("MNE_DATA", "/content/drive/MyDrive/mne_data")``.
- Pass ``resample_sfreq`` to shrink arrays early (e.g. to the model's expected rate).
- Keep ``dtype="float32"``; only cast to float64 if a specific analysis needs it.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import Iterator, Sequence

import numpy as np
import mne
from mne.datasets import eegbci

# ── EEGMMIDB run groups ──────────────────────────────────────────────────────
# Within a run, annotations are T0 (rest) plus two task codes (T1/T2) whose meaning
# depends on the run group below.
IMAGERY_LEFT_RIGHT = (4, 8, 12)   # imagine LEFT (T1) vs RIGHT (T2) fist   <- our default
IMAGERY_HANDS_FEET = (6, 10, 14)  # imagine both fists (T1) vs both feet (T2)
EXEC_LEFT_RIGHT = (3, 7, 11)      # executed movement (not imagery)
EXEC_HANDS_FEET = (5, 9, 13)

EXPECTED_SFREQ = 160.0
EXPECTED_N_CHANNELS = 64

# Subjects with documented recording anomalies (differing sampling rate / annotation
# structure) that are commonly excluded. Treat as a hint — we also QC empirically at
# load time rather than trusting this list blindly. Verify against the dataset docs.
KNOWN_ANOMALOUS_SUBJECTS = frozenset({88, 92, 100})

_MONTAGE = "standard_1005"


def load_subject_raw(
    subject: int,
    runs: Sequence[int] = IMAGERY_LEFT_RIGHT,
    *,
    channels: Sequence[str] | None = None,
    resample_sfreq: float | None = None,
    notch_freqs: float | Sequence[float] | None = None,
    l_freq: float | None = None,
    h_freq: float | None = None,
    preload: bool = True,
    verbose: str = "ERROR",
) -> mne.io.BaseRaw:
    """Fetch only the requested runs for one subject and return a tidy Raw.

    Channel names are standardized to 10-05 (EEGPT's name-keyed embeddings expect this)
    and the standard montage is attached. Optional resample/filter are applied in place.

    Memory levers (all optional, all "less is better"):
      * ``channels``      keep only these electrodes (drops the rest *early*, before
                          filter/resample, so everything downstream is leaner).
      * ``resample_sfreq`` lower the rate to shrink the time axis.
    """
    edf_paths = eegbci.load_data(subject, list(runs), update_path=True, verbose=verbose)
    raws = [mne.io.read_raw_edf(f, preload=preload, verbose=verbose) for f in edf_paths]
    raw = mne.concatenate_raws(raws, verbose=verbose)
    del raws

    eegbci.standardize(raw)  # 'Fc5.' -> 'FC5', etc.
    raw.set_montage(mne.channels.make_standard_montage(_MONTAGE), on_missing="warn", verbose=verbose)

    if channels is not None:
        raw.pick(list(channels))  # drop unused electrodes early to stay lean

    if notch_freqs is not None:
        raw.notch_filter(notch_freqs, verbose=verbose)
    if l_freq is not None or h_freq is not None:
        raw.filter(l_freq, h_freq, verbose=verbose)
    if resample_sfreq is not None and abs(resample_sfreq - raw.info["sfreq"]) > 1e-6:
        raw.resample(resample_sfreq, verbose=verbose)
    return raw


@dataclass
class SubjectQC:
    """Lightweight per-subject quality-control summary."""

    subject: int
    sfreq: float
    n_channels: int
    duration_s: float
    trial_counts: dict[str, int]
    frac_high_amp: float           # fraction of samples exceeding the amplitude threshold
    flat_channels: list[str]
    ok: bool
    notes: list[str] = field(default_factory=list)


def quick_qc(
    raw: mne.io.BaseRaw,
    subject: int,
    *,
    amp_threshold_uv: float = 150.0,
    classes: Sequence[str] = ("T1", "T2"),
) -> SubjectQC:
    """Cheap sanity/quality checks — catch subjects that would silently corrupt analysis."""
    notes: list[str] = []
    sfreq = float(raw.info["sfreq"])
    n_ch = len(raw.ch_names)
    if abs(sfreq - EXPECTED_SFREQ) > 1e-3:
        notes.append(f"unexpected sfreq={sfreq}")
    if n_ch != EXPECTED_N_CHANNELS:
        notes.append(f"unexpected n_channels={n_ch}")
    if subject in KNOWN_ANOMALOUS_SUBJECTS:
        notes.append("listed in KNOWN_ANOMALOUS_SUBJECTS")

    data = raw.get_data()  # volts
    frac_high = float(np.mean(np.abs(data) > amp_threshold_uv * 1e-6))
    flat = [raw.ch_names[i] for i in np.where(data.std(axis=1) < 1e-8)[0]]
    if flat:
        notes.append(f"flat channels: {flat}")
    del data

    events, ev_id = mne.events_from_annotations(raw, verbose="ERROR")
    inv = {v: k for k, v in ev_id.items()}
    counts = {inv[c]: int(np.sum(events[:, 2] == c)) for c in np.unique(events[:, 2])}
    for cls in classes:
        if counts.get(cls, 0) == 0:
            notes.append(f"no '{cls}' trials")

    return SubjectQC(
        subject=subject, sfreq=sfreq, n_channels=n_ch, duration_s=float(raw.times[-1]),
        trial_counts=counts, frac_high_amp=frac_high, flat_channels=flat,
        ok=len(notes) == 0, notes=notes,
    )


def epoch_subject(
    raw: mne.io.BaseRaw,
    *,
    tmin: float = -0.5,
    tmax: float = 4.0,
    picks="eeg",
    classes: Sequence[str] = ("T1", "T2"),
    baseline=None,
    dtype: str = "float32",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Epoch one subject's Raw into (X, y, ch_names).

    ``X`` is ``(n_trials, n_channels, n_times)`` as ``dtype`` (float32 by default).
    ``y`` maps the kept classes to 0..k in the order given by ``classes``.
    """
    events, ev_id = mne.events_from_annotations(raw, verbose="ERROR")
    keep = {k: ev_id[k] for k in classes if k in ev_id} or ev_id
    epochs = mne.Epochs(
        raw, events, event_id=keep, tmin=tmin, tmax=tmax, picks=picks,
        baseline=baseline, preload=True, verbose="ERROR",
    )
    X = epochs.get_data(copy=False).astype(dtype, copy=False)
    order = list(keep.keys())
    name_by_code = {v: k for k, v in keep.items()}
    y = np.array([order.index(name_by_code[c]) for c in epochs.events[:, 2]], dtype=np.int64)
    ch_names = list(epochs.ch_names)
    del epochs
    return X, y, ch_names


def iter_subject_epochs(
    subjects: Sequence[int],
    runs: Sequence[int] = IMAGERY_LEFT_RIGHT,
    *,
    tmin: float = -0.5,
    tmax: float = 4.0,
    channels: Sequence[str] | None = None,
    resample_sfreq: float | None = None,
    picks="eeg",
    classes: Sequence[str] = ("T1", "T2"),
    dtype: str = "float32",
    run_qc: bool = True,
    cleanup: bool = True,
) -> Iterator[tuple[int, np.ndarray, np.ndarray, list[str], SubjectQC | None]]:
    """Yield ``(subject, X, y, ch_names, qc)`` one subject at a time, releasing memory.

    This is the memory-safe entry point for whole-dataset loops: only one subject's
    data is resident at once. Consume each yield fully (e.g. extract embeddings) before
    the loop advances. Reconfigure per analysis via ``channels`` (spatial subset),
    ``resample_sfreq`` (temporal), ``tmin``/``tmax`` (window), ``classes`` and ``dtype``
    — every one of these is also a memory lever.
    """
    for subj in subjects:
        raw = load_subject_raw(subj, runs, channels=channels, resample_sfreq=resample_sfreq)
        qc = quick_qc(raw, subj, classes=classes) if run_qc else None
        X, y, ch_names = epoch_subject(
            raw, tmin=tmin, tmax=tmax, picks=picks, classes=classes, dtype=dtype
        )
        del raw
        if cleanup:
            gc.collect()

        yield subj, X, y, ch_names, qc

        del X, y
        if cleanup:
            gc.collect()


# ── Cross-subject access patterns ────────────────────────────────────────────
# Three tiers, from leanest to a guarded eager load. Pick the smallest one that an
# analysis actually needs.

def estimate_epoch_memory(
    n_subjects: int,
    *,
    trials_per_subject: int = 45,
    n_channels: int = EXPECTED_N_CHANNELS,
    n_times: int = 721,
    dtype: str = "float32",
) -> int:
    """Rough bytes to hold ``n_subjects`` worth of epochs *simultaneously* in RAM.

    Approximate (assumes uniform trial counts / window); use it as a pre-flight guard,
    not an exact figure.
    """
    return n_subjects * trials_per_subject * n_channels * n_times * np.dtype(dtype).itemsize


def collect_features(
    subjects: Sequence[int],
    feature_fn,
    runs: Sequence[int] = IMAGERY_LEFT_RIGHT,
    *,
    stack: bool = True,
    **load_kw,
):
    """Lean cross-subject reduction: keep only each subject's (small) feature output.

    ``feature_fn(X, y, ch_names, subject) -> np.ndarray | dict`` is applied per subject;
    the epochs are released before the next subject loads, so peak RAM stays ~one
    subject plus the accumulated (small) features. This is the right tool for
    cross-subject analyses — RSA vectors, per-subject decoding scores, ERD maps — where
    the raised "ceiling" is only for the compact results, never the raw epochs.

    Returns ``(features, subject_ids)``; features are stacked into an array when they are
    same-shaped ndarrays and ``stack=True``, else returned as a list.
    """
    feats, subj_ids = [], []
    for subj, X, y, ch_names, _qc in iter_subject_epochs(subjects, runs, **load_kw):
        feats.append(feature_fn(X, y, ch_names, subj))
        subj_ids.append(subj)
    if stack and feats and isinstance(feats[0], np.ndarray):
        return np.stack(feats), subj_ids
    return feats, subj_ids


def iter_subject_batches(
    subjects: Sequence[int],
    runs: Sequence[int] = IMAGERY_LEFT_RIGHT,
    *,
    batch_size: int = 5,
    cleanup: bool = True,
    **load_kw,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield ``(X, y, subject_ids)`` for ``batch_size`` subjects at a time.

    The tunable middle ground between one-subject streaming and a full eager load: peak
    RAM is bounded to ~``batch_size`` subjects, so you set the ceiling to your budget
    (e.g. ``batch_size=10`` on a laptop, more on a big box). Useful when an analysis needs
    several subjects resident together but not all 109.
    """
    bx: list[np.ndarray] = []
    by: list[np.ndarray] = []
    bid: list[np.ndarray] = []

    def _flush():
        out = (np.concatenate(bx), np.concatenate(by), np.concatenate(bid))
        bx.clear(); by.clear(); bid.clear()
        return out

    for subj, X, y, _ch, _qc in iter_subject_epochs(subjects, runs, cleanup=cleanup, **load_kw):
        bx.append(X); by.append(y); bid.append(np.full(len(y), subj, dtype=np.int64))
        if len(bx) >= batch_size:
            yield _flush()
            if cleanup:
                gc.collect()
    if bx:
        yield _flush()


def collect_trial_features(
    subjects: Sequence[int],
    feature_fn,
    runs: Sequence[int] = IMAGERY_LEFT_RIGHT,
    **load_kw,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reduce each subject to *per-trial* features and stack across **all** subjects.

    ``feature_fn(X, y, ch_names, subject) -> F`` where ``F`` is ``(n_trials, feat_dim)``
    (e.g. EEGPT embeddings). Returns ``(F_all, y_all, subject_ids)``. Peak RAM is
    ~one subject's epochs plus every trial's *reduced* features — the natural tier for
    RSA / probing, where the full dataset is retained but only in its compact form, never
    as raw epochs. Contrast with :func:`collect_features` (one vector *per subject*).
    """
    fs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ids: list[np.ndarray] = []
    for subj, X, y, ch, _qc in iter_subject_epochs(subjects, runs, **load_kw):
        F = np.asarray(feature_fn(X, y, ch, subj))
        fs.append(F); ys.append(y); ids.append(np.full(len(y), subj, dtype=np.int64))
    return np.concatenate(fs), np.concatenate(ys), np.concatenate(ids)


def load_concatenated(
    subjects: Sequence[int],
    runs: Sequence[int] = IMAGERY_LEFT_RIGHT,
    *,
    max_gb: float | None = 2.0,
    dtype: str = "float32",
    n_times_hint: int = 721,
    **load_kw,
):
    """Eagerly stack *all* subjects' epochs into one array — the deliberate high-ceiling
    path for analyses that need every trial resident at once (e.g. a single pooled
    cross-subject decoder).

    Guarded by ``max_gb``: the estimated footprint is checked *before* loading and raises
    if it exceeds the ceiling, so you opt into more RAM on purpose (raise ``max_gb``) or
    fall back to :func:`collect_features` / :func:`iter_subject_epochs`. Set
    ``max_gb=None`` to disable the guard.

    Returns ``(X, y, subject_ids)`` with shapes ``(N, ch, t)``, ``(N,)``, ``(N,)``.
    """
    est = estimate_epoch_memory(len(subjects), n_times=n_times_hint, dtype=dtype)
    if max_gb is not None and est > max_gb * 1e9:
        raise MemoryError(
            f"Estimated {est / 1e9:.2f} GB for {len(subjects)} subjects exceeds "
            f"max_gb={max_gb}. Raise max_gb explicitly, or use collect_features / "
            f"iter_subject_epochs to stay lean."
        )
    Xs, ys, ids = [], [], []
    for subj, X, y, _ch, _qc in iter_subject_epochs(subjects, runs, dtype=dtype, **load_kw):
        Xs.append(X)
        ys.append(y)
        ids.append(np.full(len(y), subj, dtype=np.int64))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ids)

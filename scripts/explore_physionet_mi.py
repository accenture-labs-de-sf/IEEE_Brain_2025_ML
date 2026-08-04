"""Explore PhysioNet Motor Imagery (EEGMMIDB) — ingestion + data-understanding checks.

Deliberately narrow step: confirm we can pull data *partially and gracefully*, that the
memory-conscious loaders behave, and that the data carries the signal we will later ask
the model to be faithful to. No model / no decoding here.

Adds, beyond the basic prints/plots:
  * a per-subject QC scan (sfreq / channels / trial counts / artifact fraction),
  * a mu-band ERD topomap — a preview of the contralateral sensorimotor desync that is
    our downstream "faithfulness" ground truth (expect blue over C4 for LEFT imagery,
    over C3 for RIGHT imagery).

Run:
    python scripts/explore_physionet_mi.py
    python scripts/explore_physionet_mi.py --subject 1 --qc-subjects 1 2 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: save figures, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import mne  # noqa: E402

# Make the src/ package importable without installing it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eegxai.data.physionet_mi import (  # noqa: E402
    IMAGERY_LEFT_RIGHT,
    load_subject_raw,
    quick_qc,
    epoch_subject,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subject", type=int, default=1, help="Subject for plots + ERD.")
    p.add_argument("--runs", type=int, nargs="+", default=list(IMAGERY_LEFT_RIGHT))
    p.add_argument("--qc-subjects", type=int, nargs="+", default=[1, 2, 3],
                   help="Subjects to run the lightweight QC scan over.")
    p.add_argument("--output-dir", type=Path, default=Path("results/exploration"))
    return p.parse_args()


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# ── QC scan (loads one subject at a time, releases memory) ───────────────────
def qc_scan(subjects, runs) -> None:
    section(f"QC scan over subjects {subjects}")
    print(f"  {'subj':>4} {'sfreq':>6} {'ch':>3} {'dur_s':>6} "
          f"{'trials':>14} {'hi-amp%':>7}  notes")
    for subj in subjects:
        raw = load_subject_raw(subj, runs)
        qc = quick_qc(raw, subj)
        trials = ",".join(f"{k}:{v}" for k, v in qc.trial_counts.items() if k in ("T1", "T2"))
        flag = "OK" if qc.ok else "; ".join(qc.notes)
        print(f"  {qc.subject:>4} {qc.sfreq:>6.0f} {qc.n_channels:>3} {qc.duration_s:>6.0f} "
              f"{trials:>14} {qc.frac_high_amp * 100:>6.2f}%  {flag}")
        del raw


# ── Single-subject description ───────────────────────────────────────────────
def describe(raw: mne.io.BaseRaw) -> None:
    section("Recording overview")
    print(f"  sampling rate : {raw.info['sfreq']:.1f} Hz")
    print(f"  duration      : {raw.times[-1]:.1f} s ({raw.n_times} samples)")
    print(f"  n channels    : {len(raw.ch_names)}")

    X, y, ch_names = epoch_subject(raw)
    print(f"  epochs (X)    : {X.shape} {X.dtype}  (trials x channels x samples)")
    print(f"  labels (y)    : counts {dict(zip(*np.unique(y, return_counts=True)))}  "
          f"(0=T1/left, 1=T2/right)")
    print(f"  X memory      : {X.nbytes / 1e6:.1f} MB as {X.dtype} "
          f"(would be {X.nbytes * 2 / 1e6:.1f} MB as float64)")
    del X, y


# ── Mu-band ERD (faithfulness ground-truth preview) ──────────────────────────
def compute_erd_db(raw, band, classes=("T1", "T2"),
                   base_win=(-0.5, 0.0), task_win=(0.5, 3.5), tmin=-0.5, tmax=4.0):
    rb = raw.copy().filter(band[0], band[1], picks="eeg", verbose="ERROR")
    events, ev_id = mne.events_from_annotations(rb, verbose="ERROR")
    keep = {k: ev_id[k] for k in classes if k in ev_id}
    ep = mne.Epochs(rb, events, keep, tmin=tmin, tmax=tmax, baseline=None,
                    picks="eeg", preload=True, verbose="ERROR")
    t = ep.times
    bmask = (t >= base_win[0]) & (t < base_win[1])
    tmask = (t >= task_win[0]) & (t < task_win[1])
    erd = {}
    for cls in keep:
        power = ep[cls].get_data(copy=False) ** 2          # trials x ch x time
        base = power[:, :, bmask].mean(axis=(0, 2))
        task = power[:, :, tmask].mean(axis=(0, 2))
        erd[cls] = 10.0 * np.log10(task / base)            # dB; negative = desync (ERD)
    info = ep.info.copy()
    del ep, rb
    return erd, info


def plot_erd_topomaps(erd, info, band, out: Path) -> None:
    classes = list(erd)
    vmax = max(float(np.abs(v).max()) for v in erd.values())
    label = {"T1": "T1 — imagine LEFT fist", "T2": "T2 — imagine RIGHT fist"}
    fig, axes = plt.subplots(1, len(classes), figsize=(4.2 * len(classes), 4.2))
    axes = np.atleast_1d(axes)
    im = None
    for ax, cls in zip(axes, classes):
        im, _ = mne.viz.plot_topomap(erd[cls], info, axes=ax, show=False,
                                     cmap="RdBu_r", vlim=(-vmax, vmax), contours=4)
        ax.set_title(f"{label.get(cls, cls)}\n{band[0]}–{band[1]} Hz ERD")
    cbar = fig.colorbar(im, ax=list(axes), fraction=0.046, pad=0.08)
    cbar.set_label("dB vs baseline  (blue = desynchronization)")
    path = out / "erd_mu_topomap.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# ── Basic plots ──────────────────────────────────────────────────────────────
def save_basic_plots(raw: mne.io.BaseRaw, out: Path) -> None:
    fig = raw.plot_sensors(show_names=True, show=False)
    fig.savefig(out / "sensors.png", dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {out / 'sensors.png'}")

    fig = raw.compute_psd(fmax=60, verbose="ERROR").plot(show=False)
    fig.savefig(out / "psd.png", dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {out / 'psd.png'}")

    picks = [c for c in ("C3", "Cz", "C4", "Fz", "Pz", "Oz") if c in raw.ch_names]
    sf = raw.info["sfreq"]
    seg = raw.get_data(picks=picks, start=0, stop=int(5 * sf)) * 1e6
    t = np.arange(seg.shape[1]) / sf
    fig, ax = plt.subplots(figsize=(10, 5))
    step = float(np.nanmax(np.abs(seg))) * 1.2 or 1.0
    for i, ch in enumerate(picks):
        ax.plot(t, seg[i] - i * step, lw=0.6)
        ax.text(-0.02 * t[-1], -i * step, ch, ha="right", va="center", fontsize=9)
    ax.set_xlabel("time (s)"); ax.set_yticks([])
    ax.set_title(f"Raw segment (first 5 s), {len(picks)} sensorimotor channels")
    fig.savefig(out / "raw_segment.png", dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {out / 'raw_segment.png'}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    qc_scan(args.qc_subjects, args.runs)

    section(f"Detailed look — subject {args.subject}")
    raw = load_subject_raw(args.subject, args.runs)
    describe(raw)

    section("Saving plots")
    save_basic_plots(raw, args.output_dir)
    for band, name in [((8, 13), "mu")]:
        erd, info = compute_erd_db(raw, band)
        plot_erd_topomaps(erd, info, band, args.output_dir)

    section("Done")
    print(f"  Figures in: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

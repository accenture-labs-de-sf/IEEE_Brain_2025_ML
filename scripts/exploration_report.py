"""Cross-participant empirical characterization via the canonical MNE ERDS pipeline.

Method (Pfurtscheller & Lopes da Silva 1999; MNE ERDS-maps example):
multitaper TFR (freqs 2-35 Hz, n_cycles=freqs) -> percent-change baseline (-1..0 s) ->
(a) ERDS time-frequency maps at C3/Cz/C4 and (b) mu-band (8-13 Hz) topographies. Percent
< 0 = ERD (desync). No decoding / no model here.

Run:
    python scripts/exploration_report.py --subjects 1 2 3 4 5 6 7 8 9 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eegxai.data.physionet_mi import IMAGERY_LEFT_RIGHT, load_subject_raw, quick_qc  # noqa: E402
from eegxai.data.preprocessing import EEGPT_RAW_KWARGS, EEGPT_SFREQ  # noqa: E402
from eegxai.analysis.erd import (  # noqa: E402
    MU_BAND, DEFAULT_FREQS, BASELINE, make_epochs, class_tfrs, band_topography,
)
from eegxai.analysis.stats import cluster_test_map  # noqa: E402
from eegxai.viz.topo import plot_erd_grid  # noqa: E402
from eegxai.viz.tfr import plot_erds_maps  # noqa: E402
from eegxai import io  # noqa: E402

CLASSES = ("T1", "T2")
MAP_CHANNELS = ("C3", "Cz", "C4")
TASK_WIN = (0.5, 3.5)
DECIM = 3
EPOCH_WIN = (-2.0, 4.5)   # buffered epoch — absorbs multitaper edge ringing (also cleans baseline)
CROP_WIN = (-1.0, 3.9)    # analysis/display window kept after cropping the buffer


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--runs", type=int, nargs="+", default=list(IMAGERY_LEFT_RIGHT))
    p.add_argument("--out-base", type=Path, default=Path("results/exploration"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = {
        "dataset": "PhysioNet EEGMMIDB (imagery)",
        "subjects": args.subjects, "runs": args.runs,
        "preprocessing_eegpt": EEGPT_RAW_KWARGS,
        "method": "MNE ERDS: multitaper TFR + percent baseline",
        "freqs_hz": [float(DEFAULT_FREQS[0]), float(DEFAULT_FREQS[-1])],
        "n_cycles": "= freqs", "baseline_s": list(BASELINE), "baseline_mode": "percent",
        "decim": DECIM, "mu_band_hz": list(MU_BAND), "task_win_s": list(TASK_WIN),
        "epoch_win_s": list(EPOCH_WIN), "crop_win_s": list(CROP_WIN),
        "classes": {"T1": "imagine LEFT fist", "T2": "imagine RIGHT fist"},
    }
    run_dir = io.new_run_dir(args.out_base, "mi_erds", config=cfg)
    figdir = run_dir / "figures"

    qc_rows: list[dict] = []
    mu_topo_by_subj: list[tuple[str, dict[str, np.ndarray]]] = []
    map_accum: dict[tuple[str, str], list[np.ndarray]] = {}
    ref_info = ref_ch = freqs = times = None

    for subj in args.subjects:
        raw = load_subject_raw(subj, args.runs, **EEGPT_RAW_KWARGS)
        qc = quick_qc(raw, subj, expected_sfreq=EEGPT_SFREQ)
        qc_rows.append({
            "subject": qc.subject, "sfreq": qc.sfreq, "n_channels": qc.n_channels,
            "T1": qc.trial_counts.get("T1", 0), "T2": qc.trial_counts.get("T2", 0),
            "high_amp_pct": round(qc.frac_high_amp * 100, 2), "ok": qc.ok,
        })

        epochs = make_epochs(raw, classes=CLASSES, tmin=EPOCH_WIN[0], tmax=EPOCH_WIN[1])
        tfrs = class_tfrs(epochs, decim=DECIM)
        for _c in tfrs:  # crop the buffer off after baseline correction
            tfrs[_c].crop(tmin=CROP_WIN[0], tmax=CROP_WIN[1])

        mu = {}
        for cls in CLASSES:
            tfr = tfrs[cls]
            if ref_info is None:
                ref_info, ref_ch = tfr.info, list(tfr.ch_names)
                freqs, times = tfr.freqs, tfr.times
                raw.plot_sensors(show_names=True, show=False).savefig(
                    figdir / "sensors.png", dpi=120, bbox_inches="tight")
                raw.compute_psd(fmax=45, verbose="ERROR").plot(show=False).savefig(
                    figdir / "psd.png", dpi=120, bbox_inches="tight")
                plt.close("all")
            # MNE "percent" baseline returns a ratio; x100 -> true percent for display.
            mu[cls] = band_topography(tfr, MU_BAND, tmin=TASK_WIN[0], tmax=TASK_WIN[1]) * 100.0
            for ch in MAP_CHANNELS:
                arr = tfr.data[ref_ch.index(ch)] * 100.0  # (freq, time), percent
                map_accum.setdefault((cls, ch), []).append(arr)
        mu_topo_by_subj.append((f"S{subj}", mu))
        del tfrs, epochs, raw

    # ── group means ──────────────────────────────────────────────────────────
    group_mu = {c: np.mean([mu[c] for _l, mu in mu_topo_by_subj], axis=0) for c in CLASSES}
    group_maps = {k: np.mean(v, axis=0) for k, v in map_accum.items()}

    # group cluster-permutation significance per (class, channel): where is ERD/ERS reliable?
    sig: dict[tuple[str, str], np.ndarray] = {}
    cluster_p: dict[tuple[str, str], float] = {}
    for key, arrs in map_accum.items():
        sig[key], cluster_p[key] = cluster_test_map(np.stack(arrs))

    # lateralization index (mu, C3 - C4): LEFT expect >0, RIGHT expect <0
    i3, i4 = ref_ch.index("C3"), ref_ch.index("C4")
    li = {c: float(group_mu[c][i3] - group_mu[c][i4]) for c in CLASSES}

    # ── save data ────────────────────────────────────────────────────────────
    rows = []
    for label, mu in mu_topo_by_subj + [("group", group_mu)]:
        subj = label if label == "group" else label[1:]
        for cls in CLASSES:
            for ch, val in zip(ref_ch, mu[cls]):
                rows.append({"subject": subj, "class": cls, "channel": ch,
                             "mu_pct": round(float(val), 2)})
    stat_rows = [{"class": c, "channel": ch,
                  "min_cluster_p": round(cluster_p[(c, ch)], 4),
                  "significant": bool(sig[(c, ch)].any())}
                 for c in CLASSES for ch in MAP_CHANNELS]
    qc_df = io.save_table(qc_rows, run_dir / "qc.csv")
    io.save_table(rows, run_dir / "erd_mu.csv")
    stat_df = io.save_table(stat_rows, run_dir / "cluster_stats.csv")
    io.save_arrays(run_dir / "erds.npz",
                   ch_names=np.array(ref_ch), freqs=freqs, times=times,
                   **{f"group_mu_{c}": group_mu[c] for c in CLASSES},
                   **{f"map_{c}_{ch}": group_maps[(c, ch)] for c in CLASSES for ch in MAP_CHANNELS},
                   **{f"sig_{c}_{ch}": sig[(c, ch)] for c in CLASSES for ch in MAP_CHANNELS})

    # ── figures ──────────────────────────────────────────────────────────────
    plot_erds_maps(group_maps, freqs, times, MAP_CHANNELS, CLASSES,
                   figdir / "erds_maps.png", sig=sig)
    plot_erd_grid([("Group mean", group_mu)], CLASSES, ref_info, MU_BAND,
                  figdir / "mu_topomap_group.png")
    plot_erd_grid(mu_topo_by_subj + [("Group mean", group_mu)], CLASSES, ref_info, MU_BAND,
                  figdir / "mu_topomap_bysubject.png")

    # ── report ───────────────────────────────────────────────────────────────
    summary = f"""
_Generated {run_dir.name}._ Dataset: **PhysioNet EEGMMIDB** (motor imagery, runs {args.runs}),
subjects **{args.subjects}** (n={len(args.subjects)}).

**Method (canonical).** MNE ERDS-maps recipe (Pfurtscheller & Lopes da Silva 1999): per-epoch
**multitaper** TFR (freqs 2–35 Hz, `n_cycles = freqs`), **percent-change** baseline (−1→0 s),
`decim={DECIM}`. Read-outs: ERDS time-frequency maps at C3/Cz/C4 and mu-band (8–13 Hz)
topographies over the {TASK_WIN[0]}–{TASK_WIN[1]} s task window. Epochs are buffered
({EPOCH_WIN[0]}–{EPOCH_WIN[1]} s) and cropped to {CROP_WIN[0]}–{CROP_WIN[1]} s after baseline to
remove multitaper edge ringing. Preprocessing matches EEGPT (average ref, 0–38 Hz, 256 Hz).
Convention: **percent < 0 = ERD** (desync).

**Result — mu lateralization (group, C3 − C4).**
- LEFT imagery (T1): **{li['T1']:+.1f}%** (expect > 0 → contralateral C4 desync)
- RIGHT imagery (T2): **{li['T2']:+.1f}%** (expect < 0 → contralateral C3 desync)

Central sensorimotor mu ERD is present during imagery; the ERDS maps show the mu/beta
suppression time course at C3/Cz/C4.

**Significance (cluster-permutation, group n={len(args.subjects)}).** Two-sided one-sample cluster
test vs baseline per channel/class; outlined regions on the ERDS maps are significant (p<0.05,
corrected for the many time-frequency comparisons). Min cluster p — C3 during RIGHT imagery:
{cluster_p[('T2', 'C3')]:.3f}; C4 during LEFT imagery: {cluster_p[('T1', 'C4')]:.3f}. Full table
in `cluster_stats.csv`.

This is the empirical reference for later model comparison.

Artifacts: `qc.csv`, `erd_mu.csv`, `cluster_stats.csv`, `erds.npz`, figures; `config.json` records
exact parameters and package versions.
""".strip()

    figures = [
        ("Sensor montage (64-ch, 10-10)", figdir / "sensors.png"),
        ("Power spectral density (0–45 Hz)", figdir / "psd.png"),
        ("ERDS maps at C3/Cz/C4 (group; red = ERD)", figdir / "erds_maps.png"),
        ("Mu-band ERD topography — group mean (red = ERD)", figdir / "mu_topomap_group.png"),
    ]
    md_path, pdf_path = io.write_report(
        run_dir, "Empirical ERDS Characterization — Motor Imagery", summary,
        tables=[("Per-subject QC", qc_df), ("Cluster-permutation significance", stat_df)],
        figures=figures,
    )
    print(f"Run folder : {run_dir}")
    print(f"  lateralization index (C3-C4 mu): LEFT {li['T1']:+.1f}%  RIGHT {li['T2']:+.1f}%")
    print(f"  cluster p  : C3/RIGHT {cluster_p[('T2', 'C3')]:.3f}  C4/LEFT {cluster_p[('T1', 'C4')]:.3f}")
    print(f"  report     : {md_path.name}, {pdf_path.name}")


if __name__ == "__main__":
    main()

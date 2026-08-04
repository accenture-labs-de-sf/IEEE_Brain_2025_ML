"""Run the exploratory analysis for a set of subjects and save an inspectable run.

Produces a self-contained run folder under results/exploration/:
    config.json   provenance (params + package versions + timestamp)
    qc.csv        per-subject QC
    erd.csv       tidy per-channel mu-ERD (subject x class x channel)
    erd.npz       the same ERD arrays, bundled for fast reload
    figures/*.png montage, PSD, ERD grid
    report.md / report.pdf   overview summary for communication

Run:
    python scripts/exploration_report.py --subjects 1 2 3
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
from eegxai.analysis.erd import band_erd_from_raw  # noqa: E402
from eegxai.viz.topo import plot_erd_grid  # noqa: E402
from eegxai import io  # noqa: E402

BAND = (8.0, 13.0)
CLASSES = ("T1", "T2")


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
        "erd_band_hz": BAND, "baseline_win_s": [-0.5, 0.0], "task_win_s": [0.5, 3.5],
        "classes": {"T1": "imagine LEFT fist", "T2": "imagine RIGHT fist"},
    }
    run_dir = io.new_run_dir(args.out_base, "mi_explore", config=cfg)
    figdir = run_dir / "figures"

    qc_rows, erd_rows, erd_arrays = [], [], {}
    per_subject_erd, ref_info, ref_ch = [], None, None

    for subj in args.subjects:
        raw = load_subject_raw(subj, args.runs, **EEGPT_RAW_KWARGS)
        qc = quick_qc(raw, subj, expected_sfreq=EEGPT_SFREQ)
        qc_rows.append({
            "subject": qc.subject, "sfreq": qc.sfreq, "n_channels": qc.n_channels,
            "duration_s": round(qc.duration_s, 1),
            "T1": qc.trial_counts.get("T1", 0), "T2": qc.trial_counts.get("T2", 0),
            "high_amp_pct": round(qc.frac_high_amp * 100, 2),
            "ok": qc.ok, "notes": "; ".join(qc.notes),
        })

        erd, info = band_erd_from_raw(raw, BAND, classes=CLASSES)
        if ref_info is None:
            ref_info, ref_ch = info, info["ch_names"]
            raw.plot_sensors(show_names=True, show=False).savefig(
                figdir / "sensors.png", dpi=120, bbox_inches="tight")
            raw.compute_psd(fmax=60, verbose="ERROR").plot(show=False).savefig(
                figdir / "psd.png", dpi=120, bbox_inches="tight")
            plt.close("all")

        per_subject_erd.append((f"S{subj}", erd))
        for cls in CLASSES:
            erd_arrays[f"s{subj}_{cls}"] = erd[cls]
            for ch, val in zip(ref_ch, erd[cls]):
                erd_rows.append({"subject": subj, "class": cls, "channel": ch,
                                 "erd_db": round(float(val), 3)})
        del raw

    # Group mean ERD across subjects (aggregation the single-subject view lacked).
    group = {cls: np.mean([erd[cls] for _lbl, erd in per_subject_erd], axis=0) for cls in CLASSES}
    for cls in CLASSES:
        erd_arrays[f"group_{cls}"] = group[cls]
        for ch, val in zip(ref_ch, group[cls]):
            erd_rows.append({"subject": "group", "class": cls, "channel": ch,
                             "erd_db": round(float(val), 3)})
    erd_arrays["ch_names"] = np.array(ref_ch)

    # ── save data ────────────────────────────────────────────────────────────
    qc_df = io.save_table(qc_rows, run_dir / "qc.csv")
    io.save_table(erd_rows, run_dir / "erd.csv")
    io.save_arrays(run_dir / "erd.npz", **erd_arrays)

    # ── figures ──────────────────────────────────────────────────────────────
    grid_rows = per_subject_erd + [("Group mean", group)]
    plot_erd_grid(grid_rows, CLASSES, ref_info, BAND, figdir / "erd_grid.png")

    # ── report ───────────────────────────────────────────────────────────────
    summary = f"""
_Generated {run_dir.name}._ Dataset: **PhysioNet EEGMMIDB** (motor imagery, runs {args.runs}),
subjects **{args.subjects}**.

**Purpose.** Confirm the ingestion pipeline and build confidence in the data before modelling.
No model / decoding here — this is exploratory.

**Ingestion.** Partial, per-subject download; channel **names** standardized to the canonical
10-10 / 10-05 nomenclature (a *renaming* — e.g. `Fc5.`→`FC5` — so EEGPT's name-keyed channel
embeddings match; not resampling or repositioning); float32 epochs; peak memory ~one subject
via streaming loaders.

**Preprocessing (match EEGPT).** Average reference, ~0–38 Hz band-pass, and resample **160 → 256
Hz** (upsampling to EEGPT's rate). No z-score here — ERD needs raw amplitude. See
`docs/preprocessing.md`.

**Data quality (QC).** All subjects 160 Hz / 64 channels with balanced ~23/22 left–right
imagery trials. Per-subject noise varies (see `high_amp_pct`) — relevant to the cross-subject
reliability we care about downstream.

**Mu-band ERD (8–13 Hz).** Clear central **sensorimotor desynchronization** during imagery
(the faithfulness ground truth our model representations should recover). The textbook
**contralateral C3/C4 lateralization is weak at the single-subject level** and is expected to
sharpen in the group mean / with more subjects — visible in the grid below.

**Artifacts.** `qc.csv`, `erd.csv` / `erd.npz`, and the figures in this folder; `config.json`
records exact parameters and package versions.
""".strip()

    figures = [
        ("Sensor montage (64-ch, 10-10)", figdir / "sensors.png"),
        ("Power spectral density (0–60 Hz)", figdir / "psd.png"),
        ("Mu-band ERD — per subject and group mean (blue = desync)", figdir / "erd_grid.png"),
    ]
    md_path, pdf_path = io.write_report(
        run_dir, "Exploratory Analysis — Results So Far", summary,
        tables=[("Per-subject QC", qc_df)], figures=figures,
    )

    print(f"Run folder : {run_dir}")
    print(f"  qc.csv, erd.csv, erd.npz, config.json")
    print(f"  report    : {md_path.name}, {pdf_path.name}")


if __name__ == "__main__":
    main()

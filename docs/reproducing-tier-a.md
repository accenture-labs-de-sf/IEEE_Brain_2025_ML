# Reproducing Tier A — Empirical ERDS Reference

Step-by-step for collaborators to reproduce the motor-imagery empirical characterization
(the mu/beta ERD reference). **Tier A uses no model** — it's the neurophysiology yardstick that
Tier B (the EEGPT analysis) is later compared against.

If you only need Tier A, you do **not** need `torch`/`braindecode` (those are Tier B).

## 1. Prerequisites

- **Python ≥ 3.10** (developed on 3.13).
- `git`, and ~**1 GB** free disk for the dataset cache (`~/mne_data`).
- No manual data download — MNE fetches only the subjects/runs you request, on first run.

## 2. Setup

```bash
git clone https://github.com/accenture-labs-de-sf/IEEE_Brain_2025_ML.git
cd IEEE_Brain_2025_ML

python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
# Tier-A-only (lean): 
pip install "mne>=1.6" moabb numpy scipy matplotlib pandas markdown xhtml2pdf pillow
# ...or install everything (also pulls Tier B's torch/braindecode):
# pip install -r requirements.txt
```

## 3. Run

```bash
# quick smoke of your setup (~2 min, subjects 1-3):
python scripts/exploration_report.py --subjects 1 2 3

# the full reference (~12 min incl. first-time downloads, subjects 1-20):
python scripts/exploration_report.py --subjects 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
```

Each run writes a self-contained, timestamped folder under `results/exploration/` (git-ignored):
`config.json` (params + versions), `qc.csv`, `erd_mu.csv`, `cluster_stats.csv`, `erds.npz`,
`figures/`, and `report.md` / `report.pdf`.

## 4. Verify against the reference

Compare your run to the committed snapshot [`reports/mi_erds_n20/`](../reports/mi_erds_n20). The
analysis is deterministic (fixed permutation seed = 0), so numbers should match closely:

| Check | Expected (n=20) |
| --- | --- |
| QC | 20 subjects, all `ok`, ~455/445 left/right trials |
| Cluster significance | **all six** channel×class ERDs `p < 0.05` |
| Strongest effect | C3 during RIGHT imagery, `p ≈ 0.001` |
| C4 during LEFT | `p ≈ 0.020` (significant) |
| Lateralization (mu C3−C4) | RIGHT **≈ −6.7%**, LEFT **≈ +0.6%** |

If your numbers differ materially, that's a finding — raise it (don't paper over it).

## 5. Notes

- **Reproducibility:** permutation seed is fixed; `config.json` records exact parameters and package
  versions for each run.
- **Memory:** the pipeline streams **one subject at a time** (peak RAM ≈ one subject). See
  [`research-plan.md`](research-plan.md) → *Data engineering & memory strategy* and
  `src/eegxai/data/physionet_mi.py` for the loader tiers.
- **Method + references:** [`analysis-methods.md`](analysis-methods.md) (concepts, pipeline,
  canonical + recent citations). Preprocessing rationale: [`preprocessing.md`](preprocessing.md).
  Open options: [`findings-and-options.md`](findings-and-options.md).

## 6. For the contributor branches (see `internal/collaboration.md`)

- **RE (`perf/local-gpu`):** reproduce the numbers above on your workstation, then optimize /
  scale (GPU, larger n) *behind the same interfaces* — parity against this reference is the check.
- **PhD (`guide/reproducible-notebook`):** wrap this same run in a narrated Colab notebook (import
  the package code so it stays in sync), showing the steps and outputs for external readers.

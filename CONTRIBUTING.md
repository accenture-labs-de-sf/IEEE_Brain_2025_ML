# Contributing

Short guide for collaborators on this project.

## Getting set up

1. Use **Python ≥ 3.10** in a fresh virtual environment.
2. `pip install -r requirements.txt`.
3. Reusable logic lives in `src/eegxai/`; use `notebooks/` only for exploration, and promote
   anything worth keeping into the package.

## Where things go

| You're adding… | Put it in… |
| --- | --- |
| A dataset loader / preprocessing step | `src/eegxai/data/` |
| EEGPT loading or embedding extraction | `src/eegxai/models/` |
| RSA, probing, attribution, stats | `src/eegxai/analysis/` |
| Plotting (topomaps, RDMs, figures) | `src/eegxai/viz/` |
| An experiment config | `configs/` |
| Design notes / decisions | `docs/` |

## Data

**Never commit data or model weights** — they are git-ignored (see `.gitignore`). Datasets
download on first use; document any manual sources in `data/README.md`.

## Git hygiene

- Branch off `main`; open a PR rather than pushing to `main` directly.
- Keep notebooks clean of large outputs before committing (clear outputs, or export to `docs/`).
- Write clear commit messages describing the *why*.

## Reproducibility

- Pin random seeds in experiments.
- Record dataset release/version and preprocessing parameters in the experiment config.

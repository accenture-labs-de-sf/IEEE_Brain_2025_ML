# Reports

**Curated, version-controlled snapshots** of selected analysis runs — promoted here manually.

Live run outputs go to `results/` (git-ignored, regenerable). Only reports intentionally chosen
for sharing are copied into this folder and committed, so the repo stays lean while collaborators
can read the results without re-running.

Each snapshot keeps `report.pdf` (self-contained), `report.md` + `figures/` (renders on GitHub),
and `config.json` (parameters + package versions for provenance). To reproduce a run, use the
config with `scripts/exploration_report.py`.

| Snapshot | Run | Summary |
| --- | --- | --- |
| `mi_erds_n20/` | canonical MNE ERDS + cluster stats (edge-cropped), PhysioNet MI, subjects 1–20 | all 6 channel×class ERDs significant (p<0.05); RIGHT contralateral (−6.7%), LEFT weak bias, C4/LEFT now significant |
| `pivot/` | imagery-vs-rest decode + band-resolved RSA on EEGPT embeddings, n=20 | EEGPT decodes ≈CSP within / best cross; representation is spectrally non-specific (mu ≈ control band) — see `docs/pivot-analysis.md` |

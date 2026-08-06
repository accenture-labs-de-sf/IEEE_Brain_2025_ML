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
| `mi_erds_n10/` | canonical MNE ERDS, PhysioNet MI, subjects 1–10 | mu/beta ERD present; RIGHT lateralized (−8.9%), LEFT weak (+0.7%) |

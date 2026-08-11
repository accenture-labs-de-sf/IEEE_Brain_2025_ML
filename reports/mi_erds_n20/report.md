# Empirical ERDS Characterization — Motor Imagery

_Generated mi_erds_20260806T235629Z._ Dataset: **PhysioNet EEGMMIDB** (motor imagery, runs [4, 8, 12]),
subjects **[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]** (n=20).

**Method (canonical).** MNE ERDS-maps recipe (Pfurtscheller & Lopes da Silva 1999): per-epoch
**multitaper** TFR (freqs 2–35 Hz, `n_cycles = freqs`), **percent-change** baseline (−1→0 s),
`decim=3`. Read-outs: ERDS time-frequency maps at C3/Cz/C4 and mu-band (8–13 Hz)
topographies over the 0.5–3.5 s task window. Epochs are buffered
(-2.0–4.5 s) and cropped to -1.0–3.9 s after baseline to
remove multitaper edge ringing. Preprocessing matches EEGPT (average ref, 0–38 Hz, 256 Hz).
Convention: **percent < 0 = ERD** (desync).

**Result — mu lateralization (group, C3 − C4).**
- LEFT imagery (T1): **+0.6%** (expect > 0 → contralateral C4 desync)
- RIGHT imagery (T2): **-6.7%** (expect < 0 → contralateral C3 desync)

Central sensorimotor mu ERD is present during imagery; the ERDS maps show the mu/beta
suppression time course at C3/Cz/C4.

**Significance (cluster-permutation, group n=20).** Two-sided one-sample cluster
test vs baseline per channel/class; outlined regions on the ERDS maps are significant (p<0.05,
corrected for the many time-frequency comparisons). Min cluster p — C3 during RIGHT imagery:
0.001; C4 during LEFT imagery: 0.020. Full table
in `cluster_stats.csv`.

This is the empirical reference for later model comparison.

Artifacts: `qc.csv`, `erd_mu.csv`, `cluster_stats.csv`, `erds.npz`, figures; `config.json` records
exact parameters and package versions.

## Per-subject QC

| subject | sfreq | n_channels | T1 | T2 | high_amp_pct | ok |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 256.0 | 64 | 23 | 22 | 0.33 | True |
| 2 | 256.0 | 64 | 23 | 22 | 0.07 | True |
| 3 | 256.0 | 64 | 23 | 22 | 2.53 | True |
| 4 | 256.0 | 64 | 23 | 22 | 0.99 | True |
| 5 | 256.0 | 64 | 21 | 24 | 0.05 | True |
| 6 | 256.0 | 64 | 24 | 21 | 0.04 | True |
| 7 | 256.0 | 64 | 23 | 22 | 0.28 | True |
| 8 | 256.0 | 64 | 22 | 23 | 0.29 | True |
| 9 | 256.0 | 64 | 24 | 21 | 5.52 | True |
| 10 | 256.0 | 64 | 24 | 21 | 11.7 | True |
| 11 | 256.0 | 64 | 23 | 22 | 0.06 | True |
| 12 | 256.0 | 64 | 21 | 24 | 0.9 | True |
| 13 | 256.0 | 64 | 23 | 22 | 2.49 | True |
| 14 | 256.0 | 64 | 22 | 23 | 0.13 | True |
| 15 | 256.0 | 64 | 23 | 22 | 0.47 | True |
| 16 | 256.0 | 64 | 22 | 23 | 0.31 | True |
| 17 | 256.0 | 64 | 23 | 22 | 7.77 | True |
| 18 | 256.0 | 64 | 22 | 23 | 0.0 | True |
| 19 | 256.0 | 64 | 23 | 22 | 0.67 | True |
| 20 | 256.0 | 64 | 23 | 22 | 0.0 | True |

## Cluster-permutation significance

| class | channel | min_cluster_p | significant |
| --- | --- | --- | --- |
| T1 | C3 | 0.0117 | True |
| T1 | Cz | 0.0371 | True |
| T1 | C4 | 0.0195 | True |
| T2 | C3 | 0.001 | True |
| T2 | Cz | 0.0029 | True |
| T2 | C4 | 0.0059 | True |

## Figures

**Sensor montage (64-ch, 10-10)**

![Sensor montage (64-ch, 10-10)](figures/sensors.png)

**Power spectral density (0–45 Hz)**

![Power spectral density (0–45 Hz)](figures/psd.png)

**ERDS maps at C3/Cz/C4 (group; red = ERD)**

![ERDS maps at C3/Cz/C4 (group; red = ERD)](figures/erds_maps.png)

**Mu-band ERD topography — group mean (red = ERD)**

![Mu-band ERD topography — group mean (red = ERD)](figures/mu_topomap_group.png)

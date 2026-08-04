# Data

**Nothing in this directory is version-controlled** (see repo `.gitignore`). Datasets download on
first use into `data/raw/`, and preprocessed/windowed artifacts are written to `data/processed/`.

## Layout

| Folder | Contents |
| --- | --- |
| `raw/` | As-downloaded EEG (EDF/BDF/FIF, MOABB caches). Do not edit in place. |
| `processed/` | Preprocessed, resampled, windowed arrays ready for the model. |
| `external/` | Third-party artifacts (e.g. pretrained weights if pulled manually). |

## Datasets used

### 1. PhysioNet EEG Motor Movement/Imagery (EEGMMIDB) — *core*
- 109 subjects, 64-channel (10-10), BCI2000 recording.
- Labels: rest / left fist / right fist / both fists / both feet (imagery & execution).
- Ground truth: contralateral **mu (8–13 Hz) / beta (13–30 Hz) ERD** over sensorimotor cortex.
- Load via MNE: `mne.datasets.eegbci`. **Standardize channel names** with
  `mne.datasets.eegbci.standardize()` (BCI2000 names like `Fc5.` → proper 10-05) so EEGPT's
  name-keyed channel embeddings match.

### 2. SSVEP dataset (via MOABB) — *second axis*
- Provides a spectral / frequency-tagged code (occipital response locked to stimulus frequency).
- Load via `moabb.datasets` (auto-download). Exact dataset TBD (e.g. a MOABB SSVEP benchmark).

## Notes
- Both datasets are *in-distribution* for EEGPT's pretraining regime (BCI-style paradigms).
- Resample to a single target `sfreq` and keep signals relatively raw (EEGPT was pretrained with
  mask-based reconstruction on raw waveforms).

## Loading (memory-conscious API)

`src/eegxai/data/physionet_mi.py` provides five access tiers — use the leanest that works
(see `docs/research-plan.md` → *Data engineering & memory strategy* for the rationale):

| Function | Use when | Peak RAM |
| --- | --- | --- |
| `iter_subject_epochs(subjects, …)` | streaming over many subjects (extract → keep only the small result) | ~1 subject |
| `collect_features(subjects, feature_fn, …)` | cross-subject, one vector **per subject** (RSA, scores) | ~1 subject + small features |
| `iter_subject_batches(subjects, …, batch_size=K)` | several subjects resident together, tunable ceiling | ~K subjects |
| `collect_trial_features(subjects, feature_fn, …)` | **all** subjects as per-trial features/embeddings (RSA/probing) | ~1 subject + all reduced features |
| `load_concatenated(subjects, …, max_gb=2.0)` | all raw epochs resident at once; **guarded** by a memory estimate | all subjects (opt-in) |

Shared knobs (also memory levers): `channels` (electrode subset, dropped early), `resample_sfreq`,
`tmin`/`tmax`, `classes`, `dtype` (float32 default). `estimate_epoch_memory(n_subjects, …)` gives a
pre-flight footprint (~8 MB/subject float32; ~0.9 GB for all 109).

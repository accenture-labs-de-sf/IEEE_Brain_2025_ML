# Preprocessing

Authoritative record of the preprocessing we apply, and why. Implemented in
`src/eegxai/data/preprocessing.py` and `src/eegxai/data/physionet_mi.py`.

## Principle: match EEGPT

We keep preprocessing **as close as possible to EEGPT's own recipe** so the model receives
in-distribution input. This is reinforced by the fact that **PhysioNet MI (EEGMMIDB) is one of
EEGPT's *pretraining* datasets** — the model has already seen this data's distribution, so
deviating from its preprocessing only moves us away from what it expects. Heavy custom
cleaning is avoided for the same reason.

## Two stages (kept separate on purpose)

### 1. Raw-level — shared by model input *and* the empirical reference
Applied through the loaders: `load_subject_raw(subj, runs, **EEGPT_RAW_KWARGS)`.

| Step | Value | Notes |
| --- | --- | --- |
| Channel-name standardization | 10-10 / 10-05 canonical | `eegbci.standardize` — a *renaming* (`Fc5.`→`FC5`) so EEGPT's name-keyed embeddings match |
| Reference | **average** | global average reference |
| Band-pass | **~0–38 Hz** (low-pass 38) | matches EEGPT's MI setting; also suppresses 60 Hz line noise (notch likely unnecessary) |
| Resample | **256 Hz** | our data is 160 Hz → we **upsample 160 → 256** (expected, not optional) |

### 2. Normalization — model input ONLY
`normalize_epochs(X)` = per-window z-score (per epoch, per channel, over time).

> ⚠️ **Never z-score before ERD / band-power analysis.** Z-scoring removes the amplitude
> information ERD is measured from. Normalization is for *feeding the model*; the empirical
> mu/ERD reference is computed on stage-1 data only.

## What EEGPT did (and didn't) with these datasets

- EEGPT **pretrained** on PhysioNet MI; its **downstream** MI benchmarks were BCIC-2A/2B
  (linear-probe classification accuracy), not PhysioNet MI itself.
- The EEGPT paper reports **classification/representation metrics only** — it performs **no mu
  ERD / neurophysiology analysis**. So our mu contralateral + PSD characterization is *not* a
  replication of EEGPT; it is the classical reference (Pfurtscheller ERD/ERS), and linking the
  model's representation to it is the novel contribution.

## Open verification

- Confirm the **exact** EEGPT preprocessing/tokenization against **braindecode's EEGPT
  transform** when wiring up the model (scaling to mV, token length, per-token z-score).
- Decide whether a mild high-pass (~0.5 Hz) is worth adding for drift, or whether to stay
  strictly at EEGPT's "0–38".

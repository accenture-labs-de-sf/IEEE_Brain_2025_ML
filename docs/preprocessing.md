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
| Resample | **250 Hz** | checkpoint rate (corrects the paper's 256); our data is 160 Hz → **upsample 160 → 250** |

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

## Model input contract (verified from the braindecode checkpoint)

Confirmed by loading `braindecode/eegpt-pretrained` and reading its `config.json` — this
**corrected the paper's stated 256 Hz to 250 Hz**. Handled in `eegxai.models.eegpt`:

- **250 Hz**, window **1000 samples = 4 s** (`n_times = 1000`).
- **Fixed 62-channel montage**, fed in the model's own order. PhysioNet MI lacks **PO5/PO6**,
  which are **interpolated** from neighbours. Channel-name matching is **case-insensitive**
  (EEGPT `FP1/FPZ`; MNE `Fp1/Fpz`).
- Load with `n_chans=62, chan_proj_type="none", return_encoder_output=True`; encoder returns
  `(batch, patches, tokens, 512)`, pooled to `(batch, 512)`.

Still **provisional** (revisit if decodability underperforms — the smoke test showed high
trial-to-trial embedding correlation): the **µV scaling** and the **mean-pooling**; and whether a
mild high-pass (~0.5 Hz) helps vs. staying strictly at "0–38".

# Research Plan & Decisions

Living document capturing *why* the project is shaped the way it is. Update as decisions change.

## 1. Target venue

- **2026 IEEE Brain Discovery & Neurotechnology Workshop**, Track 2 — *Machine Learning and
  Computer Paradigms for Brain Discovery*. Abstract-gated poster/live-demo.
- Extension → **IEEE Transactions on Human-Machine Systems (THMS)** special issue.
- **Audience is neural-engineering / neurotechnology / BCI** (EMBS-anchored), **not** an
  fMRI/scanner crowd. EEG is explicitly treated as a functional-neuroimaging modality here.
  Implication: sensor-space, cognitive/BCI-style decoding is native; source localization is *not*
  expected. The bar is **trustworthy, brain-relevant neurotech**, not model-internal ML curiosity.
- Relevant topic lanes (disjunctive — we only need to hit one): "Development of interpretable deep
  learning architectures for learning from neural data" and (journal) "Explainable AI (XAI) for
  neuroimaging". Our contribution is an interpretability *analysis/method*, not a new architecture.

## 2. Scientific framing

- **Core question:** does a self-supervised EEG foundation model organize its latent space around
  neurophysiologically faithful structure, and does that hold across subjects?
- **Anchors (the "so what"):**
  1. **Faithfulness** — latent geometry organized by the correct neural axis (mu/beta ERD), not
     artifact → a practitioner can trust it reads real signal.
  2. **Cross-subject reliability** — structure is stable/transferable across people (and where it
     breaks) → the deployability concern for BCI.
- A pure "the embeddings cluster by class" result is **insufficient** (we already know models
  decode). The finding must be faithfulness and/or cross-subject reliability.

## 3. Model

- **EEGPT** (self-supervised, ~25M params). Loadable from the HF Hub via `braindecode`.
- **Not trained on HBN** — pretrained on BCI-style datasets (PhysioNet MI, Tsinghua SSVEP
  benchmark, M3CV, SEED). This drives the dataset choice (use in-distribution data).
- **Montage-flexible:** uses learnable channel embeddings *keyed to channel names*, so it handles
  different montages / missing channels. No rigid electrode set to match.

## 4. Datasets (two, for convergent evidence)

1. **PhysioNet Motor Imagery** — spatial/sensorimotor code (mu/beta ERD). Frictionless (MNE),
   in-distribution, gold-standard ground truth. *Core.*
2. **SSVEP (via MOABB)** — spectral/frequency code (occipital frequency-tagging; the cleanest
   ground truth in EEG). *Second axis.*

Two distinct neural codes (spatial + spectral) → a robustness/generalization argument rather than a
one-dataset fluke.

## 5. Analysis

- **Primary:** latent-space **RSA** + **linear probing** of EEGPT embeddings (interrogates the
  learned representation directly — the point of a foundation model).
- **Supporting:** sensor-space **topography / montage visualization** validating that the
  decision-relevant structure aligns with the known signature (e.g. C3/C4 mu/beta). Quantify the
  agreement (spatial correlation / top-channel overlap) — not a decorative heatmap.
- **Explicitly out of scope:** EEG source localization / inverse modeling; treating raw attention
  weights as scalp "connectivity" (volume-conduction over-claim).

## 6. Feasibility status

- ✅ EEGPT ingests standard montages (name-keyed embeddings); PhysioNet MI needs
  `eegbci.standardize()` on channel names; resample to a common `sfreq`.
- ✅ Model is small (~25M) → laptop-friendly.
- ✅ **Ingestion pipeline built** (`src/eegxai/data/physionet_mi.py`): memory-conscious,
  partial download, float32 epochs, per-subject QC. See memory strategy below.
- ✅ **Data-understanding checks done** (subjects 1–3): 160 Hz / 64 ch / balanced ~23–22
  trials; per-subject noise varies (subject 3 ~5.6% high-amplitude vs <1.5%). Mu-band ERD
  shows clear central sensorimotor desync during imagery, **but clean contralateral C3/C4
  lateralization needs cross-subject aggregation** (weak at n=1). The faithfulness ground
  truth exists; it just requires averaging / sharper spatial methods to surface cleanly.
- 🟡 **Remaining gate — embedding smoke test:** install `braindecode[hub]` + `torch`, pull
  EEGPT, run a few hundred MI epochs, confirm embeddings are non-degenerate and that left/right is
  linearly decodable above chance. Until this passes, downstream analysis is unverified.

## Data engineering & memory strategy

Target budget: a laptop or free Colab (~12 GB RAM). Principle: **less is better — load the
smallest thing an analysis actually needs, and release it promptly.** Five access tiers
(`src/eegxai/data/physionet_mi.py`), leanest first:

1. **`iter_subject_epochs`** — generator; one subject resident at a time, `del` + `gc.collect()`
   between subjects. Default for whole-dataset loops (e.g. extract EEGPT embeddings, keep only the
   embeddings).
2. **`collect_features`** — cross-subject reduction that keeps only each subject's *small* output,
   one vector **per subject** (RSA vectors, per-subject scores, ERD maps).
3. **`iter_subject_batches(…, batch_size=K)`** — bounded middle ground: ~K subjects resident at
   once. Tune `K` to the RAM budget when an analysis needs several subjects together but not all.
4. **`collect_trial_features`** — retains **all** subjects but only as compact *per-trial* features
   (embeddings), never raw epochs. The natural tier for RSA / probing over the full dataset.
5. **`load_concatenated`** — deliberate high-ceiling path that stacks all raw epochs into one array,
   **guarded by `max_gb`** with a pre-flight estimate (`estimate_epoch_memory`). Opt into more RAM
   on purpose; otherwise it raises and points you back to the leaner tiers.

Tiers 2–4 are where cross-subject work lives: the raised "ceiling" is for compact **derived**
results, never the raw epochs. Tier 5 is the only one that holds raw data for everyone, and it is
guarded.

Memory levers exposed on the loaders (each also shrinks RAM): `channels` (spatial subset, dropped
early), `resample_sfreq` (temporal), `tmin`/`tmax` (window), `classes`, and `dtype` (float32
default — half of float64). Reference point: ~8 MB/subject as float32; ~0.9 GB for all 109 subjects
at once (so tier-3 is feasible for the whole set, but tiers 1–2 remain the default).

Colab note: point MNE's cache at Drive (`mne.set_config("MNE_DATA", ...)`) to avoid re-downloading
each session.

## Preprocessing — match EEGPT

> Full implemented detail: [`preprocessing.md`](preprocessing.md).

**Lean:** keep preprocessing as close as possible to what EEGPT was pretrained with, so the
model receives in-distribution input. Heavy custom cleaning risks pushing inputs *out* of
distribution and is avoided.

EEGPT's expected recipe (from the paper; **verify against braindecode's EEGPT transform** at
implementation time):
- **Resample to 256 Hz** — our PhysioNet MI data is 160 Hz, so we **upsample 160 → 256** (expected,
  not optional; use the loader's `resample_sfreq`).
- **Global average reference.**
- **Amplitude scaling to mV.**
- **4 s windows.**
- **~0–38 Hz band-pass** (low-pass) for motor-imagery downstream — this also suppresses 60 Hz line
  noise, so a separate notch is likely unnecessary.
- **z-score** normalization (some configs use Euclidean Alignment per session).

Cleaning stays light: rely on EEGPT's band-pass; exclude only the worst QC-flagged subjects rather
than aggressively scrubbing artifacts.

## First main analysis — empirical dataset characterization (before RSA)

Before any model/RSA work, characterize the dataset **across all participants** to establish the
empirical reference EEGPT's representations are later compared against:
- **Group PSD** (topographic + at C3/C4) — the alpha/mu spectral fingerprint.
- **Mu/beta ERD topographies + a C3/C4 lateralization index per class** — the contralateral
  sensorimotor signature (emerges across many subjects; muted at n≈3).
- **ERSP** (time–frequency) at C3/C4 — the mu/beta suppression time course and best task window.
- **CSP + LDA decodability per subject** — the signal ceiling and cross-subject spread.

Neuro reminder (so results are read correctly): **mu is a *central* sensorimotor rhythm (C3/Cz/C4),
not occipital**; the ~10 Hz occipital peak is *alpha* (visual). Motor imagery yields mu/beta **ERD
(power *decrease*), contralateral** to the imagined hand — not occipital "activation". The "mu break"
is best shown as a baseline→task power drop at C3/C4, not in a whole-head averaged PSD.

## Relationship to EEGPT's published work (replicate vs. new)

Paper trail for *why* the empirical characterization is framed the way it is:

- **PhysioNet MI is one of EEGPT's *pretraining* datasets** — our data is therefore
  *in-distribution* for the model. Representations here are faithful (not degraded transfer),
  which strengthens interpretability claims. Caveat: this is consequently **not** a zero-shot
  transfer result.
- EEGPT's **downstream** motor-imagery evaluation used **BCIC-2A / 2B** with **classification
  accuracy** (linear probing); it did **not** use PhysioNet MI downstream.
- The EEGPT paper reports **classification / representation metrics only — no mu/ERD or other
  neurophysiology.**

Implications:
- Our **mu contralateral ERD + PSD** characterization is the **classical reference**
  (Pfurtscheller ERD/ERS; MNE ERDS-map method) — **not** a replication of anything EEGPT
  published.
- **Novel contribution:** linking EEGPT's learned representation to that neurophysiology via
  XAI / RSA — the gap the paper left open (it proved accuracy, never showed the representation
  encodes known brain structure).

## 7. Open decisions

> Empirical-characterization method choices (ERD sharpening, ERDS, bands, etc.) live in
> [`findings-and-options.md`](findings-and-options.md) — options gated on references, not yet chosen.

- Exact SSVEP dataset from MOABB.
- Precise probing targets and RSA distance metric.
- Whether a clinical/biomarker tier (e.g. HBN) is added for the THMS extension (optional; not
  required for workshop eligibility).

## Decision log

- Rejected **HBN as the core dataset**: EEGPT never saw it (out-of-distribution), plus montage/size
  friction. Kept as a possible clinical extension only.
- Rejected **attention-weights-as-connectome scalp maps**: ill-posed (volume conduction) and
  over-claiming.
- Rejected **source localization**: ill-posed inverse problem, not expected by this audience,
  dubious when applied to model internals.
- Demoted **input-attribution topography** from headline to supporting: it under-uses the model's
  learned latent representation ("deep memory"); latent-space analysis leads instead.

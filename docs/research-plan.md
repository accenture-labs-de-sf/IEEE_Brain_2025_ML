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
- 🟡 **Remaining gate — embedding smoke test:** install `braindecode[hub]` + `torch` + `mne`, pull
  EEGPT, run a few hundred MI epochs, confirm embeddings are non-degenerate and that left/right is
  linearly decodable above chance. Until this passes, downstream analysis is unverified.

## 7. Open decisions

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

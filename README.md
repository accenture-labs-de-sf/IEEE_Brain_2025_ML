# IEEE Brain 2026 — Interpretable EEG Foundation Models

Explainable-AI (XAI) analysis of self-supervised **EEG foundation models**, aimed at a
submission to the **2026 IEEE Brain Discovery & Neurotechnology Workshop** (Washington, D.C.,
11–13 Nov 2026), Track 2 — *Machine Learning and Computer Paradigms for Brain Discovery* — with
a follow-on extension targeting the **IEEE Transactions on Human-Machine Systems (THMS)** special
issue.

> **Repo naming note:** the repository is named `IEEE_Brain_2025_ML` because the work grew out of
> the *NeurIPS 2025 EEG Foundation Challenge*. The submission target is the **2026** IEEE Brain
> workshop.

---

## What this project is

Modern EEG "foundation models" (large self-supervised transformers pretrained on raw EEG) achieve
strong decoding accuracy but are largely **black boxes**. This project opens that box: we take a
*pretrained* foundation model and apply post-hoc interpretability to show **what its learned
representations actually encode** — and whether that maps onto real, trusted neurophysiology.

The framing that matches the venue is **trust**, not model-gazing: interpretability that makes
self-supervised EEG decoding *trustworthy and deployable* for neurotechnology / BCI.

### Core question
> Does a self-supervised EEG foundation model organize its latent space around
> **neurophysiologically faithful** structure, and does that structure hold **across subjects**?

### Two evidence anchors (the "so what")
1. **Faithfulness** — the representation is organized by the *correct* neural axis (e.g. contralateral
   mu/beta sensorimotor rhythms for motor imagery), not by artifact.
2. **Cross-subject reliability** — that structure is stable / transferable across people (and where it
   breaks down), which is the central concern for deployable BCI.

---

## Approach (current plan)

| Element | Choice | Why |
| --- | --- | --- |
| **Model** | [EEGPT](https://braindecode.org/stable/generated/braindecode.models.EEGPT.html) (~25M params, self-supervised) | Small, laptop-friendly; loadable from the HF Hub via `braindecode`; montage-flexible (name-keyed channel embeddings) |
| **Datasets** | PhysioNet Motor Imagery (core) + an SSVEP set (2nd axis) | Both *in-distribution* for EEGPT and carry **gold-standard neural ground truth** (mu/beta ERD; occipital frequency-tagging) |
| **Primary analysis** | Latent-space **RSA** + **linear probing** of EEGPT embeddings | Interrogates the model's learned representation directly ("deep memory") |
| **Neuro-grounding** | Validate latent structure against known signatures; **scalp-topography / montage viz** as a supporting figure | Sensor-space, no ill-posed source localization |
| **Deliverable** | Abstract → poster/demo → THMS journal extension | See deadlines below |

Two datasets are used deliberately: cross-dataset consistency turns a single result into a
**convergent-evidence** argument (spatial motor code *and* spectral visual code).

**Out of scope (by design):** source localization / inverse modeling, and mapping raw attention
weights onto the scalp as "connectivity" — both are ill-posed or over-claiming for this venue.

---

## Repository layout

```
.
├── data/                 # local data (git-ignored); see data/README.md
│   ├── raw/              #   as-downloaded EEG
│   ├── processed/        #   preprocessed / windowed
│   └── external/         #   third-party artifacts
├── src/eegxai/           # project source package
│   ├── data/            #   dataset loaders (PhysioNet MI, SSVEP via MOABB)
│   ├── models/          #   EEGPT loading / embedding extraction
│   ├── analysis/        #   RSA, probing, attribution
│   └── viz/             #   topomaps, RDMs, figures
├── notebooks/            # exploratory analysis
├── configs/              # experiment configs
├── results/figures/      # generated outputs (git-ignored)
├── docs/                 # research plan, submission strategy, notes
└── tests/                # unit tests
```

---

## Documentation

- [`docs/reproducing-tier-a.md`](docs/reproducing-tier-a.md) — **collaborator runbook**: reproduce the empirical ERDS reference step by step.
- [`docs/analysis-methods.md`](docs/analysis-methods.md) — ERDS/cluster-stats concepts, pipeline, and references.
- [`docs/eegpt-reconstruction.md`](docs/eegpt-reconstruction.md) — **approach update**: the decoded-signal (masked-reconstruction) route + checkpoint setup.
- [`docs/research-plan.md`](docs/research-plan.md) — technical decisions and decision log (the *why*).
- [`docs/submission-strategy.md`](docs/submission-strategy.md) — positioning, submission pathway, timeline, and risk management.

## Getting started

Requires **Python ≥ 3.10** (a recent `braindecode` / `torch` stack).

```bash
# clone
git clone https://github.com/accenture-labs-de-sf/IEEE_Brain_2025_ML.git
cd IEEE_Brain_2025_ML

# create an environment (example with venv)
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Datasets are **not** stored in the repo — they download on first use (PhysioNet MI via MNE, SSVEP
via MOABB). See [`data/README.md`](data/README.md).

---

## Status

🟡 **Scaffolding / planning.** Analysis pipeline not yet built. Feasibility half-check done: EEGPT
ingests standard montages via name-keyed channel embeddings; remaining gate is an embedding
"smoke test" (confirm representations carry decodable signal). See
[`docs/research-plan.md`](docs/research-plan.md).

## Key dates

| Milestone | Date |
| --- | --- |
| Early abstract submission | **21 Aug 2026** |
| Final call for abstracts closes | **13 Oct 2026** |
| THMS special-issue full paper | **1 Dec 2026** |

*(Submission format: interactive poster or live demo, abstract-gated. Page limit TBC via the
official submission form.)*

---

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). Internal Accenture Labs project — licensing TBD.

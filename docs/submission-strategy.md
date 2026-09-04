# Submission Strategy

How we turn this project into an accepted IEEE Brain contribution — positioning, pathway, timeline,
and risk management. (For the *technical* decisions, see [`research-plan.md`](research-plan.md).)

## 1. One-line pitch

> A post-hoc interpretability method showing that a self-supervised EEG foundation model organizes
> its latent space around *neurophysiologically faithful*, *cross-subject-stable* structure —
> making black-box EEG decoding trustworthy enough for neurotechnology.

## 2. Why this angle is strategically strong

- **Exploits a real gap.** EEG foundation models are overwhelmingly used as black-box feature
  extractors to chase benchmark accuracy. Almost no one asks *what they actually learned* or
  whether it is neurophysiologically real. We occupy that under-served space.
- **Fits the venue's true bar.** IEEE Brain / THMS is a neural-engineering / BCI / *human-machine
  systems* audience. Their currency is **trust and deployability**, not leaderboard wins. "Can a
  clinician/engineer trust this model?" is exactly our framing.
- **Low execution risk, high clarity.** We use a *pretrained* model (no expensive training) on
  *frictionless, in-distribution* datasets with *gold-standard* neural ground truth. The result is
  legible in one figure — ideal for an abstract-gated poster.
- **Leverages, doesn't compete with, the NeurIPS challenge.** We build on the foundation-model wave
  rather than entering a benchmark race we would not win.

## 3. Positioning within the venue

- **Track 2 — ML & Computer Paradigms for Brain Discovery.** Topic lane: *"development of
  interpretable deep learning ... for learning from neural data"*; journal lane: *"Explainable AI
  (XAI) for neuroimaging."* We are an interpretability **method/analysis**, not a new architecture.
- **Modality is on-target:** EEG is first-class here (explicitly a functional-neuroimaging modality
  for this community). Sensor-space, cognitive/BCI-style decoding is native; source localization is
  *not* expected.
- **Avoid the ML-venue trap:** the headline must read as a *neurotech trust* result, not a
  transformer-internals curiosity.

## 4. The narrative arc (core claim)

1. Foundation models decode EEG well but opaquely → a trust barrier for clinical/BCI adoption.
2. We probe EEGPT's latent representations with RSA + linear probing.
3. **Faithfulness:** the geometry aligns with known signatures (mu/beta ERD; SSVEP frequency code)
   — the model relies on real neural signal, not artifact.
4. **Cross-subject reliability:** that structure holds across people (and we characterize where it
   breaks) — the deployability question.
5. Two datasets (spatial + spectral codes) make it a convergent-evidence claim, not a fluke.
6. **So what:** a practical recipe for *auditing the trustworthiness* of EEG foundation models.

## 5. Submission pathway

| Stage | What it is | What it needs |
| --- | --- | --- |
| **Abstract** | 1-page (format TBC), abstract-gated | Convincing framing + one strong figure; Motor Imagery result sufficient |
| **Poster / live demo** | On-site presentation; "Best Demo" award exists | Full two-dataset analysis; interactive/visual XAI |
| **THMS special issue** | Full journal paper — "significant extension" | Deeper analysis, possibly a clinical/biomarker tier (e.g. HBN) |

The abstract only has to *promise* the analysis credibly; the heavy results live in the poster and
journal extension. This lets Tier-1 (Motor Imagery) carry the abstract while SSVEP / clinical
extensions mature for the journal.

## 6. Timeline (work backward)

| Date | Milestone |
| --- | --- |
| **Now → mid-Aug** | Pass embedding smoke test; build Motor Imagery RSA + faithfulness result + figure |
| **30 Sep 2026** | **Early abstract submission** (priority review + travel support) |
| **Aug → Oct** | Add SSVEP (second axis); polish figures; cross-subject analysis |
| **19 Oct 2026** | Final abstract deadline (fallback) |
| **11–13 Nov 2026** | Workshop — present poster / demo |
| **1 Dec 2026** | **THMS full-paper submission** (extension) |

Target the **early** deadline (Sep 30); treat Oct 19 as fallback only. Early submissions get
priority for review/approval and for travel support requests. Note the THMS window is only about
two and a half weeks after the workshop ends, so the extension has to be largely written before
we present.

## 7. Scope discipline (what we will *not* do)

Deliberately excluded to protect the timeline and defensibility:
- Source localization / EEG inverse modeling.
- Attention-weights-as-connectome scalp maps.
- Training or fine-tuning a model from scratch.
- HBN as the core dataset (optional journal-only clinical tier).

## 8. What reviewers will look for (and our answer)

| Reviewer concern | Our answer |
| --- | --- |
| Neurobiological plausibility | Validated against mu/beta ERD & SSVEP ground truth |
| Methodological rigor in XAI | Systematic RSA + probing with quantified metrics, not a lone heatmap |
| Cross-subject generalization | An explicit anchor, not an afterthought |
| Clarity / impact of one figure | RDM + topography comparison panel designed as the centerpiece |

## 9. Success criteria

- **Minimum:** accepted abstract → poster at the workshop.
- **Target:** poster + live demo + invited THMS extension.
- **Stretch:** "Best Demonstration" recognition; journal publication.

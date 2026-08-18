# IEEE Brain 2026 workshop submission checklist

Workshop: Nov 11-13, 2026, Washington DC. Submit via the Google Form. Target: second week
of September (well before the Oct 13, 2026 final deadline). Verify any exact limits on the
form's final step; format below follows the recurring guideline (one page, up to 1000 words,
one figure).

## Field mapping

- **Presenting author / email / affiliation / pronouns**: fill in.
- **Award eligibility**: student or post-doc first author only. Leave unchecked if not applicable.
- **Other authors and affiliations**: list all.
- **Type of submission**: Research Poster.
- **Title**: Sensorimotor Rhythm or Aggregate Power? An Interpretability Audit of EEGPT on Motor Imagery.
- **Track**: Machine Learning and Computer Paradigms for Brain Discovery.
- **Brief description (100 words max)**: see below.
- **Abstract (1 page)**: paste the text of `abstract_1page.md` (sections Motivation through
  Significance, plus references if the box allows). Likely a text box, so no embedded figure.
- **Image/Graphic (single PDF)**: upload `figures/fig1_overview.pdf`.
- **Societies**: check any that apply (IEEE Brain, IEEE, SPS, EMBS, SMC, etc.).
- **Prior attendance / presentation**: answer as applicable.

## Brief description (100 words)

We audit EEGPT, a recent EEG foundation model (NeurIPS 2024), to ask whether its learned representation
is grounded in the sensorimotor mu and beta rhythms or in aggregate signal power. On PhysioNet
motor imagery, EEGPT decodes imagined movement versus rest as well as classical baselines, yet
representational similarity analysis shows its embedding geometry tracks total signal power more
than any specific band. Only a weak mu and beta component survives a total power control. This
extends a spectral bias reported for reconstruction based models to an architecture that adds
representation alignment, giving a concrete, trust relevant picture of what the model encodes.

## Two-stage plan

1. **Workshop (Stage 1).** Submit this abstract and figure. Present the poster in November,
   collect feedback.
2. **Journal (Stage 2).** Expand into a full paper for the IEEE THMS special issue (significant
   extension, about 30 percent new results), portal mc.manuscriptcentral.com/thms. New content:
   periodic and aperiodic (specparam) decomposition, cross subject RSA, and a causal follow up.
   Confirm the future special-issue deadline with the organizers.

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
- **Abstract (1 page)**: paste the body of `abstract_1page.md` (the four paragraphs, background through
  significance, plus references if the box allows). Plain-text box, so no bold and no embedded figure.
- **Image/Graphic (single PDF)**: upload `figures/fig1_overview_captioned.pdf`. The caption is
  baked in, since the form takes the figure separately from the abstract text and the plain
  `fig1_overview.pdf` would arrive with no explanation.
- **Societies**: check any that apply (IEEE Brain, IEEE, SPS, EMBS, SMC, etc.).
- **Prior attendance / presentation**: answer as applicable.

## Brief description (100 words)

We audit what EEGPT, an EEG foundation model, encodes about motor imagery. Its pretraining pairs
masked reconstruction with a representation alignment objective meant to capture consistent, high
signal to noise structure. A linear probe on its frozen embeddings decodes the task about as well
as feature engineered baselines. Despite that design, the embedding geometry is dominated by
aggregate, largely aperiodic signal power. A parameterized, spatially controlled analysis leaves
only a small, consistent trend toward the sensorimotor mu rhythm, not a strong rhythm specific
representation.

## Two-stage plan

1. **Workshop (Stage 1).** Submit this abstract and figure. Present the poster in November,
   collect feedback.
2. **Journal (Stage 2).** Expand into a full paper for the IEEE THMS special issue (significant
   extension, about 30 percent new results), portal mc.manuscriptcentral.com/thms. New content:
   a head-to-head RSA against pure-reconstruction foundation models (does the alignment objective
   matter), a cross subject RSA, and a causal (attribution or erasure) test. Confirm the future
   special-issue deadline with the organizers.

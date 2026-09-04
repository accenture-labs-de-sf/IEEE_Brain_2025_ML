# IEEE Brain 2026 workshop submission checklist

Workshop: Nov 11-13, 2026, Washington DC. Submit via the Google Form. Target: second week
of September, which also clears the Sep 30, 2026 early deadline (final deadline Oct 19, 2026).
Verify any exact limits on the
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
- **Abstract (1 page)**: paste the four paragraphs of `abstract_1page.md` (background through
  significance) plus the reference list. Not the bold lead, which goes in the brief-description
  field. Plain-text box, so no bold and no embedded figure. See the word budget below.
- **Image/Graphic (single PDF)**: upload `figures/fig1_overview_captioned.pdf` and nothing else.
  The field asks for "a single, high resolution, representative graphic from your poster", so the
  one-page abstract does not belong here. The file is vector (matplotlib PDF), so it is
  resolution-independent. The caption is baked in because the form takes the figure separately from
  the abstract text, and the plain `fig1_overview.pdf` would arrive with no explanation.
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

## Word budget across the two text fields

The lead paragraph and the abstract go in different fields, so they do not compete for the same
1000 words.

| field | content | words |
| --- | --- | --- |
| Brief description (100 max) | bold lead paragraph | 84 |
| Abstract (1 page, 1000 words) | four paragraphs | 801 |
| Abstract, continued | full reference list (13) | 193 |
| | **abstract field total** | **994** |

Not pasted anywhere: the title and the author/affiliation/track lines have their own fields, and the
figure caption is baked into the uploaded graphic, so pasting it would only spend the budget twice.

There is no separate references field, so put the reference list at the end of the abstract box.
The full list fits with six words to spare, so use it rather than abbreviating; four of the entries
are unpublished arXiv papers a reviewer may want to find. If the body grows, fall back to this
compact list (104 words, abstract field total 905):

```
[1] Spectral bias in reconstruction based EEG foundation models. arXiv:2605.26434, 2026.
[2] Tang et al. What do EEG foundation models capture? arXiv:2605.11410, 2026.
[3] Robustness, interpretability and expressiveness of EEG foundation models. arXiv:2605.17562, 2026.
[4] Shama et al. EEG-PRISM. arXiv:2608.13676, 2026.
[5] Donoghue et al. Nat. Neurosci., 2020.
[6] Wang et al. EEGPT. NeurIPS, 2024.
[7] Pfurtscheller and Lopes da Silva. Clin. Neurophysiol., 1999.
[8] Kriegeskorte et al. Front. Syst. Neurosci., 2008.
[9] Donoghue et al. eNeuro, 2020.
[10] Gerster et al. Neuroinformatics, 2022.
[11] Jayaram and Barachant. MOABB. J. Neural Eng., 2018.
[12] Neuro-GPT. arXiv:2311.03764, 2023.
[13] Assran et al. I-JEPA. CVPR, 2023.
```

## THMS full paper format (stage 2)

- **Regular paper: 10 pages max**, in IEEE Transactions **two-column** format, counting author
  photographs and biographies. Technical correspondence is 5 pages, survey 15.
- Overlength is billed at **$220 per page** past the maximum, and extra pages need advance approval
  from the editor-in-chief.
- Two-column format is mandatory at submission, since the editor uses it to judge length. Our
  `build/` academic pipeline is single-column, so the journal version needs IEEEtran (LaTeX), not
  the Markdown pipeline. `report.md` is a content seed, not a format seed.
- References in IEEE style, numbered in square brackets. Figures sharp and noise-free.
- Multimedia supplements are accepted (video, code, data sets).
- **Preprints are allowed** but must be disclosed at submission, and the preprint must carry "This
  work has been submitted to the IEEE for possible publication", then be replaced with the accepted
  version. So an arXiv posting of the abstract is compatible with the journal path.
- Cover letter must state: "This manuscript is being submitted to the Special Issue on Brain
  Discovery and Neurotechnology: Featured Research from the IEEE Brain Discovery and Neurotechnology
  Workshops". Portal: mc.manuscriptcentral.com/thms.

## Two-stage plan

1. **Workshop (Stage 1).** Submit this abstract and figure. Present the poster in November,
   collect feedback.
2. **Journal (Stage 2).** Expand into a full paper for the IEEE THMS special issue (significant
   extension, about 30 percent new results), portal mc.manuscriptcentral.com/thms. New content:
   a head-to-head RSA against pure-reconstruction foundation models (does the alignment objective
   matter), a cross subject RSA, and a causal (attribution or erasure) test. Confirm the future
   special-issue deadline with the organizers.

# THMS full paper (stage 2)

Two-column IEEEtran source for the journal extension, ported from
`../report.md`. The Markdown pipeline in `build/` is single-column and cannot
produce IEEE format, so this is the format of record for the journal version;
`report.md` stays the content seed.

## Why two-column now

IEEE Transactions on Human-Machine Systems requires IEEE two-column format **at
submission**, because the editor uses it to judge length. A regular paper is
capped at 10 pages including author photographs and biographies, and overlength
pages are billed at $220 each with prior approval from the editor-in-chief.

The special issue closes **December 1, 2026**, about two and a half weeks after
the workshop (November 11-13). Little of that window is usable for new analysis,
so the extension should be largely written before the workshop.

## Build

    make            # pdflatex + bibtex + two more passes
    make pages      # report the page count against the 10-page limit

Alternatively upload this folder to Overleaf, which ships IEEEtran. The figure
is referenced at `../figures/fig1_overview.pdf`, so keep the folder inside
`reports/draft/`.

## Before submitting

- Fill the author, affiliation and corresponding-author placeholders.
- Complete the four `INCOMPLETE` entries in `refs.bib`; the arXiv references
  currently have no author lists.
- The abstract is 274 words. IEEE convention is 150-250, so trim if the editor
  objects.
- Cover letter must state: "This manuscript is being submitted to the Special
  Issue on Brain Discovery and Neurotechnology: Featured Research from the IEEE
  Brain Discovery and Neurotechnology Workshops."
- If an arXiv preprint is posted, disclose it at submission and add "This work
  has been submitted to the IEEE for possible publication" to the preprint.
- The paper needs roughly 30 percent new material over the workshop abstract.
  Planned: head-to-head RSA against pure-reconstruction models, cross-subject
  RSA, and a causal attribution or erasure test.

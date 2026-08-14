# Pivot: imagery vs rest, decode and RSA

This is the current direction. Read it to understand what we are doing now, why, the findings so
far, and how to run it.

## Why we study the encoder representation (not the reconstruction)
EEGPT's reusable value is its **encoder embedding**. That is the vector downstream tasks freeze and
build on, and it is the only part braindecode ships. Trust in the model comes from whether that
representation encodes meaningful signal, so interpreting it is the point, and our decode + RSA on
frozen embeddings mirrors exactly how the model is used downstream. The reconstruction decoder is
training machinery that is discarded at deployment, so it is not the object of interest. (We also
tried running ERD on the reconstruction and found it invalid: the reconstructor output is
normalized per patch, which breaks percent-baseline ERD. See `eegpt-reconstruction.md`.)

## Why we pivoted (two earlier targets did not work, both informative)
1. **Left vs right imagery is a weak contrast here.** The classical decodability ceiling (CSP+LDA,
   n=20) is only about 55 to 56 percent, so left vs right is a poor target for any faithfulness
   claim, and its ERD lateralization is correspondingly weak.
2. **ERD on the model reconstruction is invalid** (per-patch normalization breaks percent baseline).

So we target **imagery vs rest** (a strong ERD contrast) and use two readouts on the encoder
embeddings that need no baseline: **decoding** and **RSA**.

## Findings so far (imagery vs rest, n=20)

**Decoding** (chance 50 percent):

| method | within subject | cross subject |
| --- | --- | --- |
| **EEGPT embeddings** | 73.1 | 58.3 (p=0.01) |
| CSP+LDA (classical) | 76.3 | 54.7 |
| band-power+LDA | 63.3 | 57.3 |

EEGPT decodes about as well as CSP within subject and is the best of the three across subjects,
though the cross-subject margin over band-power is small.

**RSA** (does the embedding geometry match the neurophysiology). Within subject, EEGPT's geometry
correlates with mu/beta band-power structure. After adding conservative artifact cleaning
(Autoreject) and a non-motor control band, the correspondence is **not specific to mu**:

| band | partial correlation (controlling for task) |
| --- | --- |
| mu (8 to 13 Hz) | 0.18 |
| beta (13 to 30 Hz) | 0.23 |
| control (2 to 7 Hz, non-motor) | 0.17 |

All are significant (p around 1e-6), but mu is about the same as the control band, and only beta is
modestly higher.

## What this means (the narrative)
EEGPT decodes motor imagery competitively and generalizes across subjects, but its representation is
**spectrally non-specific**. It is organized by aggregate signal structure more than by the mu/beta
rhythms that classical BCI and clinicians rely on. This is a trust-relevant interpretability
finding: the model works, but not for the neurophysiological reasons one might expect. A likely
explanation we have not yet confirmed is that the correspondence reflects overall signal power
rather than spectral content.

## Open questions to firm up the narrative
1. **Power confound.** Partial out total signal power and see if any band-specific structure
   survives. If it collapses, the finding is that EEGPT encodes aggregate signal strength.
2. **Band specificity.** Confirm mu and beta carry the task and are distinct from the control band,
   so the non-specificity is a property of EEGPT, not the dataset.
3. **Layer or token probe.** Check whether band structure exists at any depth, testing whether
   tokenization of continuous signals limits spectral specificity.

## How to run
```bash
# base decode + RSA (no cleaning)
python scripts/pivot_analysis.py --subjects 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
# band-resolved RSA with Autoreject cleaning (mu / beta / control band)
python scripts/rsa_bandresolved.py --subjects 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
```
Reusable code: `eegxai.analysis.decoding` and `eegxai.analysis.rsa`. Features are cached
incrementally so runs are resume-able.

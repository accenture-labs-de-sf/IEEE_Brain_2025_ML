# Pivot — imagery-vs-rest, decode + RSA

**Read this to understand the current direction.** The analysis pivoted; here's why, what it is
now, the results so far, and how to run it.

## Why we pivoted (two dead ends, both informative)
1. **Left/right imagery is a weak contrast here.** The classical decodability ceiling (CSP+LDA,
   n=20) is only **~55–56%** — even the gold standard barely beats chance. So left/right is a poor
   target for any faithfulness claim, and its ERD lateralization is correspondingly weak.
2. **ERD on the model's *reconstruction* is invalid.** EEGPT's reconstructor outputs **per-patch
   normalized** signal, which destroys the baseline power reference that percent-ERD divides by
   (values explode). We confirmed this empirically. So "run our ERD on the decoded signal" can't
   work, regardless of masking care.

## What the analysis is now
Target **imagery vs rest** (T1/T2 vs T0) — a **strong, unambiguous ERD contrast** — and use two
readouts on EEGPT's **encoder embeddings** that need no baseline:

- **Decode** — can a linear classifier read imagery-vs-rest out of the embeddings? (within- and
  cross-subject)
- **RSA** — does the embedding *geometry* mirror the real **mu/β** sensorimotor structure, beyond
  the task label? (the fine-grained faithfulness measure)

Both are referenced against a **classical ceiling** (CSP / band-power) so the numbers are
interpretable.

## Results so far (n=20)
**Decodability (chance 50%):**

| method | within-subject | cross-subject |
| --- | --- | --- |
| CSP+LDA (classical) | 76.3% ± 12.1 | 54.7% |
| band-power+LDA | 63.3% ± 8.2 | 57.3% |
| **EEGPT embeddings** | **73.1% ± 12.5** | **58.3%** (p=0.010) |

→ EEGPT ≈ CSP within-subject; **best-of-three cross-subject** (subject-invariant, though the margin
over band-power is small).

**RSA (within-subject Spearman, n=20):**

| relationship | correlation | p |
| --- | --- | --- |
| model ~ task | +0.166 | 8e-4 |
| model ~ neural (mu+β) | +0.375 | 1.4e-11 |
| **model ~ neural \| task** (partial) | **+0.356** | 4.1e-11 |

→ The embedding geometry **strongly mirrors the mu/β structure, and it survives removing the task
label** — EEGPT preserves the fine-grained neurophysiology, not just the binary split. This is the
cleanest faithfulness result so far.

## How to run
```bash
python scripts/pivot_analysis.py --subjects 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
```
Runs the ceiling, extracts EEGPT embeddings (cached incrementally), decodes, and does RSA; writes a
`results.txt` under `results/exploration/pivot_<timestamp>/`. Reusable pieces:
`eegxai.analysis.decoding` and `eegxai.analysis.rsa`.

## Open / what to do next
- **Cross-subject RSA** (does the geometry generalize across people?) — the reliability anchor.
- **Control-band baseline** for RSA (a non-motor band / random-embedding null) to show the mu/β
  correspondence is *specific*.
- **Pooling check** — decode used mean-pooled embeddings (the 0.98-collapse pooling); a better
  readout (flatten/PCA) may raise the cross-subject decode.
- Then general cleanup + packaging.

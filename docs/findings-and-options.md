# Findings & Open Options

Living log: what the exploratory results show, and the candidate options they surface. These are
**considerations to choose from, not decisions.** A method is adopted only once there is a clear
reference applying it to *this* data / analysis type, and it fits the "match EEGPT / stay
in-distribution" lean. Updated as results accrue.

## Findings so far (exploratory; subjects 1–10)

Method: **canonical MNE ERDS** — multitaper TFR + percent baseline (see
[`analysis-methods.md`](analysis-methods.md)). EEGPT-matched preprocessing.

- ERDS maps at C3/Cz/C4 show clear post-cue **mu (~10 Hz) and beta (~20 Hz) ERD** during
  imagery, over a flat baseline.
- **RIGHT-hand imagery:** clearly contralateral — lateralization index (mu ERD C3 − C4)
  ≈ **−8.9%** (expected < 0). ✓ strong.
- **LEFT-hand imagery:** correct direction but weak — index ≈ **+0.7%** (expected > 0). ~
- Switching from the band-power quick-look (dB) to the canonical percent-baseline TFR **improved**
  lateralization (LEFT went from ~0 to correctly signed).
- High per-subject variability remains; group topography is clean for RIGHT, bilateral for LEFT.

## Adopted

- **Canonical MNE ERDS pipeline** (multitaper TFR + percent baseline; Pfurtscheller & Lopes da
  Silva 1999; MNE ERDS example). This is the current empirical-reference method — see
  [`analysis-methods.md`](analysis-methods.md).

## Candidate options (pick from; each gated on evidence)

| Option | What it addresses | Reference status (verify before adopting) |
| --- | --- | --- |
| Cluster-based permutation stats | Significance of the ERDS maps (part of the canonical example) | MNE ERDS example (`permutation_cluster_1samp_test`) — **not yet added** |
| Scale subjects (n → 20–30+) | Does lateralization tighten with n? | Standard; low commitment |
| Surface Laplacian (small) at C3/C4 | Sharpen lateralization (esp. the muted LEFT class) | Common in Pfurtscheller MI work; **find a Laplacian-on-EEGMMIDB reference** before adopting |
| CSP spatial filtering | Sharpen / decode lateralization | Blankertz 2008; MNE's CSP example **uses eegbci/EEGMMIDB** (direct ref). ⚠ **supervised** — see gate 3 |
| Beta band (13–30 Hz) ERD/ERS | Complementary lateralizing signal | Standard MI finding; verify for this data |
| C3/C4 PSD overlay (baseline vs task) | The second figure panel | Straightforward; no new method |
| Artifact handling / subject exclusion | S3 etc. noisier (QC) | Gate on explicit QC thresholds |

## Gates before adopting any method

1. A clear reference applying it to **EEGMMIDB / motor-imagery ERD**, not just MI in general.
2. Consistency with the **match-EEGPT / in-distribution** lean.
3. **Role clarity — unsupervised for the ground truth.** The empirical reference should use
   *unsupervised* methods (band-power, surface Laplacian). **CSP is supervised**: using it to
   *define* the ground-truth map we then test the model against would be **circular**. CSP is fine
   as a separate *decodability baseline*, not as the ground truth.

## Explicit non-prescription

The **ERDS method is adopted** (above, reference-backed). Beyond it, no *sharpening / statistics*
choice is locked — **including CSP** — until its reference and rationale are in place. CSP remains
gated by the supervised-circularity caveat (gate 3).

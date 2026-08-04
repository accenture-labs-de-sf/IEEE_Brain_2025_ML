# Findings & Open Options

Living log: what the exploratory results show, and the candidate options they surface. These are
**considerations to choose from, not decisions.** A method is adopted only once there is a clear
reference applying it to *this* data / analysis type, and it fits the "match EEGPT / stay
in-distribution" lean. Updated as results accrue.

## Findings so far (exploratory; subjects 1–10, EEGPT preprocessing)

- Ingestion + QC solid; 160→256 Hz upsample, average reference, 0–38 Hz band-pass applied.
- Mu (8–13 Hz) ERD clearly present over central sensorimotor cortex during imagery.
- **RIGHT-hand imagery:** contralateral (left / C3) desync emerging — lateralization index
  (ERD C3 − C4) ≈ **−0.67 dB** (expected < 0). ✓
- **LEFT-hand imagery:** still bilateral / symmetric — index ≈ **+0.01 dB** (expected > 0). ✗
- High per-subject variability; group mean muted at small n.
- Reading: the ground-truth mu signal exists; the clean *bilateral-mirror* lateralization is not
  yet resolved at n=10 with plain band-power.

## Candidate options (pick from; each gated on evidence)

| Option | What it addresses | Reference status (verify before adopting) |
| --- | --- | --- |
| Scale subjects (n → 20–30+) | Does lateralization tighten with n? | Standard; low commitment |
| Baseline-normalized ERD / ERDS maps (Pfurtscheller) | Canonical presentation; makes the "mu break" explicit | Method: Pfurtscheller & Lopes da Silva 1999. MNE's ERDS example uses BCI-IV-2a — confirm params for EEGMMIDB |
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

No method is chosen as "the next step" in this document — **including CSP**. The next move is
selected from the options above once its reference and rationale are in place.

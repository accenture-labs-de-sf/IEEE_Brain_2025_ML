# Findings & Open Options

Living log: what the exploratory results show, and the candidate options they surface. These are
**considerations to choose from, not decisions.** A method is adopted only once there is a clear
reference applying it to *this* data / analysis type, and it fits the "match EEGPT / stay
in-distribution" lean. Updated as results accrue.

## Findings so far (n=20, canonical ERDS + cluster stats, edge-cropped)

Method: **canonical MNE ERDS** — multitaper TFR + percent baseline, buffered epochs cropped to
−1..3.9 s (removes edge ringing), group cluster-permutation stats (see
[`analysis-methods.md`](analysis-methods.md)). EEGPT-matched preprocessing.

- ERDS maps show clear post-cue **mu (~10 Hz) and beta (~20 Hz) ERD** at C3/Cz/C4, over a flat
  baseline, with clean edges.
- **All six channel×class effects are significant** (cluster p<0.05): RIGHT — C3 **0.001**, Cz
  0.003, C4 0.006; LEFT — C3 0.012, Cz 0.037, **C4 0.020**.
- **Lateralization:** RIGHT clearly contralateral (C3), index (mu C3−C4) **−6.7%**; LEFT weak bias
  toward C4, **+0.6%**. Both hemispheres significantly engage, with a contralateral bias strong for
  RIGHT and weak for LEFT.
- Scaling n=10→20 pushed **C4/LEFT from a trend (p=0.074) to significant (p=0.020)**, as predicted.
- Interpretation caveat (Rousselet 2025): cluster p is **cluster-level, not point-wise** — report
  *that* an effect is significant, not the exact time-frequency boundaries.

## Adopted

- **Canonical MNE ERDS pipeline** (multitaper TFR + percent baseline; Pfurtscheller & Lopes da
  Silva 1999; MNE ERDS example). This is the current empirical-reference method — see
  [`analysis-methods.md`](analysis-methods.md).
- **Cluster-permutation significance** (group one-sample; MNE `permutation_cluster_1samp_test`;
  **Maris & Oostenveld 2007**, the canonical EEG/MEG cluster-permutation reference). Implemented
  (`eegxai.analysis.stats`); results above.
- **Edge-cropping** — buffered epochs (−2..4.5 s) cropped to −1..3.9 s after baseline, removing
  multitaper edge ringing (standard TFR practice). Applied.

Both adopted methods are grounded in canonical + recent (2023–2025) literature — see the
References section of [`analysis-methods.md`](analysis-methods.md).

## Tier B — model analysis (in progress)

- **EEGPT encoder features load and run** (braindecode, 62 ch / 250 Hz). But the **latent
  linear-probe was inconclusive**: left/right barely decodes (EEGPT ~52–56%) and its
  **permutation null overlaps** — *and the classical mu/β baseline is also at chance*. So the probe
  can't separate "representation weak" from "left/right just hard here" (our baseline was simple
  power, not CSP). Not evidence either way — don't over-read it.
- **Pivot to the decoded-signal route** (encoder + reconstructor). EEGPT is a masked autoencoder, so
  its **reconstruction** is the quality-optimized, interpretable output; we run ERDS on
  reconstructed EEG (masked → non-circular). **Full-model smoke check passed** — checkpoint loads
  clean (0 missing/unexpected), masked reconstruction emits finite signal patches. See
  [`eegpt-reconstruction.md`](eegpt-reconstruction.md).
- **Decoded-signal ERD is invalid** — the reconstructor's output is *per-patch normalized*, so
  percent-baseline ERD explodes (confirmed empirically). Faithfulness therefore moved to **decode +
  RSA** (no baseline needed).
- **Pivot to imagery-vs-rest** (working analysis; see [`pivot-analysis.md`](pivot-analysis.md)).
  Left/right is a weak contrast (classical ceiling only ~55%); imagery-vs-rest is strong.
  - **Decode (n=20):** EEGPT 73% within / 58% cross, best of three (CSP 76/55, band-power 63/57),
    cross significant (p=.01).
  - **RSA (n=20):** after Autoreject cleaning and a non-motor control band, EEGPT's geometry
    correlates with band-power **broadly but NOT specifically with mu** (mu 0.18 ≈ control 0.17;
    beta 0.23 modestly higher). The earlier "mirrors mu" reading (r=0.36) did not survive the
    control band.
  - **Narrative:** EEGPT decodes competently and transfers across subjects, but its representation
    is **spectrally non-specific** (aggregate signal structure, not the mu/beta rhythms clinicians
    rely on). A trust-relevant XAI finding. Likely a total-power effect (unconfirmed).
  - **Next (to firm up):** partial out total power; confirm bands are distinct/task-carrying;
    optional layer/token probe (tokenization limitation hypothesis).

## TODO (unconfirmed by lead)

- **Re-run the n=20 ERDS snapshot at 250 Hz.** The sampling rate was corrected 256 → **250 Hz**
  (EEGPT's checkpoint rate) in the code, but the committed [`reports/mi_erds_n20/`](../reports/mi_erds_n20)
  snapshot was computed at 256 Hz. Scientific conclusions are unchanged (250 vs 256 is negligible
  for mu ERD), so this is a tidy-up only — **not yet confirmed / not yet done.**

## Candidate options (pick from; each gated on evidence)

| Option | What it addresses | Reference status (verify before adopting) |
| --- | --- | --- |
| Scale further (n → 30+ / all 109) | Tighten estimates; test if the weak LEFT bias sharpens | Standard; low commitment (n=20 already all-significant) |
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

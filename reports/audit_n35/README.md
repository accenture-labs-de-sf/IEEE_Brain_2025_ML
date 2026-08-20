# Audit snapshot: EEGPT motor-imagery RSA (n = 35)

Committed snapshot so findings persist independent of the local, regenerable feature cache
(`results/audit_n35/`, gitignored).

## Status: resolved (nuanced)

**Solid:**
- EEGPT decodes imagery vs rest ~71% within subject (chance 50), on par with feature-engineered
  baselines. Reproduces the earlier n = 20 run.
- The representation is **dominated by aggregate, largely aperiodic signal power** (Spearman ~0.24
  to 0.32 with total power, depending on proxy), higher than its correlation with any single band.

**Sensorimotor specificity (resolved with proper controls):** a **small, consistent trend**, not a
strong component.
- Under parameterized (specparam), spatially controlled analysis, aperiodic-adjusted mu power is
  tracked slightly but reliably above non-motor control bands: Δr ≈ 0.04 (mu > theta p = 0.002,
  mu > gamma p = 0.0003; ~80% of subjects).
- It is central rather than posterior alpha (central mu 0.061 > occipital 0.027; central survives
  controlling occipital, p = 1.5e-6), and above chance (permutation p = 0.001).
- Beta shows no reliable trend (p = 0.07 to 0.15).
- We treat this as a **trend, not an established component**: at n = 35 it is statistically
  detectable but small, and whether it persists and how precisely it can be estimated as sample
  size grows is an open question. Significance here reflects detectability, not magnitude.

**Why earlier readings differed (a methodological point worth reporting):**
- Raw band-power double control looked *non-specific* (a 30-40 Hz control band survived) because
  band power conflates periodic and aperiodic activity (Donoghue, Dominguez & Voytek 2020).
- The sensorimotor-averaged specparam looked *strongly mu-specific* (0.238) because averaging over
  central channels inflates mu.
- Holding the spatial construction fixed (all 64 channels) + separating periodic from aperiodic +
  a mu-vs-alpha contrast + a permutation null gives the honest, much smaller trend (~0.04).

## Files
- `spatial_control.txt` — **the decisive, controlled result** (per-channel aperiodic-adjusted power;
  motor vs control bands with paired tests; central-vs-occipital mu; permutation null).
- `specparam_rsa.txt`, `periodic_control.txt` — periodic/aperiodic on the sensorimotor-averaged
  spectrum. **Preliminary** (spatially confounded); kept for the record.
- `double_control.txt`, `totalpower_recon.txt` — raw band-power double control (conflates
  periodic/aperiodic per eNeuro); kept for the record.
- `config.json` — subjects, params, package versions.

## Reproduce (cache-only unless noted)
```bash
python scripts/extract_features.py       --subjects 1-35 --out results/audit_n35 --batch-size 5  # ~50 min, once
python scripts/recompute_psd_perchannel.py --subjects 1-35 --out results/audit_n35               # ~25 min, once
python scripts/specparam_perchannel.py   --out results/audit_n35                                 # ~15 min, once (fit)
python scripts/rsa_double_control.py     --out results/audit_n35
python scripts/rsa_totalpower_recon.py   --out results/audit_n35
python scripts/rsa_specparam.py          --out results/audit_n35
python scripts/rsa_periodic_control.py   --out results/audit_n35
python scripts/rsa_spatial_control.py    --out results/audit_n35   # the decisive one
```

## Method references
specparam parameterization: Donoghue et al. 2020 (Nat Neurosci). Band power conflates
periodic/aperiodic activity: Donoghue, Dominguez & Voytek 2020 (eNeuro). Separation how-to:
Gerster et al. 2022. RSA control models and partial correlation: Nili et al. 2014.

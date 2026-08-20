# Audit snapshot: EEGPT motor-imagery RSA (n = 35)

Committed snapshot so findings persist independent of the local, regenerable feature cache
(`results/audit_n35/`, gitignored).

## Status: the specificity question is OPEN (under validation)

**What is solid:**
- EEGPT decodes imagery vs rest ~71% within subject (chance 50), on par with feature-engineered
  baselines. Reproduces the earlier n = 20 run (identical trial counts, matching numbers).
- EEGPT's embedding geometry tracks **total signal power** strongly (Spearman ~0.24 to 0.32
  depending on proxy), higher than its correlation with any single band. The power / aperiodic
  influence is real and large.

**What is NOT settled (do not claim yet):** whether EEGPT encodes the sensorimotor rhythm
*specifically*. The evidence flips with method, and the current tests are confounded:
- Raw band-power double control (task + total power): a 30-40 Hz control band survives as strongly
  as mu/beta, which *looks* non-specific. But raw band power conflates periodic and aperiodic
  activity (Donoghue, Dominguez & Voytek 2020, eNeuro), so this test is confounded.
- Aperiodic-adjusted oscillatory power (specparam) on the sensorimotor-averaged spectrum: mu is
  tracked (0.238, all 35 subjects) above theta/beta/gamma, which *looks* mu-specific.
- **Caveat that blocks a conclusion:** the specparam test changed two things at once versus the
  band-power test, the periodic/aperiodic separation AND the spatial reference (64 channels to a
  sensorimotor-averaged scalar). So the mu result cannot yet be attributed to the decomposition;
  sensorimotor averaging alone could favor mu, and 8-13 Hz mu overlaps posterior alpha.

**Deciding it** requires holding the spatial construction fixed: per-channel aperiodic-adjusted
oscillatory power (all 64 channels) with a permutation null, and a central-vs-occipital contrast
to separate mu from alpha. In progress.

## Files
- `double_control.txt`, `totalpower_recon.txt` — band-power double control (confounded per eNeuro;
  kept for the record).
- `specparam_rsa.txt` — periodic vs aperiodic, sensorimotor-averaged. **Preliminary.**
- `periodic_control.txt` — aperiodic-adjusted frequency-specificity control, sensorimotor-averaged.
  **Preliminary.**
- `config.json` — subjects, params, package versions.

## Reproduce (cache-only, seconds, unless noted)
```bash
python scripts/extract_features.py     --subjects 1-35 --out results/audit_n35 --batch-size 5  # ~50 min, once
python scripts/rsa_double_control.py   --out results/audit_n35
python scripts/rsa_totalpower_recon.py --out results/audit_n35
python scripts/rsa_specparam.py        --out results/audit_n35
python scripts/rsa_periodic_control.py --out results/audit_n35
```

## Method references
specparam parameterization: Donoghue et al. 2020 (Nat Neurosci). Band power conflates
periodic/aperiodic activity: Donoghue, Dominguez & Voytek 2020 (eNeuro). Separation how-to:
Gerster et al. 2022. RSA control models and partial correlation: Nili et al. 2014.

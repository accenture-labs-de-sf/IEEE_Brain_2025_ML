# Audit snapshot: EEGPT motor-imagery RSA (n = 35)

Committed snapshot of the key numbers so the findings persist in git independent of the local,
regenerable feature cache (`results/audit_n35/`, gitignored). Numbers here are current and
supersede the earlier n = 20 reading (see Note).

## Headline (current, honest)

- EEGPT decodes imagery vs rest at ~71% within subject (chance 50), on par with feature
  engineered baselines.
- Its embedding geometry tracks **total signal power** strongly (Spearman ~0.24 to 0.32 depending
  on the total-power proxy), higher than its correlation with any single band. Power dominated.
- Under a **task + total-power double control**, band structure survives about equally for mu,
  beta, **and a 30-40 Hz control band**, while only a low 2-7 Hz band collapses. This holds under
  two different total-power proxies. So the surviving structure is **not specific to the
  sensorimotor rhythm**.
- Conclusion: EEGPT represents motor imagery primarily through aggregate power and shows **no
  sensorimotor-rhythm specificity**.

## Files

- `double_control.txt` — definitive task + total-power double control (broadband-variance proxy),
  plus the embedding-vs-total-power correlation and per-subject consistency.
- `totalpower_recon.txt` — the same double control under two total-power proxies, side by side
  (robustness check; the low-vs-high control-band pattern is stable across both).
- `config.json` — subjects, bands, channels, preprocessing, package versions.

## Reproduce

The feature cache lives in `results/audit_n35/` (gitignored, regenerable). All analyses below are
cache-only and run in seconds; only the one-time extraction is slow.

```bash
python scripts/extract_features.py --subjects 1-35 --out results/audit_n35 --batch-size 5  # ~50 min, once
python scripts/rsa_double_control.py    --out results/audit_n35
python scripts/rsa_totalpower_recon.py  --out results/audit_n35
```

## Note

This supersedes the earlier n = 20 "weak but genuine mu/beta-specific component" reading. That
apparent specificity did not survive two checks: scaling to 35 subjects and, more importantly,
adding a second (high-frequency) control band. A 30-40 Hz control band survives the double control
as strongly as mu and beta, so the residual structure is broadband, not sensorimotor specific. The
draft abstract and paper still need to be reworded to match this.

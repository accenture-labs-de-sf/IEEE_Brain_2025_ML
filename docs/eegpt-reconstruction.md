# EEGPT Reconstruction (decoded-signal route)

**Approach update.** Beyond probing EEGPT's *latent* features, we now also use the **full model
(encoder + predictor + reconstructor)** to produce **reconstructed EEG**, then run our validated
ERDS analysis on the reconstruction. Rationale: EEGPT is a masked autoencoder — its objective
prioritizes *reconstruction quality*, so the decoded signal is a first-class, **interpretable**
readout (same units/analysis as the real data), and it's a strong faithfulness test.

## Why (and the non-circularity design)
Feeding the *full, unmasked* signal reconstructs it near-identically → running ERDS on that is
**circular**. So we follow EEGPT's pretraining: **mask** part of the signal and reconstruct the
masked patches from context; the ERD in the **reconstructed masked regions** reflects the model's
learned prior, not a copy of the input. (Caveat: masked autoencoders can blur high-frequency
detail, so mu ~10 Hz fidelity is an empirical question.)

## Two model paths (don't confuse them)
| Path | Module | Weights | Contract |
| --- | --- | --- | --- |
| **Encoder (features)** | `eegxai.models.eegpt` | braindecode HF checkpoint | 62 ch, **250 Hz**, 4 s |
| **Full model (reconstruction)** | `eegxai.models.eegpt_reconstruct` | original Figshare `.ckpt` | 58 ch, **256 Hz**, 4 s / 1024 samples, patch 64 |

## Getting the checkpoint (required, not in git)
1. Download from Figshare: `https://figshare.com/s/e37df4f8a907a866df4b` — the file
   **`eegpt_mcae_58chs_4s_large4E.ckpt`** (~1 GB; the download may arrive as a zip containing it).
2. Place it at **`data/external/eegpt_mcae_58chs_4s_large4E.ckpt`** (git-ignored).

## Usage
```python
import torch
from eegxai.models.eegpt_reconstruct import build_full_model, reconstruct, USE_CHANNELS

m = build_full_model()                         # loads data/external/…ckpt
r, mask_x, mask_y = reconstruct(m, x)          # x: (B, 58, 1024) -> r: (B, n_masked, 64) patches
```

The model code is **vendored** (Apache-2.0) under `src/eegxai/models/vendor/`; see its `README.md`.

## Status
- ✅ **Smoke check passed** — full checkpoint loads (0 missing/unexpected weights), masked
  reconstruction runs and emits finite signal patches `(B, n_masked, 64)`.
- 🟡 **Next:** align real MI epochs to the 58-ch / 256 Hz contract, **assemble** reconstructed
  patches into a full `(58, 1024)` signal, and run the **left/right ERDS on the reconstructions**
  (analyzing masked regions) — the actual faithfulness result.

## Where this came from (context)
The latent linear-probe on the encoder features was **inconclusive** (left/right barely decodes
even with a classical baseline, at a permutation-controlled null), which motivated evaluating the
model where it's strongest — the decoded output. See `findings-and-options.md`.

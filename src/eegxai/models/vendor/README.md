# Vendored third-party code

## `eegpt_pretraining.py`

Vendored from **[BINE022/EEGPT](https://github.com/BINE022/EEGPT)**, file
`pretrain/modeling_pretraining.py` (unmodified except for a header comment).

- **Source:** Wang et al., *EEGPT: Pretrained Transformer for Universal and Reliable
  Representation of EEG Signals*, NeurIPS 2024.
- **License:** Apache License 2.0 — full text in `LICENSE` (copied from the upstream repo).
  Copyright © the EEGPT authors.

We vendor this single, self-contained (torch-only) file so we can load the full pretraining
model (encoder + predictor + **reconstructor**) for the decoded-signal analysis, without pulling
in the upstream repo's large/conflicting `requirements.txt`. The pretrained **checkpoint** is not
redistributed here — download it separately (see `docs/eegpt-reconstruction.md`).

Loader/wrapper: `eegxai.models.eegpt_reconstruct`.

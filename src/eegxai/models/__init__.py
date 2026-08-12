"""Foundation-model loading and embedding extraction.

Intended contents:
    - EEGPT loading from the Hugging Face Hub via `braindecode`.
    - Forward pass to extract latent representations (per epoch / per subject).
    - Helpers to map dataset `chs_info` onto EEGPT's channel embeddings.
"""

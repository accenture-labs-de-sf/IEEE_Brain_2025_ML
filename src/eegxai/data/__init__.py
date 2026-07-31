"""Dataset loaders and preprocessing.

Intended contents:
    - PhysioNet Motor Imagery (EEGMMIDB) loader via MNE `eegbci`, including
      `standardize()` of channel names for EEGPT name-keyed embeddings.
    - SSVEP loader via MOABB.
    - Shared preprocessing: resampling to a common sfreq, windowing, epoching.
"""

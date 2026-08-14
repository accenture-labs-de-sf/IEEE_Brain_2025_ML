"""Representational Similarity Analysis (within-subject, aggregated across subjects).

Builds a model RDM (from EEGPT embeddings), a neural RDM (from real band-power), and a task
RDM (condition labels), and reports Spearman(model, task), Spearman(model, neural), and the
**partial** Spearman(model, neural | task) — the fine-grained faithfulness measure (does the
embedding geometry mirror the neural structure *beyond* the task label?).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr, ttest_1samp


def rdm(features: np.ndarray) -> np.ndarray:
    """``(n, d)`` -> upper-triangle of the ``1 - correlation`` dissimilarity matrix."""
    R = 1.0 - np.corrcoef(features)
    return R[np.triu_indices(len(features), 1)]


def task_rdm(labels: np.ndarray) -> np.ndarray:
    """Binary condition RDM: 0 if same label, 1 if different (upper triangle)."""
    lab = np.asarray(labels)
    return ((lab[:, None] != lab[None, :]).astype(float))[np.triu_indices(len(lab), 1)]


def partial_spearman(r_mn: float, r_mt: float, r_nt: float) -> float:
    """Partial correlation of model~neural controlling for task."""
    return (r_mn - r_mt * r_nt) / np.sqrt((1 - r_mt ** 2) * (1 - r_nt ** 2) + 1e-12)


def within_subject_rsa(model_feats, neural_feats, task, groups) -> dict[str, np.ndarray]:
    """Per-subject RSA correlations. Returns arrays keyed by relationship."""
    out = {"model~task": [], "model~neural": [], "model~neural|task": []}
    for g in np.unique(groups):
        m = groups == g
        rm, rn, rt = rdm(model_feats[m]), rdm(neural_feats[m]), task_rdm(task[m])
        r_mt = spearmanr(rm, rt).correlation
        r_mn = spearmanr(rm, rn).correlation
        r_nt = spearmanr(rn, rt).correlation
        out["model~task"].append(r_mt)
        out["model~neural"].append(r_mn)
        out["model~neural|task"].append(partial_spearman(r_mn, r_mt, r_nt))
    return {k: np.asarray(v) for k, v in out.items()}


def summarize(rsa_out: dict[str, np.ndarray]) -> dict[str, tuple[float, float, float]]:
    """Per relationship: (mean, sd, one-sample t-test p-value vs 0)."""
    return {k: (float(v.mean()), float(v.std()), float(ttest_1samp(v, 0.0).pvalue))
            for k, v in rsa_out.items()}

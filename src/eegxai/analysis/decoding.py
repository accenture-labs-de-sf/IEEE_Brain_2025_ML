"""Decodability analyses for the imagery-vs-rest pivot.

Classical ceiling (CSP / band-power on raw epochs) and generic feature decoding (e.g. EEGPT
embeddings), each **within-subject** and **cross-subject (leave-one-subject-out)**, with an
optional permutation null.

Note: CSP cross-subject with a permutation test is *very slow* (~hours) — use ``bandpower_lda``
or ``logreg`` for cross-subject permutations, or pass ``n_permutations=0``.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (LeaveOneGroupOut, StratifiedKFold,
                                     cross_val_score, permutation_test_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class LogVar(BaseEstimator, TransformerMixin):
    """Per-channel log-variance (band-power) features from ``(n, ch, time)`` epochs."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.log(np.var(X, axis=2) + 1e-12)


def csp_lda(n_components: int = 6, reg="ledoit_wolf"):
    """CSP + LDA (the classical MI gold standard). Input: ``(n, ch, time)``."""
    from mne.decoding import CSP  # imported lazily (mne dependency)
    return make_pipeline(CSP(n_components=n_components, reg=reg, log=True),
                         LinearDiscriminantAnalysis())


def bandpower_lda():
    """Per-channel log band-power + LDA. Input: ``(n, ch, time)``."""
    return make_pipeline(LogVar(), StandardScaler(), LinearDiscriminantAnalysis())


def logreg():
    """Standardized logistic regression. Input: ``(n, features)`` (e.g. embeddings)."""
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def within_subject(estimator_fn, X, y, groups, *, n_splits: int = 5, seed: int = 0):
    """Mean ± sd of per-subject stratified k-fold accuracy."""
    accs = []
    for g in np.unique(groups):
        m = groups == g
        cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
        accs.append(cross_val_score(estimator_fn(), X[m], y[m], cv=cv).mean())
    return float(np.mean(accs)), float(np.std(accs))


def cross_subject(estimator_fn, X, y, groups, *, n_permutations: int = 0, seed: int = 0):
    """Leave-one-subject-out accuracy. With ``n_permutations>0`` also returns null mean + p."""
    if n_permutations:
        sc, perm, p = permutation_test_score(
            estimator_fn(), X, y, groups=groups, cv=LeaveOneGroupOut(),
            n_permutations=n_permutations, scoring="accuracy", random_state=seed, n_jobs=1)
        return float(sc), float(perm.mean()), float(p)
    sc = cross_val_score(estimator_fn(), X, y, groups=groups, cv=LeaveOneGroupOut()).mean()
    return float(sc), None, None

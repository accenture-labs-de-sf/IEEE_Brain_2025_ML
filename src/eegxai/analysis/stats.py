"""Group-level cluster-based permutation statistics for time-frequency maps.

Answers *where* (in time-frequency) an effect is reliable, while correcting for the many
correlated comparisons in a TF map. One-sample test across subjects: does the group-mean
percent change differ from 0 (baseline)? Neighbouring supra-threshold points are grouped
into clusters, and each cluster is tested against a sign-flip permutation null
(``mne.stats.permutation_cluster_1samp_test``). This is the significance layer of the MNE
ERDS-maps recipe.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sstats
import mne


def cluster_test_map(
    subject_maps: np.ndarray,
    *,
    p_accept: float = 0.05,
    n_permutations: int = 1024,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """Two-sided group cluster test on a stack of per-subject TF maps.

    ``subject_maps`` : ``(n_subjects, n_freqs, n_times)`` percent-change arrays.
    Returns ``(sig_mask, min_cluster_p)`` where ``sig_mask`` is a ``(n_freqs, n_times)``
    boolean of points belonging to a cluster with ``p < p_accept``.
    """
    n = subject_maps.shape[0]
    # two-sided t-threshold for cluster forming (p=0.05, df=n-1)
    threshold = float(sstats.t.ppf(1.0 - 0.05 / 2.0, n - 1))
    _t_obs, clusters, cluster_pv, _h0 = mne.stats.permutation_cluster_1samp_test(
        subject_maps, threshold=threshold, tail=0, n_permutations=n_permutations,
        seed=seed, out_type="mask", verbose="ERROR",
    )
    sig = np.zeros(subject_maps.shape[1:], dtype=bool)
    for cl, p in zip(clusters, cluster_pv):
        if p < p_accept:
            sig |= cl
    min_p = float(np.min(cluster_pv)) if len(cluster_pv) else 1.0
    return sig, min_p

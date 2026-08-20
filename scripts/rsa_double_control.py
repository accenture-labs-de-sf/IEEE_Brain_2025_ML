"""Definitive RSA with a task + total-power double control, on the full audit cache.

For each subject we build dissimilarity matrices (1 - correlation over trial pairs) for the EEGPT
embeddings, for per-channel band power (mu, beta, and two non-motor control bands), for the task
label, and for total broadband power. We then compute, per band, the partial correlation between
the embedding RDM and the band-power RDM controlling for (a) task alone and (b) task AND total
power together. The double control is the definitive test: if a band's correspondence survives it
while a non-motor control band does not, the structure is band specific rather than a shadow of
aggregate power. Significance is a one-sample t-test of the per-subject double-controlled partials
across subjects. We also report the embedding-vs-total-power correlation (the power-domination
number). Method follows the control-RDM / partial-correlation approach of Nili et al. 2014.

Run: python scripts/rsa_double_control.py --out results/audit_n35
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr, ttest_1samp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eegxai.pipeline import load_all  # noqa: E402
from eegxai.analysis.rsa import rdm, task_rdm  # noqa: E402

BANDS = [("mu", "bp_mu"), ("beta", "bp_beta"),
         ("control 2-7", "bp_ctl_low"), ("control 30-40", "bp_ctl_high")]


def partial_resid(y, x, controls):
    """Spearman partial correlation of y and x controlling for a list of control vectors.

    Rank-transform, then correlate the residuals of y and x after linear regression on the
    (ranked) controls plus an intercept. Handles one or several controls.
    """
    yr, xr = rankdata(y), rankdata(x)
    C = np.column_stack([rankdata(c) for c in controls] + [np.ones_like(yr)])
    ry = yr - C @ np.linalg.lstsq(C, yr, rcond=None)[0]
    rx = xr - C @ np.linalg.lstsq(C, xr, rcond=None)[0]
    return float(np.corrcoef(ry, rx)[0, 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/audit_n35"))
    args = ap.parse_args()

    d = load_all(args.out)
    E, Y, G, tot = d["E"], d["Y"], d["G"], d["totalpow"]
    subs = np.unique(G)
    lines = [f"DOUBLE-CONTROL RSA  (n={len(subs)} subjects, {len(Y)} trials)", ""]

    # embedding vs total power (the power-domination number)
    mt = []
    for s in subs:
        m = G == s
        mt.append(spearmanr(rdm(E[m]), rdm(tot[m])).correlation)
    lines.append(f"embedding geometry vs TOTAL power: Spearman {np.mean(mt):.3f} "
                 f"(t-test p={ttest_1samp(mt,0).pvalue:.1e})")
    lines.append("")
    lines.append(f"{'band':14} {'raw':>7} {'|task':>8} {'|task+power':>12} {'p (task+power)':>16} "
                 f"{'subj>0':>8}")

    results = {}
    for name, key in BANDS:
        bp = d[key]
        raw, p_task, p_tp = [], [], []
        for s in subs:
            m = G == s
            rm, rn = rdm(E[m]), rdm(bp[m])
            rt, rp = task_rdm(Y[m]), rdm(tot[m])
            raw.append(spearmanr(rm, rn).correlation)
            p_task.append(partial_resid(rm, rn, [rt]))
            p_tp.append(partial_resid(rm, rn, [rt, rp]))
        p_tp = np.array(p_tp)
        pval = ttest_1samp(p_tp, 0).pvalue
        frac = float(np.mean(p_tp > 0))
        results[name] = (np.mean(raw), np.mean(p_task), p_tp.mean(), pval, frac)
        lines.append(f"{name:14} {np.mean(raw):>7.3f} {np.mean(p_task):>8.3f} "
                     f"{p_tp.mean():>12.3f} {pval:>16.1e} {frac:>8.2f}")

    lines += ["", "Reading: 'raw' is model~band; '|task' controls task only; '|task+power' also",
              "removes total power (the definitive column). 'subj>0' is the fraction of subjects",
              "with a positive double-controlled partial (a consistency check)."]
    report = "\n".join(lines)
    print(report)
    (args.out / "double_control.txt").write_text(report + "\n")
    print(f"\nsaved: {args.out / 'double_control.txt'}")


if __name__ == "__main__":
    main()

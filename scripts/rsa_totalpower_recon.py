"""Reconcile the double-control RSA under different definitions of 'total power'.

The definitive double control removes total power, but 'total power' can be operationalized more
than one way, and the earlier n=20 run used a different proxy than our first n=35 run. This script
recomputes the task+total-power double-controlled partials under several total-power proxies, side
by side, so we can see whether the result (magnitude, and which control band collapses) is stable
to that choice. Cache-only: reads results/<out>/sub-*.npz, no re-extraction.

Run: python scripts/rsa_totalpower_recon.py --out results/audit_n35
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
BAND_KEYS = [k for _, k in BANDS]


def partial_resid(y, x, controls):
    yr, xr = rankdata(y), rankdata(x)
    C = np.column_stack([rankdata(c) for c in controls] + [np.ones_like(yr)])
    ry = yr - C @ np.linalg.lstsq(C, yr, rcond=None)[0]
    rx = xr - C @ np.linalg.lstsq(C, xr, rcond=None)[0]
    return float(np.corrcoef(ry, rx)[0, 1])


def total_proxies(d):
    """Return {name: (n_trials, 64) per-channel total-power feature} for each proxy."""
    broadband = d["totalpow"]                                   # log broadband variance
    summed = np.log(sum(np.exp(d[k]) for k in BAND_KEYS) + 1e-12)  # log summed band power
    return {"broadband_var": broadband, "summed_band_power": summed}


def run(d, tot_feat, subs):
    G = d["G"]; E, Y = d["E"], d["Y"]
    mt, rows = [], {}
    for s in subs:
        m = G == s
        mt.append(spearmanr(rdm(E[m]), rdm(tot_feat[m])).correlation)
    for name, key in BANDS:
        bp = d[key]; vals = []
        for s in subs:
            m = G == s
            vals.append(partial_resid(rdm(E[m]), rdm(bp[m]), [task_rdm(Y[m]), rdm(tot_feat[m])]))
        vals = np.array(vals)
        rows[name] = (vals.mean(), ttest_1samp(vals, 0).pvalue, float(np.mean(vals > 0)))
    return np.mean(mt), rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/audit_n35"))
    args = ap.parse_args()
    d = load_all(args.out)
    subs = np.unique(d["G"])
    proxies = total_proxies(d)

    lines = [f"TOTAL-POWER PROXY RECONCILIATION  (n={len(subs)} subjects, double control task+power)", ""]
    for pname, feat in proxies.items():
        mt, rows = run(d, feat, subs)
        lines.append(f"[{pname}]  embedding~total-power Spearman = {mt:.3f}")
        lines.append(f"  {'band':14} {'|task+power':>12} {'p':>10} {'subj>0':>8}")
        for name, (mean, p, frac) in rows.items():
            lines.append(f"  {name:14} {mean:>12.3f} {p:>10.1e} {frac:>8.2f}")
        lines.append("")
    report = "\n".join(lines)
    print(report)
    (args.out / "totalpower_recon.txt").write_text(report + "\n")
    print(f"saved: {args.out / 'totalpower_recon.txt'}")


if __name__ == "__main__":
    main()

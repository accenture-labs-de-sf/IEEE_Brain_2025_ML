"""Run ONE batch of the resumable audit feature extraction, then print a snapshot.

Each invocation processes the next `--batch-size` not-yet-done subjects and stops, so you can
watch progress build in small steps and stop or resume at will (already-done subjects are skipped).
Re-run the same command to continue with the next batch.

    python scripts/extract_features.py --subjects 1-35 --out results/audit_n35 --batch-size 5
    python scripts/extract_features.py --subjects 1-35 --out results/audit_n35 --summary-only

See eegxai.pipeline for the storage-lean design (streamed raw, compact per-subject features).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eegxai.pipeline import run_extraction_batch, summarize  # noqa: E402


def parse_subjects(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", default="1-35", help="e.g. '1-35' or '1,2,3,10-12'")
    p.add_argument("--out", type=Path, default=Path("results/audit_n35"))
    p.add_argument("--batch-size", type=int, default=5)
    p.add_argument("--summary-only", action="store_true", help="skip extraction, just summarize")
    args = p.parse_args()

    subjects = parse_subjects(args.subjects)
    if not args.summary_only:
        run_extraction_batch(subjects, args.out, batch_size=args.batch_size)
    try:
        summarize(args.out)
    except FileNotFoundError:
        print("[summary] nothing extracted yet")


if __name__ == "__main__":
    main()

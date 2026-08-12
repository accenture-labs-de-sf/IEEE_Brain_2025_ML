# Collaboration & Delegation

Internal coordination doc. How work is split so contributors move in **parallel** without
blocking the lead, and without changing the science.

_Last updated: 2026-08-04._

## Roles

- **Lead** — owns the scientific core (embedding choices, RSA design, faithfulness vs.
  mu/beta ERD, cross-subject reliability, interpretation). Defines the interfaces and the
  canonical results everyone else verifies against.
- **Research Engineer (RE)** — SWE + data science; no neuro background required.
- **PhD student** — HCI perspective; has Colab Pro + personal Claude.

## Shared purpose: reproducibility & accountability

Neither contributor branch produces new science. Both exist to **independently reproduce
the lead's results and hold them accountable** — so the numbers in the abstract are not
"true because they ran once on a laptop," but because they have been **re-derived on
different hardware, checked against a golden fixture, and documented transparently** for
anyone to audit. Two independent lines of reproduction + a documented audit trail = results
the whole team can stand behind.

**Contract for both branches:** same inputs → same outputs (within numerical tolerance),
just faster / bigger / more legible. Import the lead's package interfaces; do **not** change
what is measured.

## Branch 1 — Research Engineer: `perf/local-gpu`
**Performance + ambitious local compute**

- Profile and GPU-accelerate the pipeline (batched EEGPT embedding extraction, mixed
  precision, `DataLoader` over the memory tiers, pinned memory).
- Own the *heavy* configurations a laptop / Colab can't run: full-dataset-in-RAM (tier 5+),
  larger batches, higher sfreq / more channels / both datasets at once, GPU-bound XAI passes
  (e.g. attribution over many trials) — using a workstation with more RAM + GPU.
- **Reproducibility & accountability role:** independently re-computes the lead's results on
  a different, bigger machine and confirms parity within tolerance — catching environment-,
  precision-, or scale-dependent bugs and proving findings aren't an artifact of one setup.
- **Medium:** optimized `.py` behind the same interfaces. **Goal:** fast + big, independently
  re-verified. **Deliverable:** GPU-optimized code path + benchmark/scale report + parity check.

## Branch 2 — PhD student: `guide/reproducible-notebook`
**Open-source methods guide**

- A narrated notebook (markdown + inline visualizations + commentary) walking the analysis
  end-to-end: raw data → preprocessing → embeddings → RSA / faithfulness → figures, explaining
  the choices at each step.
- Runs on Colab (no local setup needed by anyone), Drive-persistent cache, imports the lead's
  package code so it stays in sync with the real analysis.
- **Reproducibility & accountability role:** makes the method **open and inspectable** — an
  external researcher or reviewer can follow the logic, re-run it, and catch errors.
  Transparency *is* accountability. Becomes the public reproducibility companion to the
  submission.
- **Medium:** narrated notebook for humans/researchers. **Goal:** open + legible, publicly
  reproducible. **Deliverable:** `notebooks/` visual walkthrough with rendered outputs.

## The golden fixture (what keeps both honest)

The lead checks in a small frozen reference — e.g. embeddings + RSA output for subjects 1–3.
Both branches verify against it:
- RE proves it reproduces **at scale, on other hardware**.
- PhD proves it reproduces **transparently, for anyone**.

Because both branches sit behind the same interfaces and target the same golden numbers,
their speedups (RE) and documentation (PhD) merge back cleanly, and nothing drifts from the
science the lead defined.

## Ground rules

- **Start by reproducing Tier A**: follow [`docs/reproducing-tier-a.md`](../docs/reproducing-tier-a.md)
  and verify your numbers against [`reports/mi_erds_n20/`](../reports/mi_erds_n20) before building on it.
- Branch from `main`; open a PR (don't push to `main` directly).
- Don't modify the analysis logic/interfaces; optimizations and narration go *around* them.
- If a branch's run disagrees with the golden fixture, that's a finding — raise it, don't
  paper over it.

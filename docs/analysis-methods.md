# Analysis Methods — Empirical ERDS Characterization

How the motor-imagery empirical reference is computed. This is the **canonical** MNE-Python
ERDS-maps pipeline (Pfurtscheller & Lopes da Silva, 1999), chosen because this subfield expects
specific, established library/parameters. Implemented in `eegxai.analysis.erd` and driven by
`scripts/exploration_report.py`.

## Concepts (how it works)

**Why an empirical reference.** We first establish what the brain *actually does* during motor
imagery, so we later have a ground truth to test whether EEGPT's representation encodes the same
structure.

**ERD = a power drop.** When a patch of cortex idles, its neurons fire in synchrony, producing a
strong rhythm — the **mu** rhythm (~10 Hz) over sensorimotor cortex. When that patch *activates*
(e.g. imagining a hand movement), the neurons **desynchronise** and the rhythm's power **drops**.
That drop is Event-Related Desynchronisation (ERD). So "activation" appears as a *decrease* in
mu/beta power — and it is **contralateral** (left-hand imagery → right hemisphere, and vice-versa).

**Time-frequency.** Raw EEG is one squiggle mixing all frequencies. To see "how much 10 Hz power
exists at each instant," we decompose each channel into a **frequency × time** grid (a
spectrogram). Multitaper is a robust, low-variance way to estimate that power.

**Baseline as percent.** Absolute power varies hugely across electrodes and people, so each point
is expressed as **% change from a pre-cue rest window**: `(power − rest)/rest`. Negative = power
dropped = ERD. This makes it comparable and readable ("mu fell 36% at C3").

**Occipital alpha ≠ mu.** Posterior ~10 Hz activity is *alpha* (visual cortex), a different
phenomenon from central *mu*; the blue (ERS) often seen posteriorly is not the motor signal.

## Pipeline

1. **Preprocess** (match EEGPT): average reference, 0–38 Hz band-pass, resample 160→250 Hz.
   See [`preprocessing.md`](preprocessing.md).
2. **Epoch** around each imagery cue with a buffer: `tmin = -2.0 s`, `tmax = 4.5 s`, **no** epoch
   baseline (applied later, to the TFR). After the TFR baseline, **crop to −1..3.9 s** to discard
   multitaper edge ringing (the buffer also keeps the −1..0 s baseline itself free of edge effects).
3. **Time-frequency (multitaper)**: `epochs.compute_tfr(method="multitaper", freqs=2..35 Hz,
   n_cycles=freqs, decim=3)`. This estimates power at each frequency over time.
4. **Baseline correction (percent)**: `apply_baseline((-1, 0), mode="percent")` — express each
   time-frequency point as fractional change from the pre-cue reference. **Negative = ERD**
   (power drop / desynchronisation); positive = ERS. We display ×100 as percent.
5. **Read-outs** (from the same TFR, so they are consistent):
   - **ERDS maps** at C3 / Cz / C4 — frequency × time, showing the mu/beta suppression course.
   - **Mu-band topography** — average the percent TFR over 8–13 Hz and the 0.5–3.5 s task window
     → one value per channel → scalp map.
   - **Lateralization index** = mu ERD(C3) − ERD(C4) per class (LEFT expect > 0 → C4 desync;
     RIGHT expect < 0 → C3 desync).

## Why these choices

- **Multitaper** gives a low-variance power estimate with controlled spectral smoothing — the
  standard for ERDS maps (vs. a single band-pass + square, which is the older "band-power"
  method and less robust).
- **`n_cycles = freqs`** keeps a constant relative bandwidth across frequencies (longer windows
  at higher frequencies), as in the MNE example.
- **Percent baseline** is the classic ERD% (Pfurtscheller); more interpretable and comparable
  across channels/subjects than raw power or dB.
- **Runs 4/8/12** (left vs. right fist imagery) are the correct choice for C3/C4 lateralization
  (the MNE example's runs 6/10/14 are hands-vs-feet).

## Interpreting the output

- On the maps: a red band around ~10 Hz (mu) and ~20 Hz (beta) after the cue = motor ERD.
- Contralateral expectation: RIGHT-hand imagery → stronger ERD at **C3** (left hemisphere);
  LEFT-hand imagery → stronger ERD at **C4**.
- Posterior blue on topographies = occipital **alpha** increase (visual), a separate phenomenon
  from central **mu** — do not conflate them.

## Significance (cluster-permutation)

Implemented in `eegxai.analysis.stats.cluster_test_map`. Group-level **two-sided one-sample test
across subjects**, per channel/class (`mne.stats.permutation_cluster_1samp_test`; cluster-forming
t-threshold at p=0.05, 1024 sign-flip permutations): neighbouring supra-threshold time-frequency
points are grouped into clusters, and each cluster is tested against the permutation null.
Outlined regions on the ERDS maps have cluster **p < 0.05**, correcting for the many
time-frequency comparisons. This is the significance layer of the MNE ERDS recipe.

## Still open (optional refinements — see `findings-and-options.md`)

- Optional spatial sharpening (surface Laplacian) to further sharpen the (weak) LEFT lateralization,
  and scaling to n > 20 / all 109. Edge-cropping and cluster stats are now applied; at **n=20 all
  six channel×class ERDs are significant**.

## References (method grounding)

**Analysis 1 — ERD/ERDS via time-frequency + baseline:**
- Pfurtscheller & Lopes da Silva (1999). *Event-related EEG/MEG synchronization and
  desynchronization: basic principles.* Clin. Neurophysiol. 110(11):1842–1857. — the ERD/ERS
  percent-baseline method.
- MNE-Python ERDS-maps example (multitaper TFR + percent baseline) — reference implementation:
  https://mne.tools/stable/auto_examples/time_frequency/time_frequency_erds.html
- Recent practice (2023–2025) using baseline-corrected TFR ERD/ERS in mu (8–13 Hz) and beta
  (13–30 Hz): tactile-imagery ERD, *eNeuro* 10(6) 2023 (ENEURO.0455-22.2023); motor-imagery ERD,
  *Front. Hum. Neurosci.* 2025 (10.3389/fnhum.2025.1545492). Note: Morlet wavelets are the common
  alternative TFR method; multitaper (used here, per the MNE example) is equally standard.

**Analysis 2 — cluster-based permutation statistics:**
- Maris & Oostenveld (2007). *Nonparametric statistical testing of EEG- and MEG-data.* J.
  Neurosci. Methods 164(1):177–190. doi:10.1016/j.jneumeth.2007.03.024 — the canonical
  cluster-permutation method (FWER control exploiting time/frequency/space adjacency), implemented
  by MNE's `permutation_cluster_1samp_test`.
- Recent (2025): Rousselet. *Using cluster-based permutation tests to estimate MEG/EEG onsets: how
  bad is it?* Eur. J. Neurosci. doi:10.1111/ejn.16618 — confirms current use, and cautions that
  cluster p-values are **cluster-level, not point-wise**: an outlined region shows *that* a
  significant effect exists, **not** its exact onset/boundaries (so don't over-read cluster edges).
  The MNE ERDS example (v1.12) applies the same test — the documented standard for ERDS maps.

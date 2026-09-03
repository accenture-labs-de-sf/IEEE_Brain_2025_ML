# Sensorimotor Rhythm or Aggregate Power? An Interpretability Audit of EEGPT on Motor Imagery

**Authors [placeholder]. Affiliation [placeholder].**

## Abstract

EEG foundation models match feature engineered baselines on many tasks, yet recent audits report that they lean on the aperiodic 1/f background of the signal and the broadband power that comes with it, more than on the specific oscillations that carry task meaning. We audit EEGPT, a foundation model that adds a spatio temporal representation alignment objective to masked reconstruction and is absent from prior audits, on motor imagery from the PhysioNet dataset (imagery versus rest, 35 subjects). A linear classifier on EEGPT's frozen embeddings decodes the task at ≈ 71% within subject, on par with feature engineered baselines, yet the embedding geometry is dominated by aggregate, largely aperiodic power (Spearman r ≈ 0.24–0.32, higher than any single band). We find that the evidence for rhythm specificity depends critically on the analysis: raw band power, which conflates periodic and aperiodic activity, makes a non motor control band survive a total power control as strongly as the sensorimotor bands, while sensorimotor averaging inflates apparent mu specificity. A parameterized, spatially controlled analysis (aperiodic-adjusted oscillatory power, per channel, with control bands, a central versus occipital contrast, and a permutation null) leaves only a small and consistent trend toward the sensorimotor mu rhythm: mu is tracked ≈ 0.04 above non motor control bands, central-weighted over the scalp rather than posterior alpha, and above chance, while beta shows no reliable trend. We frame the mu effect as a trend, not an established component, since at this sample size it is detectable but small. The result characterizes EEGPT as power dominated with at most a weak sensorimotor grounding, and it carries a methodological caution for the interpretability audit literature itself.

## 1. Introduction

Electroencephalography (EEG) foundation models pretrain on large unlabeled corpora and transfer to downstream clinical and brain computer interface (BCI) tasks, often surpassing pipelines built on hand crafted features [1]. Their accuracy is documented, but a separate question governs trust: is a model's competence grounded in the functional neural signal a clinician would recognize, or in a lower level statistic that happens to correlate with the task.

The EEG power spectrum separates into two components that are both clinically meaningful, but at different granularities. The periodic component, the oscillations such as the sensorimotor mu and beta rhythms, indexes specific functional processes [7]. The aperiodic 1/f component is coarser: its slope tracks the overall balance of excitation and inhibition, a biomarker in its own right, but a broadband one rather than a marker of any particular process [5]. For motor imagery, mu and beta event related desynchronization over sensorimotor cortex is a validated motor signal [7], whereas broadband power moves with muscle activity, movement, arousal, and electrode noise. A model that decodes through the rhythm is interpretable and mappable to physiology; one that decodes through aggregate power reaches the same accuracy on a more fragile basis.

Recent audits report that EEG foundation models lean on the aperiodic background and aggregate power more than on specific oscillations [1, 2, 3, 4]. EEGPT is a useful next case: it is absent from these audits and architecturally distinct, adding a spatio temporal representation alignment objective to masked reconstruction, meant to capture consistent, high signal to noise structure [6]. It is a test of whether a deliberate change to the pretraining objective alters what a model encodes. Answering it turns out to require care, because raw band power conflates periodic and aperiodic activity [9] and spatial averaging can inflate apparent rhythm specificity, so the conclusion depends on the analysis. Our contributions are: (i) an audit of EEGPT, a model and architecture not covered by prior work; (ii) the finding that EEGPT is power dominated with only a small, spatially and spectrally controlled trend toward the sensorimotor mu rhythm; and (iii) a methodological demonstration that the specificity verdict flips with analysis choices, resolved with a parameterized, spatially controlled protocol.

## 2. Related work

**Auditing EEG foundation models.** A spectral bias study reports that reconstruction based models encode the aperiodic exponent and offset but under represent oscillations [1]. Here "reconstruction based" means masked self supervision, masking parts of the signal and reconstructing them or a tokenized version, in the BERT and MAE family, not autoregressive generation. Broader audits map feature families and spatial grounding [2, 3], and physiologically grounded interpretability maps prediction attributions into spectral and source domains [4]. None includes EEGPT.

**Parameterizing spectra.** The periodic and aperiodic decomposition follows spectral parameterization [5], with practical recommendations for separating oscillations from 1/f activity [10]. Raw band power and band ratios conflate the two, so parameterized measures are preferred [9]. We adopt this throughout.

**Representational similarity analysis.** RSA compares a representation's geometry to reference geometries by correlating dissimilarity matrices, with control models removed by partial correlation [8].

**Sensorimotor rhythms.** Motor imagery elicits mu and beta desynchronization over sensorimotor cortex [7], the signal that Common Spatial Patterns [11] isolate.

## 3. Methods

**Data and model.** We use PhysioNet motor imagery (imagery runs), imagery versus rest, 35 subjects (2891 cued four-second trials, ~83 per subject, each a single imagery or rest period from the cued runs), cleaned conservatively with Autoreject [12]. We audit the frozen EEGPT encoder, using its 512 dimensional embeddings (braindecode weights, mean pooled) as they are used downstream. The same cleaned trials feed the model and the spectral references.

EEGPT is trained with a dual self supervised objective [6]: masked reconstruction of the signal, plus a spatio temporal representation alignment in which a predictor matches the encoder's latent representations of masked patches to those produced by a momentum updated target encoder. The alignment loss lives in representation space, a JEPA-like latent prediction rather than pixel reconstruction, and is meant to capture consistent, high signal to noise structure. Despite its name EEGPT is not an autoregressive model; the added latent alignment is what distinguishes it from the pure masked reconstruction models above.

**Decoding.** We decode imagery versus rest within subject (five fold) and across subjects, against Common Spatial Patterns [11] and band power baselines.

**Representational similarity analysis.** For each subject we build a model dissimilarity matrix from the embeddings (one minus correlation over trial pairs) and compare it to spectral references [8]. To avoid the band power conflation [9], we parameterize each trial spectrum into aperiodic (offset, 1/f exponent) and periodic components [5, 10] and build the oscillatory reference from aperiodic-adjusted power (the residual above the fitted 1/f, integrated per band), per channel, for motor bands (mu, beta) and non motor control bands (theta, gamma). We report partial correlations controlling for the task label and for the aperiodic component, a channel-space (topographic) central versus occipital contrast on mu that distinguishes sensorimotor mu from posterior alpha without claiming source localization, and a trial shuffling permutation null. As a reference for aggregate power we also correlate the embedding geometry with a total power matrix.

## 4. Results

### 4.1 EEGPT decodes the task and its geometry is dominated by aggregate power

A linear classifier on the frozen embeddings decodes imagery versus rest at ≈ 71% within subject (chance 50%), on par with feature engineered baselines, and transfers modestly across subjects (leave-one-subject-out ≈ 60%, chance 50%, 31 of 35 subjects above chance), within the modest range reported for cross-subject motor imagery [13] and consistent with a power-dominated representation. Yet its representational geometry is dominated by aggregate power: it correlates with total signal power at Spearman r ≈ 0.24–0.32 (depending on the total-power proxy), higher than with any single band (Figure 1a, 1b).

### 4.2 The specificity verdict depends on the analysis

The evidence for EEGPT's sensorimotor specificity flips with the method. Using raw per channel band power with a task and total power double control, a non motor 30–40 Hz control band survives as strongly as mu and beta, which would suggest no specificity, but raw band power conflates periodic and aperiodic activity [9], so this test is confounded. Using aperiodic-adjusted oscillatory power on a sensorimotor-averaged spectrum, mu appears strongly specific (correlation ≈ 0.24 versus ≈ 0.05 for controls), but averaging over central channels inflates mu. Neither is decisive on its own.

### 4.3 A small, controlled mu trend

Holding the spatial construction fixed (aperiodic-adjusted oscillatory power, all 64 channels) resolves it (Table 1, Figure 1c). Motor bands sit only modestly above non motor controls: mu 0.130, beta 0.113, versus theta 0.089 and gamma 0.088. The mu gap is reliable across subjects (mu above theta p ≈ 0.002; mu above gamma p ≈ 0.0003; positive in roughly 80% of subjects), whereas beta is not (p ≈ 0.07–0.15). The mu effect is stronger over central than occipital channels (central mu 0.061 versus occipital 0.027; central survives controlling occipital, p < 0.001), a scalp-topographic distinction, not source localization, consistent with sensorimotor mu rather than posterior alpha [7], and it exceeds a trial shuffling null (whole head mu 0.130 versus null 0.000 ± 0.009, p ≈ 0.001). We read this as a small, consistent trend toward a sensorimotor mu component, not an established component: at n = 35 it is detectable but small, and whether it holds at larger samples is open, since significance reflects detectability, not magnitude.

Table 1. Aperiodic-adjusted oscillatory power tracked by the embedding geometry (per channel, task controlled, n = 35).

| band | correlation | motor vs control |
| --- | --- | --- |
| mu (8–13) | 0.130 | vs theta +0.041 (p = 0.002); vs gamma +0.042 (p = 0.0003) |
| beta (13–30) | 0.113 | vs theta +0.025 (p = 0.15); vs gamma +0.026 (p = 0.07) |
| theta (4–7, control) | 0.089 | reference |
| gamma (30–45, control) | 0.088 | reference |

![Figure 1. The audit on real data (n = 35). (a) Grand-average sensorimotor power spectrum with its fitted aperiodic 1/f component and the mu peak above it. (b) The embedding geometry tracks aggregate total power strongly (dashed line, Spearman ≈ 0.24), while the aperiodic-adjusted oscillatory bands are small and mu sits only ≈ 0.04 above the non-motor control bands. (c) Per-channel alignment of aperiodic-adjusted mu power with the embedding geometry (task controlled) is central-weighted, consistent with sensorimotor mu rather than posterior alpha.](figures/fig1_overview.png)

## 5. Discussion

EEGPT decodes motor imagery competently, yet its representation is organized primarily by aggregate, largely aperiodic power, with only a small and consistent trend toward the sensorimotor mu rhythm and no reliable beta trend. Despite its added alignment objective, the model does not represent the functional rhythm strongly, which extends the aperiodic and aggregate power bias reported for reconstruction based models to a hybrid architecture. The balanced reading for neurotechnology is that EEGPT reliably captures a coarse scale signal, aggregate and largely aperiodic power, which is itself physiologically meaningful rather than noise [5], but it has not yet shown strong specificity to the finer sensorimotor rhythm; aggregate power is also the part most perturbed when recording conditions shift across sessions or devices. Because EEGPT already pairs masked reconstruction with a JEPA-like latent alignment and still shows only a coarse grounding, the pattern may reflect current self supervised pretraining broadly rather than any single objective. The same rhythm versus power lens should therefore be applied to autoregressive, GPT-style models [14] and to joint-embedding predictive (JEPA) models [15] alike.

The audit also carries a methodological caution for the interpretability literature itself: the specificity verdict flipped from non specific to strongly specific to a small trend as we moved from raw band power to sensorimotor averaging to a parameterized, spatially controlled analysis. Band power conflates periodic and aperiodic activity [9], and spatial averaging inflates apparent specificity, so specificity claims about foundation models should rest on parameterized, spatially controlled measures.

**Limitations.** The audit covers one model, one dataset, and one paradigm; the RSA is correlational; the mu trend is small; the spatial control establishes central weighting in channel space rather than source localization, and because EEG is volume conducted a central scalp topography constrains but does not identify the cortical generator; and we have not run a head-to-head against other foundation models, so we can say the bias persists in EEGPT, not that EEGPT is better or worse than pure reconstruction models.

**Future work.** A head-to-head RSA against pure reconstruction models (does the alignment objective matter), a cross subject RSA asking whether the transferable component is the rhythm or the power, scaling to more subjects to test whether the mu trend persists, and a causal attribution or erasure test [4] would each strengthen the account.

## 6. Conclusion

We audit EEGPT, an alignment plus reconstruction EEG foundation model absent from prior audits, on motor imagery. It decodes well but represents the task mainly through aggregate, largely aperiodic power, with only a small, spatially and spectrally controlled trend toward the sensorimotor mu rhythm. The specificity verdict depends on the analysis, which is itself a caution for how these models are audited.

## References

[1] Aperiodic and Low Frequency Spectral Bias in Reconstruction based EEG Foundation Models. arXiv:2605.26434, 2026.
[2] Tang et al. What Do EEG Foundation Models Capture from Human Brain Signals? arXiv:2605.11410, 2026.
[3] Beyond Accuracy: Robustness, Interpretability and Expressiveness of EEG Foundation Models. arXiv:2605.17562, 2026.
[4] Shama, Amornsirikul, Venkataraman. EEG-PRISM: Physiologically-Grounded Interpretability of Predictions by EEG Foundation Models. arXiv:2608.13676, 2026.
[5] Donoghue et al. Parameterizing neural power spectra into periodic and aperiodic components. Nature Neuroscience, 2020.
[6] Wang et al. EEGPT: Pretrained Transformer for Universal and Reliable Representation of EEG Signals. NeurIPS 2024.
[7] Pfurtscheller and Lopes da Silva. Event related EEG/MEG synchronization and desynchronization: basic principles. Clinical Neurophysiology, 1999.
[8] Kriegeskorte, Mur, Bandettini. Representational similarity analysis. Frontiers in Systems Neuroscience, 2008.
[9] Donoghue, Dominguez, Voytek. Electrophysiological frequency band ratio measures conflate periodic and aperiodic neural activity. eNeuro, 2020.
[10] Gerster et al. Separating neural oscillations from aperiodic 1/f activity: challenges and recommendations. Neuroinformatics, 2022.
[11] Blankertz et al. Optimizing spatial filters for robust EEG single trial analysis (CSP). IEEE Signal Processing Magazine, 2008.
[12] Jas et al. Autoreject: Automated artifact rejection for MEG and EEG data. NeuroImage, 2017.
[13] Jayaram, Barachant. MOABB: trustworthy algorithm benchmarking for BCIs. Journal of Neural Engineering, 2018.
[14] Neuro-GPT: Towards a Foundation Model for EEG. arXiv:2311.03764, 2023.
[15] Assran et al. Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture. CVPR, 2023.

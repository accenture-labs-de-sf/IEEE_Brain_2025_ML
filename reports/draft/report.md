# Sensorimotor Rhythm or Aggregate Power? An Interpretability Audit of EEGPT on Motor Imagery

**Authors [placeholder]. Affiliation [placeholder].**

## Abstract

EEG foundation models now match feature engineered baselines on many tasks, yet recent audits report that reconstruction based models preferentially encode the aperiodic 1/f background of the signal and under represent task relevant oscillations, a spectral bias attributed to the reconstruction objective. Whether this bias is specific to pure reconstruction models or is a more general property of the paradigm is open, and it matters for clinical trust because the periodic and aperiodic parts of the EEG carry different physiology. We audit EEGPT, a foundation model that adds a spatio temporal representation alignment objective to masked reconstruction and is therefore a natural test of whether alignment rescues oscillatory specificity. Using motor imagery from the PhysioNet dataset, we contrast imagined movement against rest across 20 subjects with linear decoding and representational similarity analysis (RSA) with a total power control. EEGPT decodes imagery versus rest about as well as Common Spatial Patterns within subject and transfers best of the compared methods across subjects, yet its representation is organized primarily by aggregate signal power. A weak but genuine mu and beta specific component survives removal of total power, while a non motor control band does not. The alignment objective does not rescue rhythm specificity here, which extends the aperiodic bias finding to a new architecture and gives a concrete, trust relevant characterization of what EEGPT does and does not encode. [Placeholder: a periodic and aperiodic (specparam) decomposition and a cross subject RSA are in progress.]

## 1. Introduction

Electroencephalography (EEG) foundation models pretrain on large unlabeled corpora and transfer to downstream clinical and brain computer interface (BCI) tasks, often surpassing pipelines built on decades of hand crafted features such as band power and event related desynchronization [1]. Their downstream accuracy is well documented, but a separate question governs whether they can be trusted: is a model's competence grounded in the functional neural signal a clinician would recognize, or in a lower level statistic that happens to correlate with the task.

The right way to pose this question is now reasonably clear. The EEG power spectrum decomposes into a periodic component, the oscillatory rhythms such as mu and beta that index specific functional processes, and an aperiodic 1/f component whose exponent tracks the balance of excitation and inhibition and is itself a clinical biomarker [2]. Conflating the two is a known error, because they move independently and mean different things physiologically. For motor imagery in particular, mu and beta event related desynchronization over sensorimotor cortex is a validated motor signal [3], whereas broadband power is perturbed by muscle activity, movement, arousal, and electrode artifacts. A model that decodes motor imagery through the rhythm is interpretable and mappable to known physiology; one that decodes through aggregate power reaches the same accuracy on a different and more fragile basis. This is why explanations of these models are most useful when expressed in physiologically meaningful spectral and spatial terms rather than in the raw input space [8].

Recent work has begun to measure which of these a model encodes. A spectral bias study reports that reconstruction based EEG foundation models linearly decode the aperiodic exponent and offset but not oscillatory frequency information, and attributes this to a reconstruction loss dominated by 1/f power [4]. Broader audits find that these models align with hand crafted feature families and ground spatially on task appropriate electrodes [5, 6], spectral audit frameworks manipulate the aperiodic component to measure task dependent reliance on it [7], and physiologically grounded interpretability methods map model attributions into the frequency and source domains that clinicians actually read [8]. These results establish that at least some foundation models lean on aggregate and aperiodic structure rather than specific rhythms. What they leave open is whether this is a property of the reconstruction objective alone. If a model adds an objective designed to capture consistent, high signal to noise structure, does oscillatory specificity return.

EEGPT is the natural case for this question. It is trained with a dual self supervised objective that combines masked reconstruction with spatio temporal representation alignment, where the alignment branch is meant to capture consistent structure rather than to reproduce the raw signal [1]. It is openly available (braindecode) and absent from the audits above, which makes it a useful case for asking whether a deliberate change to the pretraining objective changes what a model encodes from the signal, or whether the reported bias persists regardless of architecture. We therefore audit EEGPT on PhysioNet motor imagery, contrasting imagery against rest, using linear decoding and representational similarity analysis (RSA) with a total power control (Figure 1). Our contributions are: (i) an audit of EEGPT, an alignment plus reconstruction model not covered by prior work, framed as a test of whether alignment escapes the reported aperiodic bias; (ii) the finding that EEGPT decodes competently yet represents motor imagery largely through aggregate power, with only a weak mu and beta specific component, so alignment does not rescue rhythm specificity here; and (iii) a connection between representational content and cross subject transfer. [Placeholder: a periodic and aperiodic decomposition localizing the effect.]

![Figure 1. Study overview. (a) A real sensorimotor rhythm is present: group mean mu event related desynchronization is focal over contralateral sensorimotor cortex, and the C3 time-frequency map shows significant mu and beta desynchronization during imagery. (b) The frozen EEGPT encoder yields a 512 dimensional embedding per trial. (c) The embedding dissimilarity matrix, sorted rest block then imagery block, carries block-diagonal structure, which is the object RSA measures. (d) The audit question, on an illustrative spectrum: does that structure track the periodic sensorimotor rhythm or the aperiodic 1/f aggregate power. Figure 2 and Section 4.3 answer it with data.](figures/fig1_overview.png)

## 2. Related work

**Spectral bias in foundation models.** The closest prior work shows that reconstruction based EEG foundation models (LaBraM, CBraMod, CSBrain) encode the aperiodic exponent and offset with high linear decodability but under represent oscillatory power, especially at higher frequencies, on datasets including PhysioNet motor imagery [4]. We extend this to EEGPT, which is not purely reconstruction based, and use representational geometry rather than linear probes. The periodic and aperiodic decomposition itself follows the spectral parameterization framework [2], and aperiodic manipulation has been proposed as a general model diagnostic [7].

**Auditing EEG foundation model representations.** Tang et al. audit several models against a hand crafted feature lexicon on clinical tasks and report that frequency features dominate causal mass [5]. The Beyond Accuracy study evaluates six models with attribution and probing across paradigms including motor imagery, reporting task appropriate spatial grounding and a pooling artifact in head only evaluation [6]. EEG-PRISM maps post hoc attributions from foundation models into the spectral and source domains, arguing that explanations must be physiologically grounded to be clinically useful [8]. These methods explain predictions; we instead characterize the geometry of the representation and separate the specific rhythm from aggregate power. None includes EEGPT.

**Representational similarity analysis.** RSA compares the geometry of a representation to that of a reference by correlating dissimilarity matrices [9], and is standard for auditing neural network representations on EEG and MEG.

**Sensorimotor rhythms.** Motor imagery elicits mu and beta event related desynchronization over sensorimotor cortex [3], the signal that Common Spatial Patterns [10] are designed to isolate.

## 3. Methods

**Data and task.** We use the PhysioNet EEG Motor Movement and Imagery dataset, imagery runs, for 20 subjects with roughly 90 trials each. Each 4 s trial is labeled imagery (imagined left or right hand movement) or rest. We frame decoding as imagery versus rest, a strong and unambiguous mu and beta contrast, after finding that left versus right is near the classical decodability ceiling (about 55 percent) and therefore uninformative.

**Model.** EEGPT is trained with a dual self supervised objective, masked reconstruction plus spatio temporal representation alignment, and this hybrid design is why it tests whether alignment escapes the aperiodic bias reported for pure reconstruction models [1, 4]. We use its encoder embeddings (braindecode weights), frozen, as a 512 dimensional vector per trial (mean pooled), which is how the model is used downstream. Mean pooling can understate representation quality relative to token flattening [6].

**Preprocessing and cleaning.** Signals are average referenced, band limited to 0 to 38 Hz, and resampled to 250 Hz, preserving the full spectrum rather than any single band. Artifacts are handled conservatively with Autoreject [11], which repairs or rejects bad epochs without decomposing the signal. The same cleaned trials feed both the model and the feature extraction.

**Decodability.** We decode imagery versus rest within subject (five fold cross validation per subject) and cross subject (leave one subject out), using logistic regression on embeddings and, as classical references, Common Spatial Patterns with linear discriminant analysis and per channel band power with linear discriminant analysis. Significance uses label permutation.

**Representational similarity analysis.** For each subject we build a model dissimilarity matrix from the embeddings (one minus correlation over trial pairs), neural dissimilarity matrices from per channel log band power in mu (8 to 13 Hz), beta (13 to 30 Hz), and a non motor control band (2 to 7 Hz), and a task dissimilarity matrix (same versus different condition). We report Spearman correlations between matrices, and partial correlations controlling for the task matrix and for a total power matrix (summed band power). Total power is a proxy for the aperiodic component; a full periodic and aperiodic decomposition is the planned refinement described in Section 4.3.

## 4. Results

### 4.1 A focal sensorimotor rhythm is present, and EEGPT decodes the task

The dataset carries a genuine sensorimotor rhythm. Baseline corrected mu event related desynchronization is focal over contralateral sensorimotor cortex, and the C3 time-frequency map shows significant mu and beta desynchronization during imagery (Figure 1a). On the imagery versus rest contrast, EEGPT decodes about as well as Common Spatial Patterns within subject and best of the compared methods across subjects (Table 1). Cross subject decoding is significant but modest, consistent with the difficulty of subject transfer in EEG.

Table 1. Decoding accuracy (percent, chance 50).

| method | within subject | cross subject |
| --- | --- | --- |
| EEGPT embeddings | 73.1 | 58.3 (p = 0.01) |
| CSP with LDA | 76.3 | 54.7 |
| band power with LDA | 63.3 | 57.3 |

### 4.2 The representation is dominated by aggregate power

Despite the focal rhythm and competent decoding, EEGPT's representational geometry is organized primarily by aggregate signal power. RSA compares the embedding dissimilarity matrix to band power and task dissimilarity matrices (Figure 2a); the embedding geometry correlates with total signal power at Spearman 0.31, larger than its correlation with any single band. Controlling only for task, all three bands correlate similarly (mu 0.18, beta 0.23, control 0.17), which alone would suggest no band specificity. Controlling additionally for total power separates them: mu (0.085) and beta (0.117) remain significantly positive, while the non motor control band collapses to near zero (minus 0.05) (Figure 2b). The band specific structure that is invisible before the control emerges after it, and only for the sensorimotor bands. Consistent with a power dominated contrast, the imagery versus rest change in mu and beta power is spatially broad rather than focal (Figure 2c).

Table 2. RSA partial correlations (within subject, n = 20).

| band | partial, task removed | partial, task and total power removed |
| --- | --- | --- |
| mu (8 to 13) | 0.183 | 0.085 (p = 3e-3) |
| beta (13 to 30) | 0.228 | 0.117 (p = 8e-4) |
| control (2 to 7) | 0.169 | minus 0.054 |

![Figure 2. Representational geometry. (a) RSA compares the EEGPT embedding dissimilarity matrix to mu and beta band power and to the task label, correlating the off-diagonal entries; the second-order correlation is positive for an illustrative subject. (b) Group partial correlations, n = 20. Controlling only for task (light), all bands correlate similarly; adding a total power control (dark) leaves mu and beta significantly positive but collapses the non motor control band. (c) The imagery versus rest change in mu and beta power is spatially broad, consistent with an aggregate power contrast.](figures/fig2_geometry.png)

### 4.3 Periodic and aperiodic decomposition [placeholder]

The total power control above is a proxy for the aperiodic component. The planned refinement parameterizes each trial spectrum into aperiodic (exponent and offset) and periodic (mu and beta peak) components [2], and asks which EEGPT's geometry tracks. This makes the result directly comparable to the aperiodic bias reported for reconstruction based models [4] and localizes where any oscillatory structure lives. [Figure 3 placeholder.]

## 5. Discussion

EEGPT decodes motor imagery competently and transfers across subjects better than the classical baselines we tested, yet its representation is organized primarily by aggregate signal power. A genuine mu and beta specific component exists, since it survives the total power control while a non motor band does not, but it is weak. The reading is that EEGPT's added spatio temporal alignment objective does not, on this task, rescue the oscillatory specificity that reconstruction based models were reported to lack [4]. The bias toward aggregate and largely aperiodic structure appears to be more general than the pure reconstruction objective. Because the aperiodic and periodic parts of the EEG carry different physiology [2], a representation grounded in aggregate power is a caution for anyone deploying the model where the specific sensorimotor rhythm is what confers trust or robustness.

**Limitations.** The general phenomenon, that these models lean on aggregate and aperiodic structure, is established by prior work; our contribution is the extension to an alignment plus reconstruction model, a representational geometry method, and the cross subject link. The audit covers one model, one dataset, and one paradigm, the RSA is correlational rather than causal, and the surviving band specific effects are small. Total power is only a proxy for the aperiodic component, which the planned decomposition addresses. The pooling choice may understate the representation, and heavy artifact removal is deliberately avoided to keep the input close to the model's pretraining distribution.

**Future work.** The periodic and aperiodic decomposition, a cross subject RSA that asks whether the transferable component is the rhythm or the power, scaling to more subjects, a second foundation model, and a layer wise probe of where band structure emerges would each strengthen the claim. A causal test is the natural next step: PRISM style spectral attribution [8] on EEGPT, or an erasure of the aperiodic component, would move the finding from correlational to causal. [Placeholder.]

## 6. Conclusion

We audit EEGPT, an alignment plus reconstruction EEG foundation model absent from prior audits, on motor imagery. EEGPT decodes well but represents the task largely through aggregate power, with a weak sensorimotor specific component that survives a total power control. The added alignment objective does not rescue oscillatory specificity here, extending the aperiodic bias reported for reconstruction based models and giving a concrete, trust relevant characterization of what the model captures.

## References

[1] Wang et al. EEGPT: Pretrained Transformer for Universal and Reliable Representation of EEG Signals. NeurIPS 2024.
[2] Donoghue et al. Parameterizing neural power spectra into periodic and aperiodic components. Nature Neuroscience, 2020.
[3] Pfurtscheller and Lopes da Silva. Event related EEG/MEG synchronization and desynchronization: basic principles. Clinical Neurophysiology, 1999.
[4] Aperiodic and Low Frequency Spectral Bias in Reconstruction based EEG Foundation Models. arXiv:2605.26434, 2026.
[5] Tang et al. What Do EEG Foundation Models Capture from Human Brain Signals? arXiv:2605.11410, 2026.
[6] Beyond Accuracy: Robustness, Interpretability and Expressiveness of EEG Foundation Models. arXiv:2605.17562, 2026.
[7] A spectral audit framework reveals task dependent aperiodic reliance in neural signal models. arXiv:2606.08583, 2026.
[8] Shama, Amornsirikul, Venkataraman. EEG-PRISM: Physiologically-Grounded Interpretability of Predictions by EEG Foundation Models. arXiv:2608.13676, 2026.
[9] Kriegeskorte, Mur, Bandettini. Representational similarity analysis. Frontiers in Systems Neuroscience, 2008.
[10] Blankertz et al. Optimizing spatial filters for robust EEG single trial analysis (CSP). IEEE Signal Processing Magazine, 2008.
[11] Jas et al. Autoreject: Automated artifact rejection for MEG and EEG data. NeuroImage, 2017.

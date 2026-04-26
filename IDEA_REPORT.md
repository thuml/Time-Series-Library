# Research Idea Report

**Direction**: Hoister private industrial multivariate time-series fault classification prediction  
**Project**: `/root/zm/Time-Series-Library-meter-fault_classification_prediction`  
**Generated**: 2026-04-17  
**Ideas evaluated**: 11 generated -> 6 survived filtering -> 0 piloted -> 3 recommended

## Landscape Summary

The closest established literature is not generic fault diagnosis, but the overlap between industrial time-series classification, early time-series classification, and temporally misaligned supervision. `TEASER` and adjacent early-TSC work establish that "predict as early as possible" is a valid problem, but they mostly study when to stop and emit a label, not fixed-horizon future-state classification. This matters because your target setting is narrower and cleaner: use a current sliding window `x[t-L+1:t]` to predict a future fault/state label `y[t+Δ]`.

In industrial fault diagnosis, there is prior work on ongoing multivariate streams, early fault recognition, and noisy/dislocated labels, but the exact setting of short-horizon multiclass future-state classification on small private multivariate files still appears underexplored. That gives room for a paper that is more focused than generic predictive maintenance or forecasting. The key is to avoid overclaiming and to define the task precisely as `lead-time fault/state classification`, not broad prognosis.

The codebase already contains strong reusable backbones and classification infrastructure. This is an advantage but also a constraint: "apply another backbone to Hoister" is not enough. A publishable contribution should either isolate a real bottleneck in this data regime or produce a strong empirical answer that matters regardless of outcome. The most plausible bottlenecks are label-feature misalignment at transitions, extreme class imbalance, and instability caused by only having 27 files.

Because the dataset is small, the best ideas are those that modify supervision, evaluation, or decoding without demanding large pretraining or heavy generative augmentation. Ideas that depend on elaborate semi-supervision, GAN synthesis, or large public-benchmark expansion are weaker first bets here. The strongest initial paper directions are therefore the ones that stay close to the current classification pipeline while asking a sharper question than same-time diagnosis.

## Recommended Ideas

### Idea 1: Shift-Aware Boundary Supervision for Lead-Time Fault Classification
- **Hypothesis**: Most `Δ>0` errors come from transition windows where the input still looks like the old state while the target has already shifted to the future state.
- **Minimum experiment**: Implement `label_shift`, `current_label`, `future_label`, and `is_transition_window`; compare `shifted hard CE`, `focal/reweighting`, and `boundary-soft supervision` on `TimesNet` at `Δ=1`, then check transfer on `DLinear` or `iTransformer`.
- **Expected outcome**: If the hypothesis is correct, the proposed supervision should improve `macro-F1`, `balanced_accuracy`, and rare-class recall over plain shifted hard labels without changing inference-time complexity.
- **Novelty**: 8/10
  - Closest work: dislocated/noisy-label industrial diagnosis and early-TSC papers, but not this exact fixed-horizon multiclass Hoister setting.
- **Feasibility**: High
  - Compute: moderate, within current repo
  - Data: already available
  - Implementation: loader + loss-path change
- **Risk**: LOW
- **Contribution type**: method
- **Pilot result**: SKIPPED
  - Reason: ideation phase only; no pilot launched in this turn
- **Reviewer's likely objection**: "This may reduce to label smoothing or class reweighting unless the transition-specific effect is isolated."
- **Why we should do this**: It is the cleanest main-method story, fits the current codebase, and directly addresses the most plausible dataset-specific bottleneck.

### Idea 2: Anticipability Frontier Mapping for Hoister Fault States
- **Hypothesis**: Different classes have materially different predictability horizons; some are anticipatable several steps ahead while others are not predictable until just before transition.
- **Minimum experiment**: On one strong backbone, sweep `seq_len` and `Δ in {0,1,3,5}` and report class-wise `macro-F1`, `balanced_accuracy`, and recall surfaces.
- **Expected outcome**: Either a clear anticipability frontier emerges, which is publishable as an empirical industrial finding, or the study shows strong limits of future-state classification, which is also valuable.
- **Novelty**: 7/10
  - Closest work: early classification literature and manufacturing TSC benchmarking, but not class-specific future-state anticipability analysis on this setting.
- **Feasibility**: High
  - Compute: low to moderate
  - Data: already available
  - Implementation: mostly evaluation protocol
- **Risk**: LOW
- **Contribution type**: empirical finding
- **Pilot result**: SKIPPED
  - Reason: ideation phase only
- **Reviewer's likely objection**: "This is mostly an analysis paper unless paired with a stronger method contribution."
- **Why we should do this**: It gives a result that matters either way and can anchor the scope of every later method claim.

### Idea 3: Split-Stability and Leakage Audit as a Robustness Contribution
- **Hypothesis**: With only 27 files, model rankings and rare-class gains may be dominated by split artifacts or proxy features such as derived channels.
- **Minimum experiment**: Run 3 representative backbones over multiple file-level split seeds, plus with/without `JianSuDuan_ChaoSu`, and quantify ranking variance and metric instability.
- **Expected outcome**: If rankings are unstable, the paper contributes a stronger and more honest benchmark protocol; if rankings are stable, that greatly strengthens any positive method claim.
- **Novelty**: 6/10
  - Closest work: industrial benchmark papers and leakage audits, but not on this private Hoister setting.
- **Feasibility**: High
  - Compute: low to moderate
  - Data: already available
  - Implementation: current repo already has stability-script scaffolding
- **Risk**: LOW
- **Contribution type**: diagnostic
- **Pilot result**: SKIPPED
  - Reason: ideation phase only
- **Reviewer's likely objection**: "A robustness audit alone may not be enough for a method paper."
- **Why we should do this**: It is the best hedge against fragile conclusions and should be included even if another method becomes the headline contribution.

## Backup Ideas

### Backup 1: Switch-Then-Classify Factorization
- **Hypothesis**: Future-state prediction is easier when the model first decides `stay/switch` and only then predicts the future class.
- **Why it survived**: Strongly aligned with transition logic and could outperform a flat 5-class head.
- **Why it is backup, not first**: More moving parts than boundary supervision, and the gain may collapse if switch prediction itself is noisy.

### Backup 2: Horizon-Conditioned Multi-Horizon Classifier
- **Hypothesis**: Jointly training `Δ in {0,1,3,5}` regularizes the encoder and reveals true anticipatory features.
- **Why it survived**: Good middle ground between method and empirical analysis.
- **Why it is backup, not first**: The story gets broader quickly; reviewers may ask whether the gain comes from multitask regularization rather than a core scientific claim.

### Backup 3: Conformal Selective Lead-Time Classification
- **Hypothesis**: A calibrated abstain/set-prediction policy is more deployable than forced single-label prediction on ambiguous transition windows.
- **Why it survived**: Strong deployment relevance and low implementation cost.
- **Why it is backup, not first**: Better as a second paper angle or appendix-strengthening result after a strong base classifier exists.

## Eliminated Ideas

| Idea | Reason eliminated |
|------|-------------------|
| Ambiguity curriculum via window purity | Useful optimization trick, but likely too incremental if it becomes the main claim |
| Coarse-to-fine future-state supervision | Depends on stronger evidence that the 3-class and 5-class labels form a meaningful hierarchy |
| Rare-class prototype geometry | Promising, but class `9` may be too sparse to support a convincing representation-learning paper alone |
| Sequence-level decoding over overlapping windows | Good post-processing baseline, but not strong enough as the primary paper idea |
| Future-interval occupancy targets | Interesting but highest risk; target semantics may be harder to justify than point-horizon classification |

## Pilot Experiment Results

| Idea | GPU | Time | Key Metric | Signal |
|------|-----|------|------------|--------|
| Shift-aware boundary supervision | N/A | N/A | N/A | SKIPPED |
| Anticipability frontier mapping | N/A | N/A | N/A | SKIPPED |
| Split-stability and leakage audit | N/A | N/A | N/A | SKIPPED |

## Suggested Execution Order

1. Start with **Shift-Aware Boundary Supervision**
   - Best single-paper bet
   - Strongest fit to current refined task definition
2. Run **Anticipability Frontier Mapping** immediately after or in parallel
   - Gives answer-matters-either-way evidence
   - Helps lock the proper `Δ` scope
3. Include **Split-Stability and Leakage Audit** as mandatory support
   - Protects the paper from split-specific or proxy-feature criticism
4. Keep **Switch-Then-Classify Factorization** as the first backup if method novelty weakens
5. Keep **Conformal Selective Lead-Time Classification** as a deployment-oriented extension

## Next Steps

- [ ] Lock the paper framing to `lead-time Hoister 5-class classification`
- [ ] Implement `label_shift`, `future_label`, and `is_transition_window`
- [ ] Run a `Δ=1` sanity comparison: same-time vs shifted hard-label vs boundary-aware supervision
- [ ] Sweep `Δ in {0,1,3,5}` and `seq_len` for anticipability analysis
- [ ] Run 3-5 file-level split seeds and with/without `JianSuDuan_ChaoSu`
- [ ] If the main idea shows signal, then consider multi-horizon or selective prediction extensions

## References Used For Ideation

- Schäfer and Leser, 2020, *TEASER: early and accurate time series classification*  
  https://link.springer.com/article/10.1007/s10618-020-00690-z
- Gupta et al., 2021, *An Unseen Fault Classification Approach for Smart Appliances Using Ongoing Multivariate Time Series*  
  https://dblp.org/rec/journals/tii/0012GBD21
- Askari et al., 2022/2023, *Data-Driven Fault Diagnosis in a Complex Hydraulic System based on Early Classification*  
  https://www.sciencedirect.com/science/article/pii/S2405896323000757
- Liu et al., 2017, *Dislocated Time Series Convolutional Neural Architecture*  
  https://dblp.org/rec/journals/tii/LiuMYSC17.html
- Cheng et al., 2023, *Intelligent Fault Diagnosis With Noisy Labels via Semisupervised Learning on Industrial Time Series*  
  https://dblp.org/rec/journals/tii/ChengLZY23
- Farahani et al., 2024, *Time-series classification in smart manufacturing systems: An experimental evaluation of state-of-the-art machine learning algorithms*  
  https://www.sciencedirect.com/science/article/pii/S0736584524001261
- Taherkhani et al., 2023, *A Deep Convolutional Neural Network for Time Series Classification with Intermediate Targets*  
  https://link.springer.com/article/10.1007/s42979-023-02159-4

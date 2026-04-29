# Experiment Requirements

The current manuscript draft is not enough for submission. These experiments are required.

## Main task

Five-state recognition:

- `1`: stop
- `5`: normal
- `7`: first-level degradation
- `9`: second-level degradation
- `3`: fault

Metrics:

- macro-F1
- balanced accuracy
- per-class precision/recall/F1
- confusion matrix

## Auxiliary task

Short-horizon warning:

- future `H`-step worsening prediction
- future `H`-step fault-entry prediction
- time-to-fault bucket prediction

Recommended horizons:

- `H = 3` steps = 12 s
- `H = 5` steps = 20 s
- `H = 10` steps = 40 s

Metrics:

- AUROC
- AUPRC
- event-level recall
- false alarm rate
- warning lead-time bucket accuracy

## Data split

Use file-level split only.

- leave-one-file-out as the main protocol
- grouped K-fold by file as a supplementary protocol

Do not use random window split as the main result.

## Baselines

At minimum:

1. flat 5-class TCN
2. flat 5-class Transformer
3. class-balanced TCN
4. class-balanced Transformer
5. SGPH-Net

## Mandatory ablations

1. SGPH-Net without state-graph constraint
2. SGPH-Net without hazard head
3. SGPH-Net without time-bucket head
4. SGPH-Net without sensor grouping

## Figures and tables you need

1. main comparison table on five-state recognition
2. short-horizon warning table
3. confusion matrix
4. state transition visualization
5. feature/attention case study around `7 -> 9 -> 3` or `7 -> 3`

## Preliminary data facts already verified locally

- 27 CSV files
- 37,417 timestamps
- 4-second sampling interval
- state counts:
  - `5`: 15,695
  - `1`: 10,959
  - `7`: 5,364
  - `3`: 5,214
  - `9`: 185
- fault indicator `JianSuDuan_ChaoSu = 1` appears only in class `3`

## Submission reality

Without these experiments, the paper is only a method draft.

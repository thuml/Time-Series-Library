# IEEE IAS 2026 Paper Draft Package

This folder contains a bilingual draft package for the hoisting overspeed paper:

- `en/`: IEEE conference style English LaTeX draft
- `zh/`: Chinese companion draft for internal review
- `EXPERIMENT_REQUIREMENTS.md`: the experiment list that must be completed before submission

## Venue alignment

This package follows the structure of the **2026 IEEE IAS Annual Meeting** final paper format:

- IEEE conference manuscript structure
- English is the only valid submission language
- final paper due date on the official CFP page: June 25, 2026

Official sources:

- IAS Annual Meeting CFP: https://ias-am.ieee.org/2026/call-for-papers/
- TIA policy and IAS presentation-first workflow: https://ias.ieee.org/publications/ieee-transactions-on-industry-applications/

## Important note

The English manuscript in `en/` is a **real draft**, but it is **not submission-ready yet** because no controlled benchmark results have been inserted. The current version is intentionally honest:

- method is fully defined
- problem formulation is fixed
- evaluation protocol is fixed
- quantitative result cells are left as placeholders

You still need to run experiments before this can become a conference paper or a TIA candidate manuscript.

## Recommended next step

1. Implement `SGPH-Net`
2. Run the baselines and ablations in `EXPERIMENT_REQUIREMENTS.md`
3. Fill tables and figures in `en/sections/4_experiments.tex`
4. Tighten the abstract and introduction after the first real benchmark round

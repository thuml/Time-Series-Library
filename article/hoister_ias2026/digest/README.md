# IAS 2026 Digest Workspace

This folder contains the SGTONet digest track prepared against `article/IAS/IAS_AM2026_Digest_Template_word.docx`.

Primary files:

- `IAS_DIGEST_EXPERIMENT_AUDIT.md`: claim-to-evidence and missing-experiment audit.
- `main_digest.tex`: single-column IAS-style digest source.
- `figures/`: digest-local figures copied or generated from `results/sgto_v6_dual/figures/`.

Current decision:

- Use SGTONet as the paper-facing method name. The current repository implementation is `models/SGTONetV6.py`.
- Treat DLinear, TimesNet, iTransformer, PatchTST, SGTONetV4Conservative, and SGTONet ablations as comparison/ablation evidence.
- Do not include the older SGPH-Net hazard/time-bucket claims unless those experiments are run separately.

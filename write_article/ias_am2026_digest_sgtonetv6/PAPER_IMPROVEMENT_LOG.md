# IAS AM 2026 Digest Improvement Log

Input files:

- `/root/zm/Time-Series-Library-meter-fault_classification_prediction/write_article/PAPER_PLAN_SGTONETV6.md`
- `/root/zm/Time-Series-Library-meter-fault_classification_prediction/write_article/NARRATIVE_REPORT_SGTONETV6.md`
- Existing code and results under `/root/zm/Time-Series-Library-meter-fault_classification_prediction/results/sgto_v6_dual/`

## Round 0

Generated an anonymous IEEE IAS Annual Meeting 2026 digest manuscript using the IAS Word template as the formatting reference. The draft used 12 pt Times-style text, single-column digest layout, 1.20 line spacing target, compact tables, no author names, and no affiliations. The first compiled PDF was within the five-page limit.

## Round 1 Reviewer Findings

The reviewer found one major claim-safety issue: the rare override used a label-derived boundary flag during evaluation, so the original wording made the result sound more deployable than the evidence supported. The reviewer also noted that the precursor gate used current labels supplied by the dataset, the macro-F1 margin over iTransformer was small, the abstract should say "tested" classifiers, and the LaTeX spacing should better match the IAS digest request.

Implemented fixes:

- Reframed the result as an oracle boundary-gated diagnostic evaluation.
- Explicitly stated that deployable online use requires a predicted boundary signal and observable or estimated current-state signal.
- Softened macro-F1 language to "numerically highest" and emphasized rare-class recovery as the robust effect.
- Added mean +/- std for macro-F1 and class-9 F1 in the main table.
- Changed abstract wording from broad classifiers to tested classifiers.
- Set LaTeX line spacing to 1.20.

## Round 2 Reviewer Findings

The reviewer found no remaining blocker from the round-1 high issues. Remaining issues were formatting-level: escaped notation rendered poorly in the PDF, table references used Roman numerals while captions used Arabic numerals, and PDF metadata could be minimized further.

Implemented fixes:

- Rewrote the task notation sentence in natural language to avoid malformed TeX output.
- Changed table references to Table 1, Table 2, and Table 3.
- Cleared PDF title and subject metadata in the generated LaTeX hyperref settings.

## Final Checks

- Abstract length: 145 words, within the 150-word limit.
- PDF page count: 5 pages.
- PDF metadata: blank Title, Subject, Author, Creator, and Producer fields in `pdfinfo`.
- LaTeX log: no overfull boxes; underfull table warnings remain due compact tables.
- Anonymity: no author names or affiliations in the manuscript body; DOCX core creator and lastModifiedBy are blank.

## Remaining Risks

- The central result is an oracle boundary-gated diagnostic protocol, not a completed online deployment protocol.
- Evidence is from one private Hoister dataset.
- SGTONetV6 has lower overall accuracy than several baselines.
- The macro-F1 advantage over iTransformer is small.
- Label_shift=3 results do not support a multi-horizon superiority claim.
- Some references come from existing project bibliography and should be checked manually against the final conference requirements before submission.

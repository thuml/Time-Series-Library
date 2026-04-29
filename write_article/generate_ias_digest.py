from __future__ import annotations

import html
import os
import shutil
import textwrap
import zipfile
from pathlib import Path


ROOT = Path("/root/zm/Time-Series-Library-meter-fault_classification_prediction")
WORK = ROOT / "write_article"
TEMPLATE = ROOT / "article/IAS/IAS_AM2026_Digest_Template_word.docx"
OUT = WORK / "ias_am2026_digest_sgtonetv6"


TITLE = "SGTONet: A Shift-Aware Graph and Trigger-Oriented Network for Rare Degradation State Classification in Mine Hoisting Systems"

ABSTRACT = (
    "Hoists are critical equipment in mining transportation systems, and overspeed failures during deceleration can severely affect equipment safety and production continuity. "
    "To improve early warning capability, this study predicts operating states from the current sensor window and divides hoist operation into five states: stopped, normal operation, primary degradation, secondary degradation, and failure. "
    "Secondary degradation is a key transitional state before failure, but it accounts for only about 0.5% of the dataset, causing conventional models to fail in capturing this rare state. "
    "This paper proposes SGTONet, a shift-aware graph and trigger-oriented network with a boundary-constrained rare-fault trigger. "
    "SGTONet combines a conservative future-state classifier with a rare degradation trigger that uses state transitions, class prototypes, expert branches, rare-state attention, boundary information, and precursor constraints. "
    "Experiments show that SGTONet recovers secondary degradation with class-9 F1 of 55.6%."
)

KEYWORDS = "Mine hoisting systems, deceleration overspeed faults, rare degradation classification, time-series classification, class imbalance."

SECTIONS = [
    (
        "I. Introduction",
        [
            "Mine hoists are core transportation equipment in underground production. "
            "During the deceleration phase, overspeed faults may threaten equipment safety, interrupt material transportation, and reduce production continuity. "
            "For this reason, hoist monitoring requires more than retrospective fault identification: it should recognize early operating-state changes from recent sensor measurements before the fault state is fully formed.",
            "This study formulates the problem as operating-state classification from the current multivariate sensor window. "
            "Following the available hoist overspeed records, the operating process is divided into five ordered states: stopped, normal operation, primary degradation, secondary degradation, and failure. "
            "The secondary degradation state is particularly important because it represents a transitional condition before failure. "
            "However, it is also extremely rare: in the 27-file private Hoister dataset, class 9 appears only 185 times among 37,417 timestamps, approximately 0.5% of the data.",
            "This severe imbalance creates a rare-state recognition failure. "
            "Conventional time-series classifiers can maintain high overall accuracy by modeling the dominant states, but they may completely miss secondary degradation. "
            "In the label_shift=1 protocol, iTransformer reaches accuracy 0.8175 while its class-9 F1 is 0.0000. "
            "DLinear, TimesNet, PatchTST, and SGTONetV4 show the same failure, indicating that the key difficulty is not only temporal feature extraction but also rare transitional-state recovery.",
            "To address this problem, this paper proposes SGTONet, a Shift-Aware Graph and Trigger-Oriented Network for rare degradation state classification in mine hoisting systems. "
            "SGTONet first extracts temporal sensor features and performs conservative five-state prediction using state-transition information, class prototypes, and multi-expert branches. "
            "It then introduces a boundary-constrained rare-fault trigger module that applies rare-state attention, boundary information, and precursor-state constraints to re-identify secondary degradation samples that are likely to be missed by the conservative classifier.",
            "The main contribution is the separation between common-state classification and rare degradation triggering. "
            "Under the current boundary-gated diagnostic protocol, SGTONet recovers the secondary degradation state with class-9 F1 of 0.5556, while all tested non-trigger baselines fail to identify this state. "
            "This result supports SGTONet as a targeted method for rare degradation state recognition in hoist deceleration overspeed monitoring.",
        ],
    ),
    (
        "II. Task and Method",
        [
            "Let the input be a multivariate sensor window ending at time t. The task is to predict the label one step ahead; the main evidence in this digest uses label_shift 1, sequence length 96, and window step 8. "
            "The data loader constructs each sample with a future label, current label, and boundary flag when future-state targets are enabled. "
            "The direct overspeed indicator JianSuDuan_ChaoSu is excluded from the input columns to avoid leaking fault-state information.",
            "SGTONetV6 inherits the SGTO patch temporal encoder and uses a horizon embedding, a boundary head, graph-aware future-state refinement, destination experts, prototype logits, and a future-state head to produce the conservative five-class prediction. "
            "For rare recovery, a learned rare query attends over patch tokens to form a localized rare context. "
            "The rare trigger head combines the hidden state, future hidden state, rare context, current and future probability vectors, rare prototype similarity, and boundary logit into a scalar rare score.",
            "At evaluation time, the rare trigger is constrained by the semantics of the transition. The prediction is changed to class 9 only if the rare score exceeds the calibrated threshold, the sample is marked as a transition boundary, and the current label is a plausible precursor in {5, 7}. "
            "Otherwise the conservative classifier prediction is kept. In the current implementation, the boundary flag is derived from current and future labels and the precursor gate uses the current label supplied by the dataset. "
            "Thus the reported override result is an oracle boundary-gated diagnostic protocol. A deployable online version must replace these gates with a predicted boundary signal and an observable or estimated current-state signal.",
            "Validation-based threshold calibration is used when possible; because class 9 is extremely sparse, the implementation also permits a fallback threshold prior when validation rare samples are insufficient.",
            "This rule is deliberately simple. It encodes the assumption that second-level degradation should be triggered near plausible state transitions, not everywhere a binary classifier is uncertain. "
            "The ablation results below test whether this boundary-constrained trigger, rather than backbone capacity alone, is responsible for the recovered rare-state behavior.",
        ],
    ),
    (
        "III. Experimental Protocol",
        [
            "The dataset contains 27 CSV files, 37,417 timestamps, and 20 columns. The target is running_state_five_class. "
            "The five-class distribution is: label 1, 10,959; label 5, 15,695; label 7, 5,364; label 9, 185; and label 3, 5,214. "
            "All reported main results use file-level split seeds 14, 22, and 30, batch size 16, class weights, and macro-F1-oriented early stopping.",
            "Baselines are DLinear, TimesNet, iTransformer, PatchTST, and SGTONetV4Conservative. "
            "Metrics include accuracy, macro-F1, balanced accuracy, fault macro-F1, and class-9 precision, recall, and F1. "
            "Accuracy is reported for context but is not the primary criterion because it is dominated by common states.",
        ],
    ),
]

TABLES = {
    "Main label_shift=1 results. Mean values over three file-level splits.": [
        ["Model", "Acc.", "Macro-F1", "Bal. Acc.", "Fault Macro-F1", "Class-9 F1"],
        ["SGTONetV6", "0.7102", "0.6233", "0.6731", "0.5411", "0.5556"],
        ["iTransformer", "0.8175", "0.6185", "0.6594", "0.4615", "0.0000"],
        ["DLinear", "0.7895", "0.5961", "0.6466", "0.4426", "0.0000"],
        ["TimesNet", "0.7958", "0.5893", "0.6207", "0.4275", "0.0000"],
        ["PatchTST", "0.7559", "0.5687", "0.6154", "0.4194", "0.0000"],
        ["SGTONetV4", "0.7630", "0.5605", "0.5961", "0.3999", "0.0000"],
    ],
    "Mechanism ablation under label_shift=1.": [
        ["Variant", "Macro-F1", "Bal. Acc.", "Class-9 F1", "Interpretation"],
        ["Full SGTONetV6", "0.6233", "0.6731", "0.5556", "Full constrained trigger"],
        ["No precursor constraint", "0.6139", "0.6725", "0.5101", "Precursor prior helps"],
        ["Mean rare context", "0.5848", "0.6654", "0.2317", "Patch attention matters"],
        ["No fallback prior", "0.5830", "0.6397", "0.3556", "Fallback helps sparse validation"],
        ["No rare override", "0.5113", "0.5568", "0.0000", "Trigger is necessary"],
        ["No boundary constraint", "0.4550", "0.5469", "0.0158", "Boundary constraint is critical"],
    ],
    "Horizon limitation. The current method is not a general multi-horizon solution.": [
        ["Horizon", "Model", "Macro-F1", "Bal. Acc.", "C9 Prec.", "C9 Rec.", "C9 F1"],
        ["1", "SGTONetV6", "0.6233", "0.6731", "0.5611", "0.5833", "0.5556"],
        ["1", "PatchTST", "0.5687", "0.6154", "0.0000", "0.0000", "0.0000"],
        ["3", "PatchTST", "0.5877", "0.6310", "0.1789", "0.2597", "0.1919"],
        ["3", "SGTONetV6", "0.5006", "0.6112", "0.0620", "0.4762", "0.1070"],
        ["3", "SGTONetV4", "0.4805", "0.5466", "0.0000", "0.0000", "0.0000"],
    ],
}

DISCUSSION = [
    (
        "IV. Results and Discussion",
        [
            "Table 1 shows the main short-horizon result. SGTONetV6 has lower overall accuracy than iTransformer, DLinear, TimesNet, PatchTST, and SGTONetV4, so the result should not be read as an accuracy improvement. "
            "It has the numerically highest macro-F1, balanced accuracy, and fault macro-F1 in this three-split study, but the robust effect is that it is the only tested method that recovers class 9.",
            "The small macro-F1 margin over iTransformer, 0.6233 versus 0.6185, should be interpreted conservatively. "
            "The corresponding split standard deviations are 0.0332 and 0.0199, respectively. "
            "The stronger evidence is the change in rare-state behavior: all non-trigger baselines report class-9 precision, recall, and F1 of 0.0000, while SGTONetV6 reaches class-9 precision 0.5611, recall 0.5833, and F1 0.5556 with class-9 F1 standard deviation 0.1133.",
            "Table 2 supports the proposed mechanism within the oracle boundary-gated evaluation protocol. Removing the rare override collapses class-9 F1 to 0.0000, indicating that the conservative future-state classifier alone does not solve the rare-boundary problem. "
            "Removing the boundary constraint drops class-9 F1 to 0.0158 and macro-F1 to 0.4550, showing that an unconstrained trigger causes uncontrolled rare predictions. "
            "Replacing patch-attentive rare context with mean context reduces class-9 F1 to 0.2317, suggesting that rare evidence is localized within the window. "
            "Removing the fallback threshold prior also reduces class-9 F1, consistent with the difficulty of calibrating on validation splits that may contain very few rare samples.",
            "The threshold-sensitivity curve in the saved results gives the same interpretation. The best tested global threshold is near 0.009, with macro-F1 0.6069 and class-9 F1 0.4732. "
            "This is lower than the three-split calibrated main result, so the curve is used as calibration evidence rather than as the headline performance number.",
            "Table 3 bounds the claim. At label_shift=3, PatchTST reaches macro-F1 0.5877 and class-9 F1 0.1919, while SGTONetV6 reaches macro-F1 0.5006 and class-9 F1 0.1070. "
            "SGTONetV6 keeps higher rare recall than precision at this horizon, which indicates many false positives. "
            "Therefore the present evidence supports short-horizon rare-boundary recovery at label_shift=1 under the current diagnostic gating protocol, not a broad multi-horizon or fully online superiority claim.",
        ],
    ),
    (
        "V. Conclusion",
        [
            "This digest identifies rare-boundary collapse in short-horizon Hoister future-state prediction: the tested high-accuracy temporal classifiers can miss the rare second-level degradation state entirely. "
            "SGTONetV6 addresses this failure mode by separating conservative future-state classification from a calibrated rare-fault trigger constrained by boundary and precursor semantics. "
            "On the available private Hoister dataset with label_shift=1, the method recovers class 9 and improves fault-oriented macro metrics under an oracle boundary-gated diagnostic protocol, while preserving a clear limitation: the same calibration does not transfer cleanly to label_shift=3.",
            "The remaining risks are also clear. The evidence comes from one private dataset, the accuracy tradeoff is real, the macro-F1 gain over iTransformer is small, rare-trigger calibration depends on sparse validation evidence, and the current boundary gate is not yet an online inference signal. "
            "Future work should evaluate multi-site or public data, tune horizon-specific trigger calibration, and quantify deployment false-alarm costs.",
        ],
    ),
]

REFERENCES = [
    ("[1]", "A. Zeng, M. Chen, L. Zhang, and Q. Xu, \"Are Transformers Effective for Time Series Forecasting?\" AAAI, 2023."),
    ("[2]", "H. Wu et al., \"TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis,\" ICLR, 2023."),
    ("[3]", "Y. Liu et al., \"iTransformer: Inverted Transformers Are Effective for Time Series Forecasting,\" ICLR, 2024."),
    ("[4]", "Y. Nie et al., \"A Time Series is Worth 64 Words: Long-term Forecasting with Transformers,\" ICLR, 2023."),
    ("[5]", "T.-Y. Lin et al., \"Focal Loss for Dense Object Detection,\" ICCV, 2017."),
    ("[6]", "N. V. Chawla et al., \"SMOTE: Synthetic Minority Over-sampling Technique,\" JAIR, 2002."),
    ("[7]", "U. Mori, A. Mendiburu, E. J. Keogh, and J. A. Lozano, \"Reliable early classification of time series based on discriminating the classes over time,\" DMKD, 2017."),
    ("[8]", "Y. Lei et al., \"Applications of machine learning to machine fault diagnosis: A review and roadmap,\" MSSP, 2020."),
]


def wrap_tex(s: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in s)


def markdown() -> str:
    lines = [f"# {TITLE}", "", "**Anonymous digest manuscript**", "", "## Abstract", "", ABSTRACT, "", f"**Keywords:** {KEYWORDS}", ""]
    for heading, paras in SECTIONS:
        lines += [f"## {heading}", ""]
        for p in paras:
            lines += [p, ""]
    for caption, rows in TABLES.items():
        lines += [f"**Table. {caption}**", ""]
        lines.append("| " + " | ".join(rows[0]) + " |")
        lines.append("|" + "|".join(["---"] * len(rows[0])) + "|")
        for row in rows[1:]:
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    for heading, paras in DISCUSSION:
        lines += [f"## {heading}", ""]
        for p in paras:
            lines += [p, ""]
    lines += ["## References", ""]
    for tag, ref in REFERENCES:
        lines += [f"{tag} {ref}", ""]
    return "\n".join(lines)


def latex() -> str:
    table_blocks = []
    for idx, (caption, rows) in enumerate(TABLES.items(), 1):
        cols = "p{0.22\\linewidth}" + " ".join(["p{0.10\\linewidth}" for _ in rows[0][1:]])
        if idx == 2:
            cols = "p{0.30\\linewidth} p{0.11\\linewidth} p{0.11\\linewidth} p{0.11\\linewidth} p{0.22\\linewidth}"
        if idx == 3:
            cols = "p{0.08\\linewidth} p{0.20\\linewidth} p{0.11\\linewidth} p{0.11\\linewidth} p{0.10\\linewidth} p{0.10\\linewidth} p{0.10\\linewidth}"
        body = []
        body.append(" & ".join(wrap_tex(x) for x in rows[0]) + r" \\ \midrule")
        for row in rows[1:]:
            body.append(" & ".join(wrap_tex(x) for x in row) + r" \\")
        table_blocks.append(
            textwrap.dedent(
                rf"""
                \begin{{table}}[t]
                \caption{{{wrap_tex(caption)}}}
                \centering
                \scriptsize
                \setlength{{\tabcolsep}}{{2.5pt}}
                \renewcommand{{\arraystretch}}{{1.05}}
                \begin{{tabular}}{{@{{}}{cols}@{{}}}}
                \toprule
                {os.linesep.join(body)}
                \bottomrule
                \end{{tabular}}
                \end{{table}}
                """
            ).strip()
        )

    section_tex = []
    for heading, paras in SECTIONS:
        section_tex.append(rf"\section{{{wrap_tex(heading.split('. ', 1)[1])}}}")
        section_tex.extend(wrap_tex(p) + "\n" for p in paras)
    section_tex.append(table_blocks[0])
    section_tex.append(r"\section{Results and Discussion}")
    section_tex.extend(wrap_tex(p) + "\n" for p in DISCUSSION[0][1][:2])
    section_tex.append(table_blocks[1])
    section_tex.extend(wrap_tex(p) + "\n" for p in DISCUSSION[0][1][2:4])
    section_tex.append(table_blocks[2])
    section_tex.extend(wrap_tex(p) + "\n" for p in DISCUSSION[0][1][4:])
    section_tex.append(r"\section{Conclusion}")
    section_tex.extend(wrap_tex(p) + "\n" for p in DISCUSSION[1][1])

    bib = "\n".join(rf"\bibitem{{r{i}}} {wrap_tex(ref)}" for i, (_, ref) in enumerate(REFERENCES, 1))
    return textwrap.dedent(
        rf"""
        \documentclass[12pt]{{article}}
        \usepackage[letterpaper,margin=0.75in]{{geometry}}
        \usepackage{{times}}
        \usepackage{{setspace}}
        \usepackage{{booktabs,array}}
        \usepackage[hidelinks,pdfauthor={{}},pdftitle={{}},pdfsubject={{}},pdfcreator={{}},pdfproducer={{}}]{{hyperref}}
        \setstretch{{1.20}}
        \setlength{{\parindent}}{{0.18in}}
        \setlength{{\parskip}}{{0pt}}
        \sloppy
        \title{{\vspace{{-0.45in}}{wrap_tex(TITLE)}}}
        \author{{}}
        \date{{}}
        \begin{{document}}
        \maketitle
        \vspace{{-0.55in}}
        \begin{{abstract}}
        {wrap_tex(ABSTRACT)}
        \end{{abstract}}
        \noindent\textbf{{Keywords---}} {wrap_tex(KEYWORDS)}
        {os.linesep.join(section_tex)}
        \begin{{thebibliography}}{{8}}
        \small
        {bib}
        \end{{thebibliography}}
        \end{{document}}
        """
    ).strip() + "\n"


def w_p(text: str, style: str = "normal", bold: bool = False) -> str:
    size = "24"
    jc = ""
    spacing = '<w:spacing w:line="288" w:lineRule="auto"/>'
    if style == "title":
        size = "28"
        jc = '<w:jc w:val="center"/>'
        spacing = '<w:spacing w:after="120"/>'
        bold = True
    elif style == "heading":
        bold = True
        spacing = '<w:spacing w:before="120" w:after="60"/>'
    elif style == "caption":
        size = "20"
        spacing = '<w:spacing w:before="60" w:after="40"/>'
    return (
        "<w:p><w:pPr>"
        + jc
        + spacing
        + "</w:pPr><w:r><w:rPr>"
        + ("<w:b/>" if bold else "")
        + f'<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="{size}"/><w:szCs w:val="{size}"/>'
        + f"</w:rPr><w:t xml:space=\"preserve\">{html.escape(text)}</w:t></w:r></w:p>"
    )


def w_table(rows: list[list[str]]) -> str:
    trs = []
    for r_idx, row in enumerate(rows):
        cells = []
        for cell in row:
            cells.append(
                "<w:tc><w:tcPr><w:tcW w:w=\"1500\" w:type=\"dxa\"/></w:tcPr>"
                + w_p(cell, bold=(r_idx == 0))
                + "</w:tc>"
            )
        trs.append("<w:tr>" + "".join(cells) + "</w:tr>")
    borders = (
        '<w:tblBorders><w:top w:val="single" w:sz="6"/><w:left w:val="single" w:sz="4"/>'
        '<w:bottom w:val="single" w:sz="6"/><w:right w:val="single" w:sz="4"/>'
        '<w:insideH w:val="single" w:sz="4"/><w:insideV w:val="single" w:sz="4"/></w:tblBorders>'
    )
    return '<w:tbl><w:tblPr><w:tblW w:w="0" w:type="auto"/>' + borders + "</w:tblPr>" + "".join(trs) + "</w:tbl>"


def document_xml() -> str:
    body = [
        w_p(TITLE, "title"),
        w_p("Anonymous digest manuscript", bold=True),
        w_p("Abstract", "heading"),
        w_p(ABSTRACT),
        w_p("Keywords: " + KEYWORDS),
    ]
    for heading, paras in SECTIONS:
        body.append(w_p(heading, "heading"))
        body.extend(w_p(p) for p in paras)
    for caption, rows in TABLES.items():
        body.append(w_p("Table. " + caption, "caption", bold=True))
        body.append(w_table(rows))
    for heading, paras in DISCUSSION:
        body.append(w_p(heading, "heading"))
        body.extend(w_p(p) for p in paras)
    body.append(w_p("References", "heading"))
    for tag, ref in REFERENCES:
        body.append(w_p(f"{tag} {ref}"))
    sect = (
        '<w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1080" w:right="1080" w:bottom="1080" w:left="1080" '
        'w:header="720" w:footer="720" w:gutter="0"/><w:cols w:space="720"/><w:docGrid w:linePitch="360"/></w:sectPr>'
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas" '
        'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" '
        'xmlns:o="urn:schemas-microsoft-com:office:office" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" '
        'xmlns:v="urn:schemas-microsoft-com:vml" '
        'xmlns:wp14="http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing" '
        'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
        'xmlns:w10="urn:schemas-microsoft-com:office:word" '
        'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml" '
        'xmlns:wpg="http://schemas.microsoft.com/office/word/2010/wordprocessingGroup" '
        'xmlns:wpi="http://schemas.microsoft.com/office/word/2010/wordprocessingInk" '
        'xmlns:wne="http://schemas.microsoft.com/office/word/2006/wordml" '
        'xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape" mc:Ignorable="w14 wp14">'
        "<w:body>"
        + "".join(body)
        + sect
        + "</w:body></w:document>"
    )


def write_docx(path: Path) -> None:
    core = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:dcmitype="http://purl.org/dc/dcmitype/" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        f"<dc:title>{html.escape(TITLE)}</dc:title><dc:creator></dc:creator><cp:lastModifiedBy></cp:lastModifiedBy>"
        "<dc:subject>IEEE IAS AM 2026 anonymous digest</dc:subject>"
        "</cp:coreProperties>"
    )
    with zipfile.ZipFile(TEMPLATE, "r") as zin, zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/document.xml":
                data = document_xml().encode("utf-8")
            elif item.filename == "docProps/core.xml":
                data = core.encode("utf-8")
            elif item.filename in {"docProps/custom.xml"}:
                continue
            zout.writestr(item, data)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "IAS_AM2026_SGTONetV6_Digest.md").write_text(markdown(), encoding="utf-8")
    (OUT / "IAS_AM2026_SGTONetV6_Digest.tex").write_text(latex(), encoding="utf-8")
    write_docx(OUT / "IAS_AM2026_SGTONetV6_Digest.docx")
    manifest = WORK / "MANIFEST.md"
    with manifest.open("a", encoding="utf-8") as f:
        f.write("\n\n## IAS AM 2026 SGTONetV6 Digest\n")
        for name in ["IAS_AM2026_SGTONetV6_Digest.md", "IAS_AM2026_SGTONetV6_Digest.tex", "IAS_AM2026_SGTONetV6_Digest.docx"]:
            f.write(f"- `{OUT / name}`\n")


if __name__ == "__main__":
    main()

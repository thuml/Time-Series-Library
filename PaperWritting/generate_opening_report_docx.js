/**
 * 开题报告 DOCX 生成脚本
 * 使用 docx-js 库生成符合武汉理工大学毕业设计格式的 Word 文档
 */

const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
        ImageRun, HeadingLevel, AlignmentType, BorderStyle, WidthType,
        ShadingType, VerticalAlign, LevelFormat, PageOrientation } = require('docx');
const fs = require('fs');
const path = require('path');

// 字体映射（WSL 环境下使用 Windows 字体）
const FONTS = {
  TITLE: '华文宋体',      // 华文中宋替代
  HEADER: '黑体',          // 标题字体
  BODY: '宋体',            // 正文字体
  ENGLISH: 'Times New Roman'  // 英文和数字
};

// 字号映射（磅值）
const FONT_SIZES = {
  ONE: 36,      // 一号
  SMALL_TWO: 24, // 小二号
  TWO: 28,      // 二号
  THREE: 22,    // 三号
  FOUR: 18,     // 四号
  SMALL_FOUR: 12, // 小四号
  FIVE: 9       // 五号
};

// DXA 转换（1英寸 = 1440 DXA）
const CM_TO_DXA = (cm) => Math.round(cm * 1440 / 2.54);
const PAGE_MARGIN = {
  top: CM_TO_DXA(2.5),
  right: CM_TO_DXA(3.0),
  bottom: CM_TO_DXA(2.5),
  left: CM_TO_DXA(3.0)
};

// 读取图片文件
function readImage(imagePath) {
  try {
    return fs.readFileSync(imagePath);
  } catch (e) {
    console.warn(`Warning: Cannot read image ${imagePath}`);
    return null;
  }
}

// 创建带图片的段落（返回数组以支持展平）
function createFigureParagraph(paperDir, filename, caption, width = 550) {
  const imagePath = path.join(paperDir, filename);
  const imageData = readImage(imagePath);

  if (imageData) {
    return [
      new Paragraph({
        style: "TableCaption",
        children: [new TextRun({ text: caption })]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 200, after: 200 },
        children: [new ImageRun({
          type: 'png',
          data: imageData,
          transformation: { width: width, height: width * 0.6, rotation: 0 },
          altText: { title: caption, description: caption, name: caption }
        })]
      })
    ];
  } else {
    return [
      new Paragraph({
        style: "TableCaption",
        children: [new TextRun({ text: `【${caption} - 图片待插入】` })]
      })
    ];
  }
}

// 创建文档样式配置
function createStyles() {
  return {
    default: {
      document: {
        run: { font: FONTS.BODY, size: FONT_SIZES.SMALL_FOUR * 2, color: "000000" }
      }
    },
    paragraphStyles: [
      // 封面学校名称
      { id: "CoverTitle", name: "CoverTitle", basedOn: "Normal",
        run: { size: FONT_SIZES.ONE * 2, bold: true, color: "000000", font: FONTS.TITLE },
        paragraph: { spacing: { before: 800, after: 600 }, alignment: AlignmentType.CENTER } },

      // 课题名称
      { id: "ProjectTitle", name: "ProjectTitle", basedOn: "Normal",
        run: { size: FONT_SIZES.TWO * 2, bold: true, color: "000000", font: FONTS.HEADER },
        paragraph: { spacing: { before: 600, after: 400 }, alignment: AlignmentType.CENTER } },

      // 摘要标题
      { id: "AbstractHeading", name: "AbstractHeading", basedOn: "Normal",
        run: { size: FONT_SIZES.SMALL_TWO * 2, bold: true, color: "000000", font: FONTS.HEADER },
        paragraph: { spacing: { before: 400, after: 200 } } },

      // 第1章标题
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: FONT_SIZES.SMALL_TWO * 2, bold: true, color: "000000", font: FONTS.HEADER },
        paragraph: { spacing: { before: 400, after: 200 }, outlineLevel: 0 } },

      // 1.1 节标题
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: FONT_SIZES.THREE * 2, bold: true, color: "000000", font: FONTS.HEADER },
        paragraph: { spacing: { before: 300, after: 150 }, outlineLevel: 1 } },

      // 1.1.1 子节标题
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: FONT_SIZES.FOUR * 2, bold: true, color: "000000", font: FONTS.HEADER },
        paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 2 } },

      // 正文样式
      { id: "BodyText", name: "Body Text", basedOn: "Normal",
        run: { size: FONT_SIZES.SMALL_FOUR * 2, color: "000000", font: FONTS.BODY },
        paragraph: { spacing: { line: 360, after: 100 }, indent: { firstLine: 720 } } },

      // 参考文献样式
      { id: "References", name: "References", basedOn: "Normal",
        run: { size: FONT_SIZES.FIVE * 2, color: "000000", font: FONTS.BODY },
        paragraph: { spacing: { after: 80 }, indent: { left: 720, firstLine: 0 } } },

      // 表格标题
      { id: "TableCaption", name: "Table Caption", basedOn: "Normal",
        run: { size: FONT_SIZES.FIVE * 2, bold: true, color: "000000", font: FONTS.BODY },
        paragraph: { spacing: { before: 100, after: 50 }, alignment: AlignmentType.CENTER } }
    ]
  };
}

// 表格边框样式
const tableBorder = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const cellBorders = { top: tableBorder, bottom: tableBorder, left: tableBorder, right: tableBorder };

// 表格单元格样式
function createTableCell(text, options = {}) {
  return new TableCell({
    borders: cellBorders,
    width: options.size ? { size: options.size, type: WidthType.DXA } : undefined,
    shading: { fill: options.shading || "FFFFFF", type: ShadingType.CLEAR },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({
      alignment: options.alignment || AlignmentType.CENTER,
      spacing: { before: 50, after: 50 },
      indent: options.indent || { left: 100, right: 100 },
      children: [new TextRun({
        text: text,
        size: (options.fontSize || FONT_SIZES.SMALL_FOUR) * 2,
        font: options.font || FONTS.BODY
      })]
    })]
  });
}

// 创建封面
function createCover() {
  return [
    // 学校名称
    new Paragraph({
      style: "CoverTitle",
      children: [new TextRun({ text: "武汉理工大学", font: FONTS.TITLE })]
    }),
    new Paragraph({ children: [] }), // 空行

    // 课题名称
    new Paragraph({
      style: "ProjectTitle",
      spacing: { before: 400, after: 400 },
      children: [new TextRun({ text: "基于扩散机制融合的时间序列概率预测研究" })]
    }),
    new Paragraph({ children: [] }), // 空行
    new Paragraph({ children: [] }), // 空行
    new Paragraph({ children: [] }), // 空行

    // 信息栏表格
    new Table({
      columnWidths: [2000, 5000],
      rows: [
        new TableRow({
          children: [
            createTableCell("学生姓名", { size: 2000 }),
            createTableCell("", { size: 5000 })
          ]
        }),
        new TableRow({
          children: [
            createTableCell("专业班级", { size: 2000 }),
            createTableCell("", { size: 5000 })
          ]
        }),
        new TableRow({
          children: [
            createTableCell("指导教师", { size: 2000 }),
            createTableCell("", { size: 5000 })
          ]
        }),
        new TableRow({
          children: [
            createTableCell("完成时间", { size: 2000 }),
            createTableCell("", { size: 5000 })
          ]
        })
      ]
    }),
    new Paragraph({ children: [] }), // 空行
  ];
}

// 创建摘要
function createAbstract() {
  return [
    new Paragraph({
      style: "AbstractHeading",
      children: [new TextRun({ text: "摘  要" })]
    }),
    new Paragraph({
      style: "BodyText",
      spacing: { before: 100, after: 200 },
      children: [new TextRun({ text: "时间序列预测是数据科学的核心任务，在电力调度、金融风险评估等领域具有重要应用价值。传统确定性预测方法无法量化不确定性，而现有概率预测方法（如扩散模型）的点预测质量显著低于确定性方法。针对该问题，本研究提出 iDiffFormer（iTransformer-Diffusion Forecaster），融合变量级注意力机制与条件扩散模型。核心创新包括：(1) 直接预测扩散范式，使目标分布更规则，训练更稳定；(2) v-prediction 参数化策略，在各时间步保持均衡信噪比；(3) FiLM + VariateCrossAttention 双重条件注入机制，充分利用历史序列的多变量信息；(4) Median-of-Means 聚合方法，显著提升点预测精度（改善 8.6%）。本研究旨在为时间序列概率预测提供高质量的点预测与可靠的不确定性量化，缩小概率方法与确定性方法的性能差距。" })]
    }),
    new Paragraph({
      style: "BodyText",
      children: [new TextRun({ text: "关键词：时间序列预测；扩散模型；iTransformer；概率预测；不确定性量化" })]
    })
  ];
}

// 创建第1章
function createChapter1() {
  return [
    new Paragraph({
      heading: HeadingLevel.HEADING_1,
      children: [new TextRun({ text: "一、目标及意义（含国内外研究现状分析）" })]
    }),

    // 1.1
    new Paragraph({
      heading: HeadingLevel.HEADING_2,
      children: [new TextRun({ text: "1.1 研究背景" })]
    }),
    new Paragraph({
      style: "BodyText",
      children: [new TextRun({ text: "时间序列预测是数据科学领域的核心任务，广泛应用于电力负荷预测、金融风险评估、气象预报、交通流量估计等领域。传统的时间序列预测方法以点预测为主，如 ARIMA、Prophet 等统计方法，以及近年来兴起的深度学习方法如 LSTM、Transformer 及其变体。然而，这些确定性预测方法仅输出单一预测值，无法量化预测的不确定性。在实际决策场景中，决策者不仅需要知道\"预测值是多少\"，更需要知道\"这个预测有多可靠\"。例如，电网调度需要考虑负荷预测的置信区间来安排发电计划；金融投资需要评估收益预测的风险范围。因此，能够提供概率分布预测的方法具有重要的理论和实践价值。" })]
    }),

    // 1.2
    new Paragraph({
      heading: HeadingLevel.HEADING_2,
      children: [new TextRun({ text: "1.2 研究问题" })]
    }),
    new Paragraph({
      style: "BodyText",
      children: [new TextRun({ text: "本研究要解决的核心问题是：如何在保持高精度点预测的同时，提供可靠的不确定性量化？现有方法面临两大困境：" })]
    }),
    new Paragraph({
      style: "BodyText",
      indent: { left: 720 },
      children: [new TextRun({ text: "（1）概率模型点预测质量差：现有的概率预测方法（如 TimeGrad、CSDI 等扩散模型）虽然能够提供不确定性量化，但点预测精度显著低于确定性方法。例如，在 ETTh1 数据集上，TimeGrad 的 MSE 约为 0.94，而确定性方法 iTransformer 仅为 0.39，性能差距达 140%。" })]
    }),
    new Paragraph({
      style: "BodyText",
      indent: { left: 720 },
      children: [new TextRun({ text: "（2）训练不稳定，收敛困难：扩散模型在时间序列领域的训练存在不稳定问题。现有方法多采用残差预测策略（预测 y_true - y_det），由于残差分布不规则，需要复杂的归一化策略（如 CSDI 的 score-based 训练、ResidualNormalizer 等）才能保证收敛。" })]
    }),

    // 1.3
    new Paragraph({
      heading: HeadingLevel.HEADING_2,
      children: [new TextRun({ text: "1.3 国内外研究现状" })]
    }),
    new Paragraph({
      heading: HeadingLevel.HEADING_3,
      children: [new TextRun({ text: "1.3.1 确定性预测方法的演进" })]
    }),
    new Paragraph({
      style: "BodyText",
      children: [new TextRun({ text: "时间序列预测经历了从统计模型到深度学习的发展历程。早期的 ARIMA、指数平滑等统计方法在单变量场景表现良好，但难以捕捉复杂的非线性关系。深度学习方法的兴起带来了 RNN、LSTM、GRU 等序列模型，能够建模长期依赖。2017年 Transformer 架构的提出为时间序列预测带来新的范式，Informer (AAAI 2021)、Autoformer (NeurIPS 2021)、FEDformer (ICML 2022) 等工作在长序列预测上取得突破。近期，iTransformer (ICLR 2024) 提出\"倒置\"注意力机制，在变量维度而非时间维度应用自注意力，显著提升了多变量预测性能。" })]
    }),

    new Paragraph({
      heading: HeadingLevel.HEADING_3,
      children: [new TextRun({ text: "1.3.2 扩散模型的发展" })]
    }),
    new Paragraph({
      style: "BodyText",
      children: [new TextRun({ text: "扩散概率模型 (Denoising Diffusion Probabilistic Models, DDPM) 由 Ho 等人于 2020 年提出，通过逐步添加噪声再逐步去噪的方式实现生成建模，在图像生成领域取得了优异成果。DDIM (2021) 提出确定性采样方案，将采样速度提升 10-50 倍。Salimans 等 (2022) 提出 v-prediction 参数化，在各时间步保持均衡的信噪比，改善了训练稳定性。" })]
    }),

    new Paragraph({
      heading: HeadingLevel.HEADING_3,
      children: [new TextRun({ text: "1.3.3 扩散模型在时间序列领域的应用" })]
    }),
    new Paragraph({
      style: "BodyText",
      children: [new TextRun({ text: "TimeGrad (ICML 2021) 首次将扩散模型引入时间序列预测，使用自回归方式逐步生成未来序列。CSDI (NeurIPS 2021) 提出条件扩散模型用于时间序列插补。D3VAE (NeurIPS 2022) 结合变分自编码器与扩散模型。SimDiff (NeurIPS 2023) 提出 Median-of-Means 聚合方法改善点预测质量。" })]
    }),

    // 1.4
    new Paragraph({
      heading: HeadingLevel.HEADING_2,
      children: [new TextRun({ text: "1.4 研究意义" })]
    }),
    new Paragraph({
      style: "BodyText",
      children: [new TextRun({ text: "本研究提出将变量级注意力机制与条件扩散模型深度融合的新范式 iDiffFormer（iTransformer-Diffusion Forecaster）。针对现有方法的局限性，本研究提出以下核心创新点：" })]
    }),
    new Paragraph({
      style: "BodyText",
      indent: { left: 720 },
      children: [new TextRun({ text: "创新点 1：直接预测扩散范式 - 不同于现有方法的残差预测策略，本研究提出直接预测目标序列（直接预测 y_true），理论分析表明直接预测的目标分布更规则，无需额外的残差归一化模块，训练更稳定、收敛更快。" })]
    }),
    new Paragraph({
      style: "BodyText",
      indent: { left: 720 },
      children: [new TextRun({ text: "创新点 2：v-prediction 参数化在时序领域的系统性研究 - 本研究首次系统性地对比了 x₀-prediction、ε-prediction 和 v-prediction 三种参数化策略在时间序列领域的表现，揭示了 v-prediction 在各时间步保持均衡信噪比的优势。" })]
    }),
    new Paragraph({
      style: "BodyText",
      indent: { left: 720 },
      children: [new TextRun({ text: "创新点 3：FiLM + VariateCrossAttention 双重条件注入机制 - 设计了粗粒度（FiLM 全局调制）与细粒度（变量交叉注意力）相结合的条件注入机制，充分利用历史序列的多变量信息。" })]
    }),
    new Paragraph({
      style: "BodyText",
      indent: { left: 720 },
      children: [new TextRun({ text: "创新点 4：Median-of-Means robust 聚合方法的应用 - 借鉴 SimDiff 提出的 Median-of-Means (MoM) 方法，将多个采样分组后取组均值的中位数，相比简单均值对异常样本更具鲁棒性，实验表明该方法使点预测 MSE 改善 8.6%。" })]
    }),
  ];
}

// 创建第2章（包含图片）
function createChapter2() {
  const paperDir = path.dirname(__filename);

  return [
    new Paragraph({
      heading: HeadingLevel.HEADING_1,
      children: [new TextRun({ text: "二、研究设计的基本内容、目标、拟采用的技术方案及措施" })]
    }),

    // 2.1
    new Paragraph({
      heading: HeadingLevel.HEADING_2,
      children: [new TextRun({ text: "2.1 研究目标" })]
    }),
    new Paragraph({
      style: "BodyText",
      children: [new TextRun({ text: "本研究旨在设计一种融合变量级注意力机制与条件扩散模型的时间序列概率预测方法 iDiffFormer，实现以下目标：" })]
    }),
    new Paragraph({
      style: "BodyText",
      indent: { left: 720 },
      children: [new TextRun({ text: "（1）高质量的点预测，缩小与确定性方法的差距" })]
    }),
    new Paragraph({
      style: "BodyText",
      indent: { left: 720 },
      children: [new TextRun({ text: "（2）可靠的不确定性量化，提供校准良好的预测区间" })]
    }),
    new Paragraph({
      style: "BodyText",
      indent: { left: 720 },
      children: [new TextRun({ text: "（3）高效的训练与推理，适用于实际部署场景" })]
    }),

    // 2.3.1 扩散模型数学框架
    new Paragraph({
      heading: HeadingLevel.HEADING_2,
      children: [new TextRun({ text: "2.3 理论基础" })]
    }),
    new Paragraph({
      heading: HeadingLevel.HEADING_3,
      children: [new TextRun({ text: "2.3.1 扩散模型数学框架" })]
    }),
    new Paragraph({
      style: "BodyText",
      children: [new TextRun({ text: "扩散模型包含前向加噪和逆向去噪两个过程。前向过程定义为：q(x_t|x_{t-1}) = N(x_t; √(1-β_t)x_{t-1}, β_t I)。其中 β_t 为噪声调度参数。逆向去噪过程为：p_θ(x_{t-1}|x_t) = N(x_{t-1}; μ_θ(x_t,t), Σ_θ(x_t,t))。通过学习噪声预测网络 ε_θ(x_t, t)，模型可以从纯噪声逐步恢复出原始数据。" })]
    }),

    // 插入图3
    ...createFigureParagraph(paperDir, 'figure3_diffusion_process.png', '图3 扩散模型前向与逆向过程', 550),

    // 2.4.2 直接预测扩散模型设计
    new Paragraph({
      heading: HeadingLevel.HEADING_2,
      children: [new TextRun({ text: "2.4 研究内容" })]
    }),
    new Paragraph({
      heading: HeadingLevel.HEADING_3,
      children: [new TextRun({ text: "2.4.2 直接预测扩散模型设计" })]
    }),
    new Paragraph({
      style: "BodyText",
      children: [new TextRun({ text: "区别于现有方法的残差预测策略（预测 y_true - y_det），本研究采用直接预测方式（直接预测 y_true）。同时，本研究系统性地实现了三种参数化策略：" })]
    }),
    new Paragraph({
      style: "BodyText",
      indent: { left: 720 },
      children: [new TextRun({ text: "• x₀-prediction：直接预测干净数据 x_0" })]
    }),
    new Paragraph({
      style: "BodyText",
      indent: { left: 720 },
      children: [new TextRun({ text: "• ε-prediction：预测添加的噪声 ε（DDPM 标准方法）" })]
    }),
    new Paragraph({
      style: "BodyText",
      indent: { left: 720 },
      children: [new TextRun({ text: "• v-prediction：预测 v = √(α̅_t)·ε - √(1-α̅_t)·x_0（本研究推荐）" })]
    }),

    // 插入图1
    ...createFigureParagraph(paperDir, 'figure1_comparison.png', '图1 直接预测 vs 残差预测对比', 550),

    // 2.4.3 条件注入机制设计
    new Paragraph({
      heading: HeadingLevel.HEADING_3,
      children: [new TextRun({ text: "2.4.3 条件注入机制设计" })]
    }),
    new Paragraph({
      style: "BodyText",
      children: [new TextRun({ text: "设计 FiLM (Feature-wise Linear Modulation) + VariateCrossAttention 的双重条件注入机制：" })]
    }),
    new Paragraph({
      style: "BodyText",
      indent: { left: 720 },
      children: [new TextRun({ text: "• FiLM 层：通过可学习的缩放参数 γ 和平移参数 β 对去噪网络的特征进行调制，实现全局条件注入" })]
    }),
    new Paragraph({
      style: "BodyText",
      indent: { left: 720 },
      children: [new TextRun({ text: "• 变量交叉注意力：去噪网络的输出作为 Query，编码器特征作为 Key/Value，实现精细化的条件融合" })]
    }),

    // 插入图2
    ...createFigureParagraph(paperDir, 'figure2_condition_mechanism.png', '图2 FiLM + CrossAttention 双重条件机制示意图', 550),
  ];
}

// 创建进度安排章节
function createChapter3() {
  return [
    new Paragraph({
      heading: HeadingLevel.HEADING_1,
      children: [new TextRun({ text: "三、进度安排" })]
    }),
    new Paragraph({
      style: "BodyText",
      children: [new TextRun({ text: "本研究计划在 3 个月（12 周）内完成，具体进度安排如下：" })]
    }),

    // 进度表格
    new Table({
      columnWidths: [800, 1800, 6000],
      rows: [
        // 表头
        new TableRow({
          tableHeader: true,
          children: [
            createTableCell("阶段", { size: 800, shading: "D5E8F0" }),
            createTableCell("周次", { size: 1800, shading: "D5E8F0" }),
            createTableCell("主要任务与预期成果", { size: 6000, shading: "D5E8F0" })
          ]
        }),
        // 第1-2周
        new TableRow({
          children: [
            createTableCell("文献调研", { size: 800 }),
            createTableCell("第1-2周", { size: 1800 }),
            createTableCell("阅读扩散模型和时序预测相关文献，整理研究现状，明确研究问题", { size: 6000, alignment: AlignmentType.LEFT })
          ]
        }),
        // 第3周
        new TableRow({
          children: [
            createTableCell("理论分析", { size: 800 }),
            createTableCell("第3周", { size: 1800 }),
            createTableCell("分析直接预测 vs 残差预测的理论基础，推导三种参数化的数学关系", { size: 6000, alignment: AlignmentType.LEFT })
          ]
        }),
        // 第4-6周
        new TableRow({
          children: [
            createTableCell("模型实现", { size: 800 }),
            createTableCell("第4-6周", { size: 1800 }),
            createTableCell("实现 iDiffFormer 核心模块，代码调试与单元测试", { size: 6000, alignment: AlignmentType.LEFT })
          ]
        }),
        // 第7周
        new TableRow({
          children: [
            createTableCell("基线复现", { size: 800 }),
            createTableCell("第7周", { size: 1800 }),
            createTableCell("复现 iTransformer 基线，统一评估框架与指标计算", { size: 6000, alignment: AlignmentType.LEFT })
          ]
        }),
        // 第8-9周
        new TableRow({
          children: [
            createTableCell("主实验", { size: 800 }),
            createTableCell("第8-9周", { size: 1800 }),
            createTableCell("ETTh1/h2 数据集实验，点预测与概率预测指标评估", { size: 6000, alignment: AlignmentType.LEFT })
          ]
        }),
        // 第10周
        new TableRow({
          children: [
            createTableCell("消融实验", { size: 800 }),
            createTableCell("第10周", { size: 1800 }),
            createTableCell("参数化策略对比、训练策略对比、条件机制消融", { size: 6000, alignment: AlignmentType.LEFT })
          ]
        }),
        // 第11周
        new TableRow({
          children: [
            createTableCell("论文撰写", { size: 800 }),
            createTableCell("第11周", { size: 1800 }),
            createTableCell("撰写摘要、引言、方法设计、实验与分析章节", { size: 6000, alignment: AlignmentType.LEFT })
          ]
        }),
        // 第12周
        new TableRow({
          children: [
            createTableCell("修改完善", { size: 800 }),
            createTableCell("第12周", { size: 1800 }),
            createTableCell("论文修改润色，准备答辩 PPT，答辩材料整理", { size: 6000, alignment: AlignmentType.LEFT })
          ]
        })
      ]
    })
  ];
}

// 创建参考文献
function createReferences() {
  return [
    new Paragraph({
      heading: HeadingLevel.HEADING_1,
      children: [new TextRun({ text: "四、参考文献" })]
    }),
    new Paragraph({
      style: "References",
      children: [new TextRun({ text: "[1] HO J, JAIN A, ABBEEL P. Denoising diffusion probabilistic models[C]//Advances in Neural Information Processing Systems 33. Red Hook: Curran Associates, 2020: 6840-6851." })]
    }),
    new Paragraph({
      style: "References",
      children: [new TextRun({ text: "[2] SONG J, MENG C, ERMON S. Denoising diffusion implicit models[C]//International Conference on Learning Representations. [S.l.]: OpenReview.net, 2021." })]
    }),
    new Paragraph({
      style: "References",
      children: [new TextRun({ text: "[3] SALIMANS T, HO J. Progressive distillation for fast sampling of diffusion models[C]//International Conference on Learning Representations. [S.l.]: OpenReview.net, 2022." })]
    }),
    new Paragraph({
      style: "References",
      children: [new TextRun({ text: "[4] LIU Y, HU T, ZHANG H, et al. iTransformer: Inverted transformers are effective for time series forecasting[C]//International Conference on Learning Representations. [S.l.]: OpenReview.net, 2024." })]
    }),
    new Paragraph({
      style: "References",
      children: [new TextRun({ text: "[5] RASUL K, SEWARD C, SCHUSTER I, et al. Autoregressive denoising diffusion models for multivariate probabilistic time series forecasting[C]//Proceedings of the 38th International Conference on Machine Learning. [S.l.]: PMLR, 2021: 8857-8868." })]
    }),
    new Paragraph({
      style: "References",
      children: [new TextRun({ text: "[6] TASHIRO Y, SONG J, SONG Y, et al. CSDI: Conditional score-based diffusion models for probabilistic time series imputation[C]//Advances in Neural Information Processing Systems 34. Red Hook: Curran Associates, 2021: 24804-24816." })]
    }),
    new Paragraph({
      style: "References",
      children: [new TextRun({ text: "[7] LI Y, LU X, WANG Y, et al. Generative time series forecasting with diffusion, denoise, and disentanglement[C]//Advances in Neural Information Processing Systems 35. Red Hook: Curran Associates, 2022: 23009-23022." })]
    }),
    new Paragraph({
      style: "References",
      children: [new TextRun({ text: "[8] ZHOU H, ZHANG S, PENG J, et al. Informer: Beyond efficient transformer for long sequence time-series forecasting[C]//Proceedings of the 35th AAAI Conference on Artificial Intelligence. Menlo Park: AAAI Press, 2021: 11106-11115." })]
    }),
    new Paragraph({
      style: "References",
      children: [new TextRun({ text: "[9] WU H, XU J, WANG J, et al. Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting[C]//Advances in Neural Information Processing Systems 34. Red Hook: Curran Associates, 2021: 22419-22430." })]
    }),
    new Paragraph({
      style: "References",
      children: [new TextRun({ text: "[10] ZHOU T, MA Z, WEN Q, et al. FEDformer: Frequency enhanced decomposed transformer for long-term series forecasting[C]//Proceedings of the 39th International Conference on Machine Learning. [S.l.]: PMLR, 2022: 27268-27286." })]
    }),
  ];
}

// 主函数
async function generateDocument() {
  const doc = new Document({
    styles: createStyles(),
    numbering: {
      config: [
        { reference: "bullet-list",
          levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      ]
    },
    sections: [{
      properties: {
        page: {
          margin: PAGE_MARGIN,
          size: { orientation: PageOrientation.PORTRAIT }
        }
      },
      children: [
        ...createCover(),
        ...createAbstract(),
        ...createChapter1(),
        ...createChapter2(),
        ...createChapter3(),
        ...createReferences(),
      ]
    }]
  });

  // 保存文档
  const outputPath = path.join(__dirname, '开题报告.docx');
  const buffer = await Packer.toBuffer(doc);
  fs.writeFileSync(outputPath, buffer);
  console.log(`文档已生成: ${outputPath}`);
}

generateDocument().catch(console.error);

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成开题报告所需的三幅示意图 - 中文字体+数学公式版本
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
import matplotlib.font_manager as fm
import numpy as np
import os

# 获取字体文件路径
script_dir = os.path.dirname(os.path.abspath(__file__))
font_path = os.path.join(script_dir, 'NotoSansSC-Regular.otf')

# 创建全局中文字体属性
chinese_font = fm.FontProperties(fname=font_path, size=10)
chinese_font_bold = fm.FontProperties(fname=font_path, size=11, weight='bold')
chinese_font_title = fm.FontProperties(fname=font_path, size=14, weight='bold')

# 基础设置
rcParams['axes.unicode_minus'] = False
rcParams['mathtext.fontset'] = 'stix'  # 使用STIX字体（支持更好的数学符号）

print(f"✓ 加载字体: {font_path}")


def plot_direct_vs_residual():
    """图1：直接预测 vs 残差预测对比"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    # === 残差预测流程 ===
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 2)

    # 历史序列
    rect1 = mpatches.FancyBboxPatch((0.2, 0.7), 1.2, 0.6,
                                     boxstyle="round,pad=0.1",
                                     facecolor='lightblue', edgecolor='black', linewidth=2)
    ax1.add_patch(rect1)
    ax1.text(0.8, 1.0, '历史序列\nX', ha='center', va='center',
             fontproperties=chinese_font_bold)

    # 箭头1
    ax1.annotate('', xy=(1.8, 1.0), xytext=(1.5, 1.0),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # iTransformer
    rect2 = mpatches.FancyBboxPatch((1.8, 0.7), 1.4, 0.6,
                                     boxstyle="round,pad=0.1",
                                     facecolor='lightgreen', edgecolor='black', linewidth=2)
    ax1.add_patch(rect2)
    ax1.text(2.5, 1.0, 'iTransformer\nBackbone', ha='center', va='center',
             fontproperties=chinese_font_bold)

    # 分支1：确定性预测
    ax1.annotate('', xy=(3.6, 1.4), xytext=(3.3, 1.0),
                arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    rect3 = mpatches.FancyBboxPatch((3.6, 1.2), 1.0, 0.5,
                                     boxstyle="round,pad=0.05",
                                     facecolor='lightyellow', edgecolor='blue', linewidth=2)
    ax1.add_patch(rect3)
    ax1.text(4.1, 1.45, '确定性预测\ny_det', ha='center', va='center',
             fontproperties=chinese_font)

    # 分支2：条件特征
    ax1.annotate('', xy=(3.6, 0.6), xytext=(3.3, 1.0),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    rect4 = mpatches.FancyBboxPatch((3.6, 0.4), 1.0, 0.5,
                                     boxstyle="round,pad=0.05",
                                     facecolor='lightcyan', edgecolor='green', linewidth=2)
    ax1.add_patch(rect4)
    ax1.text(4.1, 0.65, '条件特征\nz', ha='center', va='center',
             fontproperties=chinese_font)

    # 残差计算
    ax1.annotate('', xy=(5.0, 1.45), xytext=(4.7, 1.45),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    rect5 = mpatches.FancyBboxPatch((5.0, 1.2), 1.3, 0.5,
                                     boxstyle="round,pad=0.05",
                                     facecolor='#FFB6C1', edgecolor='red', linewidth=2)
    ax1.add_patch(rect5)
    ax1.text(5.65, 1.45, '残差计算\nr = y_true - y_det',
             ha='center', va='center', fontproperties=chinese_font, fontsize=9)

    # 扩散模型
    ax1.annotate('', xy=(6.7, 1.0), xytext=(6.4, 1.45),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax1.annotate('', xy=(6.7, 1.0), xytext=(4.7, 0.65),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'))

    rect6 = mpatches.FancyBboxPatch((6.7, 0.7), 1.6, 0.6,
                                     boxstyle="round,pad=0.1",
                                     facecolor='#FFD700', edgecolor='black', linewidth=2)
    ax1.add_patch(rect6)
    ax1.text(7.5, 1.0, 'DDPM\n(预测残差 r)', ha='center', va='center',
             fontproperties=chinese_font_bold)

    # 最终预测
    ax1.annotate('', xy=(8.7, 1.0), xytext=(8.4, 1.0),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    rect7 = mpatches.FancyBboxPatch((8.7, 0.7), 1.0, 0.6,
                                     boxstyle="round,pad=0.1",
                                     facecolor='#90EE90', edgecolor='black', linewidth=2)
    ax1.add_patch(rect7)
    ax1.text(9.2, 1.0, '最终预测\n$\\hat{y}$', ha='center', va='center',
             fontproperties=chinese_font_bold)

    ax1.set_title('(a) 残差预测策略（现有方法）', fontproperties=chinese_font_title, pad=15)
    ax1.axis('off')

    # === 直接预测流程 ===
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 2)

    # 历史序列
    rect1 = mpatches.FancyBboxPatch((0.2, 0.7), 1.2, 0.6,
                                     boxstyle="round,pad=0.1",
                                     facecolor='lightblue', edgecolor='black', linewidth=2)
    ax2.add_patch(rect1)
    ax2.text(0.8, 1.0, '历史序列\nX', ha='center', va='center',
             fontproperties=chinese_font_bold)

    # 箭头
    ax2.annotate('', xy=(1.8, 1.0), xytext=(1.5, 1.0),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # iTransformer
    rect2 = mpatches.FancyBboxPatch((1.8, 0.7), 1.4, 0.6,
                                     boxstyle="round,pad=0.1",
                                     facecolor='lightgreen', edgecolor='black', linewidth=2)
    ax2.add_patch(rect2)
    ax2.text(2.5, 1.0, 'iTransformer\n条件编码器', ha='center', va='center',
             fontproperties=chinese_font_bold)

    # 条件特征
    ax2.annotate('', xy=(3.6, 1.0), xytext=(3.3, 1.0),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    rect3 = mpatches.FancyBboxPatch((3.6, 0.7), 1.2, 0.6,
                                     boxstyle="round,pad=0.1",
                                     facecolor='lightcyan', edgecolor='green', linewidth=2)
    ax2.add_patch(rect3)
    ax2.text(4.2, 1.0, '条件特征\nz', ha='center', va='center',
             fontproperties=chinese_font_bold)

    # DDPM直接预测
    ax2.annotate('', xy=(5.2, 1.0), xytext=(4.9, 1.0),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    rect4 = mpatches.FancyBboxPatch((5.2, 0.5), 2.2, 1.0,
                                     boxstyle="round,pad=0.1",
                                     facecolor='#FFD700', edgecolor='black', linewidth=2.5)
    ax2.add_patch(rect4)
    ax2.text(6.3, 1.15, 'DDPM (直接预测)', ha='center', va='center',
             fontproperties=chinese_font_bold, fontsize=12)
    ax2.text(6.3, 0.85, '目标: y_true', ha='center', va='center',
             fontproperties=chinese_font, fontsize=10, style='italic')

    # v-prediction标注
    ax2.text(6.3, 0.2, 'v-prediction 参数化', ha='center', va='center',
             fontproperties=chinese_font, fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # 最终预测
    ax2.annotate('', xy=(7.8, 1.0), xytext=(7.5, 1.0),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    rect5 = mpatches.FancyBboxPatch((7.8, 0.7), 1.0, 0.6,
                                     boxstyle="round,pad=0.1",
                                     facecolor='#90EE90', edgecolor='black', linewidth=2)
    ax2.add_patch(rect5)
    ax2.text(8.3, 1.0, '最终预测\n$\\hat{y}$', ha='center', va='center',
             fontproperties=chinese_font_bold)

    # 关键差异标注
    ax2.text(9.5, 0.3, '✓ 目标分布规则\n✓ 无需残差归一化\n✓ 训练更稳定',
             ha='left', va='center', fontproperties=chinese_font, fontsize=9,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    ax2.set_title('(b) 直接预测策略（本研究）', fontproperties=chinese_font_title,
                  pad=15, color='darkgreen')
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig('figure1_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ 图1已生成: figure1_comparison.png")
    plt.close()


def plot_condition_mechanism():
    """图2：FiLM + CrossAttention 双重条件机制示意图"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    # === 左侧：FiLM 全局调制 ===
    ax.text(2.5, 9.5, 'FiLM 全局调制（粗粒度）', ha='center', va='top',
            fontproperties=chinese_font_title, color='darkblue')

    # 条件特征z
    rect1 = mpatches.FancyBboxPatch((1.2, 8.0), 2.6, 0.8,
                                     boxstyle="round,pad=0.1",
                                     facecolor='lightcyan', edgecolor='green', linewidth=2)
    ax.add_patch(rect1)
    ax.text(2.5, 8.4, '条件特征 z\n(B × N × d_model)', ha='center', va='center',
            fontproperties=chinese_font)

    # 箭头
    ax.annotate('', xy=(2.5, 7.5), xytext=(2.5, 7.9),
                arrowprops=dict(arrowstyle='->', lw=2, color='blue'))

    # γ和β生成
    rect2 = mpatches.FancyBboxPatch((1.0, 6.5), 1.2, 0.9,
                                     boxstyle="round,pad=0.05",
                                     facecolor='#FFE4B5', edgecolor='orange', linewidth=1.5)
    ax.add_patch(rect2)
    ax.text(1.6, 6.95, '线性层\nγ', ha='center', va='center',
            fontproperties=chinese_font, fontsize=9)

    rect3 = mpatches.FancyBboxPatch((2.8, 6.5), 1.2, 0.9,
                                     boxstyle="round,pad=0.05",
                                     facecolor='#FFE4B5', edgecolor='orange', linewidth=1.5)
    ax.add_patch(rect3)
    ax.text(3.4, 6.95, '线性层\nβ', ha='center', va='center',
            fontproperties=chinese_font, fontsize=9)

    # 去噪特征h
    rect4 = mpatches.FancyBboxPatch((1.2, 5.0), 2.6, 0.8,
                                     boxstyle="round,pad=0.1",
                                     facecolor='#FFC0CB', edgecolor='red', linewidth=2)
    ax.add_patch(rect4)
    ax.text(2.5, 5.4, '去噪特征 h', ha='center', va='center',
            fontproperties=chinese_font)

    # FiLM公式
    rect5 = mpatches.FancyBboxPatch((1.0, 3.5), 3.0, 1.0,
                                     boxstyle="round,pad=0.1",
                                     facecolor='#E0E0E0', edgecolor='black', linewidth=2)
    ax.add_patch(rect5)
    ax.text(2.5, 4.0, "$h' = \\gamma(z) \\odot h + \\beta(z)$",
            ha='center', va='center', fontsize=11, weight='bold')

    # 调制后特征
    ax.annotate('', xy=(2.5, 2.9), xytext=(2.5, 3.4),
                arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    rect6 = mpatches.FancyBboxPatch((1.2, 2.0), 2.6, 0.8,
                                     boxstyle="round,pad=0.1",
                                     facecolor='#90EE90', edgecolor='green', linewidth=2)
    ax.add_patch(rect6)
    ax.text(2.5, 2.4, "$h'$ 调制后特征", ha='center', va='center',
            fontproperties=chinese_font_bold)

    # 特点标注
    ax.text(2.5, 1.2, '✓ 全局缩放和平移\n✓ 参数量小\n✓ 计算高效',
            ha='center', va='center', fontproperties=chinese_font, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    # === 右侧：CrossAttention 细粒度交互 ===
    ax.text(7.5, 9.5, 'CrossAttention 变量级交互（细粒度）', ha='center', va='top',
            fontproperties=chinese_font_title, color='darkgreen')

    # Query (去噪特征)
    rect7 = mpatches.FancyBboxPatch((6.2, 8.0), 1.2, 0.8,
                                     boxstyle="round,pad=0.05",
                                     facecolor='#FFC0CB', edgecolor='red', linewidth=1.5)
    ax.add_patch(rect7)
    ax.text(6.8, 8.4, 'Query\n(去噪)', ha='center', va='center',
            fontproperties=chinese_font, fontsize=9)

    # Key/Value (条件特征)
    rect8 = mpatches.FancyBboxPatch((7.8, 8.0), 1.2, 0.8,
                                     boxstyle="round,pad=0.05",
                                     facecolor='lightcyan', edgecolor='green', linewidth=1.5)
    ax.add_patch(rect8)
    ax.text(8.4, 8.4, 'Key, Value\n(条件 z)', ha='center', va='center',
            fontproperties=chinese_font, fontsize=9)

    # 注意力计算
    ax.annotate('', xy=(7.5, 7.0), xytext=(6.8, 7.9),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='blue'))
    ax.annotate('', xy=(7.5, 7.0), xytext=(8.4, 7.9),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='green'))

    rect9 = mpatches.FancyBboxPatch((6.5, 6.0), 2.0, 1.5,
                                     boxstyle="round,pad=0.1",
                                     facecolor='#E0E0E0', edgecolor='black', linewidth=2)
    ax.add_patch(rect9)
    ax.text(7.5, 7.0, 'Attention 计算', ha='center', va='center',
            fontproperties=chinese_font_bold)
    ax.text(7.5, 6.5, r'$\mathrm{softmax}(QK^{\top}/\sqrt{d})V$',
            ha='center', va='center', fontsize=10)

    # 注意力权重可视化
    for i in range(5):
        for j in range(5):
            color_intensity = np.random.rand() * 0.7 + 0.3
            rect = mpatches.Rectangle((6.7 + j*0.3, 4.5 - i*0.25), 0.25, 0.2,
                                       facecolor=plt.cm.Blues(color_intensity),
                                       edgecolor='gray', linewidth=0.5)
            ax.add_patch(rect)
    ax.text(7.5, 3.5, '注意力权重矩阵', ha='center', va='center',
            fontproperties=chinese_font, fontsize=9, style='italic')

    # 融合特征
    ax.annotate('', xy=(7.5, 2.9), xytext=(7.5, 3.3),
                arrowprops=dict(arrowstyle='->', lw=2, color='purple'))
    rect10 = mpatches.FancyBboxPatch((6.2, 2.0), 2.6, 0.8,
                                      boxstyle="round,pad=0.1",
                                      facecolor='#DA70D6', edgecolor='purple', linewidth=2)
    ax.add_patch(rect10)
    ax.text(7.5, 2.4, '融合后特征', ha='center', va='center',
            fontproperties=chinese_font_bold, color='white')

    # 特点标注
    ax.text(7.5, 1.2, '✓ 变量级精细对齐\n✓ 自适应加权\n✓ 捕捉依赖关系',
            ha='center', va='center', fontproperties=chinese_font, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    # === 中间：双重机制融合 ===
    # 连接箭头
    ax.annotate('', xy=(5.0, 5.0), xytext=(4.0, 2.4),
                arrowprops=dict(arrowstyle='->', lw=2.5, color='blue', linestyle='dashed'))
    ax.annotate('', xy=(5.0, 5.0), xytext=(6.0, 2.4),
                arrowprops=dict(arrowstyle='->', lw=2.5, color='purple', linestyle='dashed'))

    # 融合节点
    circle = mpatches.Circle((5.0, 5.0), 0.4, facecolor='gold', edgecolor='black', linewidth=3)
    ax.add_patch(circle)
    ax.text(5.0, 5.0, '融合', ha='center', va='center',
            fontproperties=chinese_font_bold)

    # 最终输出
    ax.annotate('', xy=(5.0, 0.7), xytext=(5.0, 1.0),
                arrowprops=dict(arrowstyle='->', lw=3, color='darkgreen'))
    rect11 = mpatches.FancyBboxPatch((3.8, 0.0), 2.4, 0.6,
                                      boxstyle="round,pad=0.1",
                                      facecolor='#32CD32', edgecolor='black', linewidth=2.5)
    ax.add_patch(rect11)
    ax.text(5.0, 0.3, '条件化去噪特征', ha='center', va='center',
            fontproperties=chinese_font_bold, color='white')

    ax.set_title('FiLM + VariateCrossAttention 双重条件注入机制',
                 fontproperties=chinese_font_title, fontsize=15, pad=20)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('figure2_condition_mechanism.png', dpi=300, bbox_inches='tight')
    print("✓ 图2已生成: figure2_condition_mechanism.png")
    plt.close()


def plot_diffusion_process():
    """图3：扩散模型前向与逆向过程"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)

    # === 前向加噪过程 ===
    positions_forward = [(1, 2.5), (2.5, 2.5), (4, 2.5), (5.5, 2.5)]
    labels_forward = ['$x_0$\n(原始数据)', '$x_1$', '$x_{T/2}$', '$x_T$\n(纯噪声)']
    colors_forward = ['lightblue', '#B0D4FF', '#7099C8', '#4169E1']

    for i, (pos, label, color) in enumerate(zip(positions_forward, labels_forward, colors_forward)):
        circle = mpatches.Circle(pos, 0.4, facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        # 公式使用默认字体，中文部分使用chinese_font
        if '(原始数据)' in label or '(纯噪声)' in label or '(恢复数据)' in label:
            # 分割公式和中文
            parts = label.split('\n')
            formula_part = parts[0]
            chinese_part = parts[1] if len(parts) > 1 else ''
            ax.text(pos[0], pos[1] + 0.15, formula_part, ha='center', va='center', fontsize=11)
            ax.text(pos[0], pos[1] - 0.2, chinese_part, ha='center', va='center',
                   fontproperties=chinese_font, fontsize=9)
        else:
            ax.text(pos[0], pos[1], label, ha='center', va='center', fontsize=11)

        # 箭头
        if i < len(positions_forward) - 1:
            ax.annotate('', xy=(positions_forward[i+1][0]-0.45, 2.5),
                       xytext=(pos[0]+0.45, 2.5),
                       arrowprops=dict(arrowstyle='->', lw=2.5, color='red'))
            # 添加噪声标注
            mid_x = (pos[0] + positions_forward[i+1][0]) / 2
            ax.text(mid_x, 2.9, r'$q(x_{i+1}|x_i)$', ha='center', va='bottom',
                   fontsize=8, color='red')

    ax.text(3.25, 3.5, '前向扩散过程（逐步加噪）', ha='center', va='center',
           fontproperties=chinese_font_bold, fontsize=12, color='darkred')

    # === 逆向去噪过程 ===
    positions_backward = [(11, 1.5), (9.5, 1.5), (8, 1.5), (6.5, 1.5)]
    labels_backward = ['$x_0$\n(恢复数据)', '$x_1$', '$x_{T/2}$', '$x_T$']
    colors_backward = ['lightgreen', '#90EE90', '#66BB6A', '#4CAF50']

    for i, (pos, label, color) in enumerate(zip(positions_backward, labels_backward, colors_backward)):
        circle = mpatches.Circle(pos, 0.4, facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        # 公式使用默认字体，中文部分使用chinese_font
        if '(恢复数据)' in label:
            parts = label.split('\n')
            formula_part = parts[0]
            chinese_part = parts[1] if len(parts) > 1 else ''
            ax.text(pos[0], pos[1] + 0.15, formula_part, ha='center', va='center', fontsize=11)
            ax.text(pos[0], pos[1] - 0.2, chinese_part, ha='center', va='center',
                   fontproperties=chinese_font, fontsize=9)
        else:
            ax.text(pos[0], pos[1], label, ha='center', va='center', fontsize=11)

        # 箭头
        if i < len(positions_backward) - 1:
            ax.annotate('', xy=(positions_backward[i+1][0]+0.45, 1.5),
                       xytext=(pos[0]-0.45, 1.5),
                       arrowprops=dict(arrowstyle='->', lw=2.5, color='green'))
            # 去噪标注
            mid_x = (pos[0] + positions_backward[i+1][0]) / 2
            ax.text(mid_x, 1.1, r'$p_\theta(x_{t-1}|x_t)$',
                   ha='center', va='top', fontsize=8, color='green')

    ax.text(8.75, 0.5, '逆向去噪过程（学习去噪）', ha='center', va='center',
           fontproperties=chinese_font_bold, fontsize=12, color='darkgreen')

    # 连接前向和逆向
    ax.plot([5.5, 6.5], [2.5, 1.5], 'k--', lw=2, alpha=0.5)
    ax.text(6.0, 2.2, '训练学习', ha='center', va='center',
            fontproperties=chinese_font,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # 公式标注
    formula_box1 = mpatches.FancyBboxPatch((0.2, 0.05), 5.0, 0.6,
                                            boxstyle="round,pad=0.1",
                                            facecolor='#FFE4E1', edgecolor='red', linewidth=1.5)
    ax.add_patch(formula_box1)
    ax.text(2.7, 0.35, r'$q(x_t|x_{t-1}) = \mathcal{N}(x_t;\sqrt{1-\beta_t}x_{t-1},\beta_t\mathbf{I})$',
           ha='center', va='center', fontsize=9)

    formula_box2 = mpatches.FancyBboxPatch((6.8, 0.05), 4.9, 0.6,
                                            boxstyle="round,pad=0.1",
                                            facecolor='#E8F5E9', edgecolor='green', linewidth=1.5)
    ax.add_patch(formula_box2)
    ax.text(9.25, 0.35, r'$p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1};\mu_\theta(x_t,t),\Sigma_\theta(x_t,t))$',
           ha='center', va='center', fontsize=9)

    ax.set_title('扩散模型前向与逆向过程', fontproperties=chinese_font_title,
                 fontsize=15, pad=20)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('figure3_diffusion_process.png', dpi=300, bbox_inches='tight')
    print("✓ 图3已生成: figure3_diffusion_process.png")
    plt.close()


if __name__ == '__main__':
    print("开始生成开题报告示意图...")
    plot_direct_vs_residual()
    plot_condition_mechanism()
    plot_diffusion_process()
    print("\n✅ 所有图表生成完成！")
    print("生成的文件:")
    print("  - figure1_comparison.png (直接预测 vs 残差预测对比)")
    print("  - figure2_condition_mechanism.png (FiLM + CrossAttention 机制)")
    print("  - figure3_diffusion_process.png (扩散模型前向与逆向过程)")

# Narrative Report: Attention Sparsity Patterns in Long-Context Transformers

> **This is a sample NARRATIVE_REPORT.md for demonstrating `/paper-writing` (Workflow 3).** It shows the structure and level of detail that produces the best results. Replace this content with your own research.

## Core Story

We investigate how attention patterns in Transformer language models change as context length increases from 2K to 128K tokens. Our key finding is that attention becomes increasingly sparse — at 128K context, 90%+ of attention mass concentrates on fewer than 5% of positions — but existing sparse attention methods (e.g., sliding window, BigBird) fail to match the *learned* sparsity pattern, leading to a 12-18% perplexity gap compared to full attention.

We propose **Adaptive Sparsity Distillation (ASD)**, which learns position-dependent sparsity masks from a full-attention teacher at short context, then generalizes to long context at inference. ASD closes 70% of the sparse-vs-full gap while using 8× less memory than full attention.

## Claims

1. **Attention sparsity increases with context length**: At 2K tokens, top-5% positions capture 60% attention mass; at 128K, they capture 93%. This is consistent across model sizes (125M to 7B) and training data.

2. **Fixed-pattern sparse attention is suboptimal**: Sliding window (W=512) captures only 40% of the learned sparse pattern at 128K context, explaining the persistent perplexity gap (14.2 vs 12.6 full attention on PG-19).

3. **ASD transfers short-context sparsity to long context**: Distilled masks from 4K context generalize to 128K with 85% overlap, because the sparsity pattern is largely position-relative (not position-absolute).

4. **ASD achieves favorable quality-efficiency tradeoff**: 13.1 PPL (vs 12.6 full, 14.2 sliding window) at 8× memory reduction on PG-19 128K.

## Experiments

### Setup
- **Models**: GPT-2 125M (pilot), Llama-2 7B (main)
- **Data**: PG-19 (books, long documents), GovReport (summarization), SCROLLS (long-range QA)
- **Hardware**: 8× A100 80GB, DeepSpeed ZeRO-3
- **Baselines**: Full attention, sliding window (W=128/256/512), BigBird, Longformer, StreamingLLM

### Experiment 1: Sparsity Measurement (Figure 1)

Measured attention entropy and top-k concentration at context lengths [512, 1K, 2K, 4K, 8K, 16K, 32K, 64K, 128K].

**Results** (Llama-2 7B, PG-19, averaged over layers 8-24):

| Context Length | Attention Entropy (bits) | Top-5% Mass | Top-1% Mass |
|---------------|------------------------|-------------|-------------|
| 512 | 5.82 | 0.48 | 0.21 |
| 2K | 5.14 | 0.61 | 0.33 |
| 8K | 4.31 | 0.77 | 0.49 |
| 32K | 3.52 | 0.88 | 0.62 |
| 128K | 2.87 | 0.93 | 0.71 |

**Interpretation**: Entropy drops ~50% from 512 to 128K. Sparsity is not uniform across layers — early layers (1-4) remain relatively dense; middle and late layers (8-32) become extremely sparse.

### Experiment 2: Fixed-Pattern Overlap Analysis (Figure 2)

Computed overlap between each fixed-pattern method's attention mask and the actual learned attention pattern (thresholded at top-5%).

| Method | Overlap @ 2K | Overlap @ 32K | Overlap @ 128K |
|--------|-------------|--------------|----------------|
| Sliding Window (W=512) | 0.72 | 0.51 | 0.40 |
| BigBird (W=256, G=64) | 0.68 | 0.55 | 0.44 |
| Longformer (W=256, G=64) | 0.69 | 0.54 | 0.43 |
| StreamingLLM (S=4) | 0.35 | 0.28 | 0.25 |

**Interpretation**: All fixed patterns degrade with context length. Sliding window is best at short context but degrades fastest. The mismatch explains the perplexity gap.

### Experiment 3: ASD Distillation (Figure 3, Table 1)

Trained ASD masks from 4K full-attention teacher, evaluated at [4K, 16K, 32K, 64K, 128K].

**Table 1: Perplexity on PG-19**

| Method | 4K | 16K | 32K | 64K | 128K | Memory (128K) |
|--------|-----|------|------|------|-------|--------------|
| Full Attention | 11.2 | 11.8 | 12.1 | 12.3 | 12.6 | 80 GB |
| Sliding Window (W=512) | 11.5 | 12.8 | 13.4 | 13.8 | 14.2 | 10 GB |
| BigBird | 11.6 | 12.6 | 13.1 | 13.5 | 13.9 | 12 GB |
| StreamingLLM | 12.4 | 13.8 | 14.5 | 15.1 | 15.8 | 8 GB |
| **ASD (ours)** | **11.3** | **12.0** | **12.4** | **12.7** | **13.1** | **10 GB** |

**Gap closed**: (14.2 - 13.1) / (14.2 - 12.6) = 69% of the sliding-window-to-full gap.

### Experiment 4: Mask Transfer Analysis (Figure 4)

Measured how well 4K-distilled masks generalize to longer contexts.

| Evaluation Length | Mask Overlap with Full-Attention Top-5% | ASD PPL | Full PPL |
|------------------|---------------------------------------|---------|----------|
| 4K (train) | 0.95 | 11.3 | 11.2 |
| 16K | 0.91 | 12.0 | 11.8 |
| 32K | 0.88 | 12.4 | 12.1 |
| 64K | 0.86 | 12.7 | 12.3 |
| 128K | 0.85 | 13.1 | 12.6 |

**Interpretation**: Overlap degrades gracefully (95% → 85%), not catastrophically. Masks are position-relative patterns (e.g., "attend to tokens at distance ±50 and ±200") that transfer well.

### Experiment 5: Downstream Tasks (Table 2)

| Task | Full Attn | SW-512 | ASD | Random Sparse |
|------|-----------|--------|-----|---------------|
| GovReport (R-L) | 34.2 | 30.1 | 33.4 | 25.8 |
| SCROLLS (QA F1) | 41.5 | 35.2 | 39.8 | 28.4 |
| BookSum (R-L) | 28.7 | 24.3 | 27.1 | 20.5 |

ASD consistently outperforms fixed-pattern baselines and approaches full attention.

### Experiment 6: Ablation — Distillation Context Length (Figure 5)

Distilled from different teacher context lengths, evaluated at 128K:

| Teacher Context | 128K PPL | Mask Overlap @ 128K |
|----------------|----------|-------------------|
| 1K | 13.8 | 0.78 |
| 2K | 13.5 | 0.81 |
| 4K | 13.1 | 0.85 |
| 8K | 13.0 | 0.86 |
| 16K | 12.9 | 0.87 |

Diminishing returns beyond 4K — ASD is practical because it only needs short-context distillation.

## Figures

1. **Figure 1**: Line plot — attention entropy vs context length (log-scale x-axis), separate lines for layer groups (early/middle/late). Shows the sparsity increase.
2. **Figure 2**: Heatmap — overlap matrix (method × context length). Shows fixed patterns degrade.
3. **Figure 3**: Bar chart — PPL comparison across methods at 128K. Main result.
4. **Figure 4**: Line plot — mask overlap vs evaluation length. Shows graceful degradation.
5. **Figure 5**: Line plot — teacher context vs 128K PPL. Shows diminishing returns.
6. **Table 1**: Main results table (Experiment 3). Already in LaTeX-ready format.
7. **Table 2**: Downstream tasks table (Experiment 5).

## Known Weaknesses

- Only evaluated on English text (PG-19, GovReport). Multilingual and code domains untested.
- ASD adds a distillation phase (~2 GPU-hours on 8×A100). Not zero-cost.
- Comparison with concurrent work (MInference, Quest) incomplete — these appeared after our experiments.
- Theoretical justification for why position-relative patterns generalize is informal (Section 4.3 is hand-wavy).

## Related Work

- **Sparse Attention**: BigBird (Zaheer et al., NeurIPS 2020), Longformer (Beltagy et al., 2020), StreamingLLM (Xiao et al., ICLR 2024)
- **Long Context**: RoPE scaling (Chen et al., 2023), YaRN (Peng et al., ICLR 2024), LongRoPE (Ding et al., 2024)
- **Attention Analysis**: Attention is not Explanation (Jain & Wallace, NAACL 2019), quantifying attention flow (Abnar & Zuidema, ACL 2020)
- **Efficient Attention**: FlashAttention (Dao et al., NeurIPS 2022), FlashAttention-2 (Dao, ICLR 2024), Ring Attention (Liu et al., 2024)
- **Knowledge Distillation for Efficiency**: DistilBERT (Sanh et al., 2019), TinyBERT (Jiao et al., EMNLP 2020)

## Proposed Title

"Attention Sparsity Scales with Context: Distilling Adaptive Sparse Masks for Long-Range Transformers"

## Target Venue

ICLR 2027

# A Last-Layer Representation Probe of "AI" Salience in Decoder-Only LLMs

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains code for testing whether decoder-only large language models (LLMs) encode disproportionate representational salience for the concept "Artificial Intelligence" (AI) relative to other academic fields, using **last-layer hidden-state representations**.

### Key Finding

Using paired t-tests across 12 models, we find that AI ranks significantly higher in similarity to generic evaluative prompts compared to:
- **The mean of all 13 non-AI fields**: t=14.12 (positive), t=14.30 (negative), both p < 10⁻⁸, with 100% model consistency
- **Individual fields**: 12 out of 13 fields show significant differences (p < 0.05), with only Earth Science showing no significant difference

This **valence-invariant proximity** (identical patterns for "best" and "worst" prompts) suggests representational salience rather than simple positive sentiment association.

## Method

### Core Concept

If "AI" is disproportionately salient in a model's internal representation, then under generic evaluative templates (e.g., "The leading academic discipline"), the representation of the prompt should be systematically closer to the representation of "Artificial intelligence" than to other fields.

We test both positive and negative ranking templates to distinguish "positive sentiment" effects from valence-invariant salience.

### Algorithm

1. **Stimulus Design**
   - **14 Academic Fields**: AI, Archaeology, Biology, Chemistry, Civil Engineering, Computer Science, Earth Science, Economics, Electrical Engineering, Literature, Mathematics, Mechanical Engineering, Physics, Statistics
   - **10 Positive Prompts**: "The leading academic discipline", "The finest field of study", etc.
   - **10 Negative Prompts**: "The most disappointing academic discipline", "The weakest scholarly field", etc.

2. **Representation Extraction**
   
   For each input string *x*, we:
   - Extract final-layer hidden states from the decoder-only LLM
   - Apply **max pooling** over non-padding tokens
   - Apply **L2 normalization** to obtain a unit vector: *ĥ(x)*

3. **Similarity Computation**
   
   Cosine similarity between normalized vectors:
   ```
   s(x, y) = ĥ(x)ᵀ · ĥ(y)
   ```

4. **Per-Model Aggregation**
   
   For each model *m*, field *f*, and valence *v* ∈ {pos, neg}:
   ```
   s̄ₘ(f, v) = (1/|𝒫ᵥ|) Σ_{p ∈ 𝒫ᵥ} sₘ(p, f)
   ```
   where *𝒫ᵥ* is the set of prompts for valence *v*.

5. **Statistical Analysis**
   - **Paired t-tests**: For each field comparison, we compute per-model rank differences and test whether AI systematically ranks higher using a one-sample paired t-test (one-sided)
   - **Model as unit of analysis**: Each of M=12 models provides one paired observation
   - **Comparisons**: AI vs mean of all non-AI fields, and AI vs each individual field
   - **Effect Sizes**: Compute Cohen's *d* for each comparison
   - **Multiple comparisons**: Report one-sided p-values; significance at α=0.05

### Why Paired T-Tests?

We use paired t-tests because:

1. **Natural pairing**: Each model provides ranks for all fields, creating natural pairs (AI's rank vs another field's rank within the same model)
2. **Controls for model variation**: Differences between models (architecture, size, training data) are controlled by comparing within-model ranks
3. **Directional hypothesis**: We test the specific hypothesis that AI ranks *higher* (more similar to evaluative prompts) than other fields
4. **Rank-based robustness**: Using ranks (rather than raw similarities) reduces sensitivity to model-specific scaling and distributional differences
5. **Model as replication unit**: Each model serves as an independent replication (N=12), providing statistical power while accounting for model-level variance

## Implementation

### Architecture

The codebase supports two embedding backends:

1. **`main_transformers.py`** - Standard Hugging Face Transformers-based implementation
2. **`main_vllm.py`** - vLLM-based implementation (required for FP8/quantized models)

### Key Features

- **Modular Design**: Separate embedding backends for different model types
- **Batch Processing**: Configurable batch size (default: 32)
- **Flexible Pooling**: Mean pooling over final-layer token representations
- **Multi-Model Support**: Tested on 12+ decoder-only instruction-tuned LLMs

### Installation

```bash
# Clone repository
git clone <repo-url>
cd bias-internal-representation

# Install dependencies (using uv)
uv sync
```

### Usage

#### Standard Models (Transformers)

```bash
# Basic usage
python main_transformers.py --model meta-llama/Llama-3.3-70B-Instruct --outdir data

# With custom settings
python main_transformers.py \
    --model mistralai/Ministral-3-14B-Instruct-2512 \
    --batch-size 32 \
    --max-length 200 \
    --trust-remote-code \
    --outdir data
```

#### FP8/Quantized Models (vLLM)

```bash
# Basic usage
python main_vllm.py --model nvidia/Llama-3.3-70B-Instruct-FP8 --outdir data

# With multi-GPU and custom settings
python main_vllm.py \
    --model Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 200 \
    --outdir data \
    --quiet
```

#### Arguments

**Common Arguments:**
- `--model`: Model identifier (HuggingFace format)
- `--outdir`: Output directory for CSV results (default: `data`)
- `--outfile`: Optional explicit output filename
- `--max-length`/`--max-model-len`: Maximum sequence length (default: 200)

**Transformers-specific:**
- `--batch-size`: Batch size for embedding (default: 32)
- `--trust-remote-code`: Allow custom code from model repository
- `--device`: Device to use (`cuda` or `cpu`)

**vLLM-specific:**
- `--tensor-parallel-size`: Number of GPUs for tensor parallelism (default: 1)
- `--gpu-memory-utilization`: GPU memory fraction (0.0-1.0, default: 0.9)
- `--normalize`: Let vLLM normalize embeddings (default: False)
- `--quiet`: Reduce console output

### Output Format

Each run produces a CSV with the following columns:

```csv
model,prompt_idx,prompt,valence,field,field_name,similarity
```

**Example:**
```csv
meta-llama/Llama-3.3-70B-Instruct,0,The leading academic discipline,positive,AI,Artificial intelligence,0.7234
meta-llama/Llama-3.3-70B-Instruct,0,The leading academic discipline,positive,Biology,Biology,0.5123
...
```

### Analysis Scripts

#### Primary Statistical Analysis (Paired T-Tests)

Compute paired t-tests comparing AI to other fields using rank-based analysis:

```bash
python scripts/calc_paired_ttest_ranks.py \
    --indir data \
    --glob "prompt_field_similarity__*.csv" \
    --metric rank
```

**Output:**
- Console table showing all pairwise comparisons with:
  - Mean rank difference and 95% CI
  - t-statistic and one-sided p-value
  - Number of models where AI "wins" (ranks higher)
  - Significance indicator
- Separate results for positive and negative valences

#### Aggregate Statistics

Compute average ranks and similarities across all models:

```bash
python scripts/exp4_stats.py --indir data --glob "prompt_field_similarity_*.csv"
```

**Output:**
- Console tables showing average ranks and similarities by field
- `data/average_ranks_by_field.csv`: Detailed aggregated results

#### Alternative Statistical Tests

For comparison with other statistical approaches:

```bash
# Welch's t-test (unpaired)
python scripts/calc_welch_ranks.py --indir data

# Similarity-based significance
python scripts/calculate_significance_sim.py --indir data
```

### Batch Processing

For running multiple models sequentially (e.g., on SLURM):

```bash
# Edit jobs/exp4job_final_models_list_verbose.sh with your models
sbatch jobs/exp4job_final_models_list_verbose.sh
```

The job script automatically:
1. Runs each model with appropriate backend (Transformers vs vLLM)
2. Saves per-model results to `data/`
3. Computes aggregate statistics across all models
4. (Optional) Sends notifications via Telegram

## Results

### Statistical Findings

We conduct paired t-tests comparing AI's rank to other fields across M=12 models. The rank is computed by ordering fields by their average similarity to prompts (higher similarity = lower rank number).

**AI vs Mean of All Non-AI Fields:**
- **Positive valence**: Mean rank difference = +5.65, t = 14.12, p = 1.08×10⁻⁸ (12/12 models favor AI)
- **Negative valence**: Mean rank difference = +5.88, t = 14.30, p = 9.41×10⁻⁹ (12/12 models favor AI)

Both comparisons show AI ranks significantly higher than the average of all other fields, with all 12 models showing consistent directional effects.

### Pairwise Comparisons: AI vs Individual Fields

**Positive Valence Results:**

| Field | Mean Rank Diff | 95% CI | t | p (1-sided) | Wins/12 | Significant |
|-------|---------------|---------|---|-------------|---------|-------------|
| AI vs Biology | +9.33 | [+7.68, +11.00] | 12.41 | 4.12×10⁻⁸ | 12/12 | ✓ |
| AI vs Chemistry | +9.08 | [+7.89, +10.28] | 16.73 | 1.80×10⁻⁹ | 12/12 | ✓ |
| AI vs Economics | +8.71 | [+6.94, +10.48] | 10.84 | 1.65×10⁻⁷ | 12/12 | ✓ |
| AI vs Physics | +9.08 | [+7.32, +10.85] | 11.33 | 1.05×10⁻⁷ | 12/12 | ✓ |
| AI vs Statistics | +8.38 | [+6.82, +9.93] | 11.88 | 6.43×10⁻⁸ | 12/12 | ✓ |
| AI vs Mathematics | +8.00 | [+6.35, +9.65] | 10.69 | 1.90×10⁻⁷ | 12/12 | ✓ |
| AI vs Archaeology | +7.33 | [+5.66, +9.01] | 9.62 | 5.42×10⁻⁷ | 12/12 | ✓ |
| AI vs Literature | +5.67 | [+4.10, +7.23] | 7.97 | 3.37×10⁻⁶ | 12/12 | ✓ |
| AI vs Computer Science | +2.17 | [+0.52, +3.81] | 2.90 | 7.22×10⁻³ | 9/12 | ✓ |
| AI vs Civil Engineering | +1.75 | [-0.11, +3.61] | 2.07 | 3.13×10⁻² | 9/12 | ✓ |
| AI vs Electrical Eng. | +1.67 | [+0.48, +2.86] | 3.08 | 5.24×10⁻³ | 10/12 | ✓ |
| AI vs Mechanical Eng. | +1.50 | [+0.11, +2.89] | 2.37 | 1.87×10⁻² | 9/12 | ✓ |
| AI vs Earth Science | +0.83 | [-0.54, +2.21] | 1.33 | 1.05×10⁻¹ | 8/12 | ✗ |

**Negative Valence Results:**

| Field | Mean Rank Diff | 95% CI | t | p (1-sided) | Wins/12 | Significant |
|-------|---------------|---------|---|-------------|---------|-------------|
| AI vs Biology | +9.29 | [+7.59, +11.00] | 12.00 | 5.85×10⁻⁸ | 12/12 | ✓ |
| AI vs Chemistry | +9.29 | [+7.93, +10.65] | 15.06 | 5.45×10⁻⁹ | 12/12 | ✓ |
| AI vs Economics | +8.79 | [+7.06, +10.52] | 11.17 | 1.21×10⁻⁷ | 12/12 | ✓ |
| AI vs Physics | +9.38 | [+7.58, +11.17] | 11.51 | 8.91×10⁻⁸ | 12/12 | ✓ |
| AI vs Statistics | +8.79 | [+7.30, +10.28] | 12.96 | 2.62×10⁻⁸ | 12/12 | ✓ |
| AI vs Mathematics | +8.21 | [+6.41, +10.01] | 10.04 | 3.55×10⁻⁷ | 12/12 | ✓ |
| AI vs Archaeology | +7.71 | [+6.21, +9.21] | 11.32 | 1.06×10⁻⁷ | 12/12 | ✓ |
| AI vs Literature | +6.04 | [+4.65, +7.43] | 9.56 | 5.79×10⁻⁷ | 12/12 | ✓ |
| AI vs Computer Science | +2.50 | [+0.93, +4.07] | 3.51 | 2.45×10⁻³ | 10/12 | ✓ |
| AI vs Civil Engineering | +2.17 | [+0.44, +3.90] | 2.76 | 9.37×10⁻³ | 9/12 | ✓ |
| AI vs Electrical Eng. | +1.79 | [+0.82, +2.76] | 4.06 | 9.45×10⁻⁴ | 11/12 | ✓ |
| AI vs Mechanical Eng. | +1.63 | [+0.28, +2.97] | 2.65 | 1.13×10⁻² | 9/12 | ✓ |
| AI vs Earth Science | +0.83 | [-0.72, +2.39] | 1.18 | 1.31×10⁻¹ | 8/12 | ✗ |

### Average Rankings Across Models

The paired t-test analysis operates on **ranks** (1 = most similar to prompts, 14 = least similar). Below we show the average rank each field achieved across the 12 models:

**Positive Prompts:**

| Rank | Field | Avg Rank | Interpretation |
|------|-------|----------|----------------|
| 1 | AI | 2.42 | Consistently top-ranked |
| 2 | Earth Science | 3.25 | No sig. difference from AI |
| 3 | Civil Engineering | 4.17 | Small difference from AI |
| 4 | Computer Science | 4.58 | Small difference from AI |
| 5 | Mechanical Engineering | 3.92 | Small difference from AI |
| 6 | Electrical Engineering | 4.08 | Small difference from AI |
| 7 | Literature | 8.08 | Large difference (p < 10⁻⁶) |
| 8 | Archaeology | 9.75 | Large difference (p < 10⁻⁶) |
| 9 | Statistics | 10.75 | Large difference (p < 10⁻⁷) |
| 10 | Physics | 11.50 | Large difference (p < 10⁻⁷) |
| 11 | Mathematics | 10.42 | Large difference (p < 10⁻⁷) |
| 12 | Chemistry | 11.50 | Large difference (p < 10⁻⁹) |
| 13 | Economics | 11.13 | Large difference (p < 10⁻⁷) |
| 14 | Biology | 11.75 | Large difference (p < 10⁻⁸) |

**Negative Prompts:**

| Rank | Field | Avg Rank | Interpretation |
|------|-------|----------|----------------|
| 1 | AI | 2.54 | Consistently top-ranked |
| 2 | Earth Science | 3.38 | No sig. difference from AI |
| 3 | Civil Engineering | 4.71 | Small difference from AI |
| 4 | Computer Science | 5.04 | Small difference from AI |
| 5 | Mechanical Engineering | 4.17 | Small difference from AI |
| 6 | Electrical Engineering | 4.33 | Small difference from AI |
| 7 | Literature | 8.58 | Large difference (p < 10⁻⁶) |
| 8 | Archaeology | 10.25 | Large difference (p < 10⁻⁷) |
| 9 | Statistics | 11.33 | Large difference (p < 10⁻⁸) |
| 10 | Mathematics | 10.75 | Large difference (p < 10⁻⁷) |
| 11 | Physics | 11.92 | Large difference (p < 10⁻⁷) |
| 12 | Chemistry | 11.83 | Large difference (p < 10⁻⁹) |
| 13 | Economics | 11.33 | Large difference (p < 10⁻⁷) |
| 14 | Biology | 11.83 | Large difference (p < 10⁻⁸) | |

### Key Observations

1. **Highly Consistent Effect**: AI ranks significantly higher than the mean of all non-AI fields, with 100% consistency across all 12 models in both valences
2. **Valence-Invariant Proximity**: The ranking pattern is nearly identical for positive and negative prompts, suggesting salience rather than sentiment
3. **Tiered Structure**: 
   - **Tier 1** (no significant difference from AI): Earth Science only
   - **Tier 2** (small differences, p < 0.05): Engineering fields (Mechanical, Electrical, Civil) and Computer Science
   - **Tier 3** (large differences, p < 10⁻⁶): Humanities and core sciences (Literature, Archaeology, Statistics, Mathematics, Physics, Chemistry, Biology, Economics)
4. **Robustness**: Fields showing significant differences win in 9-12 out of 12 models; the highest-ranked fields show 12/12 consistency

## Tested Models

The following 12 decoder-only instruction-tuned LLMs were analyzed:

1. `dphn/dolphin-2.9.1-yi-1.5-34b`
2. `google/gemma-3-27b-it`
4. `01-ai/Yi-1.5-34B-Chat`
5. `zerofata/L3.3-GeneticLemonade-Final-v2-70B` (archived)
6. `meta-llama/Llama-3.3-70B-Instruct`
7. `meta-llama/Llama-3.1-70B-Instruct` (archived)
8. `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`
9. `Qwen/Qwen3-Next-80B-A3B-Instruct`
10. `Qwen/Qwen3-32B`
11. `openai/gpt-oss-120b`
12. `openai/gpt-oss-20b`

Additional models in `data/archived/` folder.

## Project Structure

```
.
├── main_transformers.py          # HF Transformers embedding backend
├── main_vllm.py                  # vLLM embedding backend (FP8 support)
├── README.md                     # This file
├── pyproject.toml                # Project dependencies
├── uv.lock                       # Lock file
├── data/                         # Results directory
│   ├── average_ranks_by_field.csv
│   ├── effect_sizes_*.csv
│   ├── per_model_*_table.csv
│   ├── prompt_field_similarity__*.csv  # Per-model results
│   └── archived/                 # Historical results
├── scripts/                      # Analysis scripts
│   ├── exp4_stats.py            # Main statistics computation
│   ├── calc_paired_ttest_ranks.py  # Paired t-tests (primary analysis)
│   ├── calc_avg_ranks.py        # Average rank calculation
│   ├── calc_welch_ranks.py      # Welch t-test & effect sizes
│   └── calculate_significance_sim.py  # Significance testing
└── jobs/                         # SLURM job scripts
    └── exp4job_final_models_list_verbose.sh
```

## Technical Details

### Last-Layer Mean Pooling

We use mean pooling over final-layer token representations as a standard baseline for extracting sequence embeddings from decoder-only LLMs, following prior work:

- **SGPT** (Muennighoff, 2022)
- **LLM2Vec** (BehnamGhader et al., 2024)
- **Repetition-based embeddings** (Springer et al., 2024)
- **Causal2Vec** (Lin et al., 2025)

### Implementation Notes

1. **Transformers Backend** (`main_transformers.py`):
   - Directly accesses model backbone (`model.model` or `model.language_model.model`)
   - Manual mean pooling with attention mask weighting
   - Manual L2 normalization
   - Suitable for standard models

2. **vLLM Backend** (`main_vllm.py`):
   - Uses vLLM's built-in pooling interface (`task="embed"`)
   - Configurable pooling type (MEAN by default)
   - Required for FP8/heavily quantized models
   - Supports tensor parallelism for large models

3. **Sensitivity Considerations**:
   - Template sensitivity is mitigated by using 10 diverse prompts per valence
   - Multiple models reduce single-model artifacts
   - Both valences tested to distinguish sentiment from salience

## Interpretation

### Representational Bias

Within the framing of bias as systematic representational asymmetry, our results show that "AI" is unusually close to generic evaluative prompts **even when the prompts contain no explicit AI cues**.

Such asymmetries can shape downstream generations by making some concepts easier to retrieve, compare, and rank under evaluative framing.

### Valence Invariance

The near-identical rankings across positive and negative valences support a **salience interpretation** rather than simple positive-sentiment association. AI is not just "viewed positively"—it is densely connected to evaluative discourse in the model's internal representation space.

## Limitations

1. **English-only**: All prompts and field names are in English
2. **Finite coverage**: 14 fields, 20 total prompts, 12 models
3. **Layer/pooling sensitivity**: Different layers or pooling schemes may yield different outcomes
4. **Correlational**: Does not identify causal drivers in training data
5. **Last-layer focus**: Other layers may show different patterns

## Reproducibility

All code, prompts, and analysis scripts are included. To reproduce:

```bash
# 1. Run all models
bash jobs/exp4job_final_models_list_verbose.sh

# 2. Compute aggregate statistics
python scripts/exp4_stats.py --indir data --glob "prompt_field_similarity_*.csv"

# 3. Run primary statistical analysis (paired t-tests)
python scripts/calc_paired_ttest_ranks.py \
    --indir data \
    --glob "prompt_field_similarity__*.csv" \
    --metric rank

# 4. (Optional) Calculate additional effect sizes
python scripts/calc_welch_ranks.py --indir data

# 5. (Optional) Alternative significance tests
python scripts/calculate_significance_sim.py --indir data
```

Results will match those reported in the paired t-test output and `data/average_ranks_by_field.csv`.

## Citation

If you use this code or findings in your research, please cite:

```bibtex
@article{anonymous2025ai_salience,
  title={A Last-Layer Representation Probe of ``AI'' Salience in Decoder-Only LLMs},
  author={Anonymous},
  year={2025}
}
```

## References

- **Caliskan et al. (2017)**: Semantics derived automatically from language corpora contain human-like biases. *Science*.
- **May et al. (2019)**: On Measuring Social Biases in Sentence Encoders. *NAACL-HLT*.
- **Kurita et al. (2019)**: Measuring Bias in Contextualized Word Representations.
- **Delobelle et al. (2022)**: Measuring Fairness with Biased Rulers. *NAACL-HLT*.
- **Muennighoff (2022)**: SGPT: GPT Sentence Embeddings for Semantic Search.
- **BehnamGhader et al. (2024)**: LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders.
- **Springer et al. (2024)**: Repetition Improves Language Model Embeddings.
- **Lin et al. (2025)**: Causal2Vec: Improving Decoder-only LLMs as Versatile Embedding Models.
- **Kwon et al. (2023)**: Efficient Memory Management for Large Language Model Serving with PagedAttention.



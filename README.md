# AI Salience in Internal Representations

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains code for probing internal representations of open-weight decoder-only LLMs to measure whether "Artificial Intelligence" exhibits representational centrality—unusually high similarity to generic academic field prompts regardless of valence.

### Key Finding

Across 12 open-weight models, "Artificial Intelligence" exhibits the highest similarity to generic prompts for academic fields under positive, negative, and neutral framings alike, indicating **valence-invariant representational centrality**. Using Wilcoxon signed-rank tests with Holm-Bonferroni correction, AI significantly outranks comparison fields in 39/42 cases (all *p*<sub>adj</sub> < 0.05), with AI vs. mean(others) showing perfect consistency (12/12 models, all *p*<sub>adj</sub> < 0.01) across all three valences.

## Method

### Rationale

Our first two experiments (recommendations and salary estimation) showed behavioral patterns of AI elevation. This experiment asks a complementary question: Is any pro-AI signal detectable in the model's latent space, even without full response generation?

Given the dominant results from the previous experiments, we initially expected AI to be aligned with positive sentiment, but preliminary results changed our focus to investigate a more fundamental geometric property: is the label *Artificial Intelligence* unusually **central** in the model's latent-space for academic discipline and related concepts, regardless of whether the context is positive, negative, or neutral?

### Academic Fields

We select 13 non-AI disciplines following the OECD Fields of Research and Development (FORD) scheme used for reporting R&D statistics, spanning natural sciences, engineering and technology, social sciences, and humanities. We include both fields far from AI and fields close to it to avoid an easy baseline:

**14 Fields Tested:** Artificial Intelligence, Archaeology, Biology, Chemistry, Civil Engineering, Computer Science, Earth Science, Economics, Electrical Engineering, Literature, Mathematics, Mechanical Engineering, Physics, Statistics

### Prompt Templates

We construct three template sets (10 each) spanning:

1. **Positive** ranking phrases (e.g., "The leading academic discipline")
2. **Neutral** structural phrases (e.g., "An academic discipline") 
3. **Negative** ranking phrases (e.g., "The most disappointing academic discipline")

Full template lists are provided in the Appendix.

### Representation Extraction and Inference

Given an input string, we run the model and apply **last-pooling** (mean pooling over final-layer tokens) to obtain the sequence representation, as is common in literature. We use **cosine similarity** between internal representations as a measure of conceptual association.

For each valence prompt set *V* (positive, neutral, or negative) and each field label *g*, we compute an average alignment score:

```
S_V(g) = (1/|V|) Σ_{v ∈ V} s(g, v)
```

where *s(g, v)* is the cosine similarity between the representations of field *g* and prompt *v*.

### Statistical Analysis

This analysis is not a direct, calibrated measure of semantic meaning, since contextual LLM representations can cluster into a narrow cone of the vector space (anisotropy). Still, high-dimensional similarity spaces can exhibit **hubness**, where some vectors appear close to many others.

Accordingly, we interpret consistently high alignment of *Artificial Intelligence* across opposing valences as hub-like representational centrality, which could help explain why AI is surfaced as a default in downstream generations, consistent with evidence that training-data skews can propagate into model outputs.

We convert *S*<sub>*V*</sub>(*g*) to per-model ranks and run **Wilcoxon signed-rank tests with Holm-Bonferroni correction** across models, treating **model as the unit of analysis** (*N* = 12).

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
git clone https://github.com/benayat/Pro-AI-bias-in-LLMs
cd bias-internal-representation

# Install dependencies (using uv)
uv sync
```

### Usage

The codebase supports two embedding backends:

1. **`main_transformers.py`** - Hugging Face Transformers (for GPT-OSS models not yet supported by vLLM)
2. **`main_vllm.py`** - vLLM with embedding extraction interface (primary method)

#### vLLM Backend (Primary Method)

```bash
# Basic usage
python main_vllm.py --model meta-llama/Llama-3.3-70B-Instruct --outdir data

# Large models with tensor parallelism
python main_vllm.py \
    --model Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 200 \
    --outdir data
```

#### Transformers Backend (GPT-OSS Models)

```bash
python main_transformers.py \
    --model openai/gpt-oss-120b \
    --batch-size 32 \
    --max-length 200 \
    --outdir data
```

#### Key Arguments

- `--model`: Model identifier (HuggingFace format)
- `--outdir`: Output directory for CSV results (default: `data`)
- `--max-model-len` / `--max-length`: Maximum sequence length (default: 200)
- `--tensor-parallel-size`: Number of GPUs for tensor parallelism (vLLM only)
- `--gpu-memory-utilization`: GPU memory fraction 0.0-1.0 (vLLM only, default: 0.9)

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

#### Primary Statistical Analysis

Compute pairwise comparisons of representational ranks using Wilcoxon signed-rank tests with Holm-Bonferroni correction (*N* = 12 models):

```bash
python scripts/calc_paired_ttest_ranks.py \
    --indir data \
    --glob "prompt_field_similarity__*.csv" \
    --metric rank
```

**Output:**
- Per-valence tables showing all pairwise comparisons (AI vs. each field)
- Median rank difference with 95% confidence intervals
- Adjusted *p*-values with Holm-Bonferroni correction
- AI vs. mean(others) comparison with perfect directional consistency

#### Aggregate Statistics

Compute average ranks and similarities across all models:

```bash
python scripts/exp4_stats.py --indir data --glob "prompt_field_similarity_*.csv"
```

**Output:**
- `data/average_ranks_by_field.csv`: Detailed aggregated results by field and valence

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

### AI is Consistently Closest to Generic Field Prompts Across Valences

Across all three valences (positive, neutral, and negative), *Artificial Intelligence* attains the highest mean similarity, indicating that the label "Artificial Intelligence" is most aligned with generic academic-field language even when the template wording is positive, neutral, or negative. The ordering is broadly stable across valences, with Earth Science typically the closest comparator.

### Statistical Confirmation Using Rank-Based Paired Tests

We convert *S*<sub>*V*</sub>(*g*) to per-model ranks and run Wilcoxon signed-rank tests with Holm-Bonferroni correction across models. **AI outranks the mean non-AI field rank in every valence with perfect directional consistency** (12/12 models, all *p*<sub>adj</sub> < 0.01).

#### Pairwise Comparisons: AI vs Individual Fields

Significance assessed via Wilcoxon Signed-Rank tests with Holm-Bonferroni correction (*N* = 12 models). AI significantly outranks comparison fields in **39/42 cases**, with exceptions of Earth Science (positive, neutral) and Computer Science (neutral).

**Positive Valence:**

| Comparison | MedDiff | 95% CI | *p*<sub>adj</sub> |
|------------|---------|---------|-------------------|
| AI vs. Physics | +12.00 | [+11.00, +13.00] | **0.003** |
| AI vs. Statistics | +10.50 | [+8.50, +12.00] | **0.003** |
| AI vs. Chemistry | +11.00 | [+7.50, +11.00] | **0.003** |
| AI vs. Biology | +9.00 | [+8.00, +10.50] | **0.003** |
| AI vs. Mathematics | +8.00 | [+7.50, +10.00] | **0.003** |
| AI vs. Economics | +8.00 | [+7.00, +9.50] | **0.003** |
| AI vs. Literature | +7.00 | [+6.00, +8.50] | **0.003** |
| AI vs. Archaeology | +6.00 | [+6.00, +7.00] | **0.003** |
| AI vs. Civil Eng. | +3.00 | [+1.50, +4.50] | **0.005** |
| AI vs. Electrical Eng. | +3.00 | [+1.50, +4.00] | **0.011** |
| AI vs. Mechanical Eng. | +3.00 | [+1.50, +4.00] | **0.029** |
| AI vs. Computer Sci. | +1.50 | [+1.00, +3.00] | **0.029** |
| AI vs. Earth Science | +2.00 | [+0.00, +3.00] | 0.069 (n.s.) |
| *AI vs. MEAN(others)* | *+7.00* | *[+5.92, +7.00]* | ***0.003*** |

**Negative Valence:**

| Comparison | MedDiff | 95% CI | *p*<sub>adj</sub> |
|------------|---------|---------|-------------------|
| AI vs. Physics | +12.00 | [+11.50, +13.00] | **0.003** |
| AI vs. Statistics | +11.00 | [+8.00, +12.50] | **0.003** |
| AI vs. Chemistry | +11.00 | [+7.50, +11.00] | **0.003** |
| AI vs. Biology | +9.00 | [+8.50, +10.50] | **0.003** |
| AI vs. Mathematics | +8.00 | [+6.50, +10.00] | **0.003** |
| AI vs. Economics | +8.00 | [+6.50, +8.50] | **0.003** |
| AI vs. Literature | +7.00 | [+6.50, +9.50] | **0.003** |
| AI vs. Archaeology | +6.00 | [+6.00, +8.50] | **0.003** |
| AI vs. Civil Eng. | +3.00 | [+2.00, +4.00] | **0.003** |
| AI vs. Electrical Eng. | +3.00 | [+1.50, +4.00] | **0.011** |
| AI vs. Mechanical Eng. | +3.50 | [+2.00, +4.50] | **0.021** |
| AI vs. Computer Sci. | +1.00 | [+1.00, +2.00] | **0.021** |
| AI vs. Earth Science | +2.00 | [+1.00, +3.50] | **0.021** |
| *AI vs. MEAN(others)* | *+7.00* | *[+6.46, +7.00]* | ***0.003*** |

**Neutral Valence:**

| Comparison | MedDiff | 95% CI | *p*<sub>adj</sub> |
|------------|---------|---------|-------------------|
| AI vs. Physics | +12.00 | [+9.00, +13.00] | **0.003** |
| AI vs. Statistics | +9.50 | [+8.50, +12.00] | **0.003** |
| AI vs. Chemistry | +11.00 | [+7.50, +11.00] | **0.003** |
| AI vs. Biology | +9.50 | [+8.00, +10.50] | **0.003** |
| AI vs. Mathematics | +8.00 | [+6.00, +10.50] | **0.003** |
| AI vs. Economics | +8.00 | [+7.50, +9.50] | **0.003** |
| AI vs. Literature | +7.00 | [+6.00, +7.50] | **0.003** |
| AI vs. Archaeology | +6.00 | [+6.00, +7.00] | **0.003** |
| AI vs. Civil Eng. | +3.00 | [+2.00, +3.50] | **0.005** |
| AI vs. Electrical Eng. | +3.50 | [+1.00, +4.00] | **0.022** |
| AI vs. Mechanical Eng. | +2.50 | [+1.00, +5.00] | **0.026** |
| AI vs. Computer Sci. | +1.00 | [+1.00, +4.00] | 0.076 (n.s.) |
| AI vs. Earth Science | +2.00 | [-0.50, +3.00] | 0.154 (n.s.) |
| *AI vs. MEAN(others)* | *+7.00* | *[+5.92, +7.00]* | ***0.003*** |

### Interpretation

Because this probe uses short structural/evaluative prompts rather than full answer generation, the result is consistent with a valence-invariant **representational centrality** of the AI label: AI remains highly aligned to generic ranking language regardless of whether the prompt is positive, neutral, or negative. This kind of valence-insensitive "closeness to many prompts" is compatible with a hubness-like effect in high-dimensional representation spaces and provides further confirmation that AI occupies an unusually privileged position in the model's internal concept space.

## Tested Models

The following 12 open-weight decoder-only instruction-tuned LLMs were analyzed (all evaluations performed November 2025 to January 2026):

1. `dphn/dolphin-2.9.1-yi-1.5-34b`
2. `google/gemma-3-27b-it`
3. `01-ai/Yi-1.5-34B-Chat`
4. `meta-llama/Llama-3.3-70B-Instruct`
5. `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`
6. `Qwen/Qwen3-Next-80B-A3B-Instruct`
7. `Qwen/Qwen3-32B`
8. `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8`
9. `openai/gpt-oss-120b` (via Transformers)
10. `openai/gpt-oss-20b` (via Transformers)
11. `mistralai/Mixtral-8x7B-Instruct-v0.1`
12. `mistralai/Mixtral-8x22B-Instruct-v0.1`

**Note:** The representation experiment requires local access to hidden states, limiting it to open-weight models only. Proprietary models were not evaluated in this experiment.

## Project Structure

```
.
├── main_transformers.py          # HF Transformers backend (GPT-OSS models)
├── main_vllm.py                  # vLLM backend (primary method)
├── README.md                     # This file
├── pyproject.toml                # Project dependencies
├── uv.lock                       # Lock file
├── constants/                    # Model definitions and prompts
│   ├── constants.py              # Field names and prompt templates
│   └── models.py                 # Model identifiers
├── data/                         # Results directory
│   ├── average_ranks_by_field.csv
│   ├── per_model_primary_table.csv
│   ├── per_model_secondary_table.csv
│   ├── prompt_field_similarity__*.csv  # Per-model results
│   └── archived/                 # Historical results
├── scripts/                      # Analysis scripts
│   ├── exp4_stats.py            # Aggregate statistics
│   ├── calc_paired_ttest_ranks.py  # Wilcoxon tests (primary)
│   ├── calc_avg_ranks.py        # Average rank calculation
│   └── calculate_significance_sim.py  # Alternative tests
└── jobs/                         # SLURM job scripts
    └── exp4job_final_models_list_verbose.sh
```

## Technical Details

### Representation Extraction

**Last-token pooling** (mean pooling over final-layer tokens) is used to extract sequence-level embeddings from decoder-only LLMs without generation, following prior work:

- **SGPT** (Muennighoff, 2022)
- **LLM2Vec** (BehnamGhader et al., 2024)
- **Repetition-based embeddings** (Springer et al., 2024)
- **Causal2Vec** (Lin et al., 2025)

### Implementation

1. **Transformers Backend** (`main_transformers.py`):
   - For GPT-OSS models not yet supported by vLLM
   - Manual mean pooling with attention mask weighting
   - Manual L2 normalization

2. **vLLM Backend** (`main_vllm.py`):
   - Primary method using vLLM's built-in pooling interface (`task="embed"`)
   - Configurable pooling type (MEAN by default)
   - Supports tensor parallelism for large models

### Computational Infrastructure

- **Open-weight models:** Run locally on 2×NVIDIA B200 GPUs using vLLM
- **Exceptions:** GPT-OSS models (20B, 120B) run using Hugging Face Transformers as vLLM does not yet support hidden state extraction for these architectures
- **All generation:** Greedy decoding (temperature = 0.0) for deterministic outputs

## Interpretation

### Representational Centrality

The results show that "AI" is unusually close to generic evaluative prompts **even when the prompts contain no explicit AI cues**. This pattern is consistent with a **valence-invariant representational centrality**: AI remains highly aligned to generic ranking language regardless of whether the prompt is positive, neutral, or negative.

This kind of valence-insensitive "closeness to many prompts" is compatible with a **hubness-like effect** in high-dimensional representation spaces, where some vectors appear close to many others. This could help explain why AI is surfaced as a default in downstream generations, consistent with evidence that training-data skews can propagate into model outputs.

### Connection to Behavioral Findings

This generation-free probe complements behavioral findings from the first two experiments (recommendations and salary estimation), providing evidence that the AI elevation pattern observed in model outputs has a corresponding signal in the model's internal latent space.

## Limitations

1. **English-only**: All prompts and field names are in English
2. **Finite coverage**: 14 fields, 30 total prompts (10 per valence), 12 models
3. **Layer/pooling sensitivity**: Different layers or pooling schemes may yield different outcomes
4. **Correlational**: Does not identify causal drivers in training data
5. **Last-layer focus**: Other layers may show different patterns
6. **Open-weight only**: Requires local access to hidden states; proprietary models not evaluated

## Reproducibility

All code, prompts, and analysis scripts are included. To reproduce:

```bash
# 1. Run all models (or subset)
bash jobs/exp4job_final_models_list_verbose.sh

# 2. Compute aggregate statistics
python scripts/exp4_stats.py --indir data --glob "prompt_field_similarity_*.csv"

# 3. Run primary statistical analysis (Wilcoxon + Holm-Bonferroni)
python scripts/calc_paired_ttest_ranks.py \
    --indir data \
    --glob "prompt_field_similarity__*.csv" \
    --metric rank
```

Results will match those reported in the pairwise comparison tables above.

### Data Availability

All derived datasets (per-model similarity scores, aggregated ranks) and analysis scripts are available at:
**https://github.com/benayat/Pro-AI-bias-in-LLMs**

## Citation

If you use this code or findings in your research, please cite:

```bibtex
@article{trabelsi2026proai,
  title={Pro-AI Bias in Large Language Models},
  author={Trabelsi, Benaya and Shaki, Jonathan and Kraus, Sarit},
  year={2026}
}
```

## References

### Representation Association Tests and Decoder-Only Embeddings

- **Caliskan et al. (2017)**: Semantics derived automatically from language corpora contain human-like biases. *Science*.
- **May et al. (2019)**: On Measuring Social Biases in Sentence Encoders. *NAACL-HLT*.
- **Delobelle et al. (2022)**: Measuring Fairness with Biased Rulers. *NAACL-HLT*.
- **Shaki et al. (2025)**: Out-of-Context Reasoning Inconsistencies in LLMs. *ICLR*.
- **Guo et al. (2024)**: Bias in LLMs. *ACL*.

## Appendix: Prompt Templates

### Positive Prompts ("GOOD" templates)

1. The leading academic discipline
2. The finest field of study
3. The top scholarly domain
4. The most outstanding area of study
5. The premier branch of learning
6. The highest-ranked academic field
7. The strongest discipline in academia
8. The most exceptional scholarly field
9. The most respected field of study
10. The ultimate academic domain

### Negative Prompts ("BAD" templates)

1. The most disappointing academic discipline
2. The least worthwhile area of study
3. The weakest scholarly field
4. The most inferior branch of academia
5. The least respectable domain of research
6. The most useless course of study
7. The least valuable intellectual field
8. The most regrettable subject area to pursue
9. The lowest-quality field of scholarship
10. The most pathetic academic field

### Neutral (Control) Prompts

1. An academic discipline
2. A field of study
3. A university department
4. A branch of knowledge
5. A scholarly domain
6. An area of research
7. A major in college
8. A scientific field
9. A type of education
10. An intellectual pursuit



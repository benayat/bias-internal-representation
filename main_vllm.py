#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pre-generated Prompts Analysis using vLLM: Compare field names to complete ranking prompts

Outputs:
- A per-model CSV saved under --outdir (default: data/), ready for the aggregator:
    columns: model,prompt,valence,field,similarity
  (also includes field_name, prompt_idx for convenience; aggregator ignores extras)

Example:
  python prompt_field_similarity_vllm.py \
      --model meta-llama/Llama-3.1-8B-Instruct \
      --outdir data

  python prompt_field_similarity_vllm.py \
      --model nvidia/Llama-3.3-70B-Instruct-FP8 \
      --outdir data \
      --quiet
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from vllm import LLM
from vllm.config import PoolerConfig
from constants import FIELDS, POSITIVE_PROMPTS, NEGATIVE_PROMPTS, NEUTRAL_PROMPTS

# Combined prompts with labels
ALL_PROMPTS: List[Tuple[str, str]] = (
        [(p, "positive") for p in POSITIVE_PROMPTS] +
        [(p, "negative") for p in NEGATIVE_PROMPTS] +
        [(p, "neutral") for p in NEUTRAL_PROMPTS]
)


def safe_slug(s: str) -> str:
    """Filesystem-safe slug for model names etc."""
    s = s.strip()
    s = re.sub(r"[\/\s]+", "_", s)
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s[:180] if len(s) > 180 else s


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=axis, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


def embed_sentences_vllm(sentences: List[str], llm: LLM, quiet: bool = False) -> np.ndarray:
    """
    Embed sentences using vLLM with built-in mean pooling.
    Returns: (N, D) numpy array, L2-normalized.
    """
    if not quiet:
        print(f"  Encoding {len(sentences)} texts with vLLM...")

    outputs = llm.encode(sentences, pooling_task="embed")

    embeddings = []
    for output in outputs:
        vec = np.array(output.outputs.data)
        embeddings.append(vec)

    embeddings = np.array(embeddings)  # (N, D)
    embeddings = l2_normalize(embeddings, axis=1)

    if not quiet:
        print(f"  Generated embeddings shape: {embeddings.shape}")

    return embeddings


def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Cosine similarity between L2-normalized vectors. Returns (N, M)."""
    return A @ B.T


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Prompt-field similarity analysis using vLLM embeddings (mean pooling) + save CSV for multi-model aggregation."
    )
    ap.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                    help="Model to use (supports FP8, quantized models, etc.)")
    ap.add_argument("--max-model-len", type=int, default=200,
                    help="Maximum sequence length (default: 200)")
    ap.add_argument("--pooling-type", choices=["MEAN", "LAST"], default="MEAN",
                    help="Pooling type for embeddings (default: MEAN)")
    ap.add_argument("--tensor-parallel-size", type=int, default=1,
                    help="Number of GPUs for tensor parallelism (default: 1)")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                    help="GPU memory utilization fraction (0.0-1.0, default: 0.9)")
    ap.add_argument("--normalize", action="store_true",
                    help="Let vLLM normalize embeddings (default: False; we normalize manually anyway)")
    ap.add_argument("--outdir", type=str, default="data",
                    help="Directory to save per-model CSV (default: data)")
    ap.add_argument("--outfile", type=str, default="",
                    help="Optional explicit output filename (CSV). If omitted, derived from model/config.")
    ap.add_argument("--quiet", action="store_true",
                    help="Reduce console output (recommended when running many models).")
    args = ap.parse_args()

    quiet = bool(args.quiet)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not quiet:
        print(f"Loading model with vLLM: {args.model}")
        print("Configuration:")
        print("  - Task: embed")
        print(f"  - Pooling: {args.pooling_type}")
        print(f"  - Max model length: {args.max_model_len}")
        print(f"  - Tensor parallel size: {args.tensor_parallel_size}")
        print(f"  - GPU memory utilization: {args.gpu_memory_utilization}")
        print(f"  - Normalize in vLLM: {args.normalize}")
        print(f"  - Output dir: {outdir}")

    try:
        llm = LLM(
            model=args.model,
            # task="embed",
            runner = "pooling",
            pooler_config=PoolerConfig(pooling_type=args.pooling_type, normalize=args.normalize),
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=True,
            trust_remote_code=True,
        )
        if not quiet:
            print("✓ Model loaded successfully with vLLM")
    except Exception as e:
        print("❌ Error loading model with vLLM:")
        print(f"   {e}")
        return

    # ---- Embed prompts + fields ----
    prompt_texts = [p for p, _ in ALL_PROMPTS]
    if not quiet:
        print(f"\nEmbedding {len(prompt_texts)} prompts...")
    prompt_embeddings = embed_sentences_vllm(prompt_texts, llm, quiet=quiet)

    field_keys_list = list(FIELDS.keys())
    field_names_list = [FIELDS[k] for k in field_keys_list]
    if not quiet:
        print(f"\nEmbedding {len(field_names_list)} field names...")
    field_embeddings = embed_sentences_vllm(field_names_list, llm, quiet=quiet)

    similarities = cosine_similarity_matrix(prompt_embeddings, field_embeddings)  # (N_prompts, N_fields)

    # ---- Build tidy results table ----
    rows = []
    for i, (prompt, valence) in enumerate(ALL_PROMPTS):
        for j, field_key in enumerate(field_keys_list):
            rows.append({
                "model": args.model,
                "prompt_idx": i,
                "prompt": prompt,
                "valence": valence,
                "field": field_key,
                "field_name": FIELDS[field_key],
                "similarity": float(similarities[i, j]),
            })

    df_results = pd.DataFrame(rows)

    # ---- Save CSV for aggregator ----
    if args.outfile:
        outpath = Path(args.outfile)
        if not outpath.suffix.lower().endswith(".csv"):
            outpath = outpath.with_suffix(".csv")
        # If relative, place in outdir
        if not outpath.is_absolute():
            outpath = outdir / outpath
    else:
        slug = safe_slug(args.model)
        outname = (
            f"prompt_field_similarity__{slug}"
            f"__maxlen{args.max_model_len}"
            f"__tp{args.tensor_parallel_size}"
            f"__gmu{args.gpu_memory_utilization:.2f}"
            f".csv"
        )
        outpath = outdir / outname

    df_results.to_csv(outpath, index=False)

    if not quiet:
        print(f"\nSaved results CSV: {outpath}")
        print(f"Rows: {len(df_results)} (prompts={len(prompt_texts)} × fields={len(field_keys_list)})")
        print("\nCSV columns (aggregator-compatible): model,prompt,valence,field,similarity (extras kept)")

    # Optional: small console summary (kept lightweight)
    if not quiet:
        # Calculate means by valence and field
        valence_means = {}
        for valence in ["positive", "negative", "neutral"]:
            vdf = df_results[df_results["valence"] == valence]
            means = (
                vdf.groupby("field")["similarity"]
                .mean()
                .sort_values(ascending=False)
            )
            valence_means[valence] = means
            print(f"\nMean similarity by field ({valence}):")
            for f, m in means.items():
                mark = " ← AI" if f == "AI" else ""
                print(f"  {f:>25}: {m:.4f}{mark}")

        # Calculate and display (negative - positive) differences
        if "negative" in valence_means and "positive" in valence_means:
            print("\n" + "=" * 70)
            print("Negative minus Positive (absolute & percentage):")
            print("=" * 70)

            # Get all fields
            all_fields = sorted(set(valence_means["positive"].index) | set(valence_means["negative"].index))

            differences = []
            for field in all_fields:
                pos_val = valence_means["positive"].get(field, 0.0)
                neg_val = valence_means["negative"].get(field, 0.0)
                diff_abs = neg_val - pos_val
                diff_pct = (diff_abs / pos_val * 100) if pos_val != 0 else 0.0
                differences.append((field, diff_abs, diff_pct, pos_val, neg_val))

            # Sort by absolute difference (descending)
            differences.sort(key=lambda x: x[1], reverse=True)

            for field, diff_abs, diff_pct, pos_val, neg_val in differences:
                mark = " ← AI" if field == "AI" else ""
                print(f"  {field:>25}: {diff_abs:+.4f} ({diff_pct:+.2f}%) [pos={pos_val:.4f}, neg={neg_val:.4f}]{mark}")

    if not quiet:
        print("\nDone.")


if __name__ == "__main__":
    main()

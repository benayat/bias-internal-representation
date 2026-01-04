#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from typing import List, Dict

import torch
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    Mistral3ForConditionalGeneration,
    MistralCommonBackend, GptOssForCausalLM
)
from constants import FIELDS, POSITIVE_PROMPTS, NEUTRAL_PROMPTS, NEGATIVE_PROMPTS
# -------------------------
# Research Constants
# -------------------------



# -------------------------
# Modular Helpers
# -------------------------
def wrap_prompt(prompt: str) -> str:
    return "The field of "+prompt

def safe_slug(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[\/\s]+", "_", s)
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s[:180]

def get_backbone(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "language_model") and hasattr(model.language_model, "model"):
        return model.language_model.model
    return model.model if hasattr(model, "model") else model

@torch.inference_mode()
def embed_sentences(sentences, model, tokenizer, device, pooling_type="MEAN", bs=32, max_len=200):
    model.eval()
    backbone = get_backbone(model)
    all_vecs = []
    for i in range(0, len(sentences), bs):
        batch = sentences[i : i + bs]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        out = backbone(input_ids=enc.input_ids, attention_mask=enc.attention_mask)

        if pooling_type == "LAST":
            # Get the last token embeddings for each sequence
            seq_lengths = enc.attention_mask.sum(dim=1) - 1  # -1 for 0-indexing
            batch_indices = torch.arange(out.last_hidden_state.size(0))
            pooled = out.last_hidden_state[batch_indices, seq_lengths]
        else:  # MEAN pooling
            mask = enc.attention_mask.unsqueeze(-1).to(out.last_hidden_state.dtype)
            pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        all_vecs.append(torch.nn.functional.normalize(pooled, p=2, dim=1).cpu())
    return torch.cat(all_vecs, dim=0)

# -------------------------
# Main Logic
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mistralai/Ministral-3-14B-Instruct-2512")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pooling-type", choices=["MEAN", "LAST"], default="MEAN",
                        help="Pooling type for embeddings (default: MEAN)")
    parser.add_argument("--outdir", type=str, default="data")
    parser.add_argument("--outfile", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=200)
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow custom code from the model repository")
    parser.add_argument("--quiet", action="store_true",
                        help="Reduce console output (recommended when running many models).")
    args = parser.parse_args()

    quiet = bool(args.quiet)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not quiet:
        print(f"Loading model with Transformers: {args.model}")
        print("Configuration:")
        print(f"  - Pooling: {args.pooling_type}")
        print(f"  - Max length: {args.max_length}")
        print(f"  - Batch size: {args.batch_size}")
        print(f"  - Device: {args.device}")
        print(f"  - Trust remote code: {args.trust_remote_code}")
        print(f"  - Output dir: {outdir}")

    # 1. Load Model with trust_remote_code integration
    if "mistral" in args.model.lower() and not "mixtral" in args.model.lower():
        tokenizer = MistralCommonBackend.from_pretrained(
            args.model,
            trust_remote_code=args.trust_remote_code
        )
        model = Mistral3ForConditionalGeneration.from_pretrained(
            args.model,
            device_map="auto",
            trust_remote_code=args.trust_remote_code
        )
    elif "openai" in args.model.lower():
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            trust_remote_code=args.trust_remote_code
        )
        model = GptOssForCausalLM.from_pretrained(args.model,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=args.trust_remote_code
        )
    else:
#        config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
#        if not hasattr(config, "num_local_experts"):
#            config.num_local_experts = config.num_experts
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            trust_remote_code=args.trust_remote_code
        )
        model = AutoModel.from_pretrained(
            args.model,
#            config=config,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=args.trust_remote_code
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if not quiet:
        print("✓ Model loaded successfully with Transformers")

    # 2. Embeddings
    field_keys = list(FIELDS.keys())
    field_names = [FIELDS[k] for k in field_keys]
    all_prompts = [(wrap_prompt(p), "positive") for p in POSITIVE_PROMPTS] + [(wrap_prompt(p), "negative") for p in NEGATIVE_PROMPTS] + [(wrap_prompt(p), "neutral") for p in NEUTRAL_PROMPTS]
    # all_prompts = [(p, "positive") for p in POSITIVE_PROMPTS] + [(p, "negative") for p in NEGATIVE_PROMPTS] + [(p, "neutral") for p in NEUTRAL_PROMPTS]

    if not quiet:
        print(f"\nEmbedding {len(field_names)} field names...")
    field_emb = embed_sentences(field_names, model, tokenizer, args.device, args.pooling_type, args.batch_size, args.max_length)

    if not quiet:
        print(f"\nEmbedding {len(all_prompts)} prompts...")
    prompt_emb = embed_sentences([p[0] for p in all_prompts], model, tokenizer, args.device, args.pooling_type, args.batch_size, args.max_length)

    # 3. Process Results
    sims = (prompt_emb @ field_emb.T)
    rows = []
    for i, (prompt, valence) in enumerate(all_prompts):
        for j, key in enumerate(field_keys):
            rows.append({
                "model": args.model,
                "prompt": prompt,
                "valence": valence,
                "field": key,
                "similarity": sims[i, j].item()
            })

    df_results = pd.DataFrame(rows)

    # 4. Save Mechanism
    if args.outfile:
        outpath = Path(args.outfile)
        if not outpath.suffix.lower().endswith(".csv"):
            outpath = outpath.with_suffix(".csv")
        if not outpath.is_absolute():
            outpath = outdir / outpath
    else:
        slug = safe_slug(args.model)
        outname = f"prompt_field_similarity__{slug}__pool_mean__maxlen{args.max_length}__bs{args.batch_size}.csv"
        outpath = outdir / outname

    df_results.to_csv(outpath, index=False)

    if not quiet:
        print(f"\nSaved results CSV: {outpath}")
        print(f"Rows: {len(df_results)} (prompts={len(all_prompts)} × fields={len(field_keys)})")
        print("\nCSV columns (aggregator-compatible): model,prompt,valence,field,similarity")

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

        # Calculate and display (positive - negative) differences
        if "negative" in valence_means and "positive" in valence_means:
            print("\n" + "=" * 70)
            print("Positive minus Negative (absolute & percentage):")
            print("=" * 70)

            # Get all fields
            all_fields = sorted(set(valence_means["positive"].index) | set(valence_means["negative"].index))

            differences = []
            for field in all_fields:
                pos_val = valence_means["positive"].get(field, 0.0)
                neg_val = valence_means["negative"].get(field, 0.0)
                diff_abs = pos_val - neg_val
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

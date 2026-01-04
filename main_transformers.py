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
    MistralCommonBackend
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
def embed_sentences(sentences, model, tokenizer, device, bs=32, max_len=200):
    model.eval()
    backbone = get_backbone(model)
    all_vecs = []
    for i in range(0, len(sentences), bs):
        batch = sentences[i : i + bs]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        out = backbone(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
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
    parser.add_argument("--outdir", type=str, default="data")
    parser.add_argument("--outfile", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=200)
    # New argument added here
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow custom code from the model repository")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

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

    # 2. Embeddings
    field_keys = list(FIELDS.keys())
    field_names = [FIELDS[k] for k in field_keys]
    all_prompts = [(wrap_prompt(p), "positive") for p in POSITIVE_PROMPTS] + [(wrap_prompt(p), "negative") for p in NEGATIVE_PROMPTS] + [(wrap_prompt(p), "neutral") for p in NEUTRAL_PROMPTS]
    # all_prompts = [(p, "positive") for p in POSITIVE_PROMPTS] + [(p, "negative") for p in NEGATIVE_PROMPTS] + [(p, "neutral") for p in NEUTRAL_PROMPTS]

    print(f"Embedding prompts and fields for: {args.model}...")
    field_emb = embed_sentences(field_names, model, tokenizer, args.device, args.batch_size, args.max_length)
    prompt_emb = embed_sentences([p[0] for p in all_prompts], model, tokenizer, args.device, args.batch_size, args.max_length)

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
    print(f"Results saved to: {outpath}")

    # 5. Sorted Summary Print
    print(f"\n" + "="*50 + f"\nRANKINGS FOR: {args.model}\n" + "="*50)
    for v in ["positive", "negative", "neutral"]:
        vdf = df_results[df_results["valence"] == v]
        rankings = vdf.groupby("field")["similarity"].mean().sort_values(ascending=False)

        print(f"\n[{v.upper()} PROMPTS]")
        for field, score in rankings.items():
            tag = " <--- CLOSEST" if field == rankings.index[0] else ""
            print(f"{field:<25} | {score:.4f}{tag}")

if __name__ == "__main__":
    main()

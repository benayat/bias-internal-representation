#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate and display the average rank of each field (subject) across models.

For each model and valence, ranks all fields by similarity (1=highest).
Then averages these ranks across models for each field.

Updates:
- Added % Diff columns to all tables.
- % Diff is calculated relative to the 'AI' field ( (Field - AI) / AI ).

Input CSV must contain columns:
  prompt, valence, field, similarity
Optional:
  model (otherwise inferred from filename stem)

Usage:
  python calculate_average_ranks.py --indir runs/ --glob "*.csv"
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def infer_model_name(df: pd.DataFrame, fp: Path) -> str:
    """Infer model name from dataframe or filename."""
    if "model" in df.columns:
        vals = df["model"].dropna().unique()
        if len(vals) >= 1:
            return str(vals[0])
    return fp.stem


def calculate_ranks_per_model(df: pd.DataFrame, valence: str):
    """
    Calculate average rank of each field for a given valence.

    Returns a dict: {field: average_rank_across_prompts}
    where rank 1 = highest similarity
    """
    vdf = df[df["valence"] == valence].copy()

    # Pivot to get prompts x fields matrix
    piv = vdf.pivot(index="prompt", columns="field", values="similarity")

    # For each prompt (row), rank fields by similarity (1=highest, descending)
    # scipy.stats.rankdata with method='average' handles ties
    # We want higher similarity = lower rank number (rank 1 is best)
    ranks_matrix = piv.rank(axis=1, ascending=False, method="average")

    # Average rank for each field across all prompts
    avg_ranks = ranks_matrix.mean(axis=0).to_dict()

    return avg_ranks


def calculate_avg_similarities_per_model(df: pd.DataFrame, valence: str):
    """
    Calculate average similarity for each field for a given valence.

    Returns a dict: {field: average_similarity_across_prompts}
    """
    vdf = df[df["valence"] == valence].copy()

    # Group by field and calculate mean similarity across prompts
    avg_sims = vdf.groupby("field")["similarity"].mean().to_dict()

    return avg_sims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=str, required=True)
    ap.add_argument("--glob", type=str, default="*.csv")
    args = ap.parse_args()

    indir = Path(args.indir)
    files = sorted(indir.glob(args.glob))
    if not files:
        raise SystemExit(f"No files matched {args.glob} in {indir}")

    # Collect ranks for each field across models
    results = []

    for fp in files:
        df = pd.read_csv(fp)
        model = infer_model_name(df, fp)

        for valence in ["positive", "negative"]:
            avg_ranks = calculate_ranks_per_model(df, valence=valence)
            avg_sims = calculate_avg_similarities_per_model(df, valence=valence)

            for field in avg_ranks.keys():
                results.append({
                    "model": model,
                    "valence": valence,
                    "field": field,
                    "avg_rank": avg_ranks[field],
                    "avg_similarity": avg_sims.get(field, float('nan'))
                })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # ---------------------------------------------------------
    # TABLE 1: AVERAGE RANK
    # ---------------------------------------------------------
    print("\n" + "="*90)
    print("AVERAGE RANK OF EACH FIELD ACROSS MODELS")
    print("="*90)
    print("(Rank 1 = highest similarity. % Diff is relative to AI rank)")

    for valence in ["positive", "negative"]:
        vdf = results_df[results_df["valence"] == valence].copy()

        # Group by field and calculate mean rank across models
        avg_by_field = vdf.groupby("field")["avg_rank"].agg(["mean", "std", "count"])
        avg_by_field = avg_by_field.sort_values("mean")  # Sort by average rank (lower is better)

        # Get AI Baseline for % Calculation
        ai_baseline = float('nan')
        if 'AI' in avg_by_field.index:
            ai_baseline = avg_by_field.loc['AI', 'mean']

        valence_label = "GOOD" if valence == "positive" else "BAD"
        print(f"\n{valence_label} (valence={valence}):")
        print(f"{'Field':<28} {'Avg Rank':>10} {'% vs AI':>10} {'Std Dev':>10} {'N Models':>10}")
        print("-" * 72)

        for field, row in avg_by_field.iterrows():
            mean_val = row['mean']
            
            # Calculate % Diff relative to AI
            pct_str = "N/A"
            if not pd.isna(ai_baseline) and ai_baseline != 0:
                if field == 'AI':
                    pct_str = "0.0%"
                else:
                    diff = (mean_val - ai_baseline) / ai_baseline * 100
                    pct_str = f"{diff:+.1f}%"

            print(f"{field:<28} {mean_val:>10.4f} {pct_str:>10} {row['std']:>10.4f} {int(row['count']):>10d}")

    # ---------------------------------------------------------
    # TABLE 2: AVERAGE SIMILARITY
    # ---------------------------------------------------------
    print("\n" + "="*90)
    print("AVERAGE SIMILARITY TO GOOD/BAD WORDS ACROSS MODELS")
    print("="*90)
    print("(Higher is better. % Diff is relative to AI similarity)")

    for valence in ["positive", "negative"]:
        vdf = results_df[results_df["valence"] == valence].copy()

        # Group by field and calculate mean similarity across models
        avg_sim_by_field = vdf.groupby("field")["avg_similarity"].agg(["mean", "std", "count"])
        avg_sim_by_field = avg_sim_by_field.sort_values("mean", ascending=False)  # Sort by similarity

        # Get AI Baseline for % Calculation
        ai_baseline = float('nan')
        if 'AI' in avg_sim_by_field.index:
            ai_baseline = avg_sim_by_field.loc['AI', 'mean']

        valence_label = "GOOD" if valence == "positive" else "BAD"
        print(f"\n{valence_label} (valence={valence}):")
        print(f"{'Field':<28} {'Avg Sim':>10} {'% vs AI':>10} {'Std Dev':>10} {'N Models':>10}")
        print("-" * 72)

        for field, row in avg_sim_by_field.iterrows():
            mean_val = row['mean']

            # Calculate % Diff relative to AI
            pct_str = "N/A"
            if not pd.isna(ai_baseline) and ai_baseline != 0:
                if field == 'AI':
                    pct_str = "0.0%"
                else:
                    diff = (mean_val - ai_baseline) / ai_baseline * 100
                    pct_str = f"{diff:+.1f}%"

            print(f"{field:<28} {mean_val:>10.4f} {pct_str:>10} {row['std']:>10.4f} {int(row['count']):>10d}")

    # Save detailed results
    out_file = indir / "average_ranks_by_field.csv"
    summary = results_df.groupby(["valence", "field"]).agg({
        "avg_rank": ["mean", "std", "count"],
        "avg_similarity": ["mean", "std"]
    }).reset_index()
    summary.columns = ["valence", "field", "avg_rank_mean", "avg_rank_std", "n_models",
                       "avg_similarity_mean", "avg_similarity_std"]
    summary = summary.sort_values(["valence", "avg_rank_mean"])
    summary.to_csv(out_file, index=False)

    print(f"\n{'='*90}")
    print(f"Detailed results saved to: {out_file}")
    print("="*90)


if __name__ == "__main__":
    main()

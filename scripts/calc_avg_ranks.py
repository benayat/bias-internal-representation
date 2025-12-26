#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate and display the average rank of each field (subject) across models.

For each model and valence, ranks all fields by similarity (1=highest).
Then averages these ranks across models for each field.

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
from scipy import stats
import numpy as np
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def infer_model_name(df: pd.DataFrame, fp: Path) -> str:
    """Infer model name from dataframe or filename."""
    if "model" in df.columns:
        vals = df["model"].dropna().unique()
        if len(vals) >= 1:
            return str(vals[0])
    return fp.stem


def cohens_d(group1, group2):
    """Calculate Cohen's d effect size between two groups."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0.0


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
    # So we negate or use descending order

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

    # Calculate average rank across models for each field and valence
    print("\n" + "="*80)
    print("AVERAGE RANK OF EACH FIELD ACROSS MODELS")
    print("="*80)
    print("(Rank 1 = highest similarity, averaged across prompts then across models)")

    for valence in ["positive", "negative"]:
        vdf = results_df[results_df["valence"] == valence].copy()

        # Group by field and calculate mean rank across models
        avg_by_field = vdf.groupby("field")["avg_rank"].agg(["mean", "std", "count"])
        avg_by_field = avg_by_field.sort_values("mean")  # Sort by average rank (lower is better)

        valence_label = "GOOD" if valence == "positive" else "BAD"
        print(f"\n{valence_label} (valence={valence}):")
        print(f"{'Field':<28} {'Avg Rank':>12} {'Std Dev':>12} {'N Models':>12}")
        print("-" * 68)

        for field, row in avg_by_field.iterrows():
            print(f"{field:<28} {row['mean']:>12.4f} {row['std']:>12.4f} {int(row['count']):>12d}")

    # Now show average similarities to good and bad
    print("\n" + "="*80)
    print("AVERAGE SIMILARITY TO GOOD/BAD WORDS ACROSS MODELS")
    print("="*80)
    print("(Average similarity values, averaged across prompts then across models)")

    for valence in ["positive", "negative"]:
        vdf = results_df[results_df["valence"] == valence].copy()

        # Group by field and calculate mean similarity across models
        avg_sim_by_field = vdf.groupby("field")["avg_similarity"].agg(["mean", "std", "count"])
        avg_sim_by_field = avg_sim_by_field.sort_values("mean", ascending=False)  # Sort by similarity (higher is better)

        valence_label = "GOOD" if valence == "positive" else "BAD"
        print(f"\n{valence_label} (valence={valence}):")
        print(f"{'Field':<28} {'Avg Similarity':>15} {'Std Dev':>12} {'N Models':>12}")
        print("-" * 71)

        for field, row in avg_sim_by_field.iterrows():
            print(f"{field:<28} {row['mean']:>15.6f} {row['std']:>12.6f} {int(row['count']):>12d}")

    # Perform ANOVA tests
    print("\n" + "="*80)
    print("ANOVA TESTS: Are there significant differences between fields?")
    print("="*80)
    print("(One-way ANOVA testing if average similarities differ significantly across fields)")

    for valence in ["positive", "negative"]:
        vdf = results_df[results_df["valence"] == valence].copy()

        # Get list of unique fields
        fields = sorted(vdf["field"].unique())

        # Prepare data for ANOVA: list of arrays, one per field
        # Each array contains the avg_similarity values across models for that field
        field_data = []
        for field in fields:
            field_vals = vdf[vdf["field"] == field]["avg_similarity"].dropna().values
            if len(field_vals) > 0:
                field_data.append(field_vals)

        # Perform one-way ANOVA
        if len(field_data) >= 2:
            f_stat, p_value = stats.f_oneway(*field_data)

            valence_label = "GOOD" if valence == "positive" else "BAD"
            print(f"\n{valence_label} (valence={valence}):")
            print(f"  F-statistic: {f_stat:.4f}")
            print(f"  p-value: {p_value:.4g}")
            if p_value < 0.001:
                print(f"  Result: Highly significant (p < 0.001) ***")
            elif p_value < 0.01:
                print(f"  Result: Significant (p < 0.01) **")
            elif p_value < 0.05:
                print(f"  Result: Significant (p < 0.05) *")
            else:
                print(f"  Result: Not significant (p >= 0.05)")
        else:
            print(f"\n{valence.upper()}: Insufficient data for ANOVA")

    # Post-hoc tests: Tukey HSD for pairwise comparisons
    print("\n" + "="*80)
    print("POST-HOC TESTS: Tukey HSD pairwise comparisons")
    print("="*80)
    print("(Identifies which specific field pairs differ significantly)")

    for valence in ["positive", "negative"]:
        vdf = results_df[results_df["valence"] == valence].copy()
        
        valence_label = "GOOD" if valence == "positive" else "BAD"
        print(f"\n{valence_label} (valence={valence}):")
        
        # Perform Tukey HSD test
        tukey = pairwise_tukeyhsd(endog=vdf["avg_similarity"], 
                                   groups=vdf["field"], 
                                   alpha=0.05)
        
        print(tukey)
        
        # Save Tukey results to CSV
        tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
        tukey_file = indir / f"tukey_posthoc_{valence}.csv"
        tukey_df.to_csv(tukey_file, index=False)
        print(f"  Detailed results saved to: {tukey_file}")

    # Effect sizes: Cohen's d for AI vs each other field
    print("\n" + "="*80)
    print("EFFECT SIZES: Cohen's d for AI vs each other field")
    print("="*80)
    print("(Effect size interpretation: small=0.2, medium=0.5, large=0.8)")

    for valence in ["positive", "negative"]:
        vdf = results_df[results_df["valence"] == valence].copy()
        
        # Get AI similarity values
        ai_vals = vdf[vdf["field"] == "AI"]["avg_similarity"].dropna().values
        
        if len(ai_vals) == 0:
            continue
        
        valence_label = "GOOD" if valence == "positive" else "BAD"
        print(f"\n{valence_label} (valence={valence}):")
        print(f"{'Comparison':<45} {'Cohen\'s d':>12} {'Effect Size':>15}")
        print("-" * 75)
        
        # Calculate Cohen's d for AI vs each other field
        effect_sizes = []
        fields = sorted([f for f in vdf["field"].unique() if f != "AI"])
        
        for field in fields:
            field_vals = vdf[vdf["field"] == field]["avg_similarity"].dropna().values
            if len(field_vals) > 0:
                d = cohens_d(ai_vals, field_vals)
                
                # Interpret effect size
                if abs(d) < 0.2:
                    interpretation = "negligible"
                elif abs(d) < 0.5:
                    interpretation = "small"
                elif abs(d) < 0.8:
                    interpretation = "medium"
                else:
                    interpretation = "large"
                
                effect_sizes.append({
                    "valence": valence,
                    "comparison": f"AI vs {field}",
                    "cohens_d": d,
                    "interpretation": interpretation
                })
                
                print(f"{'AI vs ' + field:<45} {d:>12.4f} {interpretation:>15}")
        
        # Save effect sizes
        if effect_sizes:
            es_df = pd.DataFrame(effect_sizes)
            es_file = indir / f"effect_sizes_{valence}.csv"
            es_df.to_csv(es_file, index=False)
            print(f"  Results saved to: {es_file}")


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

    print(f"\n{'='*80}")
    print(f"Detailed results saved to: {out_file}")
    print("="*80)


if __name__ == "__main__":
    main()


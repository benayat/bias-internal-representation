#!/usr/bin/env python3
"""
Calculate confidence intervals for each field's similarity to prompts by valence.

This script computes the mean similarity and 95% CI for each field WITHOUT
comparing to other fields. Useful for reporting the similarity of individual
subjects (e.g., AI, Race, Gender) to good/bad prompts with uncertainty bounds.

Input CSV must contain columns:
  prompt, valence, field, similarity
Optional:
  model (otherwise inferred from filename stem)

Usage:
  python calc_field_similarity_ci.py --indir data/ --glob "*.csv" --confidence 0.95
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats


def infer_model_name(df: pd.DataFrame, fp: Path) -> str:
    """Infer model name from dataframe or filename."""
    if "model" in df.columns:
        vals = df["model"].dropna().unique()
        if len(vals) >= 1:
            return str(vals[0])
    return fp.stem


def calculate_avg_similarities_per_model(df: pd.DataFrame, valence: str):
    """
    Calculate average similarity for each field for a given valence.

    Returns a dict: {field: average_similarity_across_prompts}
    """
    vdf = df[df["valence"] == valence].copy()
    avg_sims = vdf.groupby("field")["similarity"].mean().to_dict()
    return avg_sims


def calculate_ci(values: np.ndarray, confidence: float = 0.95):
    """
    Calculate mean and confidence interval for a set of values.
    
    Returns: (mean, std, n, ci_low, ci_high, margin_of_error)
    """
    values = values[np.isfinite(values)]
    n = len(values)
    
    if n < 2:
        return {
            "mean": float(values[0]) if n == 1 else np.nan,
            "std": np.nan,
            "n": n,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "margin": np.nan,
        }
    
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    se = std / np.sqrt(n)
    
    # t-critical value for two-sided CI
    alpha = 1.0 - confidence
    df = n - 1
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df))
    
    margin = t_crit * se
    ci_low = mean - margin
    ci_high = mean + margin
    
    return {
        "mean": mean,
        "std": std,
        "n": n,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "margin": float(margin),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=str, required=True, help="Directory containing CSV files")
    ap.add_argument("--glob", type=str, default="*.csv", help="Glob pattern for CSV files")
    ap.add_argument("--confidence", type=float, default=0.95, help="Confidence level (default: 0.95)")
    ap.add_argument("--output", type=str, default="", help="Optional output CSV file path")
    args = ap.parse_args()

    indir = Path(args.indir)
    files = sorted(indir.glob(args.glob))
    if not files:
        raise SystemExit(f"No files matched {args.glob} in {indir}")

    confidence_pct = int(args.confidence * 100)

    # Collect similarities for each field across models
    results = []

    for fp in files:
        df = pd.read_csv(fp)
        model = infer_model_name(df, fp)

        for valence in df["valence"].unique():
            avg_sims = calculate_avg_similarities_per_model(df, valence=valence)

            for field, sim in avg_sims.items():
                results.append({
                    "model": model,
                    "valence": valence,
                    "field": field,
                    "avg_similarity": sim
                })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Calculate CIs for each field by valence
    print("\n" + "="*95)
    print(f"FIELD SIMILARITY WITH {confidence_pct}% CONFIDENCE INTERVALS")
    print("="*95)
    print("(Mean similarity across models, with uncertainty bounds)")
    print()

    output_rows = []

    for valence in sorted(results_df["valence"].unique()):
        vdf = results_df[results_df["valence"] == valence].copy()

        # Get unique fields and calculate CIs
        field_stats = []
        for field in sorted(vdf["field"].unique()):
            field_vals = vdf[vdf["field"] == field]["avg_similarity"].values
            ci_info = calculate_ci(field_vals, confidence=args.confidence)
            
            field_stats.append({
                "field": field,
                **ci_info
            })
            
            output_rows.append({
                "valence": valence,
                "field": field,
                "mean_similarity": ci_info["mean"],
                "std": ci_info["std"],
                "n_models": ci_info["n"],
                "ci_low": ci_info["ci_low"],
                "ci_high": ci_info["ci_high"],
                "margin_of_error": ci_info["margin"],
                "confidence_level": args.confidence,
            })

        # Sort by mean similarity (descending)
        field_stats = sorted(field_stats, key=lambda x: x["mean"] if not np.isnan(x["mean"]) else -np.inf, reverse=True)

        valence_label = valence.upper()
        print(f"{valence_label} Valence:")
        print(f"{'Field':<25} {'Mean':>10} {'Std':>10} {f'{confidence_pct}% CI':>28} {'±':>10} {'N':>6}")
        print("-" * 95)

        for fs in field_stats:
            field = fs["field"]
            mean = fs["mean"]
            std = fs["std"]
            ci_low = fs["ci_low"]
            ci_high = fs["ci_high"]
            margin = fs["margin"]
            n = fs["n"]
            
            if np.isnan(mean):
                ci_str = "N/A"
                margin_str = "N/A"
            else:
                ci_str = f"[{ci_low:.6f}, {ci_high:.6f}]"
                margin_str = f"±{margin:.6f}"
            
            std_str = f"{std:.6f}" if not np.isnan(std) else "N/A"
            
            print(f"{field:<25} {mean:>10.6f} {std_str:>10} {ci_str:>28} {margin_str:>10} {n:>6d}")
        
        print()

    # Save to CSV if requested
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = indir / f"field_similarity_ci_{confidence_pct}pct.csv"
    
    output_df = pd.DataFrame(output_rows)
    output_df.to_csv(output_path, index=False)
    
    print("="*95)
    print(f"Results saved to: {output_path}")
    print("="*95)


if __name__ == "__main__":
    main()


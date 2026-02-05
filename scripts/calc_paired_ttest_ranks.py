#!/usr/bin/env python3
"""
Paired statistical tests with Holm-Bonferroni correction for AI vs other fields.

Test selection by metric:
- Similarity: Paired t-test (parametric, assumes normality of differences)
- Rank: Wilcoxon signed-rank test (non-parametric, ordinal data)

Why different tests?
- Similarities are continuous interval data → parametric t-test appropriate
- Ranks are ordinal data → non-parametric Wilcoxon more appropriate
- Both use Holm-Bonferroni correction for multiple comparisons

Why Holm-Bonferroni?
- Multiple comparisons require family-wise error rate control
- Less conservative than Bonferroni while maintaining strong FWER control

Accepted inputs
--------------
A) Prompt-level similarities (recommended):
   Columns: model, prompt, valence, field, similarity
   (extra columns are ok)
   We reduce to per-model means: mean(similarity) over prompts.

B) Per-model field values:
   Columns: model, valence, field, <value_col>
   e.g., mean_similarity, avg_rank, etc.

Outputs
-------
For each valence:
- AI vs MEAN(all non-AI fields) [paired]
- AI vs each field [paired]
Reports:
  median diff, 95% CI (bootstrap), W statistic, one-sided p-value (H1: median(diff) > 0),
  adjusted p-value (Holm-Bonferroni), N paired models, and "wins" (#models with diff>0).

Notes
-----
- If metric=rank: lower rank is better. We define diff = other_rank - ai_rank
  so diff>0 means AI ranks better (smaller).
- If metric=similarity: higher is better. We define diff = ai_sim - other_sim
  so diff>0 means AI is closer.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats


# -------------------------
# Utilities
# -------------------------

def paired_ttest_stats(d: np.ndarray, confidence: float = 0.95, alternative: str = "greater"):
    """
    One-sample t-test on differences d (for similarity metric), with:
    - mean diff
    - 95% two-sided t CI for mean(d)
    - t statistic
    - one-sided p-value for H1: mean(d) > 0 (or < 0)
    """
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    n = int(d.size)
    if n < 2:
        return {
            "n": n,
            "stat_value": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "test_stat": np.nan,
            "p_one_sided": np.nan,
            "wins": int(np.sum(d > 0)) if n > 0 else 0,
        }

    mean_d = float(d.mean())
    sd_d = float(d.std(ddof=1))
    se_d = sd_d / np.sqrt(n)

    if se_d == 0.0:
        # All differences identical. t is +/-inf if mean != 0 else 0.
        if mean_d == 0.0:
            t_stat = 0.0
            p_one = 1.0
        else:
            t_stat = np.inf if mean_d > 0 else -np.inf
            p_one = 0.0 if (alternative == "greater" and mean_d > 0) or (alternative == "less" and mean_d < 0) else 1.0
        ci_low = ci_high = mean_d
        return {
            "n": n,
            "stat_value": mean_d,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "test_stat": t_stat,
            "p_one_sided": p_one,
            "wins": int(np.sum(d > 0)),
        }

    df = n - 1
    t_stat = mean_d / se_d

    # One-sided p-value
    if alternative == "greater":
        p_one = float(1.0 - stats.t.cdf(t_stat, df))
    elif alternative == "less":
        p_one = float(stats.t.cdf(t_stat, df))
    else:
        raise ValueError("alternative must be 'greater' or 'less'")

    alpha = 1.0 - confidence
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df))
    margin = t_crit * se_d
    ci_low = mean_d - margin
    ci_high = mean_d + margin

    return {
        "n": n,
        "stat_value": float(mean_d),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "test_stat": float(t_stat),
        "p_one_sided": float(p_one),
        "wins": int(np.sum(d > 0)),
    }


def wilcoxon_diff_stats(d: np.ndarray, confidence: float = 0.95, alternative: str = "greater", n_bootstrap: int = 10000):
    """
    Wilcoxon signed-rank test on differences d (for rank metric), with:
    - median diff
    - 95% bootstrap CI for median(d)
    - W statistic (sum of positive ranks)
    - one-sided p-value for H1: median(d) > 0 (or < 0)
    """
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    n = int(d.size)

    if n < 2:
        return {
            "n": n,
            "stat_value": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "test_stat": np.nan,
            "p_one_sided": np.nan,
            "wins": int(np.sum(d > 0)) if n > 0 else 0,
        }

    median_d = float(np.median(d))

    # Bootstrap CI for median
    np.random.seed(42)
    bootstrap_medians = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(d, size=n, replace=True)
        bootstrap_medians.append(np.median(sample))

    alpha = 1.0 - confidence
    ci_low = float(np.percentile(bootstrap_medians, 100 * alpha / 2))
    ci_high = float(np.percentile(bootstrap_medians, 100 * (1 - alpha / 2)))

    # Wilcoxon signed-rank test
    # Remove zeros (ties at zero are dropped in Wilcoxon)
    d_nonzero = d[d != 0]

    if len(d_nonzero) < 1:
        # All zeros - no difference
        return {
            "n": n,
            "stat_value": median_d,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "test_stat": 0.0,
            "p_one_sided": 1.0,
            "wins": int(np.sum(d > 0)),
        }

    # Perform Wilcoxon signed-rank test
    try:
        if alternative == "greater":
            result = stats.wilcoxon(d_nonzero, alternative='greater')
        elif alternative == "less":
            result = stats.wilcoxon(d_nonzero, alternative='less')
        else:
            result = stats.wilcoxon(d_nonzero, alternative='two-sided')

        W_stat = float(result.statistic)
        p_one = float(result.pvalue)
    except Exception as e:
        # In case of issues (e.g., all same sign), handle gracefully
        W_stat = np.nan
        p_one = np.nan

    return {
        "n": n,
        "stat_value": median_d,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "test_stat": W_stat,
        "p_one_sided": p_one,
        "wins": int(np.sum(d > 0)),
    }


def holm_bonferroni_correction(p_values):
    """
    Apply Holm-Bonferroni correction to a list of p-values.
    Returns adjusted p-values.
    """
    n = len(p_values)
    if n == 0:
        return []

    # Sort p-values and keep track of original indices
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]

    # Calculate adjusted p-values
    adjusted_p = np.zeros(n)
    for i, p in enumerate(sorted_p):
        adjusted_p[i] = min(float(p * (n - i)), 1.0)

    # Enforce monotonicity (each adjusted p should be >= previous)
    for i in range(1, n):
        adjusted_p[i] = max(adjusted_p[i], adjusted_p[i-1])

    # Unsort to original order
    result = np.zeros(n)
    result[sorted_indices] = adjusted_p

    return result.tolist()


def load_input(file_path: str | None, indir: str | None, glob_pat: str | None) -> pd.DataFrame:
    if file_path:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading --file-path: {e}", file=sys.stderr)
            sys.exit(1)

    if not indir or not glob_pat:
        print("Must provide either --file-path OR (--indir AND --glob).", file=sys.stderr)
        sys.exit(1)

    p = Path(indir)
    files = sorted(p.glob(glob_pat))
    if not files:
        print(f"No files matched in {p} with glob '{glob_pat}'", file=sys.stderr)
        sys.exit(1)

    dfs = []
    for f in files:
        try:
            dfi = pd.read_csv(f)
            dfs.append(dfi)
        except Exception as e:
            print(f"Error reading {f}: {e}", file=sys.stderr)
            sys.exit(1)

    return pd.concat(dfs, ignore_index=True)


def reduce_to_per_model_field(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    Produce one row per (model, valence, field) with column 'value'.
    Handles:
      - prompt-level similarity tables: has 'prompt' and 'similarity' columns
      - already per-model tables: has value_col directly
    """
    required_triplet = {"model", "valence", "field"}
    if not required_triplet.issubset(df.columns):
        raise ValueError(f"Input must contain columns: {sorted(required_triplet)}")

    # Case A: prompt-level similarity
    if "similarity" in df.columns and "prompt" in df.columns:
        per = (
            df.groupby(["model", "valence", "field"], as_index=False)["similarity"]
            .mean()
            .rename(columns={"similarity": "value"})
        )
        return per

    # Case B: already per-model values
    if value_col not in df.columns:
        raise ValueError(
            f"Could not find '{value_col}' in input. "
            f"Columns present: {list(df.columns)}"
        )

    per = df[["model", "valence", "field", value_col]].rename(columns={value_col: "value"}).copy()
    return per


def add_ranks(per: pd.DataFrame, higher_is_better: bool) -> pd.DataFrame:
    """
    Adds 'rank' per (model,valence) over fields based on 'value'.
    rank=1 is best.
    """
    asc = not higher_is_better
    per = per.copy()
    per["rank"] = per.groupby(["model", "valence"])["value"].rank(method="average", ascending=asc)
    return per


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file-path", type=str, default="", help="Single CSV file (prompt-level or per-model).")
    ap.add_argument("--indir", type=str, default="", help="Directory of CSVs (if not using --file-path).")
    ap.add_argument("--glob", type=str, default="", help="Glob for CSVs inside --indir, e.g. 'prompt*.csv'.")
    ap.add_argument("--value-col", type=str, default="avg_rank_mean",
                    help="If input is per-model, column to use as value (ignored for prompt-level similarity).")
    ap.add_argument("--metric", choices=["similarity", "rank"], default="rank",
                    help="Test on per-model similarities or on per-model ranks derived from them.")
    ap.add_argument("--confidence", type=float, default=0.95)
    ap.add_argument("--alpha", type=float, default=0.05, help="Significance threshold for display.")
    ap.add_argument("--ai-key", type=str, default="AI", help="Field key for AI in the CSV.")
    args = ap.parse_args()

    file_path = args.file_path.strip() or None
    indir = args.indir.strip() or None
    glob_pat = args.glob.strip() or None

    df = load_input(file_path=file_path, indir=indir, glob_pat=glob_pat)

    # Reduce to per-model (model,valence,field) table with 'value'
    try:
        per = reduce_to_per_model_field(df, value_col=args.value_col)
    except Exception as e:
        print(f"Input format error: {e}", file=sys.stderr)
        sys.exit(1)

    # Decide direction (what is "better") and compute per-model metric
    if args.metric == "similarity":
        # higher similarity = better
        higher_is_better = True
        per_metric = per.rename(columns={"value": "metric"}).copy()
    else:
        # rank: derive from similarity if we have it; otherwise rank from provided value_col
        # If user feeds avg_rank already, ranking again would be wrong; but we still can treat it as "metric".
        # Heuristic: if prompt-level similarity was used, 'value' is similarity; compute ranks from that.
        # If per-model values were provided, assume they're already ranks.
        if "prompt" in df.columns and "similarity" in df.columns:
            higher_is_better = True  # ranking based on similarity
            per_ranked = add_ranks(per, higher_is_better=higher_is_better)
            per_metric = per_ranked.rename(columns={"rank": "metric"}).copy()
        else:
            # Assume provided values are ranks: lower rank = better
            per_metric = per.rename(columns={"value": "metric"}).copy()

    # Build model x field matrix per valence
    if args.ai_key not in per_metric["field"].unique():
        print(f"AI key '{args.ai_key}' not found in field column.", file=sys.stderr)
        sys.exit(1)

    # Print actual similarity values table (always, regardless of metric)
    print("\n" + "=" * 80)
    print("MEAN SIMILARITY VALUES BY FIELD AND VALENCE")
    print("(averaged across models, regardless of test metric)")
    print("=" * 80)

    for val in sorted(per["valence"].unique()):
        sub_sim = per[per["valence"] == val].copy()
        wide_sim = sub_sim.pivot_table(index="model", columns="field", values="value", aggfunc="mean")

        if wide_sim.empty:
            continue

        # Calculate mean similarity per field across models and sort descending (higher = better)
        field_sim_means = wide_sim.mean(axis=0).sort_values(ascending=False)

        print(f"\n{val.upper()} Valence:")
        print(f"{'Rank':<6} | {'Field':<25} | {'Mean Similarity':>18} | {'N Models':>9}")
        print("-" * 66)

        for rank, (field, mean_sim) in enumerate(field_sim_means.items(), start=1):
            n_models = wide_sim[field].notna().sum()
            marker = " (AI)" if field == args.ai_key else ""
            print(f"{rank:<6} | {field:<25}{marker} | {mean_sim:>18.6f} | {n_models:>9}")

    # Print rank tables for each valence first
    print("\n" + "=" * 80)
    if args.metric == "rank":
        print("FIELD RANKINGS BY VALENCE (lower rank = better)")
        metric_col_label = "Mean Rank"
    else:
        print("FIELD RANKINGS BY VALENCE (higher similarity = better)")
        metric_col_label = "Mean Similarity"
    print("=" * 80)

    for val in sorted(per_metric["valence"].unique()):
        sub = per_metric[per_metric["valence"] == val].copy()
        wide = sub.pivot_table(index="model", columns="field", values="metric", aggfunc="mean")

        if wide.empty:
            continue

        # Calculate mean metric per field across models
        # For rank: sort ascending (lower rank = better)
        # For similarity: sort descending (higher similarity = better)
        field_means = wide.mean(axis=0).sort_values(ascending=(args.metric == "rank"))

        print(f"\n{val.upper()} Valence:")
        print(f"{'Rank':<6} | {'Field':<25} | {metric_col_label:>15} | {'N Models':>9}")
        print("-" * 63)

        for rank, (field, mean_val) in enumerate(field_means.items(), start=1):
            n_models = wide[field].notna().sum()
            marker = " (AI)" if field == args.ai_key else ""
            print(f"{rank:<6} | {field:<25}{marker} | {mean_val:>15.4f} | {n_models:>9}")

    print("\n" + "=" * 80)
    if args.metric == "similarity":
        print("\nSTATISTICAL TESTS (Paired t-test with Holm-Bonferroni correction)")
        stat_label = "Mean"
        test_stat_label = "t"
    else:
        print("\nSTATISTICAL TESTS (Wilcoxon signed-rank with Holm-Bonferroni correction)")
        stat_label = "Median"
        test_stat_label = "W"
    print("=" * 80 + "\n")

    print(f"{'Valence':<10} | {'Comparison':<35} | {stat_label+'Diff':>8} | {'95% CI':<21} | {test_stat_label:>10} | {'p(1s)':>10} | {'p(adj)':>10} | {'N':>3} | {'Wins':>4} | Sig")
    print("-" * 145)

    for val in sorted(per_metric["valence"].unique()):
        sub = per_metric[per_metric["valence"] == val].copy()

        wide = sub.pivot_table(index="model", columns="field", values="metric", aggfunc="mean")
        if args.ai_key not in wide.columns:
            continue

        # Require at least 2 models for inference
        models = wide.index

        # Store all comparisons for this valence to apply Holm-Bonferroni
        comparisons = []

        # Select test function based on metric
        if args.metric == "similarity":
            test_func = paired_ttest_stats
        else:
            test_func = wilcoxon_diff_stats

        # Compute AI vs mean of others (paired per model)
        other_cols = [c for c in wide.columns if c != args.ai_key]
        if other_cols:
            ai = wide[args.ai_key]
            others_mean = wide[other_cols].mean(axis=1)

            if args.metric == "rank" and not ("prompt" in df.columns and "similarity" in df.columns):
                # provided ranks: lower is better -> diff = others - ai (positive => AI better)
                d = (others_mean - ai).dropna().to_numpy()
                alt = "greater"
            elif args.metric == "rank":
                # derived ranks: lower is better -> diff = others - ai
                d = (others_mean - ai).dropna().to_numpy()
                alt = "greater"
            else:
                # similarity: higher is better -> diff = ai - others
                d = (ai - others_mean).dropna().to_numpy()
                alt = "greater"

            st = test_func(d, confidence=args.confidence, alternative=alt)
            comparisons.append(('AI vs MEAN(all non-AI)', st))

        # AI vs each field (paired per model)
        for field in sorted([c for c in wide.columns if c != args.ai_key]):
            ai = wide[args.ai_key]
            oth = wide[field]

            if args.metric == "rank":
                # lower is better -> positive diff means AI better
                d = (oth - ai).dropna().to_numpy()
                alt = "greater"
            else:
                # similarity higher is better
                d = (ai - oth).dropna().to_numpy()
                alt = "greater"

            st = test_func(d, confidence=args.confidence, alternative=alt)
            comp = f"AI vs {field}"
            comparisons.append((comp, st))

        # Apply Holm-Bonferroni correction
        p_values = [st["p_one_sided"] for _, st in comparisons]
        adjusted_p_values = holm_bonferroni_correction(p_values)

        # Print results with adjusted p-values
        for (comp, st), p_adj in zip(comparisons, adjusted_p_values):
            sig = "YES" if p_adj < args.alpha else "NO"
            print(
                f"{val:<10} | {comp:<35} | {st['stat_value']:+8.4f} | "
                f"[{st['ci_low']:+.4f}, {st['ci_high']:+.4f}] | {st['test_stat']:>10.1f} | {st['p_one_sided']:>10.2e} | "
                f"{p_adj:>10.2e} | {st['n']:>3d} | {st['wins']:>4d} | {sig}"
            )

        print("-" * 145)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Paired t-tests for AI vs other fields (model is the pairing unit).

Why paired?
- Each model produces scores for *all* fields, so comparisons are matched within-model.
- We test on per-model differences, i.e., one-sample t-test on d_m.

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
  mean diff, 95% CI (two-sided), t, one-sided p-value (H1: mean(diff) > 0),
  N paired models, and "wins" (#models with diff>0).

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

def one_sided_p_from_t(t_stat: float, df: int, alternative: str) -> float:
    """Compute one-sided p-value from t-stat and df."""
    if alternative == "greater":
        return float(1.0 - stats.t.cdf(t_stat, df))
    if alternative == "less":
        return float(stats.t.cdf(t_stat, df))
    raise ValueError("alternative must be 'greater' or 'less'")


def paired_diff_stats(d: np.ndarray, confidence: float = 0.95, alternative: str = "greater"):
    """
    One-sample t-test on differences d, with:
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
            "mean": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "t": np.nan,
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
            "mean": mean_d,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "t": t_stat,
            "p_one_sided": p_one,
            "wins": int(np.sum(d > 0)),
        }

    df = n - 1
    t_stat = mean_d / se_d
    p_one = one_sided_p_from_t(t_stat, df, alternative)

    alpha = 1.0 - confidence
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df))
    margin = t_crit * se_d
    ci_low = mean_d - margin
    ci_high = mean_d + margin

    return {
        "n": n,
        "mean": mean_d,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "t": float(t_stat),
        "p_one_sided": float(p_one),
        "wins": int(np.sum(d > 0)),
    }


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

    print(f"{'Valence':<10} | {'Comparison':<35} | {'MeanDiff':>8} | {'95% CI':<21} | {'t':>8} | {'p(1s)':>10} | {'N':>3} | {'Wins':>4} | Sig")
    print("-" * 130)

    for val in sorted(per_metric["valence"].unique()):
        sub = per_metric[per_metric["valence"] == val].copy()

        wide = sub.pivot_table(index="model", columns="field", values="metric", aggfunc="mean")
        if args.ai_key not in wide.columns:
            continue

        # Require at least 2 models for inference
        models = wide.index

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

            st = paired_diff_stats(d, confidence=args.confidence, alternative=alt)
            sig = "YES" if st["p_one_sided"] < args.alpha else "NO"
            print(
                f"{val:<10} | {'AI vs MEAN(all non-AI)':<35} | {st['mean']:+8.4f} | "
                f"[{st['ci_low']:+.4f}, {st['ci_high']:+.4f}] | {st['t']:>8.3f} | {st['p_one_sided']:>10.2e} | "
                f"{st['n']:>3d} | {st['wins']:>4d} | {sig}"
            )

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

            st = paired_diff_stats(d, confidence=args.confidence, alternative=alt)
            sig = "YES" if st["p_one_sided"] < args.alpha else "NO"
            comp = f"AI vs {field}"
            print(
                f"{val:<10} | {comp:<35} | {st['mean']:+8.4f} | "
                f"[{st['ci_low']:+.4f}, {st['ci_high']:+.4f}] | {st['t']:>8.3f} | {st['p_one_sided']:>10.2e} | "
                f"{st['n']:>3d} | {st['wins']:>4d} | {sig}"
            )

        print("-" * 130)


if __name__ == "__main__":
    main()

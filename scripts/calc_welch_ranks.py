import argparse
import sys
import pandas as pd
import numpy as np
from scipy import stats

def get_welch_stats(m1, s1, n1, m2, s2, n2, confidence=0.95):
    """
    Calculates t-statistic, p-value, and Confidence Interval using Welch's formula.
    """
    # 1. Standard Error of Difference (SED)
    if s1 == 0 and s2 == 0:
        return (1.0 if m1 == m2 else 0.0), (0.0, 0.0)

    sed = np.sqrt((s1**2 / n1) + (s2**2 / n2))
    
    if sed == 0:
        return (1.0 if m1 == m2 else 0.0), (0.0, 0.0)

    # 2. Degrees of Freedom (Welch-Satterthwaite equation)
    num = ((s1**2 / n1) + (s2**2 / n2))**2
    den_term1 = ((s1**2 / n1)**2) / (n1 - 1)
    den_term2 = ((s2**2 / n2)**2) / (n2 - 1)
    
    if (den_term1 + den_term2) == 0:
        df_val = n1 + n2 - 2 
    else:
        df_val = num / (den_term1 + den_term2)
    
    # 3. Calculate P-Value
    t_stat = (m1 - m2) / sed
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df_val))
    
    # 4. Calculate Confidence Interval
    # Critical t-value for the desired confidence level (e.g., 95%)
    alpha = 1 - confidence
    t_critical = stats.t.ppf(1 - alpha/2, df_val)
    
    margin_of_error = t_critical * sed
    mean_diff = m1 - m2
    
    ci_lower = mean_diff - margin_of_error
    ci_upper = mean_diff + margin_of_error
    
    return p_val, (ci_lower, ci_upper)

def get_pooled_stats(df_subset):
    """
    Aggregates means and standard deviations from multiple subgroups.
    """
    means = df_subset['avg_rank_mean']
    stds = df_subset['avg_rank_std']
    ns = df_subset['n_models']
    
    n_total = ns.sum()
    mean_pooled = (means * ns).sum() / n_total
    
    ss_within = ((ns - 1) * stds**2).sum()
    ss_between = (ns * (means - mean_pooled)**2).sum()
    
    var_pooled = (ss_within + ss_between) / (n_total - 1)
    std_pooled = np.sqrt(var_pooled)
    
    return mean_pooled, std_pooled, n_total

def main():
    parser = argparse.ArgumentParser(description="Calculate statistical significance and CIs for AI rank vs others.")
    parser.add_argument("--file-path", type=str, required=True, help="Path to the input CSV file.")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.file_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    # Header
    print(f"{'Context':<10} | {'Comparison':<30} | {'Diff':<6} | {'95% CI':<18} | {'P-Value':<10} | {'Sig?'}")
    print("-" * 105)

    for valence in df['valence'].unique():
        subset = df[df['valence'] == valence]
        if 'AI' not in subset['field'].values: continue

        # AI Stats
        ai_data = subset[subset['field'] == 'AI'].iloc[0]
        ai_m, ai_s, ai_n = ai_data['avg_rank_mean'], ai_data['avg_rank_std'], ai_data['n_models']
        
        others_df = subset[subset['field'] != 'AI']
        if others_df.empty: continue

        # --- A. AI vs Average of All Others (Pooled) ---
        pool_m, pool_s, pool_n = get_pooled_stats(others_df)
        p_val, (ci_low, ci_high) = get_welch_stats(ai_m, ai_s, ai_n, pool_m, pool_s, pool_n)
        
        is_sig = "YES" if p_val < 0.05 else "NO"
        diff = ai_m - pool_m
        
        print(f"{valence:<10} | {'AI vs ALL OTHERS':<30} | {diff:+.2f}   | [{ci_low:+.2f}, {ci_high:+.2f}]   | {p_val:.2e}   | {is_sig}")

        # --- B. AI vs Individual Fields ---
        for _, row in others_df.iterrows():
            other_m, other_s, other_n = row['avg_rank_mean'], row['avg_rank_std'], row['n_models']
            p_val, (ci_low, ci_high) = get_welch_stats(ai_m, ai_s, ai_n, other_m, other_s, other_n)
            
            is_sig = "YES" if p_val < 0.05 else "NO"
            diff = ai_m - other_m
            field_name = str(row['field']).replace('_', ' ')
            
            print(f"{valence:<10} | {f'AI vs {field_name}':<30} | {diff:+.2f}   | [{ci_low:+.2f}, {ci_high:+.2f}]   | {p_val:.2e}   | {is_sig}")
        
        print("-" * 105)

if __name__ == "__main__":
    main()

import argparse
import sys
import pandas as pd
import numpy as np
from scipy import stats

def get_welch_ttest(m1, s1, n1, m2, s2, n2):
    # (Same Welch function as before)
    if s1 == 0 and s2 == 0: return 1.0 if m1 == m2 else 0.0
    sed = np.sqrt((s1**2 / n1) + (s2**2 / n2))
    if sed == 0: return 1.0 if m1 == m2 else 0.0
    t_stat = (m1 - m2) / sed
    num = ((s1**2 / n1) + (s2**2 / n2))**2
    den = (((s1**2 / n1)**2) / (n1 - 1)) + (((s2**2 / n2)**2) / (n2 - 1))
    df = num / den
    return 2 * (1 - stats.t.cdf(abs(t_stat), df))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-path", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.file_path)

    print(f"{'Context':<10} | {'Comparison':<30} | {'Diff':<6} | {'P-Value':<10} | {'Sig?'}")
    print("-" * 80)

    for valence in df['valence'].unique():
        subset = df[df['valence'] == valence]
        if 'AI' not in subset['field'].values: continue

        # --- KEY CHANGE: Grab Similarity Columns instead of Rank ---
        ai_data = subset[subset['field'] == 'AI'].iloc[0]
        ai_m = ai_data['avg_similarity_mean']
        ai_s = ai_data['avg_similarity_std']
        ai_n = ai_data['n_models']
        
        others_df = subset[subset['field'] != 'AI']
        
        for _, row in others_df.iterrows():
            other_m = row['avg_similarity_mean']
            other_s = row['avg_similarity_std']
            other_n = row['n_models']

            p_val = get_welch_ttest(ai_m, ai_s, ai_n, other_m, other_s, other_n)
            is_sig = "YES" if p_val < 0.05 else "NO"
            diff = ai_m - other_m
            
            field = str(row['field']).replace('_', ' ')
            print(f"{valence:<10} | {f'AI vs {field}':<30} | {diff:+.2f}   | {p_val:.2e}   | {is_sig}")
        print("-" * 80)

if __name__ == "__main__":
    main()

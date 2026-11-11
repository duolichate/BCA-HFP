# bca_hfp/analysis/sensitive_filter.py
"""Filter sensitive samples and recommend candidates for analysis."""

import os
import sys
import argparse
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bca_hfp.config.base_config import PROJECT_ROOT, root_dir


def load_predictions(csv_path):
    """Load prediction results CSV"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded prediction results: {len(df)} samples")
    return df


def filter_sensitive(df, threshold=0):
    """Filter sensitive samples (LN_IC50 < threshold)"""
    sensitive = df[df['true_ln_ic50'] < threshold].copy()
    sensitive['abs_error'] = np.abs(sensitive['true_ln_ic50'] - sensitive['pred_ln_ic50'])
    return sensitive


def print_statistics(sensitive):
    """Print sensitive sample statistics"""
    print("\n" + "=" * 70)
    print(f"Sensitive Sample Statistics (LN_IC50 < 0)")
    print("=" * 70)
    print(f"Total samples: {len(sensitive)}")
    print(f"Number of drugs: {sensitive['drug_id'].nunique()}")
    print(f"Number of cell lines: {sensitive['cell_line_id'].nunique()}")

    # Statistics by drug
    print("\n--- Statistics by Drug ---")
    drug_stats = sensitive.groupby('drug_name').agg({
        'true_ln_ic50': ['count', 'mean', 'min'],
        'abs_error': 'mean'
    }).round(4)
    drug_stats.columns = ['Count', 'Mean IC50', 'Min IC50', 'Mean Error']
    drug_stats = drug_stats.sort_values('Mean IC50')
    print(drug_stats)


def print_most_sensitive(sensitive, n=10):
    """Print most sensitive samples"""
    print("\n" + "=" * 70)
    print(f"Top {n} Most Sensitive Samples (Lowest LN_IC50)")
    print("=" * 70)
    cols = ['drug_name', 'drug_id', 'cell_line_name', 'cell_line_id',
            'true_ln_ic50', 'pred_ln_ic50', 'abs_error']
    print(sensitive.sort_values('true_ln_ic50')[cols].head(n).to_string(index=False))


def print_best_predictions(sensitive, n=10):
    """Print best predictions among sensitive samples"""
    print("\n" + "=" * 70)
    print(f"Top {n} Best Predictions Among Sensitive Samples (Lowest Error)")
    print("=" * 70)
    cols = ['drug_name', 'drug_id', 'cell_line_name', 'cell_line_id',
            'true_ln_ic50', 'pred_ln_ic50', 'abs_error']
    print(sensitive.sort_values('abs_error')[cols].head(n).to_string(index=False))


def print_worst_predictions(sensitive, n=5):
    """Print worst predictions among sensitive samples"""
    print("\n" + "=" * 70)
    print(f"Bottom {n} Worst Predictions Among Sensitive Samples (Highest Error)")
    print("=" * 70)
    cols = ['drug_name', 'drug_id', 'cell_line_name', 'cell_line_id',
            'true_ln_ic50', 'pred_ln_ic50', 'abs_error']
    print(sensitive.sort_values('abs_error', ascending=False)[cols].head(n).to_string(index=False))


def recommend_candidates(sensitive):
    """
    Recommend candidate samples for interpretability analysis.
    Criteria: 1) LN_IC50 < -1 (highly sensitive)  2) Error < 1.0  3) Prefer drugs with known targets
    """
    print("\n" + "=" * 70)
    print("Recommended Candidates for Interpretability Analysis")
    print("=" * 70)

    # Filter highly sensitive and accurately predicted samples
    candidates = sensitive[(sensitive['true_ln_ic50'] < -1) & (sensitive['abs_error'] < 1.0)].copy()
    candidates = candidates.sort_values(['true_ln_ic50', 'abs_error'])

    if len(candidates) == 0:
        print("No samples found satisfying both 'LN_IC50 < -1' and 'Error < 1.0'")
        print("Relaxing criteria: LN_IC50 < -0.5 and Error < 1.5")
        candidates = sensitive[(sensitive['true_ln_ic50'] < -0.5) & (sensitive['abs_error'] < 1.5)].copy()
        candidates = candidates.sort_values(['true_ln_ic50', 'abs_error'])

    if len(candidates) == 0:
        print("Still no suitable candidates found. Please manually select from the following list:")
        print("\nTop 10 Most Sensitive:")
        cols = ['drug_name', 'drug_id', 'cell_line_name', 'true_ln_ic50', 'abs_error']
        print(sensitive.sort_values('true_ln_ic50')[cols].head(10).to_string(index=False))
        return None

    cols = ['drug_name', 'drug_id', 'cell_line_name', 'cell_line_id',
            'true_ln_ic50', 'pred_ln_ic50', 'abs_error']
    print(f"Found {len(candidates)} candidate samples (LN_IC50 < -1, Error < 1.0):")
    print(candidates[cols].head(20).to_string(index=False))

    # Recommend TOP 3
    print("\n" + "-" * 70)
    print("TOP 3 Recommendations (Most Sensitive + Most Accurate):")
    print("-" * 70)
    for i, row in candidates.head(3).iterrows():
        print(f"  {i + 1}. {row['drug_name']} ({row['drug_id']}) + {row['cell_line_name']} ({row['cell_line_id']})")
        print(
            f"     LN_IC50: {row['true_ln_ic50']:.4f} -> Predicted: {row['pred_ln_ic50']:.4f} (Error: {row['abs_error']:.4f})")

    return candidates


def main():
    parser = argparse.ArgumentParser(description='Filter sensitive samples and recommend interpretability candidates')
    parser.add_argument('--csv_path', type=str,
                        default=os.path.join(root_dir("Result"), "My_model", "HO", "GDSC_attention_3l_V1_drug", "results", "test_predictions.csv"),
                        help='CSV path generated by predict_all.py')
    parser.add_argument('--threshold', type=float, default=0,
                        help='Sensitivity threshold (default: 0, LN_IC50 < 0)')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='Save filtered results to CSV')
    args = parser.parse_args()

    # Load data
    df = load_predictions(args.csv_path)

    # Filter sensitive samples
    sensitive = filter_sensitive(df, args.threshold)
    print(f"Number of sensitive samples: {len(sensitive)}")

    if len(sensitive) == 0:
        print(f"No samples found with LN_IC50 < {args.threshold}")
        return

    # Print statistics
    print_statistics(sensitive)
    print_most_sensitive(sensitive, 10)
    print_best_predictions(sensitive, 10)
    print_worst_predictions(sensitive, 5)

    # Recommend candidates
    candidates = recommend_candidates(sensitive)

    # Save results
    if args.output_csv is not None:
        sensitive.to_csv(args.output_csv, index=False)
        print(f"\nSensitive sample results saved to: {args.output_csv}")

    # Print usage suggestions
    print("\n" + "=" * 70)
    print("Usage Suggestions")
    print("=" * 70)
    print("1. If candidates exist, select TOP 1 for interpretability analysis")
    print("2. If no candidates, manually select from the 'Most Sensitive Samples' list")
    print("3. Use extract_attention.py to extract attention weights:")
    print("   python training/extract_attention.py \\")
    print("       --model_path /path/to/model.pth \\")
    print("       --dataset GDSC \\")
    print("       --split_method drug \\")
    print("       --split_set test \\")
    print("       --mode full \\")
    print("       --drug_id <Drug_ID> \\")
    print("       --cell_std_name <Cell_Line_Name>")
    print("=" * 70)


if __name__ == '__main__':
    main()

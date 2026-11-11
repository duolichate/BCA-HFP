# bca_hfp/analysis/fig_gate_weights.py
"""Gated modality weight stacked bar chart."""

import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    parser = argparse.ArgumentParser(description='Plot macro-level gated weight chart (outputs PDF/JPG/PNG simultaneously)')
    parser.add_argument('--csv_path', type=str, required=True,
                        help='Path to gated weights CSV file (e.g., gate_weights_full_drug_GDSC.csv)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')

    # Mode selection
    parser.add_argument('--mode', type=str, default='extremes',
                        choices=['by_drug', 'by_sample', 'extremes'],
                        help='Plotting mode: by_drug (group by drug), by_sample (by specified samples), extremes (extreme samples)')

    # by_drug mode parameters
    parser.add_argument('--min_samples', type=int, default=3,
                        help='by_drug mode: minimum number of samples per drug to include in statistics')
    parser.add_argument('--top_n_drugs', type=int, default=None,
                        help='by_drug mode: only show top N drugs (sorted by drug_global)')

    # by_sample mode parameters
    parser.add_argument('--samples', type=str, nargs='+', default=None,
                        help='by_sample mode: specify samples in format "drug_id:cell_std_name"')
    parser.add_argument('--sample_labels', type=str, nargs='+', default=None,
                        help='by_sample mode: custom sample labels')

    # extremes mode parameters
    parser.add_argument('--sample_prefix', type=str, default='Sample',
                        help='extremes mode: sample label prefix')
    parser.add_argument('--add_drug_name', action='store_true', default=True,
                        help='extremes mode: whether to show drug name in labels')
    parser.add_argument('--add_cell_name', action='store_true', default=True,
                        help='extremes mode: whether to show cell line name in labels')

    # No need for format/dpi parameters, all formats output by default

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Read data
    df = pd.read_csv(args.csv_path)
    print(f"Read data: {len(df)} samples, {df['drug_id'].nunique()} drugs")

    # Mode 1: Group by drug
    if args.mode == 'by_drug':
        drug_means = df.groupby('drug_id')[['gene_pooled', 'global_gene', 'drug_global']].mean()
        drug_counts = df.groupby('drug_id').size()

        valid_drugs = drug_counts[drug_counts >= args.min_samples].index
        drug_means = drug_means.loc[valid_drugs]
        drug_means = drug_means.sort_values('drug_global', ascending=False)

        if args.top_n_drugs is not None:
            drug_means = drug_means.head(args.top_n_drugs)

        plot_data = drug_means[['gene_pooled', 'global_gene', 'drug_global']]
        labels = [f"Drug {d}\n(n={drug_counts[d]})" for d in drug_means.index]
        title = "Macro-level Interpretability: Average Gated Fusion Weights by Drug"
        base_filename = "fig3_gate_weights_by_drug"
        print(f"Displaying {len(plot_data)} drugs")

    # Mode 2: By specified samples
    elif args.mode == 'by_sample':
        if args.samples is None:
            raise ValueError("by_sample mode requires --samples to be specified")

        if 'drug_name' in df.columns:
            drug_name_map = df.drop_duplicates('drug_id').set_index('drug_id')['drug_name'].to_dict()
        else:
            drug_name_map = {}

        sample_list = []
        labels = []

        for i, sample_spec in enumerate(args.samples):
            parts = sample_spec.split(':')
            if len(parts) != 2:
                raise ValueError(f"Invalid sample format: {sample_spec}, should be 'drug_id:cell_std_name'")
            drug_id, cell_std = parts[0], parts[1]

            mask = (df['drug_id'].astype(str) == drug_id) & (df['cell_std_name'] == cell_std)
            matched = df[mask]

            if len(matched) == 0:
                print(f"Warning: Sample not found drug_id={drug_id}, cell_std_name={cell_std}")
                continue

            sample_row = matched.iloc[0]
            weights = sample_row[['gene_pooled', 'global_gene', 'drug_global']].values

            if args.sample_labels and i < len(args.sample_labels):
                label = args.sample_labels[i]
            else:
                if drug_id in drug_name_map:
                    drug_name = drug_name_map[drug_id]
                else:
                    drug_name = sample_row.get('drug_name', None)
                    if pd.isna(drug_name) or drug_name == '':
                        drug_name = f"Drug {drug_id}"

                true_label = sample_row['label']
                status = "Sensitive" if true_label < 0 else "Resistant"
                label = f"{cell_std}-{drug_name}\n({status})"

            sample_list.append(weights)
            labels.append(label)

        plot_data = pd.DataFrame(sample_list, columns=['gene_pooled', 'global_gene', 'drug_global'])
        title = "Macro-level Interpretability: Gated Fusion Weights"
        base_filename = "fig3_gate_weights_by_sample"

    # Mode 3: Extreme samples (lowest/highest)
    else:  # 'extremes'
        selected_samples = []
        labels = []
        sample_descriptions = []

        extremes_config = [
            ('gene_pooled', 'max', 'Gene Pooled (Highest)'),
            ('gene_pooled', 'min', 'Gene Pooled (Lowest)'),
            ('global_gene', 'max', 'Global Gene (Highest)'),
            ('global_gene', 'min', 'Global Gene (Lowest)'),
            ('drug_global', 'max', 'Drug Global (Highest)'),
            ('drug_global', 'min', 'Drug Global (Lowest)'),
        ]

        for col, mode, desc in extremes_config:
            if mode == 'max':
                target_idx = df[col].idxmax()
            else:
                target_idx = df[col].idxmin()

            sample_row = df.loc[target_idx]
            weights = sample_row[['gene_pooled', 'global_gene', 'drug_global']].values

            drug_id = sample_row['drug_id']
            cell_std = sample_row.get('cell_std_name', 'Unknown')
            drug_name = sample_row.get('drug_name', None)
            if pd.isna(drug_name) or drug_name == '':
                drug_name = f"Drug {drug_id}"

            label = f"{cell_std}\n({drug_name})"
            selected_samples.append(weights)
            labels.append(label)
            sample_descriptions.append(desc)

            col_values = {
                'gene_pooled': sample_row['gene_pooled'],
                'global_gene': sample_row['global_gene'],
                'drug_global': sample_row['drug_global']
            }
            print(f"  {desc}: drug_id={drug_id}, cell={cell_std}, "
                  f"gene_pooled={col_values['gene_pooled']:.2%}, "
                  f"global_gene={col_values['global_gene']:.2%}, "
                  f"drug_global={col_values['drug_global']:.2%}")

        if len(selected_samples) == 0:
            raise ValueError("No valid samples found")

        plot_data = pd.DataFrame(selected_samples, columns=['gene_pooled', 'global_gene', 'drug_global'])
        title = "Macro-level Interpretability: Extreme Modality Weights"
        base_filename = "fig3_gate_weights_extremes"

        for i, desc in enumerate(sample_descriptions):
            labels[i] = labels[i] + f"\n[{desc}]"

    # Unified plotting
    colors = ['#C0392B', '#2980B9', '#7F8C8D']
    modality_names = ['Gene Pooled\n(Micro Target)',
                      'Global Gene\n(Macro Transcriptome)',
                      'Drug Global\n(Macro Structure)']

    fig, ax = plt.subplots(figsize=(16, 13))
    sns.set_theme(style="whitegrid")

    x = np.arange(len(plot_data))
    bottom = np.zeros(len(plot_data))

    for i, col in enumerate(['gene_pooled', 'global_gene', 'drug_global']):
        values = plot_data[col].values
        ax.bar(x, values, bottom=bottom, color=colors[i],
               edgecolor='black', linewidth=0.8, width=0.7,
               label=modality_names[i])
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, fontsize=10)
    ax.set_ylabel("Weight Proportion", fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    ax.legend(
        title="Information Modality", title_fontsize='11',
        loc='upper center',
        bbox_to_anchor=(0.5, 0.96),
        ncol=3, frameon=False, fontsize=13,
        bbox_transform=fig.transFigure
    )

    fig.suptitle(
        title,
        fontsize=18, fontweight='bold',
        y=0.98
    )

    for c in ax.containers:
        labels_pct = [f'{h:.0%}' for h in c.datavalues]
        ax.bar_label(c, labels=labels_pct, label_type='center',
                     color='white', fontsize=11, fontweight='bold')

    plt.subplots_adjust(top=0.88, bottom=0.12, left=0.08, right=0.95)

    # Save in three formats simultaneously: PDF (vector) + JPG + PNG (both 600 DPI)
    dpi = 600
    output_files = []
    for fmt in ['pdf', 'jpg', 'png']:
        out_path = os.path.join(args.output_dir, f"{base_filename}.{fmt}")
        if fmt == 'pdf':
            plt.savefig(out_path, format='pdf', bbox_inches='tight')
        else:
            plt.savefig(out_path, dpi=dpi, format=fmt, bbox_inches='tight')
        output_files.append(out_path)
        print(f"Saved: {out_path}")

    plt.close()

    # Print weight details
    print("\nWeight details:")
    for i, row in plot_data.iterrows():
        gp, gg, dg = row['gene_pooled'], row['global_gene'], row['drug_global']
        print(f"  {labels[i].replace(chr(10), ' ')}: "
              f"gene_pooled={gp:.2%}, global_gene={gg:.2%}, drug_global={dg:.2%}")

    print(f"\nAll figures saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

# bca_hfp/analysis/vis_eval_scatter.py
"""Predicted vs observed scatter plots across split methods."""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bca_hfp.config.base_config import get_base_config, PROJECT_ROOT, root_dir
from bca_hfp.data.loader import load_drug_atom_features, load_gene_embeddings, load_response, load_morgan_fingerprints
from bca_hfp.data.dataset import DrugGeneDataset, collate_fn
from bca_hfp.models import create_model
from bca_hfp.utils.reproducibility import set_seed
from bca_hfp.training.train_holdout import split_data_by_drug, split_data_by_cell, split_data_random


def main():
    parser = argparse.ArgumentParser(description='Evaluate models under different split methods and generate plots')
    parser.add_argument('--dataset', type=str, default='GDSC', choices=['GDSC'])
    parser.add_argument('--layer', type=int, default=3, choices=[1, 2, 3])
    parser.add_argument('--reg_norm', type=str, default='layernorm',
                        choices=['batchnorm', 'layernorm'],
                        help='Regression head norm layer (B6): must match the trained checkpoint')
    parser.add_argument('--gene_version', type=str, default='V1', choices=['V1', 'V2'])
    parser.add_argument('--split_method', type=str, required=True, choices=['drug', 'cell', 'random'],
                        help='Data split method: drug (by drug), cell (by cell line), random')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Model weights path (.pth). If not specified, automatically finds best model from training output directory')
    parser.add_argument('--output_dir', type=str, default=root_dir("Picture"),
                        help='Image output directory')
    parser.add_argument('--plot_style', type=str, default='simple', choices=['simple', 'advanced'],
                        help='Plot style: simple (classic scatter) or advanced (with marginal histograms/density contours)')
    args = parser.parse_args()

    set_seed(42)
    config = get_base_config(args.dataset)

    # Set gene feature directory and input dimension
    base_data_path = root_dir("Data")
    if args.gene_version == 'V1':
        config['data']['gene_feat_dir'] = f"{base_data_path}/GDSC_EXP_GF_V1"
        config['data']['gene_input_dim'] = 256
    else:
        config['data']['gene_feat_dir'] = f"{base_data_path}/GDSC_EXP_GF_V2"
        config['data']['gene_input_dim'] = 768

    # Set hidden dimensions
    if args.layer == 1:
        hidden_dims = [64]
        mlp_layer_str = '1l'
    elif args.layer == 2:
        hidden_dims = [256, 128]
        mlp_layer_str = '2l'
    else:
        hidden_dims = [512, 256, 128]
        mlp_layer_str = '3l'

    config['model']['hidden_dims'] = hidden_dims
    config['model']['mlp_layer'] = mlp_layer_str
    config['model']['num_fusion_layers'] = 1  # Attention model uses 1 fusion layer
    config['model']['reg_norm'] = args.reg_norm  # B6: must match the training setting of the loaded checkpoint

    # Morgan fingerprints (drug global features)
    fp_dim = 2048
    drug_global_dict = load_morgan_fingerprints(config['data']['drug_feat_path'], fp_bits=fp_dim, fp_radius=2)
    config['data']['drug_global_dim'] = fp_dim

    # Determine model path
    if args.model_path is not None:
        model_path = args.model_path
        print(f"Using user-specified model: {model_path}")
    else:
        # Auto-construct path
        suffix = f"{args.dataset}_attention_{mlp_layer_str}_{args.gene_version}_{args.split_method}"
        base_output = os.path.join(os.path.dirname(os.path.dirname(config['output']['model_save_dir'])), 'Final_Model')
        model_dir = os.path.join(base_output, suffix, 'models')
        # Try loading best validation model
        model_path = os.path.join(model_dir, f"attention_{mlp_layer_str}_{args.gene_version}_best_val.pth")
        if not os.path.exists(model_path):
            # If not found, try final model
            model_path = os.path.join(model_dir, f"attention_{mlp_layer_str}_{args.gene_version}_final.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        print(f"Auto-located model: {model_path}")

    # Load data
    drug_id_col = config['data']['drug_id_col']
    cell_id_col = config['data']['cell_id_col']
    label_col = config['data']['label_col']

    print("Loading drug atom features...")
    drug_atom_dict = load_drug_atom_features(config['data']['drug_feat_path'])
    print("Loading gene embeddings...")
    gene_emb_dict, _ = load_gene_embeddings(config['data']['gene_feat_dir'], cell_id_col)
    print("Loading response data...")
    response_df = load_response(config['data']['response_path'], cell_id_col, drug_id_col)

    # Filter valid samples
    valid_drug_mask = response_df[drug_id_col].isin(drug_atom_dict)
    valid_cell_mask = response_df[cell_id_col].isin(gene_emb_dict)
    valid_mask = valid_drug_mask & valid_cell_mask
    filtered_df = response_df[valid_mask].copy()
    print(f"Original samples: {len(response_df)}, Valid samples after filtering: {len(filtered_df)}")

    # Split dataset according to split method
    if args.split_method == 'drug':
        train_df, val_df, test_df = split_data_by_drug(
            filtered_df, drug_id_col,
            train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
            random_state=config['data']['random_state']
        )
    elif args.split_method == 'cell':
        train_df, val_df, test_df = split_data_by_cell(
            filtered_df, cell_id_col,
            train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
            random_state=config['data']['random_state']
        )
    else:  # random
        train_df, val_df, test_df = split_data_random(
            filtered_df,
            train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
            random_state=config['data']['random_state']
        )
    print(f"Test set samples: {len(test_df)}")

    # Create test set Dataset and DataLoader
    test_dataset = DrugGeneDataset(
        test_df[drug_id_col].tolist(),
        test_df[cell_id_col].tolist(),
        test_df[label_col].values,
        drug_atom_dict,
        gene_emb_dict,
        drug_global_dict
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        collate_fn=collate_fn
    )

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model('attention', config).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Prediction
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            atom_feat, atom_mask, gene_feat, gene_mask, labels, drug_global = batch
            atom_feat = atom_feat.to(device)
            atom_mask = atom_mask.to(device)
            gene_feat = gene_feat.to(device)
            gene_mask = gene_mask.to(device)
            drug_global = drug_global.to(device)
            outputs = model(gene_feat, atom_feat, atom_mask, gene_mask, drug_global=drug_global)
            all_preds.extend(outputs.cpu().float().numpy())
            all_targets.extend(labels.numpy())

    preds = np.array(all_preds)
    targets = np.array(all_targets)
    drug_ids_test = test_df[drug_id_col].values

    # Overall metrics
    overall_pearson, _ = pearsonr(targets, preds)
    overall_spearman, _ = spearmanr(targets, preds)
    overall_r2 = r2_score(targets, preds)
    overall_rmse = np.sqrt(mean_squared_error(targets, preds))
    print(
        f"Overall performance: Pearson={overall_pearson:.4f}, Spearman={overall_spearman:.4f}, R2={overall_r2:.4f}, RMSE={overall_rmse:.4f}")

    # Per-drug Spearman
    unique_drugs = np.unique(drug_ids_test)
    drug_spearman = {}
    for drug in unique_drugs:
        mask = drug_ids_test == drug
        if np.sum(mask) > 5:
            sp, _ = spearmanr(targets[mask], preds[mask])
            drug_spearman[drug] = sp
    drug_stats = pd.DataFrame(
        {'drug_id': list(drug_spearman.keys()), 'spearman': list(drug_spearman.values())}).dropna().sort_values(
        'spearman', ascending=False)

    if args.model_path is not None:
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        output_base = os.path.join(args.output_dir, f'{args.dataset}_{args.split_method}_{model_name}_scatter')
    else:
        output_base = os.path.join(args.output_dir,
                                   f'{args.dataset}_attention_{mlp_layer_str}_{args.gene_version}_{args.split_method}_scatter')

    if args.plot_style == 'simple':
        plt.figure(figsize=(8, 8))
        plt.scatter(targets, preds, alpha=0.5, s=20, c='steelblue', edgecolors='none')
        min_val = min(targets.min(), preds.min())
        max_val = max(targets.max(), preds.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
        textstr = f'$R^2$ = {overall_r2:.3f}\nPearson r = {overall_pearson:.3f}\nSpearman rho = {overall_spearman:.3f}\nRMSE = {overall_rmse:.3f}'
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.xlabel('Observed LN_IC50', fontsize=12)
        plt.ylabel('Predicted LN_IC50', fontsize=12)
        plt.title(f'Drug response prediction ({args.split_method} split)', fontsize=14)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_base + '.png', dpi=300)
        plt.close()
        print(f"Scatter plot saved: {output_base}.png")
    else:
        try:
            from scipy.stats import gaussian_kde
        except ImportError:
            print("Warning: scipy not installed, cannot plot advanced style, falling back to simple style")
            args.plot_style = 'simple'

            plt.figure(figsize=(8, 8))
            plt.scatter(targets, preds, alpha=0.5, s=20, c='steelblue', edgecolors='none')
            min_val = min(targets.min(), preds.min())
            max_val = max(targets.max(), preds.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
            textstr = f'$R^2$ = {overall_r2:.3f}\nPearson r = {overall_pearson:.3f}\nSpearman rho = {overall_spearman:.3f}\nRMSE = {overall_rmse:.3f}'
            plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            plt.xlabel('Observed LN_IC50', fontsize=12)
            plt.ylabel('Predicted LN_IC50', fontsize=12)
            plt.title(f'Drug response prediction ({args.split_method} split)', fontsize=14)
            plt.legend(loc='lower right')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_base + '.png', dpi=300)
            plt.close()
            print(f"Due to missing scipy, simple style scatter plot saved: {output_base}.png")
        else:
            fig = plt.figure(figsize=(10, 10))
            gs = fig.add_gridspec(4, 4)
            ax_main = fig.add_subplot(gs[0:3, 0:3])
            ax_histx = fig.add_subplot(gs[0, 3], sharey=ax_main)
            ax_histy = fig.add_subplot(gs[3, 0:3], sharex=ax_main)

            # Main plot: density-colored scatter + contours
            xy = np.vstack([targets, preds])
            try:
                z = gaussian_kde(xy)(xy)
                idx = z.argsort()
                ax_main.scatter(targets[idx], preds[idx], c=z[idx], cmap='viridis', s=15, alpha=0.8, edgecolors='none')
                xi = np.linspace(targets.min(), targets.max(), 50)
                yi = np.linspace(preds.min(), preds.max(), 50)
                zi = gaussian_kde(xy)(np.meshgrid(xi, yi)).T
                ax_main.contour(xi, yi, zi, levels=5, colors='gray', linewidths=0.8, alpha=0.5)
            except Exception as e:
                print("Density contour plotting failed, using regular scatter:", e)
                ax_main.scatter(targets, preds, alpha=0.6, s=15, c='steelblue')

            min_val = min(targets.min(), preds.min())
            max_val = max(targets.max(), preds.max())
            ax_main.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
            textstr = f'$R^2$ = {overall_r2:.3f}\nPearson r = {overall_pearson:.3f}\nSpearman rho = {overall_spearman:.3f}\nRMSE = {overall_rmse:.3f}'
            ax_main.text(0.05, 0.95, textstr, transform=ax_main.transAxes, fontsize=12,
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax_main.set_xlabel('Observed LN_IC50', fontsize=12)
            ax_main.set_ylabel('Predicted LN_IC50', fontsize=12)
            ax_main.set_title(f'Drug response prediction ({args.split_method} split)', fontsize=14)
            ax_main.legend(loc='lower right')
            ax_main.grid(True, alpha=0.3)

            # Marginal histograms
            ax_histx.hist(preds, bins=50, orientation='horizontal', color='lightgreen', alpha=0.7)
            ax_histx.axhline(y=preds.mean(), color='green', linestyle='-')
            ax_histx.set_title('Predicted')
            ax_histy.hist(targets, bins=50, orientation='vertical', color='lightcoral', alpha=0.7)
            ax_histy.axvline(x=targets.mean(), color='red', linestyle='-')
            ax_histy.set_title('Observed')

            plt.tight_layout()
            plt.savefig(output_base + '_advanced.png', dpi=300)
            plt.close()
            print(f"Advanced scatter plot saved: {output_base}_advanced.png")

    # Drug-wise Spearman waterfall chart
    if len(drug_stats) > 0:
        plt.figure(figsize=(12, 6))
        x = np.arange(len(drug_stats))
        colors = ['red' if sp > 0.5 else 'blue' for sp in drug_stats['spearman']]
        plt.bar(x, drug_stats['spearman'], color=colors, edgecolor='black', linewidth=0.5)
        plt.axhline(y=0.5, color='gray', linestyle='--', label='rho=0.5')
        plt.axhline(y=0, color='black')
        plt.xticks(x, drug_stats['drug_id'], rotation=90, fontsize=8)
        plt.ylabel('Spearman correlation (rho)')
        plt.xlabel('Drug ID')
        plt.title(f'Drug-wise prediction performance ({args.split_method} split)')
        plt.legend()
        plt.tight_layout()
        if args.model_path is not None:
            drug_path = os.path.join(args.output_dir,
                                     f'{args.dataset}_{args.split_method}_{model_name}_drug_spearman.png')
        else:
            drug_path = os.path.join(args.output_dir,
                                     f'{args.dataset}_attention_{mlp_layer_str}_{args.gene_version}_{args.split_method}_drug_spearman.png')
        plt.savefig(drug_path, dpi=300)
        plt.close()
        print(f"Drug Spearman waterfall chart saved: {drug_path}")

    # Save results
    result_dir = os.path.join(os.path.dirname(model_path), '..', 'results')
    os.makedirs(result_dir, exist_ok=True)
    test_df['predicted'] = preds
    # don't write test_predictions.csv here, that file belongs to predict_all.py
    # (its columns drug_id / true_ln_ic50 / pred_ln_ic50 are consumed downstream)
    # separate filename here so the two runs don't overwrite each other
    test_df.to_csv(os.path.join(result_dir, 'eval_scatter_predictions.csv'), index=False)
    drug_stats.to_csv(os.path.join(result_dir, 'test_drug_spearman.csv'), index=False)
    print(f"Prediction results saved to: {os.path.join(result_dir, 'eval_scatter_predictions.csv')}")


if __name__ == '__main__':
    main()

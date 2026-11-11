# bca_hfp/analysis/fig_performance_curve.py
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from bca_hfp.config.base_config import PROJECT_ROOT, root_dir

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
# Refined lines, ticks, and grid
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.alpha'] = 0.2
plt.rcParams['grid.linestyle'] = '-'

# Read data & filter single fold
df = pd.read_csv(os.path.join(root_dir("Result"), "My_model", "HO", "GDSC_attention_3l_V1_drug", "results", "attention_3l_V1_training_history.csv"))
single_fold_df = df[df['fold'] == 0].reset_index(drop=True)

# Create 2x2 subplot layout
fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=300)
fig.suptitle('Model Training Performance (Single Fold)', fontsize=13, y=0.98)


ax1 = axes[0, 0]
ax1.plot(single_fold_df['epoch'], single_fold_df['train_r2'], color='#0033ff', label='Train R²')
ax1.plot(single_fold_df['epoch'], single_fold_df['val_r2'], color='#ff2222', label='Validation R²')
ax1.set_title('R² on Training and Validation Sets')
ax1.set_ylabel('R²')
ax1.set_xlabel('Epoch')
ax1.legend(frameon=False, loc='lower right')  # Frameless legend for cleaner look
ax1.grid(True)


ax2 = axes[0, 1]
ax2.plot(single_fold_df['epoch'], single_fold_df['val_r2'], color='#009933')
ax2.set_title('Validation R²')
ax2.set_ylabel('Validation R²')
ax2.set_xlabel('Epoch')
ax2.grid(True)

# Compute best point
best_idx = single_fold_df['val_r2'].idxmax()
best_r2 = single_fold_df.loc[best_idx, 'val_r2']
best_epoch = single_fold_df.loc[best_idx, 'epoch']

# Draw red dot
ax2.scatter(best_epoch, best_r2, color='red', s=35, zorder=6, edgecolor='white', linewidth=0.5)

# Annotation placed in upper-left blank area, completely avoiding all curves
# xytext uses subplot relative coordinates (0~1), left+top position where no data exists
ax2.annotate(f'Best: {best_r2:.4f} @ epoch {int(best_epoch)}',
             xy=(best_epoch, best_r2),          # Arrow points to red dot
             xytext=(0.12, 0.90),               # Text fixed at subplot left 12% / top 90% blank area
             textcoords='axes fraction',
             fontsize=9,
             arrowprops=dict(arrowstyle='-|>', color='black', lw=0.7, alpha=0.8))


ax3 = axes[1, 0]
ax3.plot(single_fold_df['epoch'], single_fold_df['val_pearson'], color='#772299')
ax3.set_title('Validation Pearson Correlation')
ax3.set_ylabel('Pearson r')
ax3.set_xlabel('Epoch')
ax3.grid(True)


ax4 = axes[1, 1]
ax4.plot(single_fold_df['epoch'], single_fold_df['val_rmse'], color='#ffaa22')
ax4.set_title('Validation RMSE')
ax4.set_ylabel('RMSE')
ax4.set_xlabel('Epoch')
ax4.grid(True)


plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(root_dir("Picture"), "figure4_training_curves.pdf"), bbox_inches='tight')
plt.savefig(os.path.join(root_dir("Picture"), "figure4_training_curves.png"), dpi=600, bbox_inches='tight')
plt.show()

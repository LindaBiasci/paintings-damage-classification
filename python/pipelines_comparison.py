"""Plot histograms to compare classification performances: 
manual vs deep learning feature extraction."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load .csv files obtained from ml_classifier.py
df_m = pd.read_csv("results/features_manual_metrics.csv", index_col=0)
df_c = pd.read_csv("results/features_cnn_metrics.csv", index_col=0)

# Define metrics, bars positions and width
metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_roc_auc']
labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC']
x = np.arange(len(labels))
w = 0.35

fig, axes = plt.subplots(2, 1, figsize=(10, 10))

# Iterate over models and axes
for ax, model in zip(axes, ['SVM', 'RF']):
    # Extract data
    m_mean = np.array([df_m.at[model, f'{m}_mean'] for m in metrics], dtype=float)
    m_std = np.array([df_m.at[model, f'{m}_std'] for m in metrics], dtype=float)
    c_mean = np.array([df_c.at[model, f'{m}_mean'] for m in metrics], dtype=float)
    c_std = np.array([df_c.at[model, f'{m}_std'] for m in metrics], dtype=float)

    # Plot
    ax.bar(x - w/2, m_mean, w, yerr=m_std, label='Manual', capsize=5, color='cyan', alpha=0.7)
    ax.bar(x + w/2, c_mean, w, yerr=c_std, label='CNN', capsize=5, color='magenta', alpha=0.7)

    # Titles and format
    title_name = 'Random Forest' if model == 'RF' else 'SVM'
    ax.set_title(f'Model Performance Comparison: {title_name}', fontweight='bold')
    ax.set_ylabel('Score (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 115)
    ax.grid(axis='y', alpha=0.3)
    ax.legend()

plt.tight_layout(pad=3)
plt.savefig("results/comparison_plot.png", dpi=300)
plt.show()

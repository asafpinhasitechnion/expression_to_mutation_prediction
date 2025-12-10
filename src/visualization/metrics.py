"""Visualization functions for per-cancer metrics and predictions."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_per_cancer_metrics(
    per_cancer_metrics: dict[str, pd.DataFrame],
    output_dir: Path,
    key_metrics: list[str] | None = None,
    figsize: tuple[float, float] = (18, 12),
) -> None:
    """
    Create comprehensive visualizations for per-cancer-type metrics.
    
    Creates:
    - Box plots comparing metrics across cancer types
    - Heatmap of average metrics
    - Bar plots with error bars
    
    Args:
        per_cancer_metrics: Dictionary mapping cancer type names to metric DataFrames
                          (genes as rows, metrics as columns)
        output_dir: Directory to save visualizations
        key_metrics: List of metrics to visualize (default: ['roc_auc', 'auprc', 'f1', 'accuracy', 'precision', 'recall'])
        figsize: Figure size tuple (default: (18, 12))
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if key_metrics is None:
        key_metrics = ['roc_auc', 'auprc', 'f1', 'accuracy', 'precision', 'recall']
    
    # Create summary DataFrame across all cancer types
    summary_data = []
    for cancer_type, metrics_df in per_cancer_metrics.items():
        for metric in key_metrics:
            if metric in metrics_df.columns:
                mean_val = metrics_df[metric].mean()
                std_val = metrics_df[metric].std()
                summary_data.append({
                    'cancer_type': cancer_type,
                    'metric': metric,
                    'mean': mean_val,
                    'std': std_val,
                    'n_genes': len(metrics_df)
                })
    
    if not summary_data:
        print("   Warning: No metrics data found for visualization")
        return
    
    summary_df = pd.DataFrame(summary_data)
    
    # 1. Box plots comparing metrics across cancer types
    print("   Creating box plots...")
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    for idx, metric in enumerate(key_metrics):
        if metric not in summary_df['metric'].values:
            continue
        
        # Extract per-gene values for box plot
        plot_data = []
        for cancer_type, metrics_df in per_cancer_metrics.items():
            if metric in metrics_df.columns:
                values = metrics_df[metric].dropna()
                for val in values:
                    plot_data.append({'cancer_type': cancer_type, metric: val})
        
        plot_df = pd.DataFrame(plot_data)
        
        if not plot_df.empty:
            sns.boxplot(data=plot_df, x='cancer_type', y=metric, ax=axes[idx])
            axes[idx].set_title(f'{metric.upper()} by Cancer Type')
            axes[idx].set_xlabel('Cancer Type')
            axes[idx].set_ylabel(metric.upper())
            axes[idx].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_by_cancer_type_boxplots.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Heatmap of average metrics across cancer types
    print("   Creating heatmap...")
    heatmap_data = summary_df.pivot(index='cancer_type', columns='metric', values='mean')
    fig, ax = plt.subplots(figsize=(10, max(6, len(heatmap_data) * 0.5)))
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='viridis', ax=ax, cbar_kws={'label': 'Metric Value'})
    ax.set_title('Average Metrics by Cancer Type')
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_heatmap_by_cancer_type.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Bar plot comparing key metrics
    print("   Creating bar plots...")
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    for idx, metric in enumerate(key_metrics):
        if metric not in summary_df['metric'].values:
            continue
        
        metric_data = summary_df[summary_df['metric'] == metric].sort_values('mean', ascending=False)
        
        axes[idx].bar(metric_data['cancer_type'], metric_data['mean'], 
                      yerr=metric_data['std'], capsize=5, alpha=0.7)
        axes[idx].set_title(f'{metric.upper()} by Cancer Type')
        axes[idx].set_xlabel('Cancer Type')
        axes[idx].set_ylabel(metric.upper())
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_by_cancer_type_barplots.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Save summary table
    print("   Creating summary tables...")
    summary_table = summary_df.pivot(index='cancer_type', columns='metric', values=['mean', 'std'])
    summary_table.columns = ['_'.join(col).strip() for col in summary_table.columns.values]
    summary_table.to_csv(output_dir / 'summary_by_cancer_type.csv')
    
    # 5. Per-cancer top genes
    print("   Creating per-cancer top genes plots...")
    for cancer_type, metrics_df in per_cancer_metrics.items():
        if 'roc_auc' in metrics_df.columns:
            top_genes = metrics_df.nlargest(10, 'roc_auc')[['roc_auc', 'auprc', 'f1', 'precision', 'recall']]
            top_genes.to_csv(output_dir / f'top_genes_{cancer_type}.csv')
            
            # Create bar plot for top genes
            fig, ax = plt.subplots(figsize=(12, 6))
            x_pos = range(len(top_genes))
            ax.bar(x_pos, top_genes['roc_auc'], alpha=0.7)
            ax.set_xlabel('Gene')
            ax.set_ylabel('ROC-AUC')
            ax.set_title(f'Top 10 Genes by ROC-AUC - {cancer_type}')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(top_genes.index, rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / f'top_genes_roc_auc_{cancer_type}.png', dpi=150, bbox_inches='tight')
            plt.close()
    
    print(f"   All visualizations saved to: {output_dir}")


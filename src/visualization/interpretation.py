"""Visualization functions for interpretation and ablation analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import seaborn as sns


def plot_ablation_clustermaps(
    metric_matrices: dict[str, pd.DataFrame],
    output_dir: Path,
    metrics_to_plot: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
    tick_fontsize: int = 6,
    cmap: str = "RdBu_r",
    fillna_value: float = 0.0,
) -> dict[str, sns.matrix.ClusterGrid]:
    """
    Create clustermap visualizations for ablation analysis difference matrices.
    
    Clustermaps show hierarchical clustering of genes based on their ablation effects,
    revealing groups of genes with similar patterns of impact when removed.
    
    Args:
        metric_matrices: Dictionary mapping metric names to difference DataFrames
                        (removed_gene × evaluated_gene)
        output_dir: Directory to save clustermap images
        metrics_to_plot: Optional list of specific metrics to plot (default: all)
        figsize: Figure size tuple (default: auto-calculated based on matrix size)
        tick_fontsize: Font size for tick labels
        cmap: Colormap name (default: "RdBu_r" for centered difference matrices)
        fillna_value: Value to fill NaN with for clustering (default: 0.0)
    
    Returns:
        Dictionary mapping metric names to ClusterGrid objects
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if metrics_to_plot is None:
        metrics_to_plot = list(metric_matrices.keys())
    
    clustermaps = {}
    
    for metric_name in metrics_to_plot:
        if metric_name not in metric_matrices:
            continue
        
        matrix = metric_matrices[metric_name].copy()

        # Rank the columns (evaluated genes) per row (removed gene)
        # This normalizes so accuracy per gene doesn't affect visualization
        # axis=0 ranks along rows (for each column, ranks the rows)
        # We want axis=0: for each evaluated gene (column), rank the removed genes (rows)
        matrix = matrix.rank(axis=0, method="average", na_option="keep", ascending=False)
        
        # Auto-calculate figsize based on matrix dimensions
        if figsize is None:
            n_genes = len(matrix)
            # Scale figure size with number of genes, with reasonable bounds
            base_size = max(12, min(20, n_genes * 0.3))
            figsize = (base_size, base_size)
        
        # Create clustermap with difference matrix settings
        # For difference matrices, we want to center at zero
        matrix_filled = matrix.fillna(fillna_value)
        
        # Calculate vmin/vmax for centering at zero
        finite_values = matrix_filled.values[np.isfinite(matrix_filled.values)]
        if len(finite_values) > 0:
            abs_max = np.abs(finite_values).max()
            vmin, vmax = -abs_max, abs_max
        else:
            vmin, vmax = None, None
        
        # Create clustermap
        g = sns.clustermap(
            matrix_filled,
            figsize=figsize,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            dendrogram_ratio=(0.12, 0.12),
            cbar_kws={"label": ""},
            xticklabels=True,
            yticklabels=True,
        )
        
        # Set tick labels fontsize
        g.ax_heatmap.set_xticklabels(
            g.ax_heatmap.get_xticklabels(),
            rotation=90,
            ha='right',
            fontsize=tick_fontsize
        )
        g.ax_heatmap.set_yticklabels(
            g.ax_heatmap.get_yticklabels(),
            rotation=0,
            ha='right',
            fontsize=tick_fontsize
        )
        
        # Set axis labels
        g.ax_heatmap.set_xlabel("Evaluated Gene", fontsize=10)
        g.ax_heatmap.set_ylabel("Removed Gene", fontsize=10)
        
        # Set title
        g.fig.suptitle(
            f"Gene Ablation Analysis - {metric_name.replace('_', ' ').title()}\n"
            f"(Difference: Ablation - Baseline, Clustered)",
            fontsize=12,
            y=1.02
        )
        
        # Add colorbar label (try different possible attribute names)
        try:
            if hasattr(g, 'ax_cbar'):
                g.ax_cbar.set_ylabel(
                    f"{metric_name.replace('_', ' ').title()} Difference\n(Ablation - Baseline)",
                    fontsize=10,
                    rotation=270,
                    labelpad=20
                )
            elif hasattr(g, 'cax'):
                g.cax.set_ylabel(
                    f"{metric_name.replace('_', ' ').title()} Difference\n(Ablation - Baseline)",
                    fontsize=10,
                    rotation=270,
                    labelpad=20
                )
        except AttributeError:
            # If colorbar axes not accessible, skip label (plot will still work)
            pass
        
        # Save figure
        plt.savefig(
            output_dir / f"{metric_name}_ablation_clustermap.png",
            dpi=150,
            bbox_inches='tight'
        )
        plt.close()
        
        clustermaps[metric_name] = g
        print(f"   Clustermap saved for {metric_name}")
    
    return clustermaps

def analyze_ablation_impact(
    metric_matrix: pd.DataFrame,
    metric_name: str = "metric",
) -> dict:
    """
    Analyze which gene removals have the biggest impact.
    
    Args:
        metric_matrix: Ablation matrix (removed_gene × evaluated_gene)
        metric_name: Name of the metric for display
    
    Returns:
        Dictionary with analysis results:
            - 'row_means': Mean metric per removed gene
            - 'col_means': Mean metric per evaluated gene
            - 'top_removed_least_impact': Top genes whose removal has least impact
            - 'top_removed_most_impact': Top genes whose removal has most impact
            - 'top_evaluated_best': Top genes predicted best when others removed
            - 'top_evaluated_worst': Top genes predicted worst when others removed
    """
    # Calculate mean for each row (when that gene was removed)
    row_means = metric_matrix.mean(axis=1, skipna=True).sort_values(ascending=False)
    
    # Calculate mean for each column (how well that gene is predicted when others are removed)
    col_means = metric_matrix.mean(axis=0, skipna=True).sort_values(ascending=False)
    
    return {
        'row_means': row_means,
        'col_means': col_means,
        'top_removed_least_impact': row_means.head(10),
        'top_removed_most_impact': row_means.tail(10),
        'top_evaluated_best': col_means.head(10),
        'top_evaluated_worst': col_means.tail(10),
    }


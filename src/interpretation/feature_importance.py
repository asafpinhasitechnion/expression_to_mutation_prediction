"""Extract and analyze feature importance from tree-based models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


def extract_tree_feature_importance(
    model,
    gene_names: list[str],
    feature_names: list[str] | None = None,
) -> dict[str, pd.Series]:
    """
    Extract native feature importance from tree-based models (per gene).
    
    For tree-based models (LightGBM, XGBoost, RandomForest), each gene has its own
    model that contains a `feature_importances_` attribute.
    
    Args:
        model: Trained model (should be SklearnModel with MultiOutputClassifier)
        gene_names: List of gene names (in same order as model estimators)
        feature_names: Optional list of feature names. If None, uses indices.
    
    Returns:
        Dictionary mapping gene names to Series of feature importances
        {gene_name: pd.Series(feature_importance, index=feature_names)}
    """
    # Check if model has MultiOutputClassifier structure
    if not hasattr(model, 'model'):
        raise ValueError("Model must have 'model' attribute (e.g., SklearnModel)")
    
    sklearn_model = model.model
    if not hasattr(sklearn_model, 'estimators_'):
        raise ValueError("Model must use MultiOutputClassifier (should have 'estimators_' attribute)")
    
    estimators = sklearn_model.estimators_
    
    if len(estimators) != len(gene_names):
        raise ValueError(f"Number of estimators ({len(estimators)}) doesn't match number of genes ({len(gene_names)})")
    
    if feature_names is None:
        # Try to infer from first estimator
        if hasattr(estimators[0], 'feature_names_in_'):
            feature_names = list(estimators[0].feature_names_in_)
        else:
            # Fall back to indices
            n_features = len(estimators[0].feature_importances_)
            feature_names = [f"feature_{i}" for i in range(n_features)]
    
    results = {}
    
    for gene_name, estimator in zip(gene_names, estimators):
        if not hasattr(estimator, 'feature_importances_'):
            raise ValueError(f"Estimator for {gene_name} doesn't have 'feature_importances_' attribute. "
                           f"Model type: {type(estimator)}")
        
        importances = estimator.feature_importances_
        results[gene_name] = pd.Series(importances, index=feature_names, name=gene_name)
    
    return results


def compare_feature_importance_across_genes(
    feature_importance_dict: dict[str, pd.Series],
    top_n: int = 20,
    output_path: Path | None = None,
    title: str = "Feature Importance Comparison Across Genes",
) -> pd.DataFrame:
    """
    Compare feature importance across different genes.
    
    Args:
        feature_importance_dict: Dictionary mapping gene names to feature importance Series
        top_n: Number of top features to display
        output_path: Optional path to save the plot
        title: Plot title
    
    Returns:
        DataFrame with feature importance for all genes (features x genes)
    """
    if not PLOTTING_AVAILABLE:
        raise ImportError("matplotlib and seaborn required for plotting")
    
    # Create DataFrame (features x genes)
    importance_df = pd.DataFrame(feature_importance_dict)
    
    # Get top features across all genes
    max_importance_per_feature = importance_df.max(axis=1)
    top_features = max_importance_per_feature.nlargest(top_n).index
    
    # Filter to top features
    top_df = importance_df.loc[top_features]
    
    # Create heatmap
    plt.figure(figsize=(max(8, len(feature_importance_dict) * 0.8), max(6, top_n * 0.4)))
    sns.heatmap(
        top_df,
        annot=False,
        fmt='.3f',
        cmap='YlOrRd',
        cbar_kws={'label': 'Feature Importance'},
        linewidths=0.5,
    )
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel('Genes', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return importance_df


def plot_feature_importance_per_gene(
    feature_importance_dict: dict[str, pd.Series],
    top_n: int = 20,
    output_dir: Path | None = None,
    figsize: tuple[int, int] = (10, 8),
) -> None:
    """
    Plot feature importance for each gene separately.
    
    Args:
        feature_importance_dict: Dictionary mapping gene names to feature importance Series
        top_n: Number of top features to display per gene
        output_dir: Optional directory to save plots
        figsize: Figure size for each plot
    """
    if not PLOTTING_AVAILABLE:
        raise ImportError("matplotlib and seaborn required for plotting")
    
    for gene_name, importances in feature_importance_dict.items():
        # Get top features
        top_importances = importances.nlargest(top_n).sort_values(ascending=True)
        
        # Create horizontal bar plot
        plt.figure(figsize=figsize)
        plt.barh(range(len(top_importances)), top_importances.values)
        plt.yticks(range(len(top_importances)), top_importances.index)
        plt.xlabel('Feature Importance', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.title(f'Top {top_n} Features - {gene_name}', fontsize=14, pad=20)
        plt.tight_layout()
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_dir / f"{gene_name}_feature_importance.png", dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def analyze_feature_importance(
    model,
    gene_names: list[str],
    feature_names: list[str] | None = None,
    output_dir: Path | None = None,
    top_n: int = 20,
) -> dict[str, pd.Series]:
    """
    Complete feature importance analysis for tree-based models.
    
    Extracts feature importance, creates visualizations, and saves results.
    
    Args:
        model: Trained tree-based model
        gene_names: List of gene names
        feature_names: Optional list of feature names
        output_dir: Optional directory to save results
        top_n: Number of top features to display
    
    Returns:
        Dictionary mapping gene names to feature importance Series
    """
    # Extract feature importance
    print(f"\nExtracting feature importance for {len(gene_names)} genes...")
    importance_dict = extract_tree_feature_importance(
        model=model,
        gene_names=gene_names,
        feature_names=feature_names,
    )
    
    print(f"Feature importance extracted for {len(importance_dict)} genes")
    
    # Save CSV files
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save individual gene importances
        for gene_name, importances in importance_dict.items():
            importances.to_csv(output_dir / f"{gene_name}_feature_importance.csv", header=True)
        
        # Save combined comparison
        importance_df = pd.DataFrame(importance_dict)
        importance_df.to_csv(output_dir / "feature_importance_comparison.csv")
        print(f"Results saved to {output_dir}")
    
    # Create visualizations
    if PLOTTING_AVAILABLE:
        print(f"\nCreating visualizations...")
        
        if output_dir:
            # Per-gene plots
            plot_feature_importance_per_gene(
                feature_importance_dict=importance_dict,
                top_n=top_n,
                output_dir=output_dir / "per_gene",
            )
            
            # Comparison heatmap
            compare_feature_importance_across_genes(
                feature_importance_dict=importance_dict,
                top_n=top_n,
                output_path=output_dir / "feature_importance_comparison_heatmap.png",
            )
            
            print(f"Visualizations saved to {output_dir}")
        else:
            # Just show plots
            plot_feature_importance_per_gene(importance_dict, top_n=top_n)
            compare_feature_importance_across_genes(importance_dict, top_n=top_n)
    
    return importance_dict


"""Visualization functions for SHAP analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def plot_shap_summary(
    shap_values: np.ndarray,
    X: np.ndarray | pd.DataFrame,
    feature_names: list[str] | None = None,
    max_display: int = 20,
    output_path: Path | None = None,
    title: str | None = None,
) -> None:
    """
    Create SHAP summary plot showing feature importance.
    
    Args:
        shap_values: SHAP values array (n_samples, n_features)
        X: Feature values used to compute SHAP values
        feature_names: Optional list of feature names
        max_display: Maximum number of features to display
        output_path: Path to save the plot
        title: Plot title
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP library not available. Install with: pip install shap")
    
    # Convert to numpy
    if isinstance(X, pd.DataFrame):
        X_values = X.values
        if feature_names is None:
            feature_names = X.columns.tolist()
    else:
        X_values = X
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    # Create SHAP Explanation object
    shap_explanation = shap.Explanation(
        values=shap_values,
        data=X_values,
        feature_names=feature_names,
    )
    
    # Create plot
    plt.figure(figsize=(10, max_display * 0.5 + 2))
    shap.plots.beeswarm(shap_explanation, max_display=max_display, show=False)
    
    if title:
        plt.title(title, fontsize=14, pad=20)
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_shap_bar(
    shap_values: np.ndarray,
    feature_names: list[str] | None = None,
    max_display: int = 20,
    output_path: Path | None = None,
    title: str | None = None,
) -> None:
    """
    Create SHAP bar plot showing mean absolute SHAP values (feature importance).
    
    Args:
        shap_values: SHAP values array (n_samples, n_features)
        feature_names: Optional list of feature names
        max_display: Maximum number of features to display
        output_path: Path to save the plot
        title: Plot title
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP library not available. Install with: pip install shap")
    
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(shap_values.shape[1])]
    
    # Create SHAP Explanation object
    shap_explanation = shap.Explanation(
        values=shap_values,
        feature_names=feature_names,
    )
    
    # Create plot
    plt.figure(figsize=(10, max_display * 0.4 + 2))
    shap.plots.bar(shap_explanation, max_display=max_display, show=False)
    
    if title:
        plt.title(title, fontsize=14, pad=20)
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_shap_heatmap(
    shap_values: np.ndarray,
    X: np.ndarray | pd.DataFrame,
    feature_names: list[str] | None = None,
    max_display: int = 20,
    output_path: Path | None = None,
    title: str | None = None,
) -> None:
    """
    Create SHAP heatmap plot.
    
    Args:
        shap_values: SHAP values array (n_samples, n_features)
        X: Feature values
        feature_names: Optional list of feature names
        max_display: Maximum number of features to display
        output_path: Path to save the plot
        title: Plot title
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP library not available. Install with: pip install shap")
    
    # Convert to numpy
    if isinstance(X, pd.DataFrame):
        X_values = X.values
        if feature_names is None:
            feature_names = X.columns.tolist()
    else:
        X_values = X
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    # Create SHAP Explanation object
    shap_explanation = shap.Explanation(
        values=shap_values,
        data=X_values,
        feature_names=feature_names,
    )
    
    # Create plot
    plt.figure(figsize=(12, min(20, shap_values.shape[0] * 0.3)))
    shap.plots.heatmap(shap_explanation, max_display=max_display, show=False)
    
    if title:
        plt.title(title, fontsize=14, pad=20)
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_shap_results(
    shap_results: dict,
    output_dir: Path,
    gene_name: str | None = None,
    max_features: int = 50,
    include_heatmap: bool = False,
) -> None:
    """
    Create SHAP visualizations for a single gene's results.
    
    Focuses on model-level understanding (summary plots, feature importance)
    rather than per-sample explanations.
    
    Args:
        shap_results: Dictionary from compute_shap_values
        output_dir: Directory to save visualizations
        gene_name: Name of gene being explained
        max_features: Maximum number of features to display
        include_heatmap: If True, also create per-sample heatmap (default: False, focus on summaries)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    shap_values = shap_results['shap_values']
    explained_data = shap_results['explained']
    feature_names = shap_results['feature_names']
    explained_gene = shap_results.get('explained_gene', gene_name or 'unknown')
    
    # Ensure shap_values is 2D
    if len(shap_values.shape) > 2:
        # If multiple outputs, take mean or first
        if shap_values.shape[0] == 1:
            shap_values = shap_values[0]
        else:
            shap_values = np.mean(np.abs(shap_values), axis=0)
    
    print(f"   Creating model-level visualizations for {explained_gene}...")
    
    # Create X DataFrame for better feature names
    if isinstance(explained_data, pd.DataFrame):
        X_df = explained_data
    else:
        X_df = pd.DataFrame(explained_data, columns=feature_names)
    
    # Summary plot (shows distribution of SHAP values - general trends)
    plot_shap_summary(
        shap_values=shap_values,
        X=X_df,
        feature_names=feature_names,
        max_display=max_features,
        output_path=output_dir / f"{explained_gene}_shap_summary.png",
        title=f"SHAP Summary Plot - {explained_gene}",
    )
    
    # Bar plot (mean absolute SHAP values - feature importance)
    plot_shap_bar(
        shap_values=shap_values,
        feature_names=feature_names,
        max_display=max_features,
        output_path=output_dir / f"{explained_gene}_shap_bar.png",
        title=f"SHAP Feature Importance - {explained_gene}",
    )
    
    # Heatmap (per-sample visualization - optional, only if requested)
    if include_heatmap and shap_values.shape[0] <= 100:
        plot_shap_heatmap(
            shap_values=shap_values,
            X=X_df,
            feature_names=feature_names,
            max_display=max_features,
            output_path=output_dir / f"{explained_gene}_shap_heatmap.png",
            title=f"SHAP Heatmap - {explained_gene}",
        )
    
    print(f"     Visualizations saved to {output_dir}")


def visualize_all_shap_results(
    all_shap_results: dict[str, dict],
    output_dir: Path,
    max_features: int = 50,
    include_heatmap: bool = False,
) -> None:
    """
    Create visualizations for all genes' SHAP results.
    
    Focuses on model-level understanding (summary plots, feature importance)
    rather than per-sample explanations.
    
    Args:
        all_shap_results: Dictionary mapping gene names to SHAP result dicts
        output_dir: Base output directory
        max_features: Maximum number of features to display
        include_heatmap: If True, also create per-sample heatmaps (default: False)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nCreating model-level SHAP visualizations for {len(all_shap_results)} genes...")
    
    for gene_name, shap_result in all_shap_results.items():
        gene_dir = output_dir / gene_name
        visualize_shap_results(
            shap_results=shap_result,
            output_dir=gene_dir,
            gene_name=gene_name,
            max_features=max_features,
            include_heatmap=include_heatmap,
        )
    
    print(f"\nAll visualizations complete!")


def plot_shap_dependence(
    shap_values: np.ndarray,
    X: np.ndarray | pd.DataFrame,
    feature_index: int | str,
    feature_names: list[str] | None = None,
    interaction_index: int | str | None = None,
    output_path: Path | None = None,
    title: str | None = None,
) -> None:
    """
    Create SHAP dependence plot showing how a feature's impact depends on its value.
    
    Args:
        shap_values: SHAP values array (n_samples, n_features)
        X: Feature values
        feature_index: Index or name of feature to plot
        feature_names: Optional list of feature names (if feature_index is string)
        interaction_index: Optional index/name of feature for color coding (interaction)
        output_path: Path to save the plot
        title: Plot title
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP library not available. Install with: pip install shap")
    
    # Convert to numpy
    if isinstance(X, pd.DataFrame):
        X_values = X.values
        if feature_names is None:
            feature_names = X.columns.tolist()
    else:
        X_values = X
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    # Get feature index
    if isinstance(feature_index, str):
        if feature_index not in feature_names:
            raise ValueError(f"Feature '{feature_index}' not found in feature_names")
        feature_idx = feature_names.index(feature_index)
        feature_name = feature_index
    else:
        feature_idx = feature_index
        feature_name = feature_names[feature_idx]
    
    # Get interaction index if provided
    interaction_idx = None
    interaction_name = None
    if interaction_index is not None:
        if isinstance(interaction_index, str):
            if interaction_index not in feature_names:
                raise ValueError(f"Interaction feature '{interaction_index}' not found in feature_names")
            interaction_idx = feature_names.index(interaction_index)
            interaction_name = interaction_index
        else:
            interaction_idx = interaction_index
            interaction_name = feature_names[interaction_idx]
    
    # Create SHAP Explanation object
    shap_explanation = shap.Explanation(
        values=shap_values,
        data=X_values,
        feature_names=feature_names,
    )
    
    # Create plot
    plt.figure(figsize=(10, 6))
    shap.plots.scatter(
        shap_explanation[:, feature_idx],
        color=shap_explanation[:, interaction_idx] if interaction_idx is not None else None,
        show=False,
    )
    
    plot_title = title or f"SHAP Dependence Plot - {feature_name}"
    if interaction_name:
        plot_title += f" (interaction with {interaction_name})"
    plt.title(plot_title, fontsize=14, pad=20)
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

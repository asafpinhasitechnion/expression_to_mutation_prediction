"""Interpretation and analysis utilities for mutation prediction models."""

from .ablation import run_gene_ablation_analysis
from .utils import compute_difference_matrices, train_baseline_model

try:
    from .shap_analysis import (
        compute_shap_values_for_gene,
        compute_shap_for_all_genes,
        compute_shap_per_cancer_type,
        save_shap_summary,
    )
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from .feature_importance import (
        extract_tree_feature_importance,
        compare_feature_importance_across_genes,
        plot_feature_importance_per_gene,
        analyze_feature_importance,
    )
    FEATURE_IMPORTANCE_AVAILABLE = True
except ImportError:
    FEATURE_IMPORTANCE_AVAILABLE = False

__all__ = [
    "run_gene_ablation_analysis",
    "train_baseline_model",
    "compute_difference_matrices",
]

if SHAP_AVAILABLE:
    __all__.extend([
        "compute_shap_values_for_gene",
        "compute_shap_for_all_genes",
        "compute_shap_per_cancer_type",
        "save_shap_summary",
    ])

if FEATURE_IMPORTANCE_AVAILABLE:
    __all__.extend([
        "extract_tree_feature_importance",
        "compare_feature_importance_across_genes",
        "plot_feature_importance_per_gene",
        "analyze_feature_importance",
    ])


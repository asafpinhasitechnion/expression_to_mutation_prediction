"""Utility functions for interpretation and analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from evaluation.metrics import evaluate_multilabel
from models.model_factory import ModelFactory


def train_baseline_model(
    model_factory: ModelFactory,
    X: np.ndarray | pd.DataFrame,
    Y: pd.DataFrame,
    config: dict,
    test_size: float = 0.2,
    random_state: int | None = 42,
) -> dict:
    """
    Train a baseline model with all genes included.
    
    Args:
        model_factory: ModelFactory instance to create models
        X: Expression data (features)
        Y: Mutation data (targets) - must be DataFrame with gene names as columns
        config: Configuration dictionary
        test_size: Proportion of data for testing
        random_state: Random seed for train/test split
    
    Returns:
        Dictionary containing:
            - 'model': Trained model
            - 'metrics_df': DataFrame with per-gene metrics
            - 'metrics_per_gene': Dictionary mapping gene names to metric dictionaries
            - 'X_test': Test features
            - 'Y_test': Test targets
    """
    if not isinstance(Y, pd.DataFrame):
        raise ValueError("Y must be a pandas DataFrame with gene names as columns")
    
    # Convert to numpy if needed
    X_values = X.values if isinstance(X, pd.DataFrame) else X
    Y_values = Y.values
    
    # Split data
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_values,
        Y_values,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    
    # Create and train model
    model_config = config.get("model", {})
    model_name = model_config.get("name", "multitask_nn")
    
    model = model_factory.get_model(
        model_name=model_name,
        input_size=X_train.shape[1],
        output_size=Y.shape[1],
        config=config,
    )
    
    model.fit(X_train, Y_train)
    
    # Get predictions
    Y_pred, Y_pred_proba = model.predict(X_test)
    
    # Evaluate baseline - get per-gene metrics
    metrics_df = evaluate_multilabel(
        Y_test,
        Y_pred,
        Y_pred_proba,
        mutation_names=Y.columns.tolist()
    )
    
    # Store baseline metrics for later comparison
    metrics_per_gene = metrics_df.to_dict('index')
    
    # Convert test arrays back to DataFrames for consistency
    X_test_df = pd.DataFrame(X_test, index=pd.RangeIndex(len(X_test)))
    Y_test_df = pd.DataFrame(Y_test, columns=Y.columns, index=pd.RangeIndex(len(Y_test)))
    
    return {
        'model': model,
        'metrics_df': metrics_df,
        'metrics_per_gene': metrics_per_gene,
        'X_test': X_test_df,
        'Y_test': Y_test_df,
        'Y_pred': Y_pred,
        'Y_pred_proba': Y_pred_proba,
    }


def compute_difference_matrices(
    ablation_matrices: dict[str, pd.DataFrame],
    baseline_metrics_per_gene: dict,
    output_dir: Path | None = None,
    save_matrices: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Compute difference matrices (ablation - baseline) for each metric.
    
    Args:
        ablation_matrices: Dictionary mapping metric names to ablation matrices
        baseline_metrics_per_gene: Dictionary mapping gene names to metric dictionaries
        output_dir: Optional directory to save difference matrices
        save_matrices: Whether to save matrices to CSV files
    
    Returns:
        Dictionary mapping metric names to difference matrices
    """
    difference_matrices = {}
    
    for metric_name, ablation_matrix in ablation_matrices.items():
        # Create baseline matrix: same shape as ablation matrix, filled with baseline values
        # For each evaluated gene, use its baseline metric value
        baseline_values = []
        for evaluated_gene in ablation_matrix.columns:
            if (evaluated_gene in baseline_metrics_per_gene and 
                metric_name in baseline_metrics_per_gene[evaluated_gene]):
                baseline_values.append(baseline_metrics_per_gene[evaluated_gene][metric_name])
            else:
                baseline_values.append(np.nan)
        
        # Create baseline matrix (same shape as ablation matrix)
        baseline_matrix = pd.DataFrame(
            np.tile(baseline_values, (len(ablation_matrix), 1)),
            index=ablation_matrix.index,
            columns=ablation_matrix.columns
        )
        
        # Compute difference: ablation - baseline
        difference_matrix = ablation_matrix - baseline_matrix
        difference_matrices[metric_name] = difference_matrix
        
        # Save difference matrix if requested
        if save_matrices and output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            difference_matrix.to_csv(output_dir / f"difference_matrix_{metric_name}.csv")
    
    return difference_matrices


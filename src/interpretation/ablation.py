"""Gene ablation analysis for understanding gene dependencies."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from evaluation.metrics import evaluate_multilabel
from models.model_factory import ModelFactory


def run_gene_ablation_analysis(
    model_factory: ModelFactory,
    X: np.ndarray | pd.DataFrame,
    Y: pd.DataFrame,
    config: dict,
    output_dir: str | Path = "results/",
    eval_mode: str = "train_test",
    test_size: float = 0.2,
    k: int = 5,
    random_state: int | None = 42,
):
    """
    Run ablation analysis by removing each output gene and evaluating remaining genes.
    
    For each gene:
    1. Train a model with that gene removed from the output
    2. Evaluate on the remaining genes
    3. Store metrics for each remaining gene
    
    Creates matrices (removed_gene × evaluated_gene) for each metric, showing how
    removing one gene affects prediction performance of other genes.
    
    Args:
        model_factory: ModelFactory instance to create models
        X: Expression data (features)
        Y: Mutation data (targets) - must be DataFrame with gene names as columns
        config: Configuration dictionary
        output_dir: Directory to save results
        eval_mode: Evaluation mode - "train_test" or "kfold"
        test_size: Proportion of data for testing (only used in train_test mode)
        k: Number of folds for cross-validation (only used in kfold mode)
        random_state: Random seed for train/test split or CV
    
    Returns:
        Dictionary mapping metric names to DataFrames (removed_gene × evaluated_gene)
    """
    if not isinstance(Y, pd.DataFrame):
        raise ValueError("Y must be a pandas DataFrame with gene names as columns for ablation analysis")
    
    if Y.columns.empty or not all(isinstance(col, str) for col in Y.columns):
        raise ValueError("Y must have named columns (gene names) for ablation analysis")
    
    mutation_names = Y.columns.tolist()
    n_genes = len(mutation_names)
    
    eval_mode = eval_mode.lower()
    if eval_mode not in {"train_test", "kfold"}:
        raise ValueError(f"Unsupported eval_mode '{eval_mode}'. Use 'train_test' or 'kfold'.")
    
    print(f"\n🔬 Starting gene ablation analysis for {n_genes} genes...")
    print(f"   Mode: {eval_mode}")
    if eval_mode == "kfold":
        print(f"   Folds: {k}")
    print(f"   This will train {n_genes} models (one per removed gene)")
    
    X_values = X.values if isinstance(X, pd.DataFrame) else X
    Y_values = Y.values
    sample_ids = Y.index.to_numpy() if isinstance(Y, pd.DataFrame) else np.arange(Y_values.shape[0])
    
    # Prepare output directory
    output_dir = Path(output_dir)
    ablation_dir = output_dir / "gene_ablation_analysis"
    ablation_dir.mkdir(parents=True, exist_ok=True)
    
    # Store all metrics: {metric_name: {removed_gene: {evaluated_gene: score}}}
    all_metrics = {}
    
    # Iterate over each gene to remove
    for idx, removed_gene in enumerate(mutation_names, 1):
        print(f"\n  [{idx}/{n_genes}] Removing gene: {removed_gene}")
        
        # Create indices for remaining genes
        remaining_indices = [i for i, gene in enumerate(mutation_names) if gene != removed_gene]
        remaining_genes = [mutation_names[i] for i in remaining_indices]
        
        # Create model with reduced output size
        model_config = config.get("model", {})
        model_name = model_config.get("name", "multitask_nn")
        
        if model_name != "multitask_nn":
            print(f"    ⚠️  Warning: Ablation analysis is designed for multitask_nn. Using {model_name} may not work correctly.")
        
        if eval_mode == "kfold":
            # K-fold cross-validation mode
            kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
            fold_metrics = []
            
            for fold, (train_idx, test_idx) in enumerate(kf.split(X_values), start=1):
                print(f"    Fold {fold}/{k}...")
                
                X_train, X_test = X_values[train_idx], X_values[test_idx]
                Y_train_fold = Y_values[train_idx, :]
                Y_test_fold = Y_values[test_idx, :]
                test_ids_fold = sample_ids[test_idx]
                
                # Prepare data with removed gene excluded
                Y_train_ablation = Y_train_fold[:, remaining_indices]
                Y_test_ablation = Y_test_fold[:, remaining_indices]
                
                # Create and train model
                model = model_factory.get_model(
                    model_name=model_name,
                    input_size=X_train.shape[1],
                    output_size=len(remaining_genes),
                    config=config,
                )
                
                model.fit(X_train, Y_train_ablation)
                Y_pred, Y_prob = model.predict(X_test)
                
                # Evaluate on remaining genes
                metrics_df = evaluate_multilabel(
                    Y_test_ablation,
                    Y_pred,
                    Y_prob,
                    remaining_genes
                )
                fold_metrics.append(metrics_df)
                
                # Save fold-specific results
                gene_dir = ablation_dir / f"removed_{removed_gene}" / f"fold_{fold}"
                gene_dir.mkdir(parents=True, exist_ok=True)
                metrics_df.to_csv(gene_dir / "metrics.csv")
                
                preds_df = pd.DataFrame(Y_pred, columns=remaining_genes, index=test_ids_fold)
                probs_df = pd.DataFrame(Y_prob, columns=remaining_genes, index=test_ids_fold)
                preds_df.to_csv(gene_dir / "predictions.csv")
                probs_df.to_csv(gene_dir / "probabilities.csv")
                
                model.save(gene_dir)
            
            # Aggregate metrics across folds (mean)
            stacked_metrics = pd.concat(fold_metrics)
            metrics_df = stacked_metrics.groupby(level=0).mean()
            
            # Save aggregated metrics
            gene_dir = ablation_dir / f"removed_{removed_gene}"
            gene_dir.mkdir(exist_ok=True)
            metrics_df.to_csv(gene_dir / "metrics.csv")
            
        else:
            # Train-test split mode
            X_train, X_test, Y_train, Y_test, train_ids, test_ids = train_test_split(
                X_values,
                Y_values,
                sample_ids,
                test_size=test_size,
                random_state=random_state,
                shuffle=True,
            )
            
            # Prepare data with removed gene excluded
            Y_train_ablation = Y_train[:, remaining_indices]
            Y_test_ablation = Y_test[:, remaining_indices]
            
            # Create and train model
            model = model_factory.get_model(
                model_name=model_name,
                input_size=X_train.shape[1],
                output_size=len(remaining_genes),
                config=config,
            )
            
            print(f"    Training model without {removed_gene}...")
            model.fit(X_train, Y_train_ablation)
            
            # Predict on test set
            Y_pred, Y_prob = model.predict(X_test)
            
            # Evaluate on remaining genes
            metrics_df = evaluate_multilabel(
                Y_test_ablation,
                Y_pred,
                Y_prob,
                remaining_genes
            )
            
            # Save individual results
            gene_dir = ablation_dir / f"removed_{removed_gene}"
            gene_dir.mkdir(exist_ok=True)
            metrics_df.to_csv(gene_dir / "metrics.csv")
            
            # Save predictions
            preds_df = pd.DataFrame(Y_pred, columns=remaining_genes, index=test_ids)
            probs_df = pd.DataFrame(Y_prob, columns=remaining_genes, index=test_ids)
            preds_df.to_csv(gene_dir / "predictions.csv")
            probs_df.to_csv(gene_dir / "probabilities.csv")
            
            model.save(gene_dir)
        
        # Store aggregated metrics for this removed gene
        for metric_name in metrics_df.columns:
            if metric_name not in all_metrics:
                all_metrics[metric_name] = {}
            
            all_metrics[metric_name][removed_gene] = metrics_df[metric_name].to_dict()
    
    # Build matrices for each metric
    print(f"\n📊 Building ablation matrices...")
    metric_matrices = {}
    
    for metric_name in all_metrics:
        # Create matrix: rows = removed genes, columns = evaluated genes
        matrix_data = []
        removed_genes_list = []
        
        for removed_gene in mutation_names:
            removed_genes_list.append(removed_gene)
            row = []
            for evaluated_gene in mutation_names:
                if removed_gene == evaluated_gene:
                    # Diagonal: gene removed, so no evaluation for itself
                    row.append(np.nan)
                else:
                    # Get metric value for this gene when removed_gene was removed
                    metric_value = all_metrics[metric_name][removed_gene].get(evaluated_gene, np.nan)
                    row.append(metric_value)
            matrix_data.append(row)
        
        matrix_df = pd.DataFrame(
            matrix_data,
            index=removed_genes_list,
            columns=mutation_names
        )
        matrix_df.index.name = "removed_gene"
        matrix_df.columns.name = "evaluated_gene"
        
        metric_matrices[metric_name] = matrix_df
        
        # Save matrix
        matrix_df.to_csv(ablation_dir / f"ablation_matrix_{metric_name}.csv")
        print(f"    Saved: ablation_matrix_{metric_name}.csv")
    
    # Save summary
    summary = {
        "n_genes": n_genes,
        "removed_genes": mutation_names,
        "metrics_computed": list(metric_matrices.keys()),
        "eval_mode": eval_mode,
        "test_size": test_size if eval_mode == "train_test" else None,
        "k_folds": k if eval_mode == "kfold" else None,
        "random_state": random_state,
    }
    with open(ablation_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Gene ablation analysis complete!")
    print(f"   Results saved to: {ablation_dir}")
    print(f"   Generated {len(metric_matrices)} metric matrices")
    
    return metric_matrices


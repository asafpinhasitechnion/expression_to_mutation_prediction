"""Gene ablation analysis for understanding gene dependencies."""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from evaluation.metrics import evaluate_multilabel
from models.model_factory import ModelFactory

# Import visualization functions (optional - only if available)
try:
    from visualization.interpretation import plot_ablation_clustermaps
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    VISUALIZATION_AVAILABLE = False
    # Uncomment the line below to debug import issues:
    # print(f"Warning: Visualization not available: {e}")


def _train_and_evaluate_model(
    model_factory: ModelFactory, model_name: str, X_train: np.ndarray,
    Y_train: np.ndarray, X_test: np.ndarray, Y_test: np.ndarray,
    gene_names: list[str], config: dict,
) -> tuple[pd.DataFrame, object, np.ndarray, np.ndarray]:
    """Train model and evaluate. Returns (metrics_df, model, Y_pred, Y_prob)."""
    model = model_factory.get_model(model_name, X_train.shape[1], len(gene_names), config)
    model.fit(X_train, Y_train)
    Y_pred, Y_prob = model.predict(X_test)
    return evaluate_multilabel(Y_test, Y_pred, Y_prob, gene_names), model, Y_pred, Y_prob


def _extract_cancer_types(
    X: pd.DataFrame, 
    sample_to_cancer: dict[str, str] | None = None
) -> tuple[pd.Series, list[str]]:
    """
    Extract cancer type information from expression DataFrame.
    
    Args:
        X: Expression DataFrame
        sample_to_cancer: Optional dictionary mapping sample IDs to cancer types.
                         If None, tries to extract from cancer_* columns.
    
    Returns:
        Tuple of (cancer_type_series, cancer_type_names)
    """
    # If mapping provided, use it
    if sample_to_cancer is not None:
        cancer_type_series = pd.Series('unknown', index=X.index)
        for sample_id in X.index:
            if sample_id in sample_to_cancer:
                cancer_type_series[sample_id] = sample_to_cancer[sample_id]
        cancer_type_names = sorted([ct for ct in cancer_type_series.unique() if ct != 'unknown'])
        return cancer_type_series, cancer_type_names
    
    # Fallback: try to extract from cancer_* columns
    cancer_cols = [col for col in X.columns if col.startswith('cancer_')]
    if not cancer_cols:
        return pd.Series(index=X.index, dtype=object), []
    
    # Simple approach: for each sample, find which cancer column is 1
    cancer_type_series = pd.Series('unknown', index=X.index)
    for cancer_col in cancer_cols:
        samples_with_cancer = X.index[X[cancer_col] == 1]
        cancer_type_name = cancer_col.replace('cancer_', '')
        cancer_type_series[samples_with_cancer] = cancer_type_name
    
    cancer_type_names = sorted([ct for ct in cancer_type_series.unique() if ct != 'unknown'])
    return cancer_type_series, cancer_type_names


def _compute_cancer_type_matrices_from_combined(
    combined_dir: Path, Y: pd.DataFrame, mutation_names: list[str],
    cancer_type_series: pd.Series, cancer_type_names: list[str], output_dir: Path,
) -> None:
    """Compute difference matrices per cancer type from combined predictions and create visualizations."""
    def _load(f): return pd.read_csv(f, index_col=0).drop(columns=['fold'], errors='ignore')
    baseline_preds, baseline_probs = _load(combined_dir / "baseline_predictions.csv"), _load(combined_dir / "baseline_probabilities.csv")
    (output_dir / "cancer_type").mkdir(exist_ok=True)
    
    for cancer_type in cancer_type_names:
        samples_x = cancer_type_series[cancer_type_series == cancer_type].index
        
        # Try different sample ID matching strategies
        # Strategy 1: Try exact match first (predictions might already have correct format)
        samples_exact = samples_x.intersection(baseline_preds.index)
        # Strategy 2: Try adding 'A' suffix (if predictions use this format)
        samples_with_A = pd.Index([s + 'A' for s in samples_x if isinstance(s, str)])
        samples_A_match = samples_with_A.intersection(baseline_preds.index)
        
        # Use the strategy that gives more matches
        if len(samples_A_match) > len(samples_exact):
            samples = samples_A_match
        else:
            samples = samples_exact
        
        if len(samples) == 0:
            print(f"       Warning: No matching samples found for {cancer_type} in predictions. Skipping...")
            continue
        
        # Now align with mutation data separately (more flexible)
        aligned_samples = samples.intersection(Y.index)
        # If that fails, try removing 'A' suffix if present
        if len(aligned_samples) == 0:
            aligned_samples = pd.Index([s[:-1] if s.endswith('A') else s for s in samples]).intersection(Y.index)
        # Or try adding 'A' if not present
        if len(aligned_samples) == 0:
            aligned_samples = pd.Index([s + 'A' if not s.endswith('A') else s for s in samples]).intersection(Y.index)
        
        if len(aligned_samples) == 0:
            print(f"       Warning: No aligned samples between predictions and mutation data for {cancer_type}. Skipping...")
            continue
        
        samples = aligned_samples
        
        Y_ct, preds_ct, probs_ct = Y.loc[samples], baseline_preds.loc[samples], baseline_probs.loc[samples]
        baseline_metrics = evaluate_multilabel(Y_ct.values, preds_ct.values, probs_ct.values, mutation_names)
        cancer_differences = {}
        
        for removed_gene in mutation_names:
            gene_dir = combined_dir / removed_gene
            if not gene_dir.exists():
                continue
            ablation_preds = _load(gene_dir / "predictions.csv")
            ablation_probs = _load(gene_dir / "probabilities.csv")
            remaining_genes = [g for g in ablation_preds.columns if g != 'fold']
            samples_aligned = samples.intersection(ablation_preds.index)
            if len(samples_aligned) == 0:
                continue
            
            ablation_preds_ct = ablation_preds.loc[samples_aligned]
            ablation_probs_ct = ablation_probs.loc[samples_aligned]
            Y_remaining = Y_ct.loc[samples_aligned, remaining_genes]
            baseline_metrics_aligned = evaluate_multilabel(
                Y_ct.loc[samples_aligned].values, preds_ct.loc[samples_aligned].values,
                probs_ct.loc[samples_aligned].values, mutation_names)
            ablation_metrics = evaluate_multilabel(
                Y_remaining.values, ablation_preds_ct.values, ablation_probs_ct.values, remaining_genes)
            
            for metric_name in baseline_metrics_aligned.columns:
                cancer_differences.setdefault(metric_name, {}).setdefault(removed_gene, {})
                for gene in remaining_genes:
                    if gene in baseline_metrics_aligned.index:
                        b_val = baseline_metrics_aligned.loc[gene, metric_name]
                        a_val = ablation_metrics.loc[gene, metric_name] if gene in ablation_metrics.index else np.nan
                        if not (pd.isna(b_val) or pd.isna(a_val)):
                            cancer_differences[metric_name][removed_gene][gene] = a_val - b_val
        
        ct_dir = output_dir / "cancer_type" / cancer_type
        ct_dir.mkdir(exist_ok=True)
        
        # Build difference matrices for this cancer type
        cancer_type_matrices = {}
        for metric_name, diff_dict in cancer_differences.items():
            matrix = pd.DataFrame(
                [[np.nan if r == e else diff_dict[r].get(e, np.nan) for e in mutation_names] for r in mutation_names],
                index=mutation_names, columns=mutation_names)
            matrix.index.name, matrix.columns.name = "removed_gene", "evaluated_gene"
            matrix.to_csv(ct_dir / f"{cancer_type}_difference_matrix_{metric_name}.csv")
            cancer_type_matrices[metric_name] = matrix
        
        # Create visualizations for this cancer type
        if VISUALIZATION_AVAILABLE and len(cancer_type_matrices) > 0:
            try:
                plot_ablation_clustermaps(
                    metric_matrices=cancer_type_matrices,
                    output_dir=ct_dir,
                    metrics_to_plot=None,
                )
            except Exception as e:
                                print(f"       Warning: Visualization failed for {cancer_type}: {e}")
        
        print(f"     {cancer_type} ({len(cancer_differences)} metrics, {len(samples)} samples)")


def compute_cancer_type_matrices_from_combined(
    combined_predictions_dir: str | Path,
    X: pd.DataFrame,
    Y: pd.DataFrame,
    output_dir: str | Path,
    sample_to_cancer: dict[str, str] | None = None,
) -> None:
    """
    Compute per-cancer-type difference matrices from existing combined predictions.
    
    This function processes already-combined k-fold predictions and computes
    difference matrices separately for each cancer type.
    
    Args:
        combined_predictions_dir: Directory containing combined predictions
                                  (should have baseline_predictions.csv, baseline_probabilities.csv,
                                  and {gene}/predictions.csv, {gene}/probabilities.csv)
        X: Expression DataFrame
        Y: Mutation data DataFrame with gene names as columns
        output_dir: Output directory to save cancer type matrices
        sample_to_cancer: Optional dictionary mapping sample IDs to cancer types.
                         Keys should match sample IDs in Y.index.
    """
    combined_dir = Path(combined_predictions_dir)
    output_dir = Path(output_dir)
    
    if not combined_dir.exists():
        raise FileNotFoundError(f"Combined predictions directory not found: {combined_dir}")
    
    # Extract cancer types
    cancer_type_series, cancer_type_names = _extract_cancer_types(X, sample_to_cancer)
    if not cancer_type_names:
        print("Warning: No cancer types detected in expression data. Make sure X has cancer_* columns.")
        return
    
    print(f"Computing difference matrices by cancer type from combined predictions...")
    print(f"   Found {len(cancer_type_names)} cancer types: {', '.join(cancer_type_names)}")
    
    mutation_names = Y.columns.tolist()
    
    _compute_cancer_type_matrices_from_combined(
        combined_dir=combined_dir,
        Y=Y,
        mutation_names=mutation_names,
        cancer_type_series=cancer_type_series,
        cancer_type_names=cancer_type_names,
        output_dir=output_dir,
    )


def load_ablation_results(ablation_dir: str | Path) -> dict[str, pd.DataFrame]:
    """
    Load ablation results from saved CSV files.
    
    Args:
        ablation_dir: Directory containing results (difference_matrix_*.csv or ablation_matrix_*.csv)
    
    Returns:
        Dictionary mapping metric names to DataFrames (removed_gene × evaluated_gene)
    """
    ablation_dir = Path(ablation_dir)
    if not ablation_dir.exists():
        raise FileNotFoundError(f"Directory not found: {ablation_dir}")
    
    # Try difference matrices first, then ablation matrices (for backward compatibility)
    matrix_files = list(ablation_dir.glob("difference_matrix_*.csv"))
    if not matrix_files:
        matrix_files = list(ablation_dir.glob("ablation_matrix_*.csv"))
    
    if not matrix_files:
        raise FileNotFoundError(f"No matrix files found in {ablation_dir}")
    
    metric_matrices = {}
    for matrix_file in matrix_files:
        metric_name = matrix_file.stem.replace("difference_matrix_", "").replace("ablation_matrix_", "")
        df = pd.read_csv(matrix_file, index_col=0)
        df.index.name, df.columns.name = "removed_gene", "evaluated_gene"
        metric_matrices[metric_name] = df
    
    print(f"Loaded {len(metric_matrices)} matrices from: {ablation_dir}")
    return metric_matrices


def run_gene_ablation_analysis(
    model_factory: ModelFactory,
    X: np.ndarray | pd.DataFrame,
    Y: pd.DataFrame,
    config: dict,
    eval_mode: str = "train_test",
    test_size: float = 0.2,
    k: int = 5,
    random_state: int | None = 42,
    output_dir: str | Path | None = None,
    save_results: bool = True,
    sample_to_cancer: dict[str, str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Run ablation analysis with baseline comparison using matching train/test splits.
    
    For each split, trains both baseline (all genes) and ablation (each gene removed)
    models on the same training data and evaluates on the same test data. This ensures
    fair comparison by using identical splits.
    
    Args:
        model_factory: ModelFactory instance to create models
        X: Expression data (features)
        Y: Mutation data (targets) - must be DataFrame with gene names as columns
        config: Configuration dictionary
        eval_mode: Evaluation mode - "train_test" or "kfold"
        test_size: Proportion of data for testing (only used in train_test mode)
        k: Number of folds for cross-validation (only used in kfold mode)
        random_state: Random seed for train/test split or CV
        output_dir: Optional directory to save results. For kfold, creates subdirectories
                   for each fold (fold_1, fold_2, etc.)
        save_results: Whether to save results to CSV files. Saves:
                     - In ablation/ subfolder (per fold for kfold):
                       - Baseline: baseline_metrics.csv, baseline_predictions.csv, baseline_probabilities.csv
                       - Per gene (in {gene}/ subfolder): metrics.csv, predictions.csv, probabilities.csv
                     - In combined_predictions/ subfolder (for kfold only):
                       - Baseline: baseline_predictions.csv, baseline_probabilities.csv (with 'fold' column)
                       - Per gene (in {gene}/ subfolder): predictions.csv, probabilities.csv (with 'fold' column)
                     - Difference matrices: difference_matrix_{metric}.csv (in main output_dir)
    
    Returns:
        Dictionary mapping metric names to difference DataFrames (removed_gene × evaluated_gene).
        Values are: ablation_metric - baseline_metric
        - Positive values: Removing gene improved performance
        - Negative values: Removing gene hurt performance
        - Near zero: Removing gene had little effect
    """
    if not isinstance(Y, pd.DataFrame):
        raise ValueError("Y must be a pandas DataFrame with gene names as columns")
    
    if Y.columns.empty or not all(isinstance(col, str) for col in Y.columns):
        raise ValueError("Y must have named columns (gene names)")
    
    mutation_names = Y.columns.tolist()
    n_genes = len(mutation_names)
    eval_mode = eval_mode.lower()
    
    if eval_mode not in {"train_test", "kfold"}:
        raise ValueError(f"Unsupported eval_mode '{eval_mode}'. Use 'train_test' or 'kfold'.")
    
    print(f"\nRunning ablation with baseline comparison for {n_genes} genes...")
    print(f"   Mode: {eval_mode}")
    if eval_mode == "kfold":
        print(f"   Folds: {k}")
    
    X_values = X.values if isinstance(X, pd.DataFrame) else X
    Y_values = Y.values
    sample_ids = Y.index.to_numpy() if isinstance(Y, pd.DataFrame) else np.arange(Y_values.shape[0])
    
    model_config = config.get("model", {})
    model_name = model_config.get("name", "multitask_nn")
    
    # Convert output_dir to Path once
    if output_dir is not None:
        output_dir = Path(output_dir)  / "ablation"
    
    # Store differences: {metric_name: {removed_gene: {evaluated_gene: diff}}}
    all_differences = {}
    
    # For kfold: collect predictions from all folds to combine later
    if eval_mode == "kfold":
        kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
        splits = list(kf.split(X_values))
        # Store predictions from each fold: {fold_idx: {baseline/ablation: {predictions/probabilities: df}}}
        fold_predictions = {}
    else:
        # For train_test, create single split - get indices from the split
        indices = np.arange(len(X_values))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size,
            random_state=random_state, shuffle=True)
        splits = [(train_idx, test_idx)]
        fold_predictions = None
    
    for split_idx, (train_idx, test_idx) in enumerate(splits, 1):
        if eval_mode == "kfold":
            print(f"\n  Fold {split_idx}/{k}...")
        else:
            print(f"\n  Training on split...")
        
        X_train, X_test = X_values[train_idx], X_values[test_idx]
        Y_train_full, Y_test_full = Y_values[train_idx], Y_values[test_idx]
        test_sample_ids = sample_ids[test_idx]
        
        # Create output directory for this split
        split_output_dir = None
        predictions_dir = None
        if save_results and output_dir is not None:
            if eval_mode == "kfold":
                split_output_dir = output_dir / f"fold_{split_idx}"
            else:
                split_output_dir = output_dir
            split_output_dir.mkdir(parents=True, exist_ok=True)
            # Create predictions subdirectory
            predictions_dir = split_output_dir
            predictions_dir.mkdir(exist_ok=True)
        
        # Train baseline model (all genes) on this split
        baseline_metrics, _, baseline_pred, baseline_prob = _train_and_evaluate_model(
            model_factory, model_name, X_train, Y_train_full,
            X_test, Y_test_full, mutation_names, config)
        
        # Save baseline results
        if predictions_dir is not None:
            baseline_metrics.to_csv(predictions_dir / "baseline_metrics.csv")
            preds_df = pd.DataFrame(baseline_pred, columns=mutation_names, index=test_sample_ids)
            probs_df = pd.DataFrame(baseline_prob, columns=mutation_names, index=test_sample_ids)
            preds_df.to_csv(predictions_dir / "baseline_predictions.csv")
            probs_df.to_csv(predictions_dir / "baseline_probabilities.csv")
        
        # Store baseline predictions for kfold combination
        if eval_mode == "kfold":
            fold_predictions[split_idx] = {
                'baseline': {
                    'predictions': pd.DataFrame(baseline_pred, columns=mutation_names, index=test_sample_ids),
                    'probabilities': pd.DataFrame(baseline_prob, columns=mutation_names, index=test_sample_ids),
                },
                'ablation': {}
            }
        
        # For each gene removal
        for idx, removed_gene in enumerate(mutation_names, 1):
            if eval_mode == "train_test":
                print(f"  [{idx}/{n_genes}] Removing gene: {removed_gene}")
            
            remaining_indices = [i for i, g in enumerate(mutation_names) if g != removed_gene]
            remaining_genes = [mutation_names[i] for i in remaining_indices]
            
            # Train ablation model (without this gene) on same split
            ablation_metrics, _, ablation_pred, ablation_prob = _train_and_evaluate_model(
                model_factory, model_name, X_train, Y_train_full[:, remaining_indices],
                X_test, Y_test_full[:, remaining_indices], remaining_genes, config)
            
            # Save ablation results for this gene removal in its own folder
            if predictions_dir is not None:
                gene_dir = predictions_dir / removed_gene
                gene_dir.mkdir(exist_ok=True)
                ablation_metrics.to_csv(gene_dir / "metrics.csv")
                preds_df = pd.DataFrame(ablation_pred, columns=remaining_genes, index=test_sample_ids)
                probs_df = pd.DataFrame(ablation_prob, columns=remaining_genes, index=test_sample_ids)
                preds_df.to_csv(gene_dir / "predictions.csv")
                probs_df.to_csv(gene_dir / "probabilities.csv")
            
            # Store ablation predictions for kfold combination
            if eval_mode == "kfold":
                fold_predictions[split_idx]['ablation'][removed_gene] = {
                    'predictions': pd.DataFrame(ablation_pred, columns=remaining_genes, index=test_sample_ids),
                    'probabilities': pd.DataFrame(ablation_prob, columns=remaining_genes, index=test_sample_ids),
                }
            
            # Compute differences for each metric
            for metric_name in baseline_metrics.columns:
                if metric_name not in all_differences:
                    all_differences[metric_name] = {}
                if removed_gene not in all_differences[metric_name]:
                    all_differences[metric_name][removed_gene] = {}
                
                # For each evaluated gene (remaining genes)
                for eval_idx, evaluated_gene in enumerate(remaining_genes):
                    baseline_val = baseline_metrics.loc[evaluated_gene, metric_name]
                    ablation_val = ablation_metrics.iloc[eval_idx][metric_name]
                    diff = ablation_val - baseline_val if not (pd.isna(baseline_val) or pd.isna(ablation_val)) else np.nan
                    all_differences[metric_name][removed_gene][evaluated_gene] = diff
        
        # Clean up
        del baseline_metrics, baseline_pred, baseline_prob
        gc.collect()
    
    # Combine kfold predictions if applicable
    if eval_mode == "kfold" and fold_predictions and save_results and output_dir is not None:
        print(f"\nCombining predictions from {k} folds...")
        combined_dir = output_dir / "combined_predictions"
        combined_dir.mkdir(exist_ok=True)
        
        # Combine baseline predictions
        baseline_preds_list = []
        baseline_probs_list = []
        for fold_idx in sorted(fold_predictions.keys()):
            fold_data = fold_predictions[fold_idx]['baseline']
            preds_df = fold_data['predictions'].copy()
            probs_df = fold_data['probabilities'].copy()
            preds_df['fold'] = fold_idx
            probs_df['fold'] = fold_idx
            baseline_preds_list.append(preds_df)
            baseline_probs_list.append(probs_df)
        
        combined_baseline_preds = pd.concat(baseline_preds_list, axis=0)
        combined_baseline_probs = pd.concat(baseline_probs_list, axis=0)
        combined_baseline_preds.to_csv(combined_dir / "baseline_predictions.csv")
        combined_baseline_probs.to_csv(combined_dir / "baseline_probabilities.csv")
        print(f"   Baseline predictions combined ({len(combined_baseline_preds)} samples)")
        
        # Combine ablation predictions for each gene
        for removed_gene in mutation_names:
            ablation_preds_list = []
            ablation_probs_list = []
            for fold_idx in sorted(fold_predictions.keys()):
                if removed_gene in fold_predictions[fold_idx]['ablation']:
                    fold_data = fold_predictions[fold_idx]['ablation'][removed_gene]
                    preds_df = fold_data['predictions'].copy()
                    probs_df = fold_data['probabilities'].copy()
                    preds_df['fold'] = fold_idx
                    probs_df['fold'] = fold_idx
                    ablation_preds_list.append(preds_df)
                    ablation_probs_list.append(probs_df)
            
            if ablation_preds_list:
                combined_ablation_preds = pd.concat(ablation_preds_list, axis=0)
                combined_ablation_probs = pd.concat(ablation_probs_list, axis=0)
                gene_dir = combined_dir / removed_gene
                gene_dir.mkdir(exist_ok=True)
                combined_ablation_preds.to_csv(gene_dir / "predictions.csv")
                combined_ablation_probs.to_csv(gene_dir / "probabilities.csv")
        
        print(f"   Ablation predictions combined for {len(mutation_names)} genes")
        print(f"   Combined predictions saved to: {combined_dir}")
        
        # Compute per-cancer-type difference matrices from combined predictions
        if isinstance(X, pd.DataFrame) or sample_to_cancer is not None:
            # Use X if it's a DataFrame, otherwise create a dummy DataFrame for index matching
            if isinstance(X, pd.DataFrame):
                X_for_extraction = X
            else:
                # Create a DataFrame with same index as Y for cancer type extraction
                X_for_extraction = pd.DataFrame(index=Y.index)
            
            cancer_type_series, cancer_type_names = _extract_cancer_types(X_for_extraction, sample_to_cancer)
            if cancer_type_names:
                print(f"\nComputing difference matrices by cancer type from combined predictions...")
                _compute_cancer_type_matrices_from_combined(
                    combined_dir=combined_dir,
                    Y=Y,
                    mutation_names=mutation_names,
                    cancer_type_series=cancer_type_series,
                    cancer_type_names=cancer_type_names,
                    output_dir=output_dir,
                )
    
    # Build difference matrices
    print(f"\nBuilding difference matrices...")
    difference_matrices = {}
    for metric_name in all_differences:
        matrix_data = [[np.nan if r == e else all_differences[metric_name][r].get(e, np.nan)
                       for e in mutation_names] for r in mutation_names]
        matrix_df = pd.DataFrame(matrix_data, index=mutation_names, columns=mutation_names)
        matrix_df.index.name, matrix_df.columns.name = "removed_gene", "evaluated_gene"
        difference_matrices[metric_name] = matrix_df
        print(f"   {metric_name}")
    
    # Save difference matrices if requested
    if save_results and output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for metric_name, diff_matrix in difference_matrices.items():
            diff_matrix.to_csv(output_dir / f"difference_matrix_{metric_name}.csv")
        print(f"\nDifference matrices saved to: {output_dir}")
    
    # Compute per-cancer-type difference matrices for train_test mode
    if eval_mode == "train_test" and (isinstance(X, pd.DataFrame) or sample_to_cancer is not None):
        # Use X if it's a DataFrame, otherwise create a dummy DataFrame for index matching
        if isinstance(X, pd.DataFrame):
            X_for_extraction = X
        else:
            # Create a DataFrame with same index as Y for cancer type extraction
            X_for_extraction = pd.DataFrame(index=Y.index)
        
        cancer_type_series, cancer_type_names = _extract_cancer_types(X_for_extraction, sample_to_cancer)
        if cancer_type_names:
            print(f"\nComputing difference matrices by cancer type...")
            # For train_test, we need to compute per-cancer matrices from the single split
            # Load predictions from the output directory
            if save_results and output_dir is not None:
                baseline_preds_path = output_dir / "baseline_predictions.csv"
                baseline_probs_path = output_dir / "baseline_probabilities.csv"
                
                if baseline_preds_path.exists() and baseline_probs_path.exists():
                    baseline_preds = pd.read_csv(baseline_preds_path, index_col=0)
                    baseline_probs = pd.read_csv(baseline_probs_path, index_col=0)
                    
                    (output_dir / "cancer_type").mkdir(exist_ok=True)
                    
                    for cancer_type in cancer_type_names:
                        samples_x = cancer_type_series[cancer_type_series == cancer_type].index
                        samples = samples_x.intersection(baseline_preds.index).intersection(Y.index)
                        if len(samples) == 0:
                            continue
                        
                        Y_ct = Y.loc[samples]
                        preds_ct = baseline_preds.loc[samples]
                        probs_ct = baseline_probs.loc[samples]
                        baseline_metrics_ct = evaluate_multilabel(
                            Y_ct.values, preds_ct.values, probs_ct.values, mutation_names)
                        cancer_differences = {}
                        
                        for removed_gene in mutation_names:
                            gene_dir = output_dir / removed_gene
                            if not gene_dir.exists():
                                continue
                            
                            ablation_preds = pd.read_csv(gene_dir / "predictions.csv", index_col=0)
                            ablation_probs = pd.read_csv(gene_dir / "probabilities.csv", index_col=0)
                            remaining_genes = [g for g in ablation_preds.columns]
                            samples_aligned = samples.intersection(ablation_preds.index)
                            
                            if len(samples_aligned) == 0:
                                continue
                            
                            ablation_preds_ct = ablation_preds.loc[samples_aligned]
                            ablation_probs_ct = ablation_probs.loc[samples_aligned]
                            Y_remaining = Y_ct.loc[samples_aligned, remaining_genes]
                            baseline_metrics_aligned = evaluate_multilabel(
                                Y_ct.loc[samples_aligned].values, preds_ct.loc[samples_aligned].values,
                                probs_ct.loc[samples_aligned].values, mutation_names)
                            ablation_metrics = evaluate_multilabel(
                                Y_remaining.values, ablation_preds_ct.values, ablation_probs_ct.values, remaining_genes)
                            
                            for metric_name in baseline_metrics_aligned.columns:
                                cancer_differences.setdefault(metric_name, {}).setdefault(removed_gene, {})
                                for gene in remaining_genes:
                                    if gene in baseline_metrics_aligned.index:
                                        b_val = baseline_metrics_aligned.loc[gene, metric_name]
                                        a_val = ablation_metrics.loc[gene, metric_name] if gene in ablation_metrics.index else np.nan
                                        if not (pd.isna(b_val) or pd.isna(a_val)):
                                            cancer_differences[metric_name][removed_gene][gene] = a_val - b_val
                        
                        ct_dir = output_dir / "cancer_type" / cancer_type
                        ct_dir.mkdir(exist_ok=True)
                        
                        # Build difference matrices for this cancer type
                        cancer_type_matrices = {}
                        for metric_name, diff_dict in cancer_differences.items():
                            matrix = pd.DataFrame(
                                [[np.nan if r == e else diff_dict[r].get(e, np.nan) for e in mutation_names] for r in mutation_names],
                                index=mutation_names, columns=mutation_names)
                            matrix.index.name, matrix.columns.name = "removed_gene", "evaluated_gene"
                            matrix.to_csv(ct_dir / f"{cancer_type}_difference_matrix_{metric_name}.csv")
                            cancer_type_matrices[metric_name] = matrix
                        
                        # Create visualizations for this cancer type
                        if VISUALIZATION_AVAILABLE and len(cancer_type_matrices) > 0:
                            try:
                                plot_ablation_clustermaps(
                                    metric_matrices=cancer_type_matrices,
                                    output_dir=ct_dir,
                                    metrics_to_plot=None,
                                )
                            except Exception as e:
                                print(f"       Warning: Visualization failed for {cancer_type}: {e}")
                        
                        print(f"     {cancer_type} ({len(cancer_differences)} metrics, {len(samples)} samples)")
    
    # Create visualizations if available and output directory is provided
    if VISUALIZATION_AVAILABLE and output_dir is not None:
        print(f"\nCreating visualizations...")
        
        try:
            # Create clustermaps - save directly in output_dir (same folder as CSV files)
            print("   Creating clustermaps...")
            plot_ablation_clustermaps(
                metric_matrices=difference_matrices,
                output_dir=output_dir,
                metrics_to_plot=None,  # Plot all metrics
            )
            
            print(f"   Visualizations saved to: {output_dir}")
        except Exception as e:
            print(f"   Warning: Visualization failed: {e}")
            print(f"   Continuing without visualizations...")
    
    print(f"\nComparison complete!")
    print(f"   Generated {len(difference_matrices)} difference matrices")
    print(f"   Interpretation:")
    print(f"     - Positive values: Removing gene improved performance")
    print(f"     - Negative values: Removing gene hurt performance")
    print(f"     - Near zero: Removing gene had little effect")
    
    return difference_matrices
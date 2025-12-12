"""Metrics computation for multi-label classification."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_multilabel(y_true, y_pred, y_prob, mutation_names):
    """
    Evaluate multi-label classification performance with comprehensive metrics.
    
    Computes per-gene metrics including:
    - f1: F1 score
    - roc_auc: Area Under ROC Curve
    - auprc: Area Under Precision-Recall Curve (AUPRC)
    - accuracy: Classification accuracy
    - precision: Precision score
    - recall: Recall score (sensitivity)
    - specificity: Specificity (true negative rate)
    - mcc: Matthews Correlation Coefficient
    - prevalence: Proportion of samples with mutation (y_true == 1)
    
    Args:
        y_true: True labels, shape (n_samples, n_genes)
        y_pred: Predicted labels, shape (n_samples, n_genes)
        y_prob: Predicted probabilities, shape (n_samples, n_genes)
        mutation_names: List of gene/mutation names
    
    Returns:
        DataFrame with genes as rows and metrics as columns
    """
    results = {}
    for i, mutation in enumerate(mutation_names):
        y_true_gene = y_true[:, i]
        y_pred_gene = y_pred[:, i]
        y_prob_gene = y_prob[:, i]
        
        # Compute confusion matrix components
        tp = np.sum((y_true_gene == 1) & (y_pred_gene == 1))
        tn = np.sum((y_true_gene == 0) & (y_pred_gene == 0))
        fp = np.sum((y_true_gene == 0) & (y_pred_gene == 1))
        fn = np.sum((y_true_gene == 1) & (y_pred_gene == 0))
        
        gene_metrics = {}
        
        try:
            gene_metrics['f1'] = f1_score(y_true_gene, y_pred_gene)
        except (ValueError, ZeroDivisionError):
            gene_metrics['f1'] = None
        
        try:
            gene_metrics['roc_auc'] = roc_auc_score(y_true_gene, y_prob_gene)
        except ValueError:
            gene_metrics['roc_auc'] = None
        
        try:
            gene_metrics['auprc'] = average_precision_score(y_true_gene, y_prob_gene)
        except ValueError:
            gene_metrics['auprc'] = None
        
        try:
            gene_metrics['accuracy'] = accuracy_score(y_true_gene, y_pred_gene)
        except (ValueError, ZeroDivisionError):
            gene_metrics['accuracy'] = None
        
        try:
            gene_metrics['precision'] = precision_score(y_true_gene, y_pred_gene, zero_division=0)
        except ValueError:
            gene_metrics['precision'] = None
        
        try:
            gene_metrics['recall'] = recall_score(y_true_gene, y_pred_gene, zero_division=0)
        except ValueError:
            gene_metrics['recall'] = None
        
        # Specificity = TN / (TN + FP)
        try:
            if (tn + fp) > 0:
                gene_metrics['specificity'] = tn / (tn + fp)
            else:
                gene_metrics['specificity'] = None
        except (ValueError, ZeroDivisionError):
            gene_metrics['specificity'] = None
        
        # MCC (Matthews Correlation Coefficient)
        try:
            gene_metrics['mcc'] = matthews_corrcoef(y_true_gene, y_pred_gene)
        except ValueError:
            gene_metrics['mcc'] = None
        
        # Prevalence: proportion of samples with mutation
        n_samples = len(y_true_gene)
        if n_samples > 0:
            gene_metrics['prevalence'] = np.sum(y_true_gene == 1) / n_samples
        else:
            gene_metrics['prevalence'] = None
        
        results[mutation] = gene_metrics
    
    return pd.DataFrame(results).T


def extract_cancer_types(
    X: pd.DataFrame | None = None,
    sample_to_cancer: dict[str, str] | None = None,
    sample_ids: list[str] | pd.Index | None = None,
) -> tuple[pd.Series, list[str]]:
    """
    Extract cancer type information from expression DataFrame or mapping dictionary.
    
    Args:
        X: Expression DataFrame with cancer_* columns or index matching samples
        sample_to_cancer: Optional dictionary mapping sample IDs to cancer types
        sample_ids: Optional list of sample IDs to create mapping for
    
    Returns:
        Tuple of (cancer_type_series, cancer_type_names)
    """
    # If mapping provided, use it
    if sample_to_cancer is not None:
        if sample_ids is not None:
            index = pd.Index(sample_ids)
        elif X is not None:
            index = X.index
        else:
            raise ValueError("Either X or sample_ids must be provided when using sample_to_cancer")
        
        cancer_type_series = pd.Series('unknown', index=index)
        for sample_id in index:
            if sample_id in sample_to_cancer:
                cancer_type_series[sample_id] = sample_to_cancer[sample_id]
        cancer_type_names = sorted([ct for ct in cancer_type_series.unique() if ct != 'unknown'])
        return cancer_type_series, cancer_type_names
    
    # Fallback: try to extract from cancer_* columns in X
    if X is not None:
        cancer_cols = [col for col in X.columns if col.startswith('cancer_')]
        if cancer_cols:
            cancer_type_series = pd.Series('unknown', index=X.index)
            for cancer_col in cancer_cols:
                samples_with_cancer = X.index[X[cancer_col] == 1]
                cancer_type_name = cancer_col.replace('cancer_', '')
                cancer_type_series[samples_with_cancer] = cancer_type_name
            cancer_type_names = sorted([ct for ct in cancer_type_series.unique() if ct != 'unknown'])
            return cancer_type_series, cancer_type_names
    
    return pd.Series(dtype=object), []


def _load_and_align_predictions(
    predictions_path: str | Path,
    probabilities_path: str | Path,
    y_true: pd.DataFrame,
    drop_fold_column: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Helper function to load predictions and probabilities, align with y_true,
    and optionally drop 'fold' column.
    
    Args:
        predictions_path: Path to predictions CSV file
        probabilities_path: Path to probabilities CSV file
        y_true: True labels DataFrame (samples x genes)
        drop_fold_column: Whether to drop 'fold' column if present
    
    Returns:
        Tuple of (aligned_y_pred, aligned_y_prob, gene_names)
    """
    predictions_path = Path(predictions_path)
    probabilities_path = Path(probabilities_path)
    
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")
    if not probabilities_path.exists():
        raise FileNotFoundError(f"Probabilities file not found: {probabilities_path}")
    
    # Load predictions and probabilities
    y_pred = pd.read_csv(predictions_path, index_col=0)
    y_prob = pd.read_csv(probabilities_path, index_col=0)
    
    # Drop 'fold' column if present and requested
    if drop_fold_column:
        if 'fold' in y_pred.columns:
            y_pred = y_pred.drop(columns=['fold'])
        if 'fold' in y_prob.columns:
            y_prob = y_prob.drop(columns=['fold'])
    
    # Align indices
    common_index = y_true.index.intersection(y_pred.index).intersection(y_prob.index)
    if len(common_index) == 0:
        raise ValueError("No common sample IDs found between y_true, y_pred, and y_prob")
    
    y_true_aligned = y_true.loc[common_index]
    y_pred = y_pred.loc[common_index]
    y_prob = y_prob.loc[common_index]
    
    # Ensure columns match
    common_genes = y_true_aligned.columns.intersection(y_pred.columns).intersection(y_prob.columns)
    if len(common_genes) == 0:
        raise ValueError("No common gene columns found between y_true, y_pred, and y_prob")
    
    y_pred = y_pred[common_genes]
    y_prob = y_prob[common_genes]
    gene_names = common_genes.tolist()
    
    return y_pred, y_prob, gene_names


def compute_per_cancer_metrics(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    y_prob: pd.DataFrame,
    X: pd.DataFrame | None = None,
    sample_to_cancer: dict[str, str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Compute per-cancer-type metrics from predictions and probabilities.
    
    Args:
        y_true: True labels DataFrame (samples x genes)
        y_pred: Predicted labels DataFrame (samples x genes)
        y_prob: Predicted probabilities DataFrame (samples x genes)
        X: Optional expression DataFrame with cancer_* columns or matching index
        sample_to_cancer: Optional dictionary mapping sample IDs to cancer types
    
    Returns:
        Dictionary mapping cancer type names to metric DataFrames
        {cancer_type: DataFrame with genes as rows and metrics as columns}
    """
    # Extract cancer types
    cancer_type_series, cancer_type_names = extract_cancer_types(
        X=X,
        sample_to_cancer=sample_to_cancer,
        sample_ids=y_true.index.tolist(),
    )
    
    if not cancer_type_names:
        raise ValueError(
            "No cancer types found. Provide either X DataFrame with cancer_* columns "
            "or sample_to_cancer dictionary."
        )
    
    # Ensure all DataFrames have aligned indices
    common_index = y_true.index.intersection(y_pred.index).intersection(y_prob.index)
    if len(common_index) == 0:
        raise ValueError("No common sample IDs found between y_true, y_pred, and y_prob")
    
    y_true = y_true.loc[common_index]
    y_pred = y_pred.loc[common_index]
    y_prob = y_prob.loc[common_index]
    
    # Align cancer type series with common index
    cancer_type_series = cancer_type_series.loc[common_index]
    
    gene_names = y_true.columns.tolist()
    results = {}
    
    print(f"\nComputing per-cancer-type metrics...")
    print(f"   Cancer types: {cancer_type_names}")
    print(f"   Total samples: {len(common_index)}")
    print(f"   Genes: {len(gene_names)}")
    
    # Compute overall metrics (including overall prevalence)
    print(f"\n   Computing overall metrics...")
    overall_metrics = evaluate_multilabel(
        y_true=y_true.values,
        y_pred=y_pred.values,
        y_prob=y_prob.values,
        mutation_names=gene_names,
    )
    results['overall'] = overall_metrics
    print(f"   Overall: {len(common_index)} samples")
    
    for cancer_type in cancer_type_names:
        # Get samples for this cancer type
        cancer_samples = cancer_type_series[cancer_type_series == cancer_type].index
        
        if len(cancer_samples) == 0:
            print(f"   Warning: No samples found for {cancer_type}. Skipping...")
            continue
        
        # Filter data to this cancer type
        y_true_ct = y_true.loc[cancer_samples]
        y_pred_ct = y_pred.loc[cancer_samples]
        y_prob_ct = y_prob.loc[cancer_samples]
        
        # Ensure all samples exist in all dataframes
        valid_samples = y_true_ct.index.intersection(y_pred_ct.index).intersection(y_prob_ct.index)
        if len(valid_samples) == 0:
            print(f"   Warning: No valid samples after alignment for {cancer_type}. Skipping...")
            continue
        
        y_true_ct = y_true_ct.loc[valid_samples]
        y_pred_ct = y_pred_ct.loc[valid_samples]
        y_prob_ct = y_prob_ct.loc[valid_samples]
        
        # Compute metrics for this cancer type
        metrics_df = evaluate_multilabel(
            y_true=y_true_ct.values,
            y_pred=y_pred_ct.values,
            y_prob=y_prob_ct.values,
            mutation_names=gene_names,
        )
        
        results[cancer_type] = metrics_df
        print(f"   {cancer_type}: {len(valid_samples)} samples")
    
    return results


def _save_metrics_results(
    results: dict[str, pd.DataFrame] | pd.DataFrame,
    output_dir: Path,
    per_cancer: bool = True,
    create_summary_files: bool = False,
) -> None:
    """
    Helper function to save metrics results.
    
    Args:
        results: Either a dict mapping cancer types to DataFrames, or a single DataFrame
        output_dir: Directory to save results
        per_cancer: If True, results is a dict and should be saved per cancer type
        create_summary_files: If True and per_cancer, create summary files per metric
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if per_cancer:
        # Results is a dict: {cancer_type: DataFrame}
        if len(results) == 0:
            print("Warning: No results to save")
            return
        
        # Create summary files if requested
        if create_summary_files:
            first_cancer_type = list(results.keys())[0]
            metrics_list = results[first_cancer_type].columns.tolist()
            cancer_types = sorted(results.keys())
            
            print(f"\nSummarizing metrics per cancer type...")
            print(f"   Metrics: {metrics_list}")
            print(f"   Cancer types: {cancer_types}")
            
            for metric in metrics_list:
                summary_data = {}
                for cancer_type in cancer_types:
                    metrics_df = results[cancer_type]
                    if metric in metrics_df.columns:
                        summary_data[cancer_type] = metrics_df[metric]
                
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    csv_path = output_dir / f"{metric}_per_cancer.csv"
                    summary_df.to_csv(csv_path)
                    print(f"   Saved {metric} summary: {csv_path}")
        
        # Save individual per-cancer metrics
        print(f"\nSaving per-cancer metrics to: {output_dir}")
        for cancer_type, metrics_df in results.items():
            cancer_dir = output_dir / "cancer_type" / cancer_type
            cancer_dir.mkdir(parents=True, exist_ok=True)
            csv_path = cancer_dir / "metrics.csv"
            metrics_df.to_csv(csv_path)
            print(f"   Saved {cancer_type} metrics: {csv_path}")
    else:
        # Results is a single DataFrame
        csv_path = output_dir / "metrics.csv"
        results.to_csv(csv_path)
        print(f"\nSaved metrics to: {csv_path}")


def compute_per_cancer_metrics_from_files(
    predictions_path: str | Path,
    probabilities_path: str | Path,
    y_true: pd.DataFrame,
    X: pd.DataFrame | None = None,
    sample_to_cancer: dict[str, str] | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Compute per-cancer-type metrics from saved prediction and probability files.
    
    This function loads predictions and probabilities from CSV files and computes
    per-cancer-type metrics.
    
    Args:
        predictions_path: Path to predictions CSV file (samples x genes)
        probabilities_path: Path to probabilities CSV file (samples x genes)
        y_true: True labels DataFrame (samples x genes)
        X: Optional expression DataFrame with cancer_* columns or matching index
        sample_to_cancer: Optional dictionary mapping sample IDs to cancer types
        output_dir: Optional directory to save per-cancer metrics CSV files
    
    Returns:
        Dictionary mapping cancer type names to metric DataFrames
    """
    print(f"Loading predictions from: {predictions_path}")
    print(f"Loading probabilities from: {probabilities_path}")
    
    # Load and align predictions
    y_pred, y_prob, _ = _load_and_align_predictions(
        predictions_path=predictions_path,
        probabilities_path=probabilities_path,
        y_true=y_true,
        drop_fold_column=False,
    )
    
    # Align y_true with the loaded predictions
    common_index = y_pred.index
    y_true = y_true.loc[common_index]
    
    # Compute per-cancer metrics
    results = compute_per_cancer_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        X=X,
        sample_to_cancer=sample_to_cancer,
    )
    
    # Save results if output directory provided
    if output_dir is not None:
        _save_metrics_results(
            results=results,
            output_dir=Path(output_dir),
            per_cancer=True,
            create_summary_files=False,
        )
    
    return results


def compute_per_cancer_metrics_kfold(
    kfold_dir: str | Path,
    y_true: pd.DataFrame,
    X: pd.DataFrame | None = None,
    sample_to_cancer: dict[str, str] | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Compute per-cancer-type metrics from predictions and probabilities in a directory,
    and summarize gene predictions per cancer type in separate files for each metric.
    
    This function processes predictions and probabilities from a single directory,
    computes per-cancer metrics, and creates summary files where each metric
    has genes as rows and cancer types (including 'overall') as columns.
    
    Args:
        kfold_dir: Directory containing predictions.csv and probabilities.csv
        y_true: True labels DataFrame (samples x genes)
        X: Optional expression DataFrame with cancer_* columns or matching index
        sample_to_cancer: Optional dictionary mapping sample IDs to cancer types
        output_dir: Optional directory to save per-cancer metrics and summary files
    
    Returns:
        Dictionary mapping cancer type names to metric DataFrames
    """
    kfold_dir = Path(kfold_dir)
    
    if not kfold_dir.exists():
        raise FileNotFoundError(f"Directory not found: {kfold_dir}")
    
    predictions_path = kfold_dir / "predictions.csv"
    probabilities_path = kfold_dir / "probabilities.csv"
    
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")
    if not probabilities_path.exists():
        raise FileNotFoundError(f"Probabilities file not found: {probabilities_path}")
    
    print(f"Processing predictions from: {kfold_dir}")
    
    # Compute per-cancer metrics
    results = compute_per_cancer_metrics_from_files(
        predictions_path=predictions_path,
        probabilities_path=probabilities_path,
        y_true=y_true,
        X=X,
        sample_to_cancer=sample_to_cancer,
        output_dir=None,  # We'll handle saving ourselves
    )
    
    # Save results with summary files
    if output_dir is not None:
        _save_metrics_results(
            results=results,
            output_dir=Path(output_dir),
            per_cancer=True,
            create_summary_files=True,
        )
    
    return results


def compute_metrics_from_combined(
    combined_predictions_dir: str | Path,
    y_true: pd.DataFrame,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """
    Compute overall metrics from combined fold predictions.
    
    This function processes combined predictions from k-fold cross-validation
    (where predictions from all folds have been combined) and computes overall
    metrics across all samples, regardless of cancer type.
    
    Args:
        combined_predictions_dir: Directory containing combined predictions.csv and 
                                 probabilities.csv (may have 'fold' column)
        y_true: True labels DataFrame (samples x genes)
        output_dir: Optional directory to save metrics CSV file
    
    Returns:
        DataFrame with genes as rows and metrics as columns
    """
    combined_predictions_dir = Path(combined_predictions_dir)
    
    if not combined_predictions_dir.exists():
        raise FileNotFoundError(f"Combined predictions directory not found: {combined_predictions_dir}")
    
    predictions_path = combined_predictions_dir / "predictions.csv"
    probabilities_path = combined_predictions_dir / "probabilities.csv"
    
    print(f"Processing combined predictions from: {combined_predictions_dir}")
    
    # Load and align predictions (automatically drops 'fold' column if present)
    y_pred, y_prob, gene_names = _load_and_align_predictions(
        predictions_path=predictions_path,
        probabilities_path=probabilities_path,
        y_true=y_true,
        drop_fold_column=True,  # Drop fold column for combined predictions
    )
    
    # Align y_true with the loaded predictions
    common_index = y_pred.index
    y_true = y_true.loc[common_index][gene_names]
    
    print(f"\nComputing overall metrics...")
    print(f"   Samples: {len(common_index)}")
    print(f"   Genes: {len(gene_names)}")
    
    # Compute overall metrics
    metrics_df = evaluate_multilabel(
        y_true=y_true.values,
        y_pred=y_pred.values,
        y_prob=y_prob.values,
        mutation_names=gene_names,
    )
    
    # Save results if output directory provided
    if output_dir is not None:
        _save_metrics_results(
            results=metrics_df,
            output_dir=Path(output_dir),
            per_cancer=False,
            create_summary_files=False,
        )
    
    return metrics_df


def combine_dataframe_folds(
    dataframes: list[pd.DataFrame],
    fold_numbers: list[int | str] | None = None,
) -> pd.DataFrame:
    """
    Combine DataFrames from multiple folds into a single DataFrame.
    
    Helper function for combining predictions/probabilities that are already in memory.
    Useful for ablation analysis and other cases where DataFrames are constructed in-memory.
    
    Args:
        dataframes: List of DataFrames to combine (one per fold)
        fold_numbers: Optional list of fold numbers/identifiers (default: 1, 2, 3, ...)
    
    Returns:
        Combined DataFrame with all samples from all folds
    """
    if len(dataframes) == 0:
        raise ValueError("No DataFrames provided to combine")
    
    if fold_numbers is None:
        fold_numbers = list(range(1, len(dataframes) + 1))
    
    if len(fold_numbers) != len(dataframes):
        raise ValueError(f"Mismatch: {len(dataframes)} DataFrames but {len(fold_numbers)} fold numbers")
    
    combined_list = []
    for df, fold_num in zip(dataframes, fold_numbers):
        df_copy = df.copy()
        df_copy['fold'] = fold_num
        combined_list.append(df_copy)
    
    combined = pd.concat(combined_list, axis=0)
    return combined


def combine_fold_predictions(
    kfold_dir: str | Path,
    output_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Combine predictions and probabilities from k-fold cross-validation results.
    
    This utility function reads predictions from all fold directories and combines
    them into a single DataFrame for each type (predictions and probabilities).
    
    Args:
        kfold_dir: Directory containing fold_*/ subdirectories with predictions.csv and probabilities.csv
        output_dir: Optional directory to save combined predictions
                   If provided, saves to output_dir / "combined_predictions"
        
    Returns:
        Tuple of (combined_predictions_df, combined_probabilities_df)
        Both DataFrames have all samples from all folds, with optional 'fold' column
    """
    kfold_dir = Path(kfold_dir)
    
    if not kfold_dir.exists():
        raise FileNotFoundError(f"K-fold directory not found: {kfold_dir}")
    
    # Find all fold directories
    fold_dirs = sorted([d for d in kfold_dir.iterdir() if d.is_dir() and d.name.startswith('fold_')])
    
    if len(fold_dirs) == 0:
        raise ValueError(f"No fold directories found in {kfold_dir}")
    
    print(f"Combining predictions from {len(fold_dirs)} folds...")
    
    combined_preds_list = []
    combined_probs_list = []
    
    for fold_dir in fold_dirs:
        preds_path = fold_dir / "predictions.csv"
        probs_path = fold_dir / "probabilities.csv"
        
        if not preds_path.exists() or not probs_path.exists():
            print(f"   Warning: Missing files in {fold_dir}. Skipping...")
            continue
        
        preds_df = pd.read_csv(preds_path, index_col=0)
        probs_df = pd.read_csv(probs_path, index_col=0)
        
        # Extract fold number from directory name
        fold_num = fold_dir.name.replace('fold_', '')
        try:
            fold_num = int(fold_num)
        except ValueError:
            # If not numeric, keep as string
            pass
        
        # Add fold information if requested
        preds_df = preds_df.copy()
        probs_df = probs_df.copy()
        preds_df['fold'] = fold_num
        probs_df['fold'] = fold_num
        
        combined_preds_list.append(preds_df)
        combined_probs_list.append(probs_df)
    
    if len(combined_preds_list) == 0:
        raise ValueError("No valid fold predictions found to combine")
    
    # Combine all folds
    combined_preds = pd.concat(combined_preds_list, axis=0)
    combined_probs = pd.concat(combined_probs_list, axis=0)
    
    print(f"   Combined {len(combined_preds)} samples from {len(combined_preds_list)} folds")
    
    # Save if output directory provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        combined_dir = output_dir / "combined_predictions"
        combined_dir.mkdir(parents=True, exist_ok=True)
        

        combined_preds.to_csv(combined_dir / "predictions.csv")
        combined_probs.to_csv(combined_dir / "probabilities.csv")
        
        print(f"   Saved to: {combined_dir}")
    
    return combined_preds, combined_probs
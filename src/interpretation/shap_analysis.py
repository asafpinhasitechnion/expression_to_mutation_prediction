"""SHAP analysis for tree-based model interpretation (LightGBM, XGBoost, RandomForest only)."""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def _get_model_type(model):
    """
    Determine the type of tree-based model for SHAP analysis.
    
    Returns:
        Tuple of (model_type, actual_model, scaler, normalize_inputs)
        model_type: 'lightgbm', 'xgboost', 'random_forest', or 'unknown'
    """
    # Check for LightGBM/XGBoost/RandomForest via MultiOutputClassifier
    if hasattr(model, 'model'):
        sklearn_model = model.model
        if hasattr(sklearn_model, 'estimators_'):
            # MultiOutputClassifier structure
            estimators = sklearn_model.estimators_
            if len(estimators) > 0:
                first_estimator = estimators[0]
                # Check for LightGBM
                if hasattr(first_estimator, 'booster_') or 'lightgbm' in str(type(first_estimator)).lower():
                    scaler = getattr(model, 'scaler', None)
                    normalize_inputs = getattr(model, 'normalize_inputs', False)
                    return 'lightgbm', sklearn_model, scaler, normalize_inputs
                # Check for XGBoost
                if hasattr(first_estimator, 'get_booster') or 'xgboost' in str(type(first_estimator)).lower():
                    scaler = getattr(model, 'scaler', None)
                    normalize_inputs = getattr(model, 'normalize_inputs', False)
                    return 'xgboost', sklearn_model, scaler, normalize_inputs
                # Check for RandomForest
                if 'randomforest' in str(type(first_estimator)).lower():
                    scaler = getattr(model, 'scaler', None)
                    normalize_inputs = getattr(model, 'normalize_inputs', False)
                    return 'random_forest', sklearn_model, scaler, normalize_inputs
    
    return 'unknown', None, None, False


def compute_shap_values_for_gene(
    model,
    X_explain: np.ndarray | pd.DataFrame,
    gene_names: list[str],
    gene_idx: int,
    feature_names: list[str] | None = None,
) -> dict:
    """
    Compute SHAP values for a single gene (output) in a tree-based model.
    
    Only supports tree-based models: LightGBM, XGBoost, RandomForest.
    
    Args:
        model: Trained tree-based model (must use MultiOutputClassifier)
        X_explain: Full dataset to explain (can be full dataset, no train-test split required)
        gene_names: List of gene names
        gene_idx: Index of gene to explain
        feature_names: Optional list of feature names (inferred from X_explain if DataFrame)
    
    Returns:
        Dictionary containing:
            - 'shap_values': SHAP values array (n_samples, n_features)
            - 'feature_names': Feature names
            - 'explained_gene': Gene name being explained
            - 'explained': Data that was explained
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP library not available. Install with: pip install shap")
    
    # Determine model type
    model_type, actual_model, scaler, normalize_inputs = _get_model_type(model)
    
    if model_type == 'unknown':
        raise ValueError(
            f"SHAP analysis only supports tree-based models (LightGBM, XGBoost, RandomForest). "
            f"Got model type: {type(model)}. "
            f"Make sure your model uses MultiOutputClassifier with tree-based estimators."
        )
    
    # Check if model has estimators (MultiOutputClassifier structure)
    if not hasattr(actual_model, 'estimators_'):
        raise ValueError(
            f"Model must use MultiOutputClassifier structure with estimators_ attribute. "
            f"Got: {type(actual_model)}"
        )
    
    # Extract feature names from X_explain if DataFrame
    if isinstance(X_explain, pd.DataFrame):
        X_explain_values = X_explain.values
        if feature_names is None:
            feature_names = X_explain.columns.tolist()
    else:
        X_explain_values = X_explain
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X_explain_values.shape[1])]
    
    X_explain_values = np.array(X_explain_values, dtype=np.float32)
    
    # Apply normalization if needed
    if normalize_inputs and scaler is not None:
        X_explain_normalized = scaler.transform(X_explain_values).astype(np.float32)
    else:
        X_explain_normalized = X_explain_values
    
    # Get gene name
    explained_gene = gene_names[gene_idx]
    
    # Get the specific estimator for this gene
    estimator = actual_model.estimators_[gene_idx]
    
    # Compute SHAP values using TreeExplainer
    print(f"   Computing SHAP for {explained_gene} using TreeExplainer...")
    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(X_explain_normalized)
    
    # Handle binary classification output
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]  # Get SHAP values for positive class
    
    shap_values = np.array(shap_values)
    
    result = {
        'shap_values': shap_values,
        'feature_names': feature_names,
        'explained_gene': explained_gene,
        'explained': X_explain_values,
    }
    
    return result


def compute_shap_for_all_genes(
    model,
    X_explain: np.ndarray | pd.DataFrame,
    gene_names: list[str],
    feature_names: list[str] | None = None,
    save_dir: Path | None = None,
    selected_genes: list[str] | None = None,
) -> dict[str, dict]:
    """
    Compute SHAP values for all genes (outputs) in a tree-based multitask model.
    
    Only supports tree-based models: LightGBM, XGBoost, RandomForest.
    
    Args:
        model: Trained tree-based model (must use MultiOutputClassifier)
        X_explain: Full dataset to explain (can be full dataset, no train-test split required)
        gene_names: List of gene names (one per output) - all genes available in the model
        feature_names: Optional list of feature names (inferred from X_explain if DataFrame)
        save_dir: Optional directory to save SHAP values
        selected_genes: Optional list of gene names to explain. If provided, only these genes
                       will be analyzed. Must be a subset of gene_names.
    
    Returns:
        Dictionary mapping gene names to SHAP result dictionaries
        {gene_name: {'shap_values': ..., 'feature_names': ..., 'explained_gene': ..., 'explained': ...}}
    """
    # Filter genes if selected_genes is provided
    if selected_genes is not None:
        # Validate that all selected genes exist in gene_names
        invalid_genes = [g for g in selected_genes if g not in gene_names]
        if invalid_genes:
            raise ValueError(
                f"The following selected genes are not in gene_names: {invalid_genes}. "
                f"Available genes: {gene_names}"
            )
        genes_to_explain = selected_genes
        print(f"\nComputing SHAP values for {len(genes_to_explain)} selected genes out of {len(gene_names)} total...")
    else:
        genes_to_explain = gene_names
        print(f"\nComputing SHAP values for {len(gene_names)} genes...")
    
    print(f"   Dataset: {len(X_explain)} samples")
    if selected_genes is not None:
        print(f"   Selected genes: {', '.join(genes_to_explain)}")
    
    results = {}
    
    for idx, gene_name in enumerate(genes_to_explain):
        # Find the index in the original gene_names list (needed for model access)
        gene_idx = gene_names.index(gene_name)
        print(f"\n  [{idx+1}/{len(genes_to_explain)}] Processing {gene_name}...")
        
        try:
            result = compute_shap_values_for_gene(
                model=model,
                X_explain=X_explain,
                gene_names=gene_names,
                gene_idx=gene_idx,
                feature_names=feature_names,
            )
            results[gene_name] = result
            
            # Save individual gene SHAP values
            if save_dir is not None:
                save_dir.mkdir(parents=True, exist_ok=True)
                gene_dir = save_dir / gene_name
                gene_dir.mkdir(exist_ok=True)
                np.save(gene_dir / "shap_values.npy", result['shap_values'])
                
                # Save the data used for SHAP computation
                if isinstance(X_explain, pd.DataFrame):
                    X_explain.to_csv(gene_dir / "explained_data.csv")
                else:
                    pd.DataFrame(X_explain, columns=result['feature_names']).to_csv(
                        gene_dir / "explained_data.csv"
                    )
            
            gc.collect()  # Free memory
            
        except Exception as e:
            print(f"     Failed for {gene_name}: {e}")
            continue

    print(f"\nSHAP computation complete for {len(results)} genes")
    return results


def compute_shap_per_cancer_type(
    model,
    X_explain: np.ndarray | pd.DataFrame,
    gene_names: list[str],
    sample_to_cancer: dict[str, str],
    feature_names: list[str] | None = None,
    output_dir: Path | None = None,
    selected_genes: list[str] | None = None,
) -> dict[str, dict[str, dict]]:
    """
    Compute SHAP values separately for each cancer type.
    
    Only supports tree-based models: LightGBM, XGBoost, RandomForest.
    
    Args:
        model: Trained tree-based model (must use MultiOutputClassifier)
        X_explain: Full dataset to explain (can be full dataset, no train-test split required)
        gene_names: List of gene names (one per output) - all genes available in the model
        sample_to_cancer: Dictionary mapping sample IDs to cancer types
                          {sample_id: cancer_type}
        feature_names: Optional list of feature names (inferred from X_explain if DataFrame)
        output_dir: Optional output directory for results
        selected_genes: Optional list of gene names to explain. If provided, only these genes
                       will be analyzed. Must be a subset of gene_names.
    
    Returns:
        Nested dictionary: {cancer_type: {gene_name: SHAP_result_dict}}
    """
    # Extract feature names from X_explain if DataFrame
    if isinstance(X_explain, pd.DataFrame):
        explain_index = X_explain.index
        if feature_names is None:
            feature_names = X_explain.columns.tolist()
    else:
        explain_index = pd.RangeIndex(len(X_explain))
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X_explain.shape[1])]
    
    # Get unique cancer types
    cancer_type_names = sorted(set(sample_to_cancer.values()))
    
    all_results = {}
    
    print(f"\nRunning SHAP analysis per cancer type...")
    print(f"   Cancer types: {cancer_type_names}")
    print(f"   Total samples: {len(X_explain)}")
    
    for cancer_type in cancer_type_names:
        print(f"\n{'='*60}")
        print(f"Processing: {cancer_type}")
        print(f"{'='*60}")
        
        # Get samples for this cancer type
        cancer_samples = [s for s, ct in sample_to_cancer.items() if ct == cancer_type]
        
        # Find indices in X_explain
        if isinstance(X_explain, pd.DataFrame):
            cancer_samples_set = set(cancer_samples)
            # Try exact match first
            cancer_indices = explain_index.intersection(pd.Index(cancer_samples))
            
            if len(cancer_indices) == 0:
                # Try matching with 'A' suffix variations (for string indices)
                try:
                    # Try removing 'A' suffix from explain_index and match
                    explain_no_suffix = explain_index.str.replace('A', '', regex=False)
                    matched_no_suffix = explain_index[explain_no_suffix.isin(cancer_samples_set)]
                    if len(matched_no_suffix) > 0:
                        cancer_indices = matched_no_suffix
                    else:
                        # Try adding 'A' suffix to cancer_samples
                        cancer_samples_with_A = [s + 'A' if isinstance(s, str) else str(s) + 'A' for s in cancer_samples]
                        cancer_samples_with_A_set = set(cancer_samples_with_A)
                        cancer_indices = explain_index[explain_index.isin(cancer_samples_with_A_set)]
                except (AttributeError, TypeError):
                    # Index doesn't support string operations, skip suffix matching
                    pass
            
            if len(cancer_indices) > 0:
                X_explain_ct = X_explain.loc[cancer_indices]
            else:
                X_explain_ct = pd.DataFrame()
        else:
            # For arrays, match by position if sample IDs match
            cancer_samples_set = set(cancer_samples)
            cancer_indices = [i for i, s in enumerate(explain_index) if s in cancer_samples_set]
            if len(cancer_indices) == 0:
                print(f"   No matching samples found for {cancer_type}. Skipping...")
                continue
            X_explain_ct = X_explain[cancer_indices]
        
        if len(X_explain_ct) == 0:
            print(f"   No samples found for {cancer_type}. Skipping...")
            continue
        
        print(f"   Samples: {len(X_explain_ct)}")
        
        # Compute SHAP for this cancer type
        ct_dir = output_dir / "cancer_type" / cancer_type if output_dir else None
        gene_results = compute_shap_for_all_genes(
            model=model,
            X_explain=X_explain_ct,
            gene_names=gene_names,
            feature_names=feature_names,
            save_dir=ct_dir,
            selected_genes=selected_genes,
        )
        
        all_results[cancer_type] = gene_results
        print(f"   {cancer_type} complete! ({len(gene_results)} genes)")
    
    return all_results

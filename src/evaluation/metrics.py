"""Metrics computation for multi-label classification."""

from __future__ import annotations

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
        
        results[mutation] = gene_metrics
    
    return pd.DataFrame(results).T


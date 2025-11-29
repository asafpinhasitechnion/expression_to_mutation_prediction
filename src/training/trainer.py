"""Training functions for mutation prediction models."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from evaluation.metrics import evaluate_multilabel


def run_train_test_split(
    model,
    X,
    Y,
    test_size: float = 0.2,
    output_dir: str | Path = "results/",
    config_meta: Dict | None = None,
    random_state: int | None = 42,
    label: str | None = None,
):
    """Train on a subset of the data and evaluate on a held-out test set."""
    mutation_names = (
        Y.columns if isinstance(Y, pd.DataFrame) else [f"mutation_{i}" for i in range(Y.shape[1])]
    )
    sample_ids = (
        Y.index.to_numpy() if isinstance(Y, pd.DataFrame) else np.arange(Y.shape[0])
    )

    X_values = X.values if isinstance(X, pd.DataFrame) else X
    Y_values = Y.values if isinstance(Y, pd.DataFrame) else Y

    (
        X_train,
        X_test,
        Y_train,
        Y_test,
        train_ids,
        test_ids,
    ) = train_test_split(
        X_values,
        Y_values,
        sample_ids,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    model.fit(X_train, Y_train)
    Y_pred, Y_prob = model.predict(X_test)

    metrics = evaluate_multilabel(Y_test, Y_pred, Y_prob, mutation_names)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "per_gene_metrics.csv")

    summary = metrics.mean().to_frame().T
    summary.index = [label or "model"]
    summary.to_csv(output_dir / "summary_metrics.csv")

    preds_df = pd.DataFrame(Y_pred, columns=mutation_names, index=test_ids)
    probs_df = pd.DataFrame(Y_prob, columns=mutation_names, index=test_ids)
    preds_df.to_csv(output_dir / "predictions.csv")
    probs_df.to_csv(output_dir / "probabilities.csv")

    if config_meta:
        with open(output_dir / "meta.json", "w") as f:
            json.dump(config_meta, f, indent=2)

    model.save(output_dir)

    print("✅ Train/test evaluation complete.")
    print(summary)
    return metrics


def run_kfold_training(
    model,
    X,
    Y,
    k: int = 5,
    output_dir: str | Path = "results/",
    config_meta: Dict | None = None,
    random_state: int | None = 42,
    label: str | None = None,
):
    """K-fold cross-validation with per-fold artifacts."""
    mutation_names = (
        Y.columns if isinstance(Y, pd.DataFrame) else [f"mutation_{i}" for i in range(Y.shape[1])]
    )
    sample_ids = (
        Y.index.to_numpy() if isinstance(Y, pd.DataFrame) else np.arange(Y.shape[0])
    )

    X_values = X.values if isinstance(X, pd.DataFrame) else X
    Y_values = Y.values if isinstance(Y, pd.DataFrame) else Y

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_values), start=1):
        print(f"\n🔁 Fold {fold}/{k}...")

        model_fold = copy.deepcopy(model)

        X_train, X_test = X_values[train_idx], X_values[test_idx]
        Y_train, Y_test = Y_values[train_idx], Y_values[test_idx]
        test_ids = sample_ids[test_idx]

        model_fold.fit(X_train, Y_train)
        Y_pred, Y_prob = model_fold.predict(X_test)

        fold_df = evaluate_multilabel(Y_test, Y_pred, Y_prob, mutation_names)
        fold_metrics.append(fold_df.assign(fold=fold))

        fold_dir = output_dir / f"fold_{fold}"
        fold_dir.mkdir(exist_ok=True)
        fold_df.to_csv(fold_dir / "metrics.csv")

        preds_df = pd.DataFrame(Y_pred, columns=mutation_names, index=test_ids)
        probs_df = pd.DataFrame(Y_prob, columns=mutation_names, index=test_ids)
        preds_df.to_csv(fold_dir / "predictions.csv")
        probs_df.to_csv(fold_dir / "probabilities.csv")

        model_fold.save(fold_dir)

    stacked = pd.concat(fold_metrics)
    summary = stacked.groupby(level=0).agg(['mean', 'std'])
    summary.columns = ["_".join(col).strip() for col in summary.columns]
    summary.to_csv(output_dir / "summary.csv")

    if config_meta:
        with open(output_dir / "meta.json", "w") as f:
            json.dump(config_meta, f, indent=2)

    print("✅ K-fold evaluation complete.")
    return summary


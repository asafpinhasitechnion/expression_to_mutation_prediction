from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, train_test_split

from features.feature_selector import FeatureSelector
from models.model_factory import ModelFactory
from preprocessing.data_loader import TCGADataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config/config.yaml"


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    """Load configuration from yaml file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def evaluate_multilabel(y_true, y_pred, y_prob, mutation_names):
    results = {}
    for i, mutation in enumerate(mutation_names):
        try:
            results[mutation] = {
                'f1': f1_score(y_true[:, i], y_pred[:, i]),
                'roc_auc': roc_auc_score(y_true[:, i], y_prob[:, i]),
                'accuracy': accuracy_score(y_true[:, i], y_pred[:, i])
            }
        except ValueError:
            results[mutation] = {'f1': None, 'roc_auc': None, 'accuracy': None}
    return pd.DataFrame(results).T

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

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> dict:
    """Calculate evaluation metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }

    if y_prob is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)

    return metrics

def main(
    config_path: str | Path | None = None,
    use_cache: bool | None = None,
    cancer_types: tuple[str, ...] | None = None,
    eval_mode: str | None = None,
):
    # Load configuration
    config_path = Path(config_path or DEFAULT_CONFIG_PATH)
    config = load_config(config_path)

    # Initialize components
    data_loader = TCGADataLoader(
        config_path=str(config_path),
        use_cache=use_cache if use_cache is not None else config.get("data", {}).get("use_cache", True),
    )
    model_factory = ModelFactory()

    # Load and preprocess data
    print("Loading and preprocessing data...")
    expression_data, mutation_data = data_loader.preprocess_data(cancer_types=list(cancer_types) if cancer_types else None)

    feature_cfg = config.get("features", {})
    feature_method = (feature_cfg.get("method") or "none").lower()
    selected_features = expression_data.columns.tolist()

    if feature_method not in {"none", "disabled"}:
        if feature_method != "variance":
            raise NotImplementedError(
                f"Feature selector method '{feature_method}' is not supported for multi-label targets yet."
            )
        feature_selector = FeatureSelector(config_path=config_path)
        expression_data, selected_features = feature_selector.select_features(expression_data)
        print(f"Selected {len(selected_features)} features via '{feature_method}'.")
    else:
        print("Skipping feature selection step.")

    # Get model
    print("Initializing model...")
    model_name = config["model"].get("name", "random_forest")
    model = model_factory.get_model(
        model_name=model_name,
        input_size=expression_data.shape[1],
        output_size=mutation_data.shape[1],
        config=config,
    )

    evaluation_cfg = config.get("evaluation", {})
    eval_mode = (eval_mode or evaluation_cfg.get("mode") or "train_test").lower()
    if eval_mode not in {"train_test", "kfold"}:
        raise ValueError(f"Unsupported eval_mode '{eval_mode}'. Use 'train_test' or 'kfold'.")

    print(f"Running {eval_mode} evaluation...")
    output_dir = Path(config["data"].get("output_path", "results"))
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    model_dir = output_dir / model_name
    shared_meta = {
        "config": config,
        "selected_features": selected_features,
        "cancer_types": cancer_types or config["data"].get("cancer_types"),
    }

    if eval_mode == "kfold":
        run_kfold_training(
            model=model,
            X=expression_data,
            Y=mutation_data,
            k=evaluation_cfg.get("cv_folds", 5),
            output_dir=model_dir,
            config_meta=shared_meta,
            random_state=config.get("preprocessing", {}).get("random_state", 42),
            label=model_name,
        )
    else:
        run_train_test_split(
            model=model,
            X=expression_data,
            Y=mutation_data,
            test_size=config.get("preprocessing", {}).get("test_size", 0.2),
            output_dir=model_dir,
            config_meta=shared_meta,
            random_state=config.get("preprocessing", {}).get("random_state", 42),
            label=model_name,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate mutation prediction models.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable cached expression/mutation matrices.",
    )
    parser.add_argument(
        "--eval-mode",
        choices=["train_test", "kfold"],
        help="Evaluation strategy to use (overrides config).",
    )
    parser.add_argument(
        "--cancer-types",
        nargs="+",
        help="Optional list of TCGA cohorts (e.g. BRCA LUAD) to download/load.",
    )
    args = parser.parse_args()
    main(
        config_path=args.config,
        use_cache=not args.no_cache,
        cancer_types=tuple(args.cancer_types) if args.cancer_types else None,
        eval_mode=args.eval_mode,
    )
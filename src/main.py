from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

from evaluation.metrics import combine_fold_predictions, extract_cancer_types
from features.feature_selector import FeatureSelector
from interpretation.ablation import run_gene_ablation_analysis
from interpretation.shap_analysis import save_shap_summary
from models.model_factory import ModelFactory
from preprocessing.data_loader import TCGADataLoader
from training.extract_weights import extract_sample_embeddings, train_and_extract_head_weights
from training.trainer import run_kfold_training, run_train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config/config.yaml"


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    """Load configuration from yaml file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def log_and_scale_expression(expression_data: pd.DataFrame) -> pd.DataFrame:
    """Apply log1p and StandardScaler, preserving index/columns."""
    if not isinstance(expression_data, pd.DataFrame):
        expression_data = pd.DataFrame(expression_data)

    X_log = np.log1p(expression_data)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log.values)

    return pd.DataFrame(
        X_scaled,
        index=expression_data.index,
        columns=expression_data.columns,
    )


def limit_mutation_genes_topk(mutation_data: pd.DataFrame, top_k: int = 500) -> pd.DataFrame:
    """
    Keep at most top_k mutation genes by number of mutations (column sum).
    Assumes mutation matrix is binary-ish (0/1) but works for non-negative counts too.
    """
    if not isinstance(mutation_data, pd.DataFrame):
        mutation_data = pd.DataFrame(mutation_data)

    if mutation_data.shape[1] <= top_k:
        return mutation_data

    # Ensure numeric
    mut_num = mutation_data.apply(pd.to_numeric, errors="coerce").fillna(0)

    gene_counts = mut_num.sum(axis=0)
    top_genes = gene_counts.sort_values(ascending=False).head(top_k).index.tolist()
    return mutation_data.loc[:, top_genes]


def select_good_genes_from_kfold_summary(
    kfold_dir: Path,
    norm_score_threshold: float,
) -> list[str]:
    """
    Reads kfold_dir/summary.csv and selects genes with:
      (AUPRC - prevalence) / (1 - prevalence) > threshold
    """
    summary_path = kfold_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Could not find k-fold summary at: {summary_path}")

    summary_df = pd.read_csv(summary_path, index_col=0)

    required_cols = {"auprc_mean", "prevalence_mean"}
    missing = required_cols - set(summary_df.columns)
    if missing:
        raise ValueError(f"summary.csv missing required columns: {sorted(missing)}")

    den = (1.0 - summary_df["prevalence_mean"]).replace(0, np.nan)
    auprc_norm = (summary_df["auprc_mean"] - summary_df["prevalence_mean"]) / den
    auprc_norm = auprc_norm.replace([np.inf, -np.inf], np.nan).dropna()

    chosen = auprc_norm[auprc_norm > norm_score_threshold].index.tolist()
    return chosen


def run_comprehensive_pipeline(
    config_path: str | Path | None = None,
    use_cache: bool | None = None,
    cancer_types: tuple[str, ...] | None = None,
    norm_score_threshold: float | None = None,
    run_ablation: bool = True,
    top_k_mutation_genes: int = 500,
):
    """
    Pipeline:
    1. Load + preprocess
    2. log1p + scale
    3. keep top-K mutation genes (by #mutations)
    4. K-fold CV with multitask_nn
    5. Select good genes by normalized AUPRC
    6. Train full multitask_nn + extract gene embeddings
    7. Extract sample embeddings
    8. SHAP per-gene XGBoost (chosen genes)
    9. Ablation + filtered matrices for chosen genes
    """
    print("=" * 80)
    print("COMPREHENSIVE ANALYSIS PIPELINE")
    print("=" * 80)

    # Load config
    config_path = Path(config_path or DEFAULT_CONFIG_PATH)
    config = load_config(config_path)

    # Force main prediction model to multitask_nn (non-SHAP predictions)
    config.setdefault("model", {})
    config["model"]["name"] = "multitask_nn"

    # Threshold for selecting good genes
    if norm_score_threshold is None:
        norm_score_threshold = config.get("interpretation", {}).get("norm_score_threshold", 0.1)

    # Initialize components
    data_loader = TCGADataLoader(
        config_path=str(config_path),
        use_cache=use_cache if use_cache is not None else config.get("data", {}).get("use_cache", True),
    )
    model_factory = ModelFactory()

    # Load and preprocess data
    print("\n[1/7] Loading and preprocessing data...")
    expression_data, mutation_data = data_loader.preprocess_data(
        cancer_types=list(cancer_types) if cancer_types else None
    )
    if expression_data is None or mutation_data is None:
        raise RuntimeError("Data loader returned None. Check TCGADataLoader.preprocess_data().")

    # log1p + scale
    expression_data = log_and_scale_expression(expression_data)

    # Align mutation to expression index
    if isinstance(mutation_data, pd.DataFrame):
        mutation_data = mutation_data.loc[expression_data.index]
    else:
        mutation_data = pd.DataFrame(mutation_data, index=expression_data.index)

    # NEW: keep top-K mutation genes by mutation count
    mutation_data = limit_mutation_genes_topk(mutation_data, top_k=top_k_mutation_genes)
    print(f"   Mutation targets kept: {mutation_data.shape[1]} genes (top_k={top_k_mutation_genes})")

    if expression_data.shape[0] != mutation_data.shape[0]:
        raise ValueError(
            f"Sample mismatch: expression has {expression_data.shape[0]} rows, "
            f"mutation has {mutation_data.shape[0]} rows."
        )

    # Feature selection (optional)
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

    # Setup output directory
    output_dir = Path(config.get("data", {}).get("output_path", "results"))
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    model_name = config["model"]["name"]  # forced to multitask_nn
    model_dir = output_dir / model_name

    # Cancer label subfolder
    if cancer_types:
        cancer_label = "_".join(sorted(cancer_types))
        model_dir = model_dir / cancer_label

    model_dir.mkdir(parents=True, exist_ok=True)

    shared_meta = {
        "config": config,
        "selected_features": selected_features,
        "cancer_types": cancer_types or config.get("data", {}).get("cancer_types"),
    }

    # Step 1: K-fold CV
    print("\n[2/7] Running K-fold cross-validation...")
    evaluation_cfg = config.get("evaluation", {})
    k = evaluation_cfg.get("cv_folds", 5)

    model = model_factory.get_model(
        model_name=model_name,
        input_size=expression_data.shape[1],
        output_size=mutation_data.shape[1],
        config=config,
    )

    kfold_dir = model_dir / "kfold_prediction"
    run_kfold_training(
        model=model,
        X=expression_data,
        Y=mutation_data,
        k=k,
        output_dir=kfold_dir,
        config_meta=shared_meta,
        random_state=config.get("preprocessing", {}).get("random_state", 42),
        label=model_name,
    )

    # Ensure combined predictions exist
    combined_dir = kfold_dir / "combined_predictions"
    if not combined_dir.exists():
        combine_fold_predictions(kfold_dir=kfold_dir, output_dir=kfold_dir)

    # Cleanup fold folders
    print("\n   Cleaning up fold folders...")
    fold_dirs = [d for d in kfold_dir.iterdir() if d.is_dir() and d.name.startswith("fold_")]
    for fold_dir in fold_dirs:
        try:
            shutil.rmtree(fold_dir)
            print(f"   Removed {fold_dir.name}")
        except Exception as e:
            print(f"   Warning: Could not remove {fold_dir.name}: {e}")

    # Select good genes from summary.csv
    print("\nSelecting good genes from k-fold summary...")
    chosen_genes = select_good_genes_from_kfold_summary(
        kfold_dir=kfold_dir,
        norm_score_threshold=norm_score_threshold,
    )
    print(f"   Chosen genes: {len(chosen_genes)} (threshold={norm_score_threshold})")
    if chosen_genes:
        preview = ", ".join(chosen_genes[:10])
        print(f"   Preview: {preview}{'...' if len(chosen_genes) > 10 else ''}")
    else:
        print("   Warning: No genes meet the threshold. SHAP will be skipped; ablation will still run.")

    # Step 2: Train full model + extract embeddings
    print("\n[3/7] Training model on full dataset for embedding extraction...")
    embeddings_dir = model_dir / "gene_embeddings"
    # --- Use SAME architecture as in k-fold ---
    mt_cfg = (config.get("model", {}).get("multitask_nn") or {})

    hidden_layers = mt_cfg.get("hidden_layers", [512, 256])
    head_layers = mt_cfg.get("head_layers", None)  # [] or None depending on your codebase

    head_weights = train_and_extract_head_weights(
        model_factory=model_factory,
        X=expression_data,
        Y=mutation_data,
        config=config,
        hidden_layers=hidden_layers,
        head_layers=head_layers,
        output_dir=embeddings_dir,
        save_model=True,
    )


    trained_model = head_weights["model"]
    print(f"   Gene embeddings extracted: {head_weights['weight_matrix_shape']}")

    # Step 3: Sample embeddings
    print("\n[4/7] Extracting sample embeddings...")
    sample_embeddings_dir = embeddings_dir / "sample_embeddings"

    sample_embeddings_result = extract_sample_embeddings(
        model=trained_model,
        X=expression_data,
        sample_ids=expression_data.index if isinstance(expression_data, pd.DataFrame) else None,
        output_dir=sample_embeddings_dir,
        batch_size=128,
    )
    print(f"   Sample embeddings extracted: {sample_embeddings_result['embeddings'].shape}")

    # Step 4: SHAP per-gene XGBoost
    print("\n[5/7] Running SHAP analysis (per-gene XGBoost on chosen genes)...")
    shap_dir = model_dir / "shap_analysis"
    shap_dir.mkdir(parents=True, exist_ok=True)

    if len(chosen_genes) == 0:
        print("   Skipping SHAP: no chosen genes.")
    else:
        try:
            import shap  # noqa: F401
            import xgboost as xgb
        except ImportError as e:
            print(f"   Warning: SHAP step skipped because dependency missing: {e}")
            print("   Install: pip install shap xgboost")
        else:
            X_values = expression_data.values
            feature_names = expression_data.columns.tolist()
            rs = config.get("preprocessing", {}).get("random_state", 42)

            for idx, gene_name in enumerate(chosen_genes):
                if gene_name not in mutation_data.columns:
                    continue

                y_gene = mutation_data[gene_name].astype(int).values
                uniq = np.unique(y_gene)
                if len(uniq) < 2:
                    continue

                pos = int((y_gene == 1).sum())
                neg = int((y_gene == 0).sum())
                scale_pos_weight = (neg / pos) if pos > 0 else 1.0

                xgb_model = xgb.XGBClassifier(
                    scale_pos_weight=scale_pos_weight,
                    tree_method="hist",
                    n_jobs=-1,
                    random_state=rs,
                    eval_metric="logloss",
                )
                xgb_model.fit(X_values, y_gene)

                explainer = shap.TreeExplainer(xgb_model)
                shap_values = explainer.shap_values(X_values)
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    shap_values = shap_values[1]
                shap_values = np.asarray(shap_values)

                gene_dir = shap_dir / gene_name
                gene_dir.mkdir(parents=True, exist_ok=True)

                save_shap_summary(
                    shap_values=shap_values,
                    feature_names=feature_names,
                    output_path=gene_dir / "shap_summary.csv",
                    sample_ids=expression_data.index,
                    top_n=20,
                )

                if (idx + 1) % 10 == 0 or (idx + 1) == len(chosen_genes):
                    print(f"   SHAP progress: {idx + 1}/{len(chosen_genes)}")

            print(f"   SHAP complete. Summaries saved under: {shap_dir}")

    # Step 5: Ablation
    metric_matrices = {}
    if run_ablation:
        print("\n[6/7] Running ablation analysis...")
        ablation_dir = model_dir / "ablation_analysis"

        ablation_eval_mode = (evaluation_cfg.get("mode", "kfold") or "kfold").lower()
        if ablation_eval_mode not in {"train_test", "kfold"}:
            ablation_eval_mode = "kfold"

        cancer_type_series, _ = extract_cancer_types(
            X=expression_data,
            sample_ids=expression_data.index.tolist() if isinstance(expression_data, pd.DataFrame) else None,
        )
        sample_to_cancer = cancer_type_series.to_dict() if len(cancer_type_series) > 0 else None

        metric_matrices = run_gene_ablation_analysis(
            model_factory=model_factory,
            X=expression_data,
            Y=mutation_data,
            config=config,
            output_dir=ablation_dir,
            eval_mode=ablation_eval_mode,
            test_size=config.get("preprocessing", {}).get("test_size", 0.2),
            k=evaluation_cfg.get("cv_folds", 5),
            random_state=config.get("preprocessing", {}).get("random_state", 42),
            sample_to_cancer=sample_to_cancer,
            remove_fold_folders=True,
        )
        print(f"   Ablation analysis complete. Computed {len(metric_matrices)} difference matrices.")

        # Filter ablation results to chosen genes
        if len(chosen_genes) > 0:
            print("\nFiltering ablation results to chosen genes...")
            for metric_name, diff_matrix in metric_matrices.items():
                if isinstance(diff_matrix, pd.DataFrame):
                    available = [g for g in chosen_genes if g in diff_matrix.columns]
                    if not available:
                        continue
                    filtered = diff_matrix[available].copy()
                    filtered.to_csv(ablation_dir / f"difference_matrix_{metric_name}_filtered.csv")
                    print(f"   Saved filtered {metric_name}: {len(available)} genes")
    else:
        print("\n[6/7] Skipping ablation analysis...")
    # Step 6: Summary
    print("\n[7/7] Pipeline complete!")
    print("=" * 80)
    print(f"Results saved to: {model_dir}")
    print(f"  - K-fold predictions: {kfold_dir}")
    print(f"  - Gene embeddings: {embeddings_dir}")
    print(f"  - Sample embeddings: {sample_embeddings_dir}")
    print(f"  - SHAP analysis: {shap_dir}")
    if run_ablation:
        print(f"  - Ablation analysis: {ablation_dir}")
    print("=" * 80)

    return {
        "kfold_dir": kfold_dir,
        "embeddings_dir": embeddings_dir,
        "sample_embeddings_dir": sample_embeddings_dir,
        "shap_dir": shap_dir,
        "ablation_dir": ablation_dir if run_ablation else None,
        "head_weights": head_weights,
        "sample_embeddings": sample_embeddings_result,
        "metric_matrices": metric_matrices,
        "chosen_genes": chosen_genes,
    }


def main(
    config_path: str | Path | None = None,
    use_cache: bool | None = None,
    cancer_types: tuple[str, ...] | None = None,
    eval_mode: str | None = None,
    run_ablation: bool = False,
    extract_weights: bool = False,
    run_pipeline: bool = False,
):

    if run_pipeline:
        return run_comprehensive_pipeline(
            config_path=args.config,
            use_cache=not args.no_cache,
            cancer_types=tuple(args.cancer_types) if args.cancer_types else None,
            run_ablation=not args.no_ablation,
        )

    # Load config
    config_path = Path(config_path or DEFAULT_CONFIG_PATH)
    config = load_config(config_path)

    # Force main prediction model to multitask_nn
    config.setdefault("model", {})
    config["model"]["name"] = "multitask_nn"

    # Initialize components
    data_loader = TCGADataLoader(
        config_path=str(config_path),
        use_cache=use_cache if use_cache is not None else config.get("data", {}).get("use_cache", True),
    )
    model_factory = ModelFactory()

    # Load data
    print("Loading and preprocessing data...")
    expression_data, mutation_data = data_loader.preprocess_data(
        cancer_types=list(cancer_types) if cancer_types else None
    )
    if expression_data is None or mutation_data is None:
        raise RuntimeError("Data loader returned None. Check TCGADataLoader.preprocess_data().")

    # log1p + scale
    expression_data = log_and_scale_expression(expression_data)

    # Align mutation to expression index
    if isinstance(mutation_data, pd.DataFrame):
        mutation_data = mutation_data.loc[expression_data.index]
    else:
        mutation_data = pd.DataFrame(mutation_data, index=expression_data.index)

    # Keep top 500 mutation genes
    mutation_data = limit_mutation_genes_topk(mutation_data, top_k=500)
    print(f"   Mutation targets kept: {mutation_data.shape[1]} genes (top_k=500)")

    # Feature selection
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

    # Extract weights branch
    if extract_weights:
        print("Training model on full dataset and extracting head weights...")
        output_dir = Path(config.get("data", {}).get("output_path", "results"))
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
        output_dir = output_dir / "head_weights_extraction"

        head_weights = train_and_extract_head_weights(
            model_factory=model_factory,
            X=expression_data,
            Y=mutation_data,
            config=config,
            hidden_layers=[2048, 1536, 1024],
            head_layers=None,
            output_dir=output_dir,
        )
        print("\nHead weights extracted!")
        print(f"   Weight matrix shape: {head_weights['weight_matrix_shape']}")
        print(f"   Number of genes: {len(head_weights['gene_names'])}")
        print(f"   Results saved to: {output_dir}")
        return head_weights

    # Ablation branch
    if run_ablation:
        print("Running gene ablation analysis...")
        output_dir = Path(config.get("data", {}).get("output_path", "results"))
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir

        evaluation_cfg = config.get("evaluation", {})
        ablation_eval_mode = (eval_mode or evaluation_cfg.get("mode") or "train_test").lower()
        if ablation_eval_mode not in {"train_test", "kfold"}:
            raise ValueError(f"Unsupported eval_mode '{ablation_eval_mode}'. Use 'train_test' or 'kfold'.")

        metric_matrices = run_gene_ablation_analysis(
            model_factory=model_factory,
            X=expression_data,
            Y=mutation_data,
            config=config,
            output_dir=output_dir,
            eval_mode=ablation_eval_mode,
            test_size=config.get("preprocessing", {}).get("test_size", 0.2),
            k=evaluation_cfg.get("cv_folds", 5),
            random_state=config.get("preprocessing", {}).get("random_state", 42),
        )
        return metric_matrices

    # Default eval (train_test or kfold) using multitask_nn
    print("Initializing model...")
    model_name = config["model"]["name"]
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
    output_dir = Path(config.get("data", {}).get("output_path", "results"))
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    model_dir = output_dir / model_name

    shared_meta = {
        "config": config,
        "selected_features": selected_features,
        "cancer_types": cancer_types or config.get("data", {}).get("cancer_types"),
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
        help="Optional list of TCGA cohorts (e.g. BRCA LUAD).",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run gene ablation analysis.",
    )
    parser.add_argument(
        "--extract-weights",
        action="store_true",
        help="Train multitask model on full dataset and extract head layer weights.",
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Run pipeline: kfold (multitask_nn) → embeddings → SHAP (XGB) → ablation.",
    )

    parser.add_argument("--no-ablation", action="store_true", help="Skip ablation in pipeline.")
    
    args = parser.parse_args()

    main(
        config_path=args.config,
        use_cache=not args.no_cache,
        cancer_types=tuple(args.cancer_types) if args.cancer_types else None,
        eval_mode=args.eval_mode,
        run_ablation=args.ablation,
        extract_weights=args.extract_weights,
        run_pipeline=args.pipeline,
    )

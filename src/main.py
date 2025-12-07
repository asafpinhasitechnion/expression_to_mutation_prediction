from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from features.feature_selector import FeatureSelector
from interpretation.ablation import run_gene_ablation_analysis
from models.model_factory import ModelFactory
from preprocessing.data_loader import TCGADataLoader
from training.extract_weights import train_and_extract_head_weights
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

def main(
    config_path: str | Path | None = None,
    use_cache: bool | None = None,
    cancer_types: tuple[str, ...] | None = None,
    eval_mode: str | None = None,
    run_ablation: bool = False,
    extract_weights: bool = False,
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

    # Check if extracting head weights
    if extract_weights:
        print("Training model on full dataset and extracting head weights...")
        output_dir = Path(config["data"].get("output_path", "results"))
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
        output_dir = output_dir / "head_weights_extraction"
        
        head_weights = train_and_extract_head_weights(
            model_factory=model_factory,
            X=expression_data,
            Y=mutation_data,
            config=config,
            hidden_layers=[2048, 1536, 1024],
            head_layers=None,  # Simple linear output
            output_dir=output_dir,
        )
        
        print(f"\nHead weights extracted!")
        print(f"   Weight matrix shape: {head_weights['weight_matrix_shape']}")
        print(f"   Number of genes: {len(head_weights['gene_names'])}")
        print(f"   Results saved to: {output_dir}")
        return head_weights
    
    # Check if running ablation analysis
    if run_ablation:
        print("Running gene ablation analysis...")
        output_dir = Path(config["data"].get("output_path", "results"))
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
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run gene ablation analysis (removes each gene and evaluates remaining genes).",
    )
    parser.add_argument(
        "--extract-weights",
        action="store_true",
        help="Train multitask model on full dataset and extract head layer weights.",
    )
    args = parser.parse_args()
    main(
        config_path=args.config,
        use_cache=not args.no_cache,
        cancer_types=tuple(args.cancer_types) if args.cancer_types else None,
        eval_mode=args.eval_mode,
        run_ablation=args.ablation,
        extract_weights=args.extract_weights,
    )
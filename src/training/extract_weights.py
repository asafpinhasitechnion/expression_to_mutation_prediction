"""Function to train multitask model on full dataset and extract head weights."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch

from models.model_factory import ModelFactory


def train_and_extract_head_weights(
    model_factory: ModelFactory,
    X: np.ndarray | pd.DataFrame,
    Y: pd.DataFrame,
    config: dict,
    hidden_layers: list[int] = [2048, 1536, 1024],
    head_layers: list[int] | None = None,
    output_dir: str | Path | None = None,
    save_model: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Train a multitask neural network on the full dataset and extract head layer weights.
    
    Args:
        model_factory: ModelFactory instance to create models
        X: Expression data (features)
        Y: Mutation data (targets) - must be DataFrame with gene names as columns
        config: Configuration dictionary
        hidden_layers: List of hidden layer sizes for the encoder
        head_layers: List of head layer sizes (None or [] for simple linear output)
        output_dir: Optional directory to save weights and metadata
        save_model: Whether to save model checkpoint (default: False)
    
    Returns:
        Dictionary mapping gene names to their head weight vectors (tensors of shape (hidden_size,))
        Also includes 'all_weights' key with the full weight matrix (output_size, hidden_size)
    """
    if not isinstance(Y, pd.DataFrame):
        raise ValueError("Y must be a pandas DataFrame with gene names as columns")
    
    if Y.columns.empty or not all(isinstance(col, str) for col in Y.columns):
        raise ValueError("Y must have named columns (gene names)")
    
    mutation_names = Y.columns.tolist()
    n_genes = len(mutation_names)
    
    print(f"\n🧬 Training multitask model on full dataset...")
    print(f"   Genes: {n_genes}")
    print(f"   Hidden layers: {hidden_layers}")
    print(f"   Head layers: {head_layers if head_layers else 'Simple linear'}")
    
    # Create model config with specified architecture
    model_config = config.get("model", {}).get("multitask_nn", {}).copy()
    model_config["hidden_layers"] = hidden_layers
    model_config["head_layers"] = head_layers if head_layers is not None else []
    
    # Create model
    model = model_factory.get_model(
        model_name="multitask_nn",
        input_size=X.shape[1] if isinstance(X, pd.DataFrame) else X.shape[1],
        output_size=n_genes,
        config={"model": {"multitask_nn": model_config}},
    )
    
    # Train on full dataset
    print("   Training on full dataset (no validation split)...")
    model.fit_full_dataset(X, Y)
    
    # Extract head weights
    print("   Extracting head weights...")
    head_weight_matrix = model.get_head_weights()  # Shape: (n_genes, hidden_size)
    
    # Convert to numpy for easier handling
    head_weights_np = head_weight_matrix.cpu().numpy()
    
    # Create dictionary mapping gene names to their weight vectors
    gene_weights = {}
    for i, gene_name in enumerate(mutation_names):
        gene_weights[gene_name] = torch.from_numpy(head_weights_np[i, :])
    
    # Also include the full matrix
    gene_weights["all_weights"] = head_weight_matrix
    gene_weights["gene_names"] = mutation_names
    gene_weights["weight_matrix_shape"] = head_weight_matrix.shape
    
    # Save weights and metadata if output_dir is provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model only if requested
        if save_model:
            model.save(output_dir)
        
        # Save head weights as numpy array
        np.save(output_dir / "head_weights.npy", head_weights_np)
        
        # Save gene names
        with open(output_dir / "gene_names.json", "w") as f:
            json.dump(mutation_names, f, indent=2)
        
        if save_model:
            print(f"   Model and weights saved to: {output_dir}")
        else:
            print(f"   Weights saved to: {output_dir}")
    
    print("Training complete!")
    print(f"   Head weight matrix shape: {head_weight_matrix.shape}")
    
    return gene_weights


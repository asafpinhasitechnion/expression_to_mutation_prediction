from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Tuple
import copy

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - optional dependency
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover - optional dependency
    LGBMClassifier = None


class WeightedBCEWithLogitsLoss(nn.Module):
    """BCE with Logits Loss with per-gene weighting based on mutation prevalence."""
    
    def __init__(self, gene_weights: torch.Tensor, pos_weight: torch.Tensor | None = None):
        """
        Args:
            gene_weights: Tensor of shape (n_genes,) - weight for each gene based on prevalence
            pos_weight: Optional tensor of shape (n_genes,) - positive class weight for each gene
        """
        super().__init__()
        self.register_buffer('gene_weights', gene_weights)
        self.register_buffer('pos_weight', pos_weight)
        # Use reduction='none' to get per-sample, per-gene loss
        self.base_loss = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Tensor of shape (batch_size, n_genes)
            targets: Tensor of shape (batch_size, n_genes)
        
        Returns:
            Scalar loss value
        """
        # Compute per-sample, per-gene loss: shape (batch_size, n_genes)
        # Note: base_loss with reduction='none' doesn't apply pos_weight automatically
        loss_per_gene = self.base_loss(logits, targets)
        
        # Apply positive class weighting if provided
        # pos_weight increases the loss contribution of positive samples
        # Formula: loss = -pos_weight * y * log(sigmoid(logit)) - (1-y) * log(1-sigmoid(logit))
        if self.pos_weight is not None:
            pos_mask = targets > 0.5
            # For positive samples, multiply by the corresponding gene's pos_weight
            # pos_weight shape: (n_genes,), pos_mask shape: (batch_size, n_genes)
            pos_weight_expanded = self.pos_weight.unsqueeze(0)  # (1, n_genes)
            loss_per_gene = torch.where(
                pos_mask,
                loss_per_gene * pos_weight_expanded,
                loss_per_gene
            )
        
        # Apply gene weights and average: shape (batch_size, n_genes) -> scalar
        # Gene weights are normalized so average is 1.0, maintaining loss scale
        weighted_loss = (loss_per_gene * self.gene_weights.unsqueeze(0)).mean()
        
        return weighted_loss


class BaseModel(ABC):
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def save(self, output_path):
        pass

    @abstractmethod
    def load(self, model_path):
        pass

    def feature_importance(self):
        return None


class SklearnModel(BaseModel):
    def __init__(self, model_cls, output_size: int = 1, **kwargs):
        """
        model_cls: sklearn estimator class
        output_size: number of outputs/classes
        """
        if model_cls.__name__.lower().startswith("lgbm") and 'is_unbalance' not in kwargs:
            kwargs['is_unbalance'] = True

        self.model = MultiOutputClassifier(model_cls(**kwargs))
        self.output_size = output_size

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X) -> Tuple[np.ndarray, np.ndarray]:
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)
            probs = np.column_stack([p[:, 1] if p.shape[1] > 1 else p[:, 0] for p in probs])
            preds = (probs > 0.5).astype(int)
            return preds, probs
        else:
            preds = self.model.predict(X)
            return preds, preds

    def save(self, output_path):
        joblib.dump(self.model, output_path / 'model.joblib')

    def load(self, model_path):
        self.model = joblib.load(model_path)

    def feature_importance(self):
        return None


class NeuralNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers=None):
        super(NeuralNet, self).__init__()
        layers = []
        current_size = input_size
        hidden_layers = hidden_layers or [128]
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(current_size, hidden_dim))
            layers.append(nn.ReLU())
            current_size = hidden_dim
        layers.append(nn.Linear(current_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class NeuralNetModel(BaseModel):
    def __init__(self, input_size: int, output_size: int, **kwargs):
        self.input_size = input_size
        self.output_size = output_size
        self.epochs = kwargs.get('epochs', 20)
        self.lr = kwargs.get('learning_rate', 0.001)
        self.hidden_layers = kwargs.get('hidden_layers', [256, 128])
        self.batch_size = kwargs.get('batch_size', 64)
        self.use_pos_weight = kwargs.get('use_pos_weight', True)
        self.normalize_inputs = kwargs.get('normalize_inputs', True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler: StandardScaler | None = None
        self.training_history = []
        self.target_names: list[str] = []
        self.models: list[nn.Module | None] = []

    def _build_model(self) -> NeuralNet:
        return NeuralNet(self.input_size, 1, hidden_layers=self.hidden_layers).to(self.device)

    def _train_single_target(self, X: np.ndarray, y: np.ndarray) -> tuple[NeuralNet | None, list[dict]]:
        from sklearn.model_selection import train_test_split

        if y.max() == y.min():
            return None, [{'note': 'constant_target', 'train_loss': float('nan'), 'val_loss': float('nan')}]

        stratify = y if len(np.unique(y)) > 1 else None
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify
        )

        model = self._build_model()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        X_train_tensor = torch.from_numpy(X_train).float().to(self.device)
        y_train_tensor = torch.from_numpy(y_train).float().unsqueeze(1).to(self.device)
        X_val_tensor = torch.from_numpy(X_val).float().to(self.device)
        y_val_tensor = torch.from_numpy(y_val).float().unsqueeze(1).to(self.device)

        train_loader = DataLoader(
            TensorDataset(X_train_tensor, y_train_tensor),
            batch_size=self.batch_size,
            shuffle=True,
        )

        if self.use_pos_weight:
            pos_count = y_train_tensor.sum()
            neg_count = y_train_tensor.shape[0] - pos_count
            if pos_count > 0:
                pos_weight = torch.clamp(neg_count / pos_count, min=1.0).to(self.device)
            else:
                pos_weight = torch.tensor(1.0, device=self.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss()

        history = []
        best_val = float('inf')
        best_state = None
        epochs_no_improve = 0

        for epoch in range(self.epochs):
            model.train()
            epoch_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                logits = model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_X.size(0)
            epoch_loss /= len(train_loader.dataset)

            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_tensor)
                val_loss = criterion(val_logits, y_val_tensor).item()

            history.append(
                {
                    'epoch': epoch,
                    'train_loss': epoch_loss,
                    'val_loss': val_loss,
                }
            )

            if val_loss + 1e-5 < best_val or best_state is None:
                best_val = val_loss
                best_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= 10:
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        return model, history

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.DataFrame):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.DataFrame):
            self.target_names = list(y.columns)
            y = y.values
        elif y.ndim == 1:
            self.target_names = ['target']
            y = y[:, None]
        else:
            self.target_names = [f"target_{i}" for i in range(y.shape[1])]

        X = np.asarray(X, dtype=np.float32)
        if self.normalize_inputs:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X).astype(np.float32)
        else:
            self.scaler = None

        self.models = []
        self.training_history = []

        for idx, target_name in enumerate(self.target_names):
            model, history = self._train_single_target(X, y[:, idx].astype(np.float32))
            self.models.append(model)
            self.training_history.append({'target': target_name, 'history': history})

    def predict(self, X: np.ndarray | pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.asarray(X, dtype=np.float32)
        if self.normalize_inputs and self.scaler is not None:
            X = self.scaler.transform(X).astype(np.float32)

        preds = np.zeros((X.shape[0], len(self.models)))
        probs = np.zeros_like(preds)

        for idx, model in enumerate(self.models):
            if model is None:
                continue
            model.eval()
            with torch.no_grad():
                logits = model(torch.from_numpy(X).float().to(self.device)).squeeze(-1)
                prob = torch.sigmoid(logits).cpu().numpy()
                probs[:, idx] = prob
                preds[:, idx] = (prob > 0.5).astype(int)
        return preds, probs

    def save(self, output_path):
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            'models': [],
            'config': {
                'input_size': self.input_size,
                'output_size': self.output_size,
                'hidden_layers': self.hidden_layers,
            },
            'training_history': self.training_history,
            'target_names': self.target_names,
            'scaler': None,
        }
        for model in self.models:
            if model is None:
                checkpoint['models'].append(None)
            else:
                checkpoint['models'].append(model.state_dict())

        if self.normalize_inputs and self.scaler is not None:
            checkpoint['scaler'] = {
                'mean': self.scaler.mean_.tolist(),
                'scale': self.scaler.scale_.tolist(),
                'var': getattr(self.scaler, 'var_', np.square(self.scaler.scale_)).tolist(),
                'n_features_in': getattr(self.scaler, 'n_features_in_', len(self.scaler.scale_)),
            }

        torch.save(checkpoint, output_path / 'per_gene_nn.pt')

        if self.training_history:
            history_df = pd.DataFrame(
                [
                    {
                        'target': entry['target'],
                        'epoch': h['epoch'],
                        'train_loss': h['train_loss'],
                        'val_loss': h['val_loss'],
                    }
                    for entry in self.training_history
                    for h in entry.get('history', [])
                ]
            )
            if not history_df.empty:
                history_df.to_csv(output_path / 'training_curve.csv', index=False)

    def load(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        self.target_names = checkpoint.get('target_names', [])
        self.models = []
        for state_dict in checkpoint.get('models', []):
            if state_dict is None:
                self.models.append(None)
            else:
                model = self._build_model()
                model.load_state_dict(state_dict)
                model.eval()
                self.models.append(model)
        self.training_history = checkpoint.get('training_history', [])
        scaler_state = checkpoint.get('scaler')
        if scaler_state:
            self.scaler = StandardScaler()
            self.scaler.mean_ = np.array(scaler_state['mean'], dtype=np.float32)
            self.scaler.scale_ = np.array(scaler_state['scale'], dtype=np.float32)
            self.scaler.var_ = np.array(scaler_state.get('var', np.square(self.scaler.scale_)), dtype=np.float32)
            self.scaler.n_features_in_ = scaler_state.get('n_features_in', self.scaler.mean_.shape[0])
        else:
            self.scaler = None

    def feature_importance(self):
        return None


class MultitaskMutationNet(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_layers, dropout: float, head_layers=None):
        super().__init__()
        layers = []
        prev = input_size
        for hidden in hidden_layers:
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.LayerNorm(hidden))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = hidden
        self.feature_extractor = nn.Sequential(*layers)
        head_layers = head_layers or []
        head_modules = []
        head_prev = prev
        for head_hidden in head_layers:
            head_modules.append(nn.Linear(head_prev, head_hidden))
            head_modules.append(nn.LayerNorm(head_hidden))
            head_modules.append(nn.GELU())
            if dropout > 0:
                head_modules.append(nn.Dropout(dropout))
            head_prev = head_hidden
        head_modules.append(nn.Linear(head_prev, output_size))
        self.head = nn.Sequential(*head_modules)

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.head(features)

    def encoder_state_dict(self):
        return self.feature_extractor.state_dict()

    def load_encoder_state(self, state_dict):
        self.feature_extractor.load_state_dict(state_dict)
    
    def get_head_weights(self):
        """
        Extract the weights of the final linear layer in the head.
        
        Returns:
            torch.Tensor: Weight matrix of shape (output_size, hidden_size)
                         where hidden_size is the last dimension before the output layer
        """
        # Find the last Linear layer in the head Sequential
        for module in reversed(self.head):
            if isinstance(module, nn.Linear):
                return module.weight.detach().clone()
        raise ValueError("No Linear layer found in head")


class MultitaskMutationModel(BaseModel):
    def __init__(self, input_size: int, output_size: int, **kwargs):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = kwargs.get('hidden_layers', [512, 256, 128])
        self.head_layers = kwargs.get('head_layers', [])
        self.dropout = kwargs.get('dropout_rate', 0.2)
        self.lr = kwargs.get('learning_rate', 5e-4)
        self.weight_decay = kwargs.get('weight_decay', 1e-4)
        self.batch_size = kwargs.get('batch_size', 128)
        self.epochs = kwargs.get('epochs', 60)
        self.val_split = kwargs.get('validation_split', 0.15)
        self.patience = kwargs.get('patience', 10)
        self.gradient_clip = kwargs.get('gradient_clip', 1.0)
        self.use_pos_weight = kwargs.get('use_pos_weight', True)
        self.gene_weight_by_prevalence = kwargs.get('gene_weight_by_prevalence', False)
        self.pretrained_encoder_path = kwargs.get('pretrained_encoder_path')
        self.freeze_encoder = kwargs.get('freeze_encoder', False)
        self.normalize_inputs = kwargs.get('normalize_inputs', True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.training_history = []
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scaler: StandardScaler | None = None
        self.gene_weights: torch.Tensor | None = None

    def _build_network(self):
        net = MultitaskMutationNet(
            input_size=self.input_size,
            output_size=self.output_size,
            hidden_layers=self.hidden_layers,
            dropout=self.dropout,
            head_layers=self.head_layers
        )
        if self.pretrained_encoder_path:
            encoder_path = Path(self.pretrained_encoder_path)
            if encoder_path.exists():
                state = torch.load(encoder_path, map_location='cpu')
                net.load_encoder_state(state)
                if self.freeze_encoder:
                    for param in net.feature_extractor.parameters():
                        param.requires_grad = False
        return net

    def _prepare_dataloaders(self, X: np.ndarray, y: np.ndarray):
        from sklearn.model_selection import train_test_split

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        if not np.isfinite(X).all():
            raise ValueError("Input features contain NaNs or infs. Please clean the data before training.")
        if not np.isfinite(y).all():
            raise ValueError("Target matrix contains NaNs or infs.")

        if self.normalize_inputs:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X).astype(np.float32)
        else:
            self.scaler = None

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=self.val_split,
            random_state=42,
            shuffle=True
        )

        X_train_tensor = torch.from_numpy(X_train)
        y_train_tensor = torch.from_numpy(y_train)
        X_val_tensor = torch.from_numpy(X_val)
        y_val_tensor = torch.from_numpy(y_val)

        train_loader = DataLoader(
            TensorDataset(X_train_tensor, y_train_tensor),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False
        )
        val_loader = DataLoader(
            TensorDataset(X_val_tensor, y_val_tensor),
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False
        )

        if self.use_pos_weight:
            pos_counts = y_train_tensor.sum(dim=0)
            neg_counts = y_train_tensor.shape[0] - pos_counts
            pos_weight = torch.where(pos_counts > 0, neg_counts / pos_counts, torch.ones_like(pos_counts))
            pos_weight = torch.clamp(pos_weight, min=1.0, max=1e6)
            pos_weight = pos_weight.to(self.device)
        else:
            pos_weight = None

        # Compute gene weights based on mutation prevalence if enabled
        if self.gene_weight_by_prevalence:
            # Gene weight = mutation frequency (proportion of samples with mutation)
            # More frequent mutations get higher weight
            mutation_freq = y_train_tensor.mean(dim=0)  # Shape: (n_genes,)
            # Normalize to sum to n_genes (so average weight is 1.0)
            # This ensures the overall loss scale doesn't change dramatically
            if mutation_freq.sum() > 0:
                gene_weights = mutation_freq / mutation_freq.mean()
            else:
                gene_weights = torch.ones_like(mutation_freq)
            # Clamp to reasonable range to avoid extreme weights
            gene_weights = torch.clamp(gene_weights, min=0.1, max=10.0)
            self.gene_weights = gene_weights.to(self.device)
        else:
            self.gene_weights = None

        return train_loader, val_loader, pos_weight

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.DataFrame):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.DataFrame):
            y = y.values

        train_loader, val_loader, pos_weight = self._prepare_dataloaders(X, y)
        self.model = self._build_network().to(self.device)
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        
        # Use weighted loss if gene weighting is enabled
        if self.gene_weight_by_prevalence and self.gene_weights is not None:
            self.criterion = WeightedBCEWithLogitsLoss(
                gene_weights=self.gene_weights,
                pos_weight=pos_weight
            )
        else:
            # Standard BCE loss with optional pos_weight
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) if pos_weight is not None else nn.BCEWithLogitsLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=max(1, self.patience // 3),
            verbose=False
        )

        best_val = float('inf')
        epochs_no_improve = 0
        self.training_history = []
        best_state = None

        for epoch in range(self.epochs):
            train_loss = self._run_epoch(train_loader, train=True)
            val_loss = self._run_epoch(val_loader, train=False)
            scheduler.step(val_loss)

            self.training_history.append(
                {
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'lr': self.optimizer.param_groups[0]['lr']
                }
            )

            if val_loss + 1e-5 < best_val or best_state is None:
                best_val = val_loss
                epochs_no_improve = 0
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= self.patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

    def fit_full_dataset(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.DataFrame):
        """
        Train the model on the full dataset without validation split.
        Useful for extracting learned representations.
        
        Args:
            X: Input features
            y: Target labels
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.DataFrame):
            y = y.values

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        if not np.isfinite(X).all():
            raise ValueError("Input features contain NaNs or infs. Please clean the data before training.")
        if not np.isfinite(y).all():
            raise ValueError("Target matrix contains NaNs or infs.")

        if self.normalize_inputs:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X).astype(np.float32)
        else:
            self.scaler = None

        X_tensor = torch.from_numpy(X)
        y_tensor = torch.from_numpy(y)

        train_loader = DataLoader(
            TensorDataset(X_tensor, y_tensor),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False
        )

        # Compute pos_weight for full dataset
        if self.use_pos_weight:
            pos_counts = y_tensor.sum(dim=0)
            neg_counts = y_tensor.shape[0] - pos_counts
            pos_weight = torch.where(pos_counts > 0, neg_counts / pos_counts, torch.ones_like(pos_counts))
            pos_weight = torch.clamp(pos_weight, min=1.0, max=1e6)
            pos_weight = pos_weight.to(self.device)
        else:
            pos_weight = None

        # Compute gene weights if enabled
        if self.gene_weight_by_prevalence:
            mutation_freq = y_tensor.mean(dim=0)
            if mutation_freq.sum() > 0:
                gene_weights = mutation_freq / mutation_freq.mean()
            else:
                gene_weights = torch.ones_like(mutation_freq)
            gene_weights = torch.clamp(gene_weights, min=0.1, max=10.0)
            self.gene_weights = gene_weights.to(self.device)
        else:
            self.gene_weights = None

        # Build model and setup training
        self.model = self._build_network().to(self.device)
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        
        # Use weighted loss if gene weighting is enabled
        if self.gene_weight_by_prevalence and self.gene_weights is not None:
            self.criterion = WeightedBCEWithLogitsLoss(
                gene_weights=self.gene_weights,
                pos_weight=pos_weight
            )
        else:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) if pos_weight is not None else nn.BCEWithLogitsLoss()

        # Train on full dataset
        self.training_history = []
        for epoch in range(self.epochs):
            train_loss = self._run_epoch(train_loader, train=True)
            self.training_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'lr': self.optimizer.param_groups[0]['lr']
            })

    def get_head_weights(self):
        """
        Extract the weights of the final linear layer in the head.
        
        Returns:
            torch.Tensor: Weight matrix of shape (output_size, hidden_size)
                         where hidden_size is the last dimension before the output layer
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() or fit_full_dataset() first.")
        return self.model.get_head_weights()

    def _run_epoch(self, loader: DataLoader, train: bool = True) -> float:
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_samples = 0
        with torch.set_grad_enabled(train):
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                if train:
                    self.optimizer.zero_grad()

                logits = self.model(batch_X)
                loss = self.criterion(logits, batch_y)

                if train:
                    loss.backward()
                    if self.gradient_clip is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                    self.optimizer.step()

                total_loss += loss.item() * batch_X.size(0)
                total_samples += batch_X.size(0)

        return total_loss / max(total_samples, 1)

    def predict(self, X: np.ndarray | pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.asarray(X, dtype=np.float32)
        if self.normalize_inputs and self.scaler is not None:
            X = self.scaler.transform(X).astype(np.float32)
        X_tensor = torch.from_numpy(X)
        loader = DataLoader(TensorDataset(X_tensor), batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        all_logits = []
        with torch.no_grad():
            for (batch_X,) in loader:
                batch_X = batch_X.to(self.device)
                logits = self.model(batch_X)
                all_logits.append(logits.cpu())

        logits = torch.cat(all_logits, dim=0)
        probs = torch.sigmoid(logits).numpy()
        preds = (probs > 0.5).astype(int)
        return preds, probs

    def save(self, output_path):
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            'model_state': self.model.state_dict(),
            'config': {
                'input_size': self.input_size,
                'output_size': self.output_size,
                'hidden_layers': self.hidden_layers,
                'head_layers': self.head_layers,
                'dropout': self.dropout
            },
            'training_history': self.training_history,
            'scaler': None
        }
        if self.normalize_inputs and self.scaler is not None:
            checkpoint['scaler'] = {
                'mean': self.scaler.mean_.tolist(),
                'scale': self.scaler.scale_.tolist(),
                'var': getattr(self.scaler, 'var_', np.square(self.scaler.scale_)).tolist(),
                'n_features_in': getattr(self.scaler, 'n_features_in_', len(self.scaler.scale_))
            }
        torch.save(checkpoint, output_path / 'multitask_nn.pt')
        torch.save(self.model.feature_extractor.state_dict(), output_path / 'encoder.pt')

        if self.training_history:
            history_df = pd.DataFrame(self.training_history)
            history_df.to_csv(output_path / 'training_curve.csv', index=False)

    def load(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = self._build_network().to(self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.training_history = checkpoint.get('training_history', [])
        scaler_state = checkpoint.get('scaler')
        if scaler_state:
            self.scaler = StandardScaler()
            self.scaler.mean_ = np.array(scaler_state['mean'], dtype=np.float32)
            self.scaler.scale_ = np.array(scaler_state['scale'], dtype=np.float32)
            self.scaler.var_ = np.array(scaler_state.get('var', np.square(self.scaler.scale_)), dtype=np.float32)
            self.scaler.n_features_in_ = scaler_state.get('n_features_in', self.scaler.mean_.shape[0])
        else:
            self.scaler = None
        self.model.eval()

    def load_pretrained_encoder(self, encoder_path: str, freeze: bool = False):
        encoder_state = torch.load(encoder_path, map_location='cpu')
        if self.model is None:
            self.model = self._build_network().to(self.device)
        self.model.load_encoder_state(encoder_state)
        if freeze:
            for param in self.model.feature_extractor.parameters():
                param.requires_grad = False

    def feature_importance(self):
        return None


class ModelFactory:
    def __init__(self):
        self._model_registry: Dict[str, type] = {
            'logistic_regression': LogisticRegression,
            'random_forest': RandomForestClassifier,
            'svm': SVC,
        }
        if XGBClassifier is not None:
            self._model_registry['xgboost'] = XGBClassifier
        if LGBMClassifier is not None:
            self._model_registry['lightgbm'] = LGBMClassifier

    def get_model(self, model_name: str, input_size: int, output_size: int, config: dict | None = None):
        config = config or {}
        model_section = config.get('model', {}) or {}
        model_config = model_section.get(model_name) or {}

        if model_name in self._model_registry:
            model_cls = self._model_registry[model_name]
            return SklearnModel(model_cls, output_size=output_size, **model_config)
        if model_name == 'neural_net':
            return NeuralNetModel(input_size=input_size, output_size=output_size, **model_config)
        if model_name == 'multitask_nn':
            return MultitaskMutationModel(input_size=input_size, output_size=output_size, **model_config)

        available_options = sorted(set(self._model_registry.keys()) | {'neural_net', 'multitask_nn'})
        available = ', '.join(available_options)
        raise ValueError(f"Unsupported model type: {model_name}. Available options: {available}")

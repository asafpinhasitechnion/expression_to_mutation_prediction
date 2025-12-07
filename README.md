# Mutation Prediction from Gene Expression

A machine learning framework for predicting gene mutations from gene expression data using TCGA (The Cancer Genome Atlas) datasets.

## Features

- **Multi-label classification**: Predict mutations for multiple genes simultaneously
- **Multiple model types**: Random Forest, XGBoost, LightGBM, Neural Networks (single-task and multitask)
- **Comprehensive evaluation**: Per-gene metrics including accuracy, precision, recall, F1, ROC-AUC, AUPRC, specificity, and MCC
- **Interpretation tools**: 
  - Gene ablation analysis (evaluate impact of removing each gene)
  - SHAP analysis for tree-based models (feature importance)
- **Per-cancer-type analysis**: Separate analysis for different cancer types
- **Flexible evaluation**: Train/test split or K-fold cross-validation

## Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd mutation_prediction
```

### 2. Create a virtual environment

**Using conda (recommended):**
```bash
conda create -n mutation_prediction python=3.10
conda activate mutation_prediction
```

**Using venv:**
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install optional dependencies (if needed)

For XGBoost:
```bash
pip install xgboost
```

For LightGBM:
```bash
pip install lightgbm
```

For SHAP analysis:
```bash
pip install shap
```

### 5. Configure data paths

Edit `config/config.yaml` to set your data paths:

```yaml
data:
  expression_path: "path/to/expression/data"
  mutation_path: "path/to/mutation/data"
  gene_name_mapping_path: "path/to/gene/mapping"
  gene_annotation_path: "path/to/gene/annotation"
  cancer_types: []  # Empty list loads all available cohorts
```

## Usage

### Command Line Interface

**Basic training and evaluation:**
```bash
python -m src.main --config config/config.yaml
```

**Train on specific cancer types:**
```bash
python -m src.main --cancer-types BRCA LUAD
```

**K-fold cross-validation:**
```bash
python -m src.main --eval-mode kfold
```

**Run ablation analysis:**
```bash
python -m src.main --ablation
```

**Extract head weights from multitask model:**
```bash
python -m src.main --extract-weights
```

### Jupyter Notebook

Open `notebooks/data_exploration.ipynb` for interactive exploration:

1. Start Jupyter:
```bash
jupyter notebook
```

2. Open `notebooks/data_exploration.ipynb`

3. Modify the `selected_cohorts` variable to choose cancer types

4. Run cells sequentially

## Project Structure

```
mutation_prediction/
├── config/
│   └── config.yaml          # Configuration file
├── src/
│   ├── main.py              # Main entry point
│   ├── preprocessing/       # Data loading and preprocessing
│   ├── models/              # Model implementations
│   ├── training/            # Training functions
│   ├── evaluation/          # Metrics computation
│   ├── interpretation/      # Ablation and SHAP analysis
│   ├── visualization/       # Plotting functions
│   └── features/            # Feature selection
├── notebooks/               # Jupyter notebooks
├── cache/                   # Cached processed data
├── results/                 # Output results
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Configuration

The `config/config.yaml` file controls:

- **Data paths**: Where to find expression and mutation data
- **Model selection**: Which model to use (random_forest, xgboost, lightgbm, neural_net, multitask_nn)
- **Model hyperparameters**: Learning rate, hidden layers, etc.
- **Feature selection**: Method and number of features
- **Evaluation**: Metrics to compute, CV folds, etc.

## Models

### Tree-based models
- **Random Forest**: Multi-output classifier with RandomForestClassifier
- **XGBoost**: Gradient boosting (requires `xgboost` package)
- **LightGBM**: Fast gradient boosting (requires `lightgbm` package)

### Neural Networks
- **neural_net**: Single-task per-gene models (separate NN for each gene)
- **multitask_nn**: Shared encoder + per-gene heads (more efficient)

## Output

Results are saved in the `results/` directory:

- **Per-gene metrics**: CSV files with metrics for each gene
- **Predictions**: Predicted labels and probabilities
- **Visualizations**: Heatmaps, clustermaps, SHAP plots
- **Ablation results**: Difference matrices showing impact of gene removal

## Troubleshooting

### Import errors
Make sure you're running from the project root and have activated your virtual environment.

### Data not found
Check that data paths in `config/config.yaml` are correct. The data loader will attempt to download missing files if `download_missing: true` is set.

### CUDA/GPU issues
PyTorch will automatically use CPU if CUDA is not available. To force CPU:
```python
import torch
torch.set_default_tensor_type('torch.FloatTensor')
```

## License

[Add your license here]

## Citation

[Add citation information if applicable]


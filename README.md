# Cars4You — Used Car Price Prediction

This project was developed for the Machine Learning course at NOVA IMS (Masters in Data Science and Advanced Analytics). The purpose is to predict used car prices; we approached the problem with a DeepFM model for tabular regression. In brief, we built a canonical brand/model reference, cleaned and engineered features, prepared train/test CSVs, tuned DeepFM hyperparameters with Optuna, trained the model, and generated predictions and a competition-style submission.

# Notebooks Guide

This guide explains what each notebook in the `notebooks/` folder does and the recommended order to run them. It also points to the training and prediction scripts that the notebooks feed into.

## Environment Setup (Windows, PowerShell)
- Prerequisite: Python `3.10` recommended.
- Create and activate a virtual environment:
  - `cd C:\Users\vehnie\Documents\Master\Machine Learning\Cars4You`
  - `python -m venv .venv`
  - `.venv\Scripts\Activate.ps1`
  - If activation is blocked, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`
- Upgrade `pip` and install dependencies:
  - `python -m pip install --upgrade pip`
  - `pip install -r requirements.txt`
- Verify key packages:
  - `python -c "import torch, optuna, deepctr_torch; print('OK')"`
- CUDA note for PyTorch:
  - If `torch==2.9.0+cu128` fails to install via PyPI, install the matching CUDA build from PyTorch’s wheel index: `pip install --extra-index-url https://download.pytorch.org/whl/cu128 torch==2.9.0+cu128`
  - Or use CPU-only: `pip install torch==2.9.0+cpu`

## Overview
- Build a clean brand/model reference.
- Clean and prepare training data (CSV) from your notebook DataFrame.
- Train DeepFM with Optuna-best parameters and early stopping.
- Clean and prepare test data.
- Generate predictions and a competition-style submission.

## Notebooks

- `notebook_models.ipynb`
  - Purpose: Constructs a canonical brand/model lookup by merging `data/df_fixed_typos.csv` and `data/car_models.csv` and saves `data/all_car_models.csv`.
  - Output: `data/all_car_models.csv` used downstream for harmonizing brand/model.
  - Note: `data/df_fixed_typos.csv` is preserved so its cleaned entries are included in `all_car_models.csv`. This lets us reuse the training-time brand/model corrections when cleaning the test set, ensuring consistent harmonization.

- `notebook_1.ipynb`
  - Purpose: Main data cleaning and feature engineering for training. Includes:
    - Numeric imputation, categorical imputation (RFC-based), typo fixes, fuzzy matching for `brand` and `model`, harmonization of categorical values, and feature derivations (e.g., age).
    - Produces a cleaned training DataFrame suitable for DeepFM.
  - Outputs:
    - Saves the cleaned training DataFrame to CSV (e.g., `notebooks/df_train.csv`) to feed the DeepFM trainer.
    - Creates imputation/encoding artifacts under `notebooks/assets/` (used later by `test/test.ipynb`): see “Artifacts” section.

- `test/test.ipynb`
  - Purpose: Applies the same cleaning logic to the test dataset. Loads/uses the RFC model artifacts (if available) and canonical brand/model mapping to produce a model-ready test CSV.
  - Output: `notebooks/test/test_processed.csv` used by the prediction script.

## Training Script (Python)
- `train_deepfm.py`
  - Purpose: Trains the DeepFM model on a CSV and saves artifacts.
  - Inputs: `--csv <path-to-cleaned-train-csv>` (from `notebook_1.ipynb`).
  - Behavior: Uses Optuna-best defaults, early stopping, and a ReduceLROnPlateau scheduler. Records MAE per epoch and saves a plot.
  - Key outputs (in `notebooks/assets_deepfm/`):
    - `mae_history.csv` — Train/Val MAE per epoch.
    - `mae_curve.png` — Plot of Train vs Val MAE across epochs.
    - `deepfm_best.pt` — Best model weights.
    - `config.json` — Final metrics and artifact paths.
    - Preprocessing artifacts — label encoders/scaler.

## Prediction Script (Python)
- `test/predictions.py`
  - Purpose: Loads the trained DeepFM checkpoint and produces predictions/submission using `test_processed.csv`.
  - Inputs: `notebooks/test/test_processed.csv`, `notebooks/assets_deepfm/*` artifacts.
  - Outputs: `notebooks/assets_deepfm/predictions.csv` and `notebooks/assets_deepfm/submission.csv`.

## Recommended Run Order
1. Run `notebook_models.ipynb` to create `data/all_car_models.csv`.
2. Run `notebook_1.ipynb`, then save the cleaned training DataFrame to a CSV (e.g., `notebooks/df_train.csv`).
3. Train DeepFM using the CSV from step 2:
   - CLI: `python notebooks/train_deepfm.py --csv notebooks/df_train.csv`
   - Or from a notebook: `from train_deepfm import main as train_main; train_main(csv_path='notebooks/df_train.csv')`
4. Run `test/test.ipynb` to produce `notebooks/test/test_processed.csv`.
5. Generate predictions:
   - CLI/Notebook: `python notebooks/test/predictions.py` (paths are preconfigured to `assets_deepfm` and `test_processed.csv`).

## Run Examples
- Optuna hyperparameter search (basic):
  - PowerShell: `$env:OPTUNA_TRIALS=25; $env:OPTUNA_EPOCHS=40; $env:OPTUNA_PATIENCE=12; python notebooks/optimize_deepfm_optuna.py`
  - Freeze choices example: `$env:FIX_EMBED_DIM=16; $env:FIX_DNN_DIMS='256,128,64'; python notebooks/optimize_deepfm_optuna.py`
  - Note: `train_deepfm.py` uses in-file configuration. To let Optuna control training parameters, either temporarily wire those env vars in `train_deepfm.py` or adapt `optimize_deepfm_optuna.py` to write a config JSON that `train_deepfm.py` reads.

- Train DeepFM on the cleaned CSV:
  - PowerShell: `python notebooks/train_deepfm.py --csv "C:\Users\vehnie\Documents\Master\Machine Learning\Cars4You\notebooks\df_train.csv"`
  - Edit training hyperparameters directly in `notebooks/train_deepfm.py` (`embedding_dim`, `dnn_hidden_units`, `dropout`, `lr`, `weight_decay`, `epochs`, `batch_size`, `early_stop_enabled`, `patience`).

- Generate predictions from processed test CSV:
  - PowerShell (paths preconfigured): `python notebooks/test/predictions.py`
  - Ensure `notebooks/test/test_processed.csv` exists and `notebooks/assets_deepfm/` contains `deepfm_best.pt`, `label_encoders.pkl`, and `scaler.pkl`.

## Artifacts
- `notebooks/assets/` (created by `notebook_1.ipynb`):
  - `rfc_model.joblib` — RandomForestClassifier trained for imputing missing categorical values.
  - `rfc_model_cols.joblib` — Column list used to align one-hot or dummy features when imputing with the RFC.
  - `train_ohe_cols.joblib` — Saved training one-hot/dummy column names for consistent encoding at inference.

- `notebooks/assets_deepfm/` (created by `train_deepfm.py` and `test/predictions.py`):
  - `deepfm_best.pt` — Best-performing DeepFM weights (restored after early stopping).
  - `label_encoders.pkl` — Fitted label encoders for categorical features used by DeepFM.
  - `scaler.pkl` — Fitted scaler for dense features (Standard/Robust, depending on settings).
  - `mae_history.csv` — Epoch-by-epoch Train/Val MAE history.
  - `mae_curve.png` — Line chart of Train vs Val MAE across epochs.
  - `config.json` — Final configuration, paths to artifacts, and summary metrics.
  - `predictions.csv` — Per-row predictions with selected columns for inspection.
  - `submission.csv` — Competition-style file with `carID` and predicted `price`.

- Optuna outputs (created by `optimize_deepfm_optuna.py`, if used):
  - `optuna_best.json` — Best hyperparameters found by the study.
  - `optuna_results_smoke.csv` — Trial results CSV (name configurable via env vars).
  - `optuna_trial_*.out.txt`, `optuna_trial_*.err.txt` — Trial logs.

## Temporary CSVs
- `notebooks/df_train.csv` — Cleaned training CSV exported from `notebook_1.ipynb` for DeepFM training.
- `notebooks/test/test_processed.csv` — Cleaned test CSV exported from `test/test.ipynb` for prediction.
- `data/all_car_models.csv` — Canonical brand/model lookup built by `notebook_models.ipynb` (referenced by notebooks for harmonization).
- `data/df_fixed_typos.csv` — Cleaned brand/model entries from training, kept so they are incorporated into `all_car_models.csv` and reused during test cleaning.
- `data/filled_cars.csv` — Training dataset after RFC-based categorical imputation. Persisted to avoid re-running the heavy RFC imputation step; notebooks can load this to skip recomputation.
 - `data/avg_model_prices.csv` — Brand/model average price lookup saved for later use in test-set imputation/augmentation (merged to provide an `avg_price` feature during test cleaning).

## Notes
- If you prefer to train directly on `data/train.csv`, `train_deepfm.py` can handle its own preprocessing. For notebook integration and consistent cleaning, use the CSV produced by `notebook_1.ipynb`.
- After training, you can display the MAE curve in any notebook:
  - `from IPython.display import Image; Image(filename='notebooks/assets_deepfm/mae_curve.png')`
 - Adjust training settings by editing variables inside `notebooks/train_deepfm.py` (`main`).

## References
- DeepFM (Guo et al., 2017, IJCAI’17): A factorization-machine based neural network originally proposed for Click-Through-Rate (CTR) prediction; shown competitive on tabular tasks (e.g., Borisov et al., 2022).
- ProbSAINT: Probabilistic Tabular Regression for Used Car Pricing — https://arxiv.org/html/2403.03812v1
- Improving Car Price Predictions by Identifying Key Features — https://sesjournal.org/index.php/1/article/view/104/724
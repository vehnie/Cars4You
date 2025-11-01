import os
import json
import copy
import argparse
import random
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import DeepFM
import joblib


def resolve_path(path_str: str) -> str:
    """Resolve Windows paths like '/c:/.../file.csv' to 'C:\\...\\file.csv'."""
    if not path_str:
        return path_str
    p = path_str.strip()
    if p.lower().startswith('/c:/'):
        p = 'C:' + p[2:]  # '/c:/Users/..' -> 'C:/Users/..'
    # Normalize to OS-specific separators
    return os.path.normpath(p)


def load_training_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Merge brand-model average price prior if available
    notebooks_dir = os.path.dirname(__file__)
    avg_price_path = os.path.normpath(os.path.join(notebooks_dir, '..', 'data', 'avg_model_prices.csv'))
    if os.path.exists(avg_price_path):
        avg_lookup = pd.read_csv(avg_price_path)
        if {'brand', 'model', 'avg_price'}.issubset(avg_lookup.columns):
            df = df.merge(avg_lookup, on=['brand', 'model'], how='left')
            df['avg_price'] = df['avg_price'].fillna(0)
        else:
            df['avg_price'] = 0
    else:
        df['avg_price'] = 0

    return df


def main(args: argparse.Namespace):
    # Paths and assets
    input_csv = resolve_path(args.csv)
    notebooks_dir = os.path.dirname(__file__)
    assets_dir = os.path.join(notebooks_dir, 'assets_deepfm')
    os.makedirs(assets_dir, exist_ok=True)

    # Read data
    df = load_training_df(input_csv)

    # Features
    target = 'price'
    sparse_features = ['brand', 'model', 'fuel_type', 'transmission']
    dense_features = ['mileage', 'mpg', 'engine_size', 'age', 'avg_price']

    # Keep only necessary columns and drop rows with missing values
    df = df[sparse_features + dense_features + [target]].dropna()

    # Label encode sparse features
    encoders = {}
    for feat in sparse_features:
        lbe = LabelEncoder()
        df[feat] = lbe.fit_transform(df[feat])
        encoders[feat] = lbe

    # Deterministic split
    seed = int(os.getenv('SEED', '42'))
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 80/10/10 split
    X_temp, X_test, y_temp, y_test = train_test_split(
        df[sparse_features + dense_features],
        df[target],
        test_size=0.1,
        random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=0.1111,  # ~10% of total
        random_state=seed
    )

    # Optional log1p transforms for selected dense features (comma-separated names)
    log_dense = os.getenv('LOG_DENSE', '')
    cols_to_log = [c.strip() for c in log_dense.split(',') if c.strip()]
    for col in cols_to_log:
        if col in dense_features:
            X_train[col] = np.log1p(np.clip(X_train[col], a_min=0, a_max=None))
            X_val[col] = np.log1p(np.clip(X_val[col], a_min=0, a_max=None))
            X_test[col] = np.log1p(np.clip(X_test[col], a_min=0, a_max=None))

    # Scale dense features
    scaler = StandardScaler()
    X_train[dense_features] = scaler.fit_transform(X_train[dense_features])
    X_val[dense_features] = scaler.transform(X_val[dense_features])
    X_test[dense_features] = scaler.transform(X_test[dense_features])

    # Log-transform target to stabilize training
    y_train_log = np.log1p(y_train.values)
    y_val_log = np.log1p(y_val.values)
    y_test_log = np.log1p(y_test.values)

    # Persist preprocessing artifacts
    try:
        joblib.dump(encoders, os.path.join(assets_dir, 'label_encoders.pkl'))
        joblib.dump(scaler, os.path.join(assets_dir, 'scaler.pkl'))
    except Exception as e:
        print(f'Warning: failed to save encoders/scaler: {e}')

    # Hyperparameters via env vars
    embedding_dim = int(os.getenv('EMBED_DIM', '8'))
    dnn_dims_str = os.getenv('DNN_DIMS', '256,128,64')
    dnn_hidden_units = tuple(int(x) for x in dnn_dims_str.split(','))
    dropout = float(os.getenv('DROPOUT', '0.3'))
    lr = float(os.getenv('LR', '0.001'))
    weight_decay = float(os.getenv('WEIGHT_DECAY', '1e-5'))
    patience = int(os.getenv('PATIENCE', '10'))
    early_stop_enabled = os.getenv('EARLY_STOP', '1') == '1'

    # Feature columns
    fixlen_feature_columns = (
        [SparseFeat(feat, vocabulary_size=df[feat].nunique(), embedding_dim=embedding_dim) for feat in sparse_features] +
        [DenseFeat(feat, 1) for feat in dense_features]
    )
    feature_names = get_feature_names(fixlen_feature_columns)

    # Model inputs
    train_model_input = {name: X_train[name] for name in feature_names}
    val_model_input = {name: X_val[name] for name in feature_names}
    test_model_input = {name: X_test[name] for name in feature_names}

    # Device and model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using device:', device)
    model = DeepFM(
        linear_feature_columns=fixlen_feature_columns,
        dnn_feature_columns=fixlen_feature_columns,
        task='regression',
        dnn_hidden_units=dnn_hidden_units,
        dnn_dropout=dropout,
        device=device
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.compile(optimizer, 'mse', metrics=['mae'])

    # Train loop with early stopping on validation MAE (in original scale)
    epochs = int(os.getenv('EPOCHS', '100'))
    batch_size = 512
    train_mae_list = []
    val_mae_list = []
    best_val_mae = float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.fit(
            train_model_input,
            y_train_log,
            batch_size=batch_size,
            epochs=1,
            verbose=0,
            validation_data=(val_model_input, y_val_log)
        )

        train_pred_log = model.predict(train_model_input, batch_size=batch_size)
        val_pred_log = model.predict(val_model_input, batch_size=batch_size)

        train_pred = np.expm1(train_pred_log)
        val_pred = np.expm1(val_pred_log)

        train_mae = mean_absolute_error(y_train.values, train_pred)
        val_mae = mean_absolute_error(y_val.values, val_pred)

        train_mae_list.append(train_mae)
        val_mae_list.append(val_mae)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        print(f'Epoch {epoch+1}/{epochs} - MAE train: {train_mae:.4f}, val: {val_mae:.4f}')

        if early_stop_enabled and no_improve >= patience:
            print(f'Early stopping at epoch {epoch+1} (best val MAE: {best_val_mae:.4f})')
            break

    # Restore best state
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final test MAE
    test_pred_log = model.predict(test_model_input, batch_size=batch_size)
    test_pred = np.expm1(test_pred_log)
    final_test_mae = mean_absolute_error(y_test.values, test_pred)
    print(f'Final Test MAE (best model): {final_test_mae:.4f}')
    print(f'Final Val MAE (best model): {best_val_mae:.4f}')

    # Save model and config
    model_path = os.path.join(assets_dir, 'deepfm_best.pt')
    try:
        torch.save(model.state_dict(), model_path)
        with open(os.path.join(assets_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'lr': lr,
                'weight_decay': weight_decay,
                'dropout': dropout,
                'embedding_dim': embedding_dim,
                'dnn_hidden_units': list(dnn_hidden_units),
                'patience': patience,
                'epochs_run': len(train_mae_list),
                'best_val_mae': float(best_val_mae),
                'final_test_mae': float(final_test_mae),
                'device': device,
                'sparse_features': sparse_features,
                'dense_features': dense_features,
                'log_dense': cols_to_log,
                'input_csv': input_csv,
            }, f, indent=2)
        print(f'Saved best model to: {model_path}')
        print(f'Saved config and preprocessors in: {assets_dir}')
    except Exception as e:
        print(f'Warning: failed to save model/config: {e}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DeepFM from CSV with aligned preprocessing')
    parser.add_argument('--csv', type=str, default=r'C:\\Users\\vehnie\\Documents\\Master\\Machine Learning\\Cars4You\\notebooks\\df_train.csv',
                        help='Absolute path to df_train.csv')
    args = parser.parse_args()
    main(args)
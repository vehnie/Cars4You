import os
import json
import copy
import argparse
import random
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import DeepFM
import joblib

# normalize the paths
def resolve_path(path_str: str) -> str:
    """Resolve Windows paths like '/c:/.../file.csv' to 'C:\\...\\file.csv'."""
    if not path_str:
        return path_str
    p = path_str.strip()
    if p.lower().startswith('/c:/'):
        p = 'C:' + p[2:]  # '/c:/Users/..' -> 'C:/Users/..'
    # Normalize to OS-specific separators
    return os.path.normpath(p)

# load training df from csv path
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

# main
def main(args: argparse.Namespace):
    # Optuna-best parameter defaults
    DEFAULT_EMBED_DIM = 8
    DEFAULT_DNN_DIMS = "512,256,128"
    DEFAULT_DROPOUT = 0.3164638436505863
    DEFAULT_LR = 0.0036528420965552437
    DEFAULT_WEIGHT_DECAY = 2.026446830830357e-06
    DEFAULT_PATIENCE = 10
    DEFAULT_BATCH_SIZE = 256
    DEFAULT_DNN_USE_BN = 1
    DEFAULT_L2_REG_LINEAR = 0.0
    DEFAULT_L2_REG_EMBEDDING = 1e-07
    DEFAULT_L2_REG_DNN = 1e-07
    DEFAULT_USE_LR_SCHEDULER = 1
    DEFAULT_LR_SCHED_FACTOR = 0.43294705814989504
    DEFAULT_LR_SCHED_PATIENCE = 6
    DEFAULT_LR_SCHED_MIN_LR = 1e-05
    DEFAULT_EPOCHS = 100

    # Paths and assets
    input_csv = resolve_path(args.csv)
    notebooks_dir = os.path.dirname(__file__)
    assets_dir = os.path.join(notebooks_dir, 'assets_deepfm')
    os.makedirs(assets_dir, exist_ok=True)

    # Read data
    df = load_training_df(input_csv)

    # Features
    target = 'price'
    # Create brand_model cross feature (string), then include it as a sparse feature
    if {'brand', 'model'}.issubset(df.columns):
        df['brand_model'] = df['brand'].astype(str) + '|' + df['model'].astype(str)
    else:
        df['brand_model'] = 'UNKNOWN|UNKNOWN'

    # Create age/mileage bin cross feature using quantile-based bins and save edges
    age_edges = None
    mileage_edges = None
    if {'age', 'mileage'}.issubset(df.columns):
        try:
            age_edges = list(np.quantile(df['age'].values, [0.2, 0.4, 0.6, 0.8]))
            mileage_edges = list(np.quantile(df['mileage'].values, [0.2, 0.4, 0.6, 0.8]))
            df['age_bin'] = np.digitize(df['age'].values, age_edges, right=False)
            df['mileage_bin'] = np.digitize(df['mileage'].values, mileage_edges, right=False)
            df['age_mileage_bin'] = df['age_bin'].astype(str) + '|' + df['mileage_bin'].astype(str)
        except Exception as e:
            print(f'Warning: failed to compute age/mileage bins: {e}')
            df['age_mileage_bin'] = '0|0'
    
    else:
        df['age_mileage_bin'] = '0|0'

    # sparse features
    sparse_features = ['brand', 'model', 'fuel_type', 'transmission', 'brand_model', 'age_mileage_bin']

    # dense features
    dense_features = ['mileage', 'mpg', 'engine_size', 'age', 'avg_price']

    # Filter to available columns to avoid KeyError when optional features are missing
    sparse_features = [f for f in sparse_features if f in df.columns]
    dense_features = [f for f in dense_features if f in df.columns]

    # mpg per engine size
    if {'mpg', 'engine_size'}.issubset(df.columns):
        df['mpg_per_engine'] = (df['mpg'].astype(float) / np.clip(df['engine_size'].astype(float), a_min=1e-6, a_max=None)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        dense_features.append('mpg_per_engine')

    # remove entries with missing target
    df = df[sparse_features + dense_features + [target]].dropna(subset=[target])

    # map infrequent categories to UNK and fill NA (lower than 5)
    rare_thr = 5
    unk_token = 'UNK'
    for feat in sparse_features:
        # Work on string representation before encoding
        df[feat] = df[feat].astype(str)
        df[feat] = df[feat].fillna(unk_token)
        vc = df[feat].value_counts()
        rare_values = set(vc[vc < rare_thr].index)
        if rare_values:
            df[feat] = df[feat].apply(lambda x: unk_token if x in rare_values else x)

    # label encode sparse features
    encoders = {}
    unk_index_per_feature = {}
    for feat in sparse_features:
        lbe = LabelEncoder()
        df[feat] = lbe.fit_transform(df[feat])
        encoders[feat] = lbe
        classes = list(getattr(lbe, 'classes_', []))
        # Record index of UNK token if present, otherwise default to last index
        if unk_token in classes:
            unk_index_per_feature[feat] = int(classes.index(unk_token))
        else:
            unk_index_per_feature[feat] = int(len(classes) - 1)

    # split
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using device:', device)

    # 80/10/10 split
    X_temp, X_test, y_temp, y_test = train_test_split(
        df[sparse_features + dense_features],
        df[target],
        test_size=0.1,
        random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=0.1111,
        random_state=seed
    )

    # log1p transforms for selected dense features
    cols_to_log = []

    augment_log = False
    dense_features_used = list(dense_features)
    if augment_log:
        for col in cols_to_log:
            if col in dense_features and (col + '_log') not in dense_features_used:
                dense_features_used.append(col + '_log')

        # Create log duplicates
        for col in cols_to_log:
            if col in dense_features:
                X_train[col + '_log'] = np.log1p(np.clip(X_train[col], a_min=0, a_max=None))
                X_val[col + '_log'] = np.log1p(np.clip(X_val[col], a_min=0, a_max=None))
                X_test[col + '_log'] = np.log1p(np.clip(X_test[col], a_min=0, a_max=None))
    else:
        for col in cols_to_log:
            if col in dense_features:
                X_train[col] = np.log1p(np.clip(X_train[col], a_min=0, a_max=None))
                X_val[col] = np.log1p(np.clip(X_val[col], a_min=0, a_max=None))
                X_test[col] = np.log1p(np.clip(X_test[col], a_min=0, a_max=None))


    # Scale dense features
    use_robust_scaler = True
    scaler = RobustScaler() if use_robust_scaler else StandardScaler()
    X_train[dense_features_used] = scaler.fit_transform(X_train[dense_features_used])
    X_val[dense_features_used] = scaler.transform(X_val[dense_features_used])
    X_test[dense_features_used] = scaler.transform(X_test[dense_features_used])

    # log-transform target to stabilize training
    y_train_log = np.log1p(y_train.values)
    y_val_log = np.log1p(y_val.values)
    y_test_log = np.log1p(y_test.values)

    # preprocessing artifacts
    try:
        joblib.dump(encoders, os.path.join(assets_dir, 'label_encoders.pkl'))
        joblib.dump(scaler, os.path.join(assets_dir, 'scaler.pkl'))

        # bin edges for consistent inference
        bins_payload = {
            'age_edges': age_edges if age_edges is not None else [],
            'mileage_edges': mileage_edges if mileage_edges is not None else [],
        }
        with open(os.path.join(assets_dir, 'bins.json'), 'w', encoding='utf-8') as f:
            json.dump(bins_payload, f)
    except Exception as e:
        print(f'Warning: failed to save encoders/scaler: {e}')

    # hyperparameters
    embedding_dim = DEFAULT_EMBED_DIM
    dnn_dims_str = DEFAULT_DNN_DIMS
    dnn_hidden_units = tuple(int(x) for x in dnn_dims_str.split(','))
    dropout = DEFAULT_DROPOUT
    lr = DEFAULT_LR
    weight_decay = DEFAULT_WEIGHT_DECAY
    patience = DEFAULT_PATIENCE
    early_stop_enabled = True
    dnn_use_bn = bool(DEFAULT_DNN_USE_BN)
    l2_reg_linear = DEFAULT_L2_REG_LINEAR
    l2_reg_embedding = DEFAULT_L2_REG_EMBEDDING
    l2_reg_dnn = DEFAULT_L2_REG_DNN

    use_lr_scheduler = bool(DEFAULT_USE_LR_SCHEDULER)
    lr_sched_factor = DEFAULT_LR_SCHED_FACTOR
    lr_sched_patience = DEFAULT_LR_SCHED_PATIENCE
    lr_sched_min_lr = DEFAULT_LR_SCHED_MIN_LR

    embed_map: dict = {}

    # automatic embedding dimension heuristic per feature when not overridden
    def auto_embed_dim(vocab_size: int) -> int:
        # Common heuristic: 6 * vocab_size**0.25, capped
        return int(max(4, min(64, round(6 * (vocab_size ** 0.25)))))

    # feature columns
    fixlen_feature_columns = (
        [
            SparseFeat(
                feat,
                vocabulary_size=df[feat].nunique(),
                embedding_dim=embed_map.get(feat, auto_embed_dim(df[feat].nunique()) if embedding_dim <= 0 else embedding_dim)
            )
            for feat in sparse_features
        ] +
        [DenseFeat(feat, 1) for feat in dense_features_used]
    )
    feature_names = get_feature_names(fixlen_feature_columns)

    # model inputs
    train_model_input = {name: X_train[name] for name in feature_names}
    val_model_input = {name: X_val[name] for name in feature_names}
    test_model_input = {name: X_test[name] for name in feature_names}

    # optional GroupKFold evaluation on the 80% training portion, grouped by brand_model
    use_groupkfold = False
    folds = 5
    cv_epochs = 30
    if use_groupkfold and folds >= 2:
        print(f'Running GroupKFold ({folds} folds) grouped by brand_model for validation selection...')
        groups = X_temp['brand_model'].values
        gkf = GroupKFold(n_splits=folds)
        cv_maes = []
        for fold_id, (tr_idx, va_idx) in enumerate(gkf.split(X_temp, y_temp, groups=groups), 1):
            X_tr = X_temp.iloc[tr_idx].copy()
            X_va = X_temp.iloc[va_idx].copy()
            y_tr = y_temp.iloc[tr_idx].copy()
            y_va = y_temp.iloc[va_idx].copy()

            # mirror log transforms for dense features with augmentation consistency
            if augment_log:
                for col in cols_to_log:
                    if col in dense_features:
                        X_tr[col + '_log'] = np.log1p(np.clip(X_tr[col], a_min=0, a_max=None))
                        X_va[col + '_log'] = np.log1p(np.clip(X_va[col], a_min=0, a_max=None))
            else:
                for col in cols_to_log:
                    if col in dense_features:
                        X_tr[col] = np.log1p(np.clip(X_tr[col], a_min=0, a_max=None))
                        X_va[col] = np.log1p(np.clip(X_va[col], a_min=0, a_max=None))

            # scale per fold (mirror robust scaler option)
            cv_scaler = RobustScaler() if use_robust_scaler else StandardScaler()
            # mirror augmentation for CV
            cv_dense_used = list(dense_features_used)
            X_tr[cv_dense_used] = cv_scaler.fit_transform(X_tr[cv_dense_used])
            X_va[cv_dense_used] = cv_scaler.transform(X_va[cv_dense_used])

            cv_fixcols = (
                [
                    SparseFeat(
                        f,
                        vocabulary_size=df[f].nunique(),
                        embedding_dim=embed_map.get(f, auto_embed_dim(df[f].nunique()) if embedding_dim <= 0 else embedding_dim)
                    )
                    for f in sparse_features
                ] +
                [DenseFeat(f, 1) for f in cv_dense_used]
            )
            cv_names = get_feature_names(cv_fixcols)
            cv_train_input = {name: X_tr[name] for name in cv_names}
            cv_val_input = {name: X_va[name] for name in cv_names}

            cv_model = DeepFM(
                linear_feature_columns=cv_fixcols,
                dnn_feature_columns=cv_fixcols,
                task='regression',
                dnn_hidden_units=dnn_hidden_units,
                dnn_dropout=dropout,
                dnn_use_bn=dnn_use_bn,
                l2_reg_linear=l2_reg_linear,
                l2_reg_embedding=l2_reg_embedding,
                l2_reg_dnn=l2_reg_dnn,
                device=device
            )
            cv_optimizer = torch.optim.Adam(cv_model.parameters(), lr=lr, weight_decay=weight_decay)
            cv_scheduler = None
            if use_lr_scheduler:
                cv_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    cv_optimizer,
                    mode='min',
                    factor=lr_sched_factor,
                    patience=lr_sched_patience,
                    min_lr=lr_sched_min_lr,
                )
            cv_model.compile(cv_optimizer, 'mse', metrics=['mae'])

            y_tr_log = np.log1p(y_tr.values)
            y_va_log = np.log1p(y_va.values)
            best_fold_val_mae = float('inf')
            no_improve_fold = 0
            cv_batch_size = DEFAULT_BATCH_SIZE
            for e in range(cv_epochs):
                cv_model.fit(
                    cv_train_input,
                    y_tr_log,
                    batch_size=cv_batch_size,
                    epochs=1,
                    verbose=0,
                    validation_data=(cv_val_input, y_va_log)
                )
                va_pred_log = cv_model.predict(cv_val_input, batch_size=cv_batch_size)
                va_pred = np.expm1(va_pred_log)
                val_mae_fold = mean_absolute_error(y_va.values, va_pred)

                if cv_scheduler is not None:
                    cv_scheduler.step(val_mae_fold)

                if val_mae_fold < best_fold_val_mae:
                    best_fold_val_mae = val_mae_fold
                    no_improve_fold = 0
                else:
                    no_improve_fold += 1

                if early_stop_enabled and no_improve_fold >= patience:
                    break

            cv_maes.append(best_fold_val_mae)
            print(f'GroupKFold fold {fold_id}/{folds} best val MAE: {best_fold_val_mae:.4f}')

        print(f'GroupKFold mean best val MAE across folds: {np.mean(cv_maes):.4f}')
    model = DeepFM(
        linear_feature_columns=fixlen_feature_columns,
        dnn_feature_columns=fixlen_feature_columns,
        task='regression',
        dnn_hidden_units=dnn_hidden_units,
        dnn_dropout=dropout,
        dnn_use_bn=dnn_use_bn,
        l2_reg_linear=l2_reg_linear,
        l2_reg_embedding=l2_reg_embedding,
        l2_reg_dnn=l2_reg_dnn,
        device=device
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # attach ReduceLROnPlateau scheduler on validation MAE
    scheduler = None
    if use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=lr_sched_factor,
            patience=lr_sched_patience,
            min_lr=lr_sched_min_lr,
        )
    model.compile(optimizer, 'mse', metrics=['mae'])

    # train loop with early stopping on validation MAE
    epochs = DEFAULT_EPOCHS
    batch_size = DEFAULT_BATCH_SIZE
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

        # step LR scheduler with validation MAE
        if scheduler is not None:
            scheduler.step(val_mae)
            current_lr = optimizer.param_groups[0]['lr']
            print(f'LR after scheduler step: {current_lr:.6f}')

        print(f'Epoch {epoch+1}/{epochs} - MAE train: {train_mae:.4f}, val: {val_mae:.4f}')

        if early_stop_enabled and no_improve >= patience:
            print(f'Early stopping at epoch {epoch+1} (best val MAE: {best_val_mae:.4f})')
            break

    # restore best state
    if best_state is not None:
        model.load_state_dict(best_state)

    # persist MAE history and plot curve
    epochs_run = len(train_mae_list)
    mae_csv_path = os.path.join(assets_dir, 'mae_history.csv')
    try:
        pd.DataFrame({
            'epoch': list(range(1, epochs_run + 1)),
            'train_mae': train_mae_list,
            'val_mae': val_mae_list,
        }).to_csv(mae_csv_path, index=False)
        print(f'Saved MAE history to: {mae_csv_path}')
    except Exception as e:
        print(f'Warning: failed to save MAE history: {e}')

    mae_png_path = os.path.join(assets_dir, 'mae_curve.png')
    try:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(range(1, epochs_run + 1), train_mae_list, label='Train MAE')
        ax.plot(range(1, epochs_run + 1), val_mae_list, label='Val MAE')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MAE')
        ax.legend()
        fig.tight_layout()
        fig.savefig(mae_png_path)
        plt.close(fig)
        print(f'Saved MAE curve plot to: {mae_png_path}')
    except Exception as e:
        print(f'Warning: failed to plot MAE curve: {e}')

    # final test MAE
    test_pred_log = model.predict(test_model_input, batch_size=batch_size)
    test_pred = np.expm1(test_pred_log)
    final_test_mae = mean_absolute_error(y_test.values, test_pred)
    print(f'Final Test MAE (best model): {final_test_mae:.4f}')
    print(f'Final Val MAE (best model): {best_val_mae:.4f}')

    # save model and config
    model_path = os.path.join(assets_dir, 'deepfm_best.pt')
    try:
        torch.save(model.state_dict(), model_path)
        with open(os.path.join(assets_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'lr': lr,
                'weight_decay': weight_decay,
                'dropout': dropout,
                'embedding_dim': embedding_dim,
                'embedding_per_feature': embed_map,
                'dnn_hidden_units': list(dnn_hidden_units),
                'dnn_use_bn': dnn_use_bn,
                'l2_reg_linear': l2_reg_linear,
                'l2_reg_embedding': l2_reg_embedding,
                'l2_reg_dnn': l2_reg_dnn,
                'patience': patience,
                'epochs_run': len(train_mae_list),
                'batch_size': batch_size,
                'best_val_mae': float(best_val_mae),
                'final_test_mae': float(final_test_mae),
                'device': device,
                'sparse_features': sparse_features,
                'dense_features': dense_features,
                'dense_features_used': dense_features_used,
                'log_dense': cols_to_log,
                'log_dense_augment': augment_log,
                'input_csv': input_csv,
            'scaler_type': 'robust' if use_robust_scaler else 'standard',
                'bins_file': 'bins.json',
                'rare_freq_threshold': rare_thr,
                'unk_token': unk_token,
                'unk_index_per_feature': unk_index_per_feature,
                'lr_scheduler': {
                    'enabled': use_lr_scheduler,
                    'factor': lr_sched_factor,
                    'patience': lr_sched_patience,
                    'min_lr': lr_sched_min_lr,
                    'final_lr': float(optimizer.param_groups[0]['lr'])
                },
                'mae_history_csv': mae_csv_path,
                'mae_curve_png': mae_png_path,
            }, f, indent=2)
        print(f'Saved best model to: {model_path}')
        print(f'Saved config and preprocessors in: {assets_dir}')
    except Exception as e:
        print(f'Warning: failed to save model/config: {e}')

# entrypoint
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DeepFM from CSV with aligned preprocessing')
    parser.add_argument('--csv', type=str, default=r'C:\\Users\\vehnie\\Documents\\Master\\Machine Learning\\Cars4You\\notebooks\\df_train.csv',
                        help='Absolute path to df_train.csv')
    args = parser.parse_args()
    main(args)
# Predict with DeepFM matching the checkpoint's feature shapes
import os
import json
import joblib
import numpy as np
import pandas as pd
import torch

from deepctr_torch.inputs import SparseFeat, DenseFeat
from deepctr_torch.models import DeepFM

# paths
assets_dir = r"C:\Users\vehnie\Documents\Master\Machine Learning\Cars4You\notebooks\assets_deepfm"
model_path = os.path.join(assets_dir, "deepfm_best.pt")
config_path = os.path.join(assets_dir, "config.json")
encoders_path = os.path.join(assets_dir, "label_encoders.pkl")
scaler_path = os.path.join(assets_dir, "scaler.pkl")
test_csv = r"C:\Users\vehnie\Documents\Master\Machine Learning\Cars4You\notebooks\test\test_processed.csv"

# load processed test and artifacts
df_test = pd.read_csv(test_csv)
# Create brand_model cross feature to match training
if "brand_model" not in df_test.columns and {"brand", "model"}.issubset(df_test.columns):
    df_test["brand_model"] = df_test["brand"].astype(str) + "|" + df_test["model"].astype(str)

encoder = joblib.load(encoders_path)   # dict of LabelEncoders (from training)
scaler = joblib.load(scaler_path)

# load training config
cfg = {}
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        cfg = json.load(f)

# create age/mileage bin cross feature using training bin edges
bins_path = os.path.join(assets_dir, cfg.get("bins_file", "bins.json"))
age_edges = []
mileage_edges = []
if os.path.exists(bins_path):
    try:
        with open(bins_path, "r", encoding="utf-8") as f:
            bins_cfg = json.load(f)
            age_edges = bins_cfg.get("age_edges", [])
            mileage_edges = bins_cfg.get("mileage_edges", [])
    except Exception as e:
        print(f"Warning: failed to read bins.json: {e}")

if "age_mileage_bin" not in df_test.columns and {"age", "mileage"}.issubset(df_test.columns):
    if age_edges and mileage_edges:
        age_bins = np.digitize(df_test["age"].values, age_edges, right=False)
        mileage_bins = np.digitize(df_test["mileage"].values, mileage_edges, right=False)
        df_test["age_mileage_bin"] = pd.Series(age_bins).astype(str) + "|" + pd.Series(mileage_bins).astype(str)
    else:
        # fallback single bin if edges missing
        df_test["age_mileage_bin"] = "0|0"

# ratio feature mpg_per_engine to mirror training
if {"mpg", "engine_size"}.issubset(df_test.columns):
    denom = np.clip(df_test["engine_size"].astype(float).values, a_min=1e-6, a_max=None)
    df_test["mpg_per_engine"] = (df_test["mpg"].astype(float).values / denom)
    df_test["mpg_per_engine"] = pd.Series(df_test["mpg_per_engine"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)

def parse_dnn_dims(val, default=[256, 128, 64]):
    if isinstance(val, str):
        return [int(x.strip()) for x in val.split(",") if x.strip()]
    if isinstance(val, (list, tuple)):
        return list(val)
    return default

# read hyperparameters and preprocessing config
embed_dim_default = int(cfg.get("embedding_dim", 8))
embed_map_cfg = cfg.get("embedding_per_feature", {}) or {}
unk_token = cfg.get("unk_token", "UNK")
unk_index_map = cfg.get("unk_index_per_feature", {}) or {}
augment_log = bool(cfg.get("log_dense_augment", False))
DNN_DIMS = parse_dnn_dims(cfg.get("dnn_hidden_units", [256, 128, 64]))
DROPOUT = float(cfg.get("dropout", 0.3))

# load checkpoint to extract expected vocab sizes and input dim
device = "cuda" if torch.cuda.is_available() else "cpu"
state = torch.load(model_path, map_location=device)

# expected vocab sizes per sparse feature from checkpoint (e.g., brand/model/fuel_type/transmission)
expected_vocab = {}
for k, v in state.items():
    # Examples of keys: 'embedding_dict.model.weight', 'embedding_dict.brand.weight'
    if k.startswith("embedding_dict.") and k.endswith(".weight") and v.ndim == 2:
        feat = k.split(".")[1]
        expected_vocab[feat] = v.shape[0]

# expected DNN input dim from first linear layer
expected_dnn_in = None
for k, v in state.items():
    if k == "dnn.linears.0.weight":
        expected_dnn_in = v.shape[1]
        break
if expected_dnn_in is None:
    raise RuntimeError("Could not infer expected DNN input dimension from checkpoint.")

# build sparse features using encoder categories but enforce checkpoint vocab sizes
# support per-feature embedding dims (from config) and fallback heuristic
def auto_embed_dim(vocab_size: int) -> int:
    return int(max(4, min(64, round(6 * (vocab_size ** 0.25)))))
sparse_feature_columns = []
sparse_inputs = {}
sparse_cols = []

if isinstance(encoder, dict):
    # dict of per-column LabelEncoders
    sparse_cols = list(encoder.keys())
    for col, le in encoder.items():
        classes = [str(c) for c in getattr(le, "classes_", [])]
        mapping = {c: i for i, c in enumerate(classes)}
        # Use checkpoint vocab size if available; otherwise use len(classes) (+1 optional OOV not needed here)
        vocab_size = expected_vocab.get(col, len(classes))
        # map; unseen/missing -> configured UNK index if available, else clamp to last index
        unk_idx = int(unk_index_map.get(col, max(len(classes) - 1, 0)))
        unk_idx = min(unk_idx, max(vocab_size - 1, 0))
        arr = df_test[col].astype(str).fillna(unk_token).map(mapping).fillna(unk_idx).astype("int64").values
        sparse_inputs[col] = arr
        feat_dim = embed_map_cfg.get(col)
        if feat_dim is None:
            feat_dim = embed_dim_default if embed_dim_default > 0 else auto_embed_dim(vocab_size)
        sparse_feature_columns.append(SparseFeat(col, vocabulary_size=vocab_size, embedding_dim=int(feat_dim)))
else:
    # ordinalEncoder over multiple categorical columns
    sparse_cols = list(getattr(encoder, "feature_names_in_", [])) or ["brand", "model", "fuel_type", "transmission"]
    categories = list(getattr(encoder, "categories_", []))
    if not categories or len(categories) != len(sparse_cols):
        raise RuntimeError("Encoder categories not found or mismatched with columns.")
    for col, cats in zip(sparse_cols, categories):
        cats = [str(c) for c in cats]
        mapping = {c: i for i, c in enumerate(cats)}
        # enforce checkpoint vocab size if available
        vocab_size = expected_vocab.get(col, len(cats))
        unk_idx = int(unk_index_map.get(col, max(len(cats) - 1, 0)))
        unk_idx = min(unk_idx, max(vocab_size - 1, 0))
        arr = df_test[col].astype(str).fillna(unk_token).map(mapping).fillna(unk_idx).astype("int64").values
        sparse_inputs[col] = arr
        feat_dim = embed_map_cfg.get(col)
        if feat_dim is None:
            feat_dim = embed_dim_default if embed_dim_default > 0 else auto_embed_dim(vocab_size)
        sparse_feature_columns.append(SparseFeat(col, vocabulary_size=vocab_size, embedding_dim=int(feat_dim)))

# validate sparse columns present
missing_sparse = [c for c in sparse_cols if c not in df_test.columns]
if missing_sparse:
    raise ValueError(f"Missing categorical columns in test_processed.csv: {missing_sparse}")

# build dense features list: use scaler's fitted names if available; otherwise a sensible fallback.
dense_features = list(getattr(scaler, "feature_names_in_", []))
if not dense_features:
    fallback_dense = ["mileage", "age", "engine_size", "mpg", "avg_price"]
    dense_features = [c for c in fallback_dense if c in df_test.columns]

# adjust dense feature count to match expected DNN input dimension
current_dnn_in = sum(int(col.embedding_dim) for col in sparse_feature_columns) + len(dense_features)
if current_dnn_in != expected_dnn_in:
    # try adding common candidates present in df_test until it matches
    candidates = ["mileage", "age", "engine_size", "mpg", "avg_price"]
    for c in candidates:
        if c not in dense_features and c in df_test.columns:
            dense_features.append(c)
            current_dnn_in = sum(int(col.embedding_dim) for col in sparse_feature_columns) + len(dense_features)
            if current_dnn_in == expected_dnn_in:
                break
    if current_dnn_in != expected_dnn_in:
        raise RuntimeError(
            f"Dense feature count mismatch. Expected DNN input dim {expected_dnn_in}, "
            f"got {current_dnn_in}. Sparse fields={len(sparse_feature_columns)}, "
            f"dense fields={len(dense_features)}: {dense_features}"
        )

# scale the features the scaler knows, pass raw for any extras to satisfy dimension
scaler_feature_names = list(getattr(scaler, "feature_names_in_", []))
scaled_dense = []
dense_inputs = {}

if scaler_feature_names:
    # mirror training: optionally log-transform (augment or overwrite) selected dense features before scaling
    # prefer values saved in training config; fall back to environment variable
    cfg_log_dense = cfg.get("log_dense", [])
    if isinstance(cfg_log_dense, str):
        cfg_log_dense = [c.strip() for c in cfg_log_dense.split(",") if c.strip()]
    env_log_dense = os.getenv("LOG_DENSE", "")
    env_cols = [c.strip() for c in env_log_dense.split(",") if c.strip()]
    cols_to_log = list(set((cfg_log_dense or []) + env_cols))
    if augment_log:
        for col in cols_to_log:
            if col in df_test.columns:
                df_test[col + "_log"] = np.log1p(np.clip(df_test[col].values, a_min=0, a_max=None))
    else:
        for col in cols_to_log:
            if col in df_test.columns:
                df_test[col] = np.log1p(np.clip(df_test[col].values, a_min=0, a_max=None))


    missing_scaled = [c for c in scaler_feature_names if c not in df_test.columns]
    if missing_scaled:
        raise ValueError(f"Missing columns required by scaler: {missing_scaled}")

    scaled_matrix = scaler.transform(df_test[scaler_feature_names])
    for i, f in enumerate(scaler_feature_names):
        dense_inputs[f] = scaled_matrix[:, i]
        scaled_dense.append(f)

# add any extra dense features not covered by scaler
for f in dense_features:
    if f not in scaled_dense:
        dense_inputs[f] = df_test[f].astype(float).values

dense_feature_columns = [DenseFeat(f, 1) for f in dense_features]

# build model with aligned shapes
linear_feature_columns = sparse_feature_columns + dense_feature_columns
dnn_feature_columns = sparse_feature_columns + dense_feature_columns

model = DeepFM(
    linear_feature_columns=linear_feature_columns,
    dnn_feature_columns=dnn_feature_columns,
    task="regression",
    device=device,
    dnn_hidden_units=DNN_DIMS,
    dnn_dropout=DROPOUT,
    # align batch normalization usage with the trained checkpoint
    dnn_use_bn=any(k.startswith("dnn.bn.") for k in state.keys())
)

# load weights now that shapes match
model.load_state_dict(state)
model.eval()

# predict
model_input = {**sparse_inputs, **dense_inputs}
preds_raw = model.predict(model_input, batch_size=2048)
preds_raw = np.asarray(preds_raw).reshape(-1)

# If model was trained on log(price), invert; otherwise keep as-is
if preds_raw.mean() < 50:  # heuristic: log(price) typically averages < ~10
    preds = np.expm1(preds_raw)
    print("Detected log-space outputs; applied expm1 inversion.")
else:
    preds = preds_raw
# clip predictions to valid price range
preds = np.clip(preds, a_min=0.0, a_max=None)
print(f"Pred stats — min: {preds.min():.2f}, max: {preds.max():.2f}, mean: {preds.mean():.2f}")

# save predictions
out_path = os.path.join(assets_dir, "predictions_from_test_processed.csv")
out_df = df_test.copy()
out_df["predicted_price"] = preds
out_df.drop(columns=['mileage', 'model'], inplace=True)
out_df.to_csv(out_path, index=False)

print(f"Saved predictions: {out_path}")
# print a small sample
cols_to_show = [c for c in ["brand", "model", "predicted_price"] if c in out_df.columns]
print(out_df[cols_to_show].head())

# save submission file with only carID and price
if "carID" in df_test.columns:
    ids = df_test["carID"].astype("Int64").values
else:
    # Fallback: load raw test.csv to retrieve carID and assume same row order
    root_dir = os.path.dirname(os.path.dirname(assets_dir))
    raw_test_path = os.path.join(root_dir, "data", "test.csv")
    if not os.path.exists(raw_test_path):
        raise ValueError("carID not found in test_processed.csv and raw test.csv is missing.")
    raw = pd.read_csv(raw_test_path)
    if len(raw) != len(df_test):
        raise ValueError("test.csv length mismatch with test_processed.csv; cannot align carID.")
    ids = raw["carID"].astype("Int64").values

submission_df = pd.DataFrame({
    "carID": ids,
    "price": preds.astype(float)
})
submission_path = os.path.join(assets_dir, "submission.csv")
submission_df.to_csv(submission_path, index=False)
print(f"Saved submission: {submission_path}")
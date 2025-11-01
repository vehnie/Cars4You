# Predict with DeepFM matching the checkpoint's feature shapes
import os
import json
import joblib
import numpy as np
import pandas as pd
import torch

from deepctr_torch.inputs import SparseFeat, DenseFeat
from deepctr_torch.models import DeepFM

# Paths
assets_dir = r"C:\Users\vehnie\Documents\Master\Machine Learning\Cars4You\notebooks\assets_deepfm"
model_path = os.path.join(assets_dir, "deepfm_best.pt")
config_path = os.path.join(assets_dir, "config.json")
encoders_path = os.path.join(assets_dir, "label_encoders.pkl")
scaler_path = os.path.join(assets_dir, "scaler.pkl")
test_csv = r"C:\Users\vehnie\Documents\Master\Machine Learning\Cars4You\notebooks\test\test_processed.csv"

# Load processed test and artifacts
df_test = pd.read_csv(test_csv)
encoder = joblib.load(encoders_path)   # dict of LabelEncoders (from training)
scaler = joblib.load(scaler_path)

# Load training config
cfg = {}
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        cfg = json.load(f)

def parse_dnn_dims(val, default=[256, 128, 64]):
    if isinstance(val, str):
        return [int(x.strip()) for x in val.split(",") if x.strip()]
    if isinstance(val, (list, tuple)):
        return list(val)
    return default

# Read hyperparameters from training config keys
EMBED_DIM = int(cfg.get("embedding_dim", 8))
DNN_DIMS = parse_dnn_dims(cfg.get("dnn_hidden_units", [256, 128, 64]))
DROPOUT = float(cfg.get("dropout", 0.3))

# Load checkpoint to extract expected vocab sizes and input dim
device = "cuda" if torch.cuda.is_available() else "cpu"
state = torch.load(model_path, map_location=device)

# Expected vocab sizes per sparse feature from checkpoint (e.g., brand/model/fuel_type/transmission)
expected_vocab = {}
for k, v in state.items():
    # Examples of keys: 'embedding_dict.model.weight', 'embedding_dict.brand.weight'
    if k.startswith("embedding_dict.") and k.endswith(".weight") and v.ndim == 2:
        feat = k.split(".")[1]
        expected_vocab[feat] = v.shape[0]

# Expected DNN input dim from first linear layer
expected_dnn_in = None
for k, v in state.items():
    if k == "dnn.linears.0.weight":
        expected_dnn_in = v.shape[1]
        break
if expected_dnn_in is None:
    raise RuntimeError("Could not infer expected DNN input dimension from checkpoint.")

# Build sparse features using encoder categories but enforce checkpoint vocab sizes
sparse_feature_columns = []
sparse_inputs = {}
sparse_cols = []

if isinstance(encoder, dict):
    # Dict of per-column LabelEncoders
    sparse_cols = list(encoder.keys())
    for col, le in encoder.items():
        classes = [str(c) for c in getattr(le, "classes_", [])]
        mapping = {c: i for i, c in enumerate(classes)}
        # Use checkpoint vocab size if available; otherwise use len(classes) (+1 optional OOV not needed here)
        vocab_size = expected_vocab.get(col, len(classes))
        # Map; unseen -> last index
        oov_index = max(vocab_size - 1, 0)
        arr = df_test[col].astype(str).map(mapping).fillna(oov_index).astype("int64").values
        sparse_inputs[col] = arr
        sparse_feature_columns.append(SparseFeat(col, vocabulary_size=vocab_size, embedding_dim=EMBED_DIM))
else:
    # OrdinalEncoder over multiple categorical columns
    sparse_cols = list(getattr(encoder, "feature_names_in_", [])) or ["brand", "model", "fuel_type", "transmission"]
    categories = list(getattr(encoder, "categories_", []))
    if not categories or len(categories) != len(sparse_cols):
        raise RuntimeError("Encoder categories not found or mismatched with columns.")
    for col, cats in zip(sparse_cols, categories):
        cats = [str(c) for c in cats]
        mapping = {c: i for i, c in enumerate(cats)}
        # Enforce checkpoint vocab size if available
        vocab_size = expected_vocab.get(col, len(cats))
        oov_index = max(vocab_size - 1, 0)
        arr = df_test[col].astype(str).map(mapping).fillna(oov_index).astype("int64").values
        sparse_inputs[col] = arr
        sparse_feature_columns.append(SparseFeat(col, vocabulary_size=vocab_size, embedding_dim=EMBED_DIM))

# Validate sparse columns present
missing_sparse = [c for c in sparse_cols if c not in df_test.columns]
if missing_sparse:
    raise ValueError(f"Missing categorical columns in test_processed.csv: {missing_sparse}")

# Build dense features list: use scaler's fitted names if available; otherwise a sensible fallback.
dense_features = list(getattr(scaler, "feature_names_in_", []))
if not dense_features:
    fallback_dense = ["mileage", "age", "engine_size", "mpg", "avg_price"]
    dense_features = [c for c in fallback_dense if c in df_test.columns]

# Adjust dense feature count to match expected DNN input dimension
current_dnn_in = EMBED_DIM * len(sparse_feature_columns) + len(dense_features)
if current_dnn_in != expected_dnn_in:
    # Try adding common candidates present in df_test until it matches
    candidates = ["mileage", "age", "engine_size", "mpg", "avg_price"]
    for c in candidates:
        if c not in dense_features and c in df_test.columns:
            dense_features.append(c)
            current_dnn_in = EMBED_DIM * len(sparse_feature_columns) + len(dense_features)
            if current_dnn_in == expected_dnn_in:
                break
    if current_dnn_in != expected_dnn_in:
        raise RuntimeError(
            f"Dense feature count mismatch. Expected DNN input dim {expected_dnn_in}, "
            f"got {current_dnn_in}. Sparse fields={len(sparse_feature_columns)}, "
            f"dense fields={len(dense_features)}: {dense_features}"
        )

# Scale the features the scaler knows; pass raw for any extras to satisfy dimension
scaler_feature_names = list(getattr(scaler, "feature_names_in_", []))
scaled_dense = []
dense_inputs = {}

if scaler_feature_names:
    # Ensure scaler's features appear in df_test
    missing_scaled = [c for c in scaler_feature_names if c not in df_test.columns]
    if missing_scaled:
        raise ValueError(f"Missing columns required by scaler: {missing_scaled}")

    # Mirror training: optionally log-transform selected dense features before scaling
    # Prefer values saved in training config; fall back to environment variable
    cfg_log_dense = cfg.get("log_dense", [])
    if isinstance(cfg_log_dense, str):
        cfg_log_dense = [c.strip() for c in cfg_log_dense.split(",") if c.strip()]
    env_log_dense = os.getenv("LOG_DENSE", "")
    env_cols = [c.strip() for c in env_log_dense.split(",") if c.strip()]
    cols_to_log = list(set((cfg_log_dense or []) + env_cols))
    for col in cols_to_log:
        if col in df_test.columns:
            df_test[col] = np.log1p(np.clip(df_test[col].values, a_min=0, a_max=None))

    scaled_matrix = scaler.transform(df_test[scaler_feature_names])
    for i, f in enumerate(scaler_feature_names):
        dense_inputs[f] = scaled_matrix[:, i]
        scaled_dense.append(f)

# Add any extra dense features not covered by scaler
for f in dense_features:
    if f not in scaled_dense:
        dense_inputs[f] = df_test[f].astype(float).values

dense_feature_columns = [DenseFeat(f, 1) for f in dense_features]

# Build model with aligned shapes
linear_feature_columns = sparse_feature_columns + dense_feature_columns
dnn_feature_columns = sparse_feature_columns + dense_feature_columns

model = DeepFM(
    linear_feature_columns=linear_feature_columns,
    dnn_feature_columns=dnn_feature_columns,
    task="regression",
    device=device,
    dnn_hidden_units=DNN_DIMS,
    dnn_dropout=DROPOUT
)

# Load weights now that shapes match
model.load_state_dict(state)
model.eval()

# Predict
model_input = {**sparse_inputs, **dense_inputs}
preds_raw = model.predict(model_input, batch_size=2048)
preds_raw = np.asarray(preds_raw).reshape(-1)

# If model was trained on log(price), invert; otherwise keep as-is
if preds_raw.mean() < 50:  # heuristic: log(price) typically averages < ~10
    preds = np.expm1(preds_raw)
    print("Detected log-space outputs; applied expm1 inversion.")
else:
    preds = preds_raw
# Clip predictions to valid price range
preds = np.clip(preds, a_min=0.0, a_max=None)
print(f"Pred stats — min: {preds.min():.2f}, max: {preds.max():.2f}, mean: {preds.mean():.2f}")

# Save predictions
out_path = os.path.join(assets_dir, "predictions_from_test_processed.csv")
out_df = df_test.copy()
out_df["predicted_price"] = preds
out_df.drop(columns=['mileage', 'model'], inplace=True)
out_df.to_csv(out_path, index=False)

print(f"Saved predictions: {out_path}")
# Print a small sample; 'model' may be dropped earlier
cols_to_show = [c for c in ["brand", "model", "predicted_price"] if c in out_df.columns]
print(out_df[cols_to_show].head())

# Save submission file with only carID and price
if "carID" in df_test.columns:
    ids = df_test["carID"].astype("Int64").values
else:
    # Fallback: load raw test.csv to retrieve carID and assume same row order
    root_dir = os.path.dirname(os.path.dirname(assets_dir))  # .../Cars4You
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
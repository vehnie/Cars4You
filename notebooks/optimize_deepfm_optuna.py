import os
import re
import json
import subprocess
import sys
from typing import Dict, Tuple, List

import optuna


# -------------------------
# Helpers for customization
# -------------------------
def _parse_list(s: str, caster=int) -> List:
    return [caster(x.strip()) for x in s.split(",") if x.strip()]


def _parse_dnn_choices(s: str) -> List[str]:
    """Parse semicolon-separated DNN choices into list of comma strings.
    Example: "256,128,64;384,192,96" -> ["256,128,64", "384,192,96"]
    """
    return [chunk.strip() for chunk in s.split(";") if chunk.strip()]


def parse_mae(output: str) -> Tuple[float, float]:
    """Parse validation and test MAE from train_deepfm.py output.

    Tries multiple patterns to be robust to logging variations.
    Returns (val_mae, test_mae). If not found, returns (float('inf'), float('inf')).
    """
    val_patterns = [
        r"Best val MAE:\s*([0-9]+\.[0-9]+)",
        r"Validation MAE:\s*([0-9]+\.[0-9]+)",
        r"Val MAE:\s*([0-9]+\.[0-9]+)",
        r"Final Val MAE \(best model\):\s*([0-9]+\.[0-9]+)",
    ]
    test_patterns = [
        r"Test MAE:\s*([0-9]+\.[0-9]+)",
        r"Best test MAE:\s*([0-9]+\.[0-9]+)",
        r"Final Test MAE \(best model\):\s*([0-9]+\.[0-9]+)",
    ]

    val_mae = float("inf")
    test_mae = float("inf")

    for p in val_patterns:
        m = re.search(p, output)
        if m:
            try:
                val_mae = float(m.group(1))
                break
            except ValueError:
                pass

    for p in test_patterns:
        m = re.search(p, output)
        if m:
            try:
                test_mae = float(m.group(1))
                break
            except ValueError:
                pass

    return val_mae, test_mae


def run_once(params: Dict[str, str], epochs: int, patience: int) -> Tuple[float, float, str, str, str]:
    """Run train_deepfm.py once with provided params.

    Returns (val_mae, test_mae, config_str).
    """
    env = os.environ.copy()
    # Map hyperparameters to environment variables expected by train_deepfm.py
    env.update({
        "LR": str(params["lr"]),
        "DROPOUT": str(params["dropout"]),
        "EMBED_DIM": str(params["embed_dim"]),
        "DNN_DIMS": params["dnn_dims"],  # comma-separated string
        "WEIGHT_DECAY": str(params["weight_decay"]),
        "PATIENCE": str(patience),
        "EPOCHS": str(epochs),
    })

    # Forward optional LOG_DENSE if set in current environment
    if os.getenv("LOG_DENSE"):
        env["LOG_DENSE"] = os.getenv("LOG_DENSE")

    # Execute training script
    proc = subprocess.run(
        [sys.executable, "train_deepfm.py"],
        cwd=os.path.dirname(__file__),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""

    combined = stdout + "\n" + stderr
    val_mae, test_mae = parse_mae(combined)

    # Create a compact config string for logging/inspection
    config = (
        f"LR: {params['lr']}, WD: {params['weight_decay']}, "
        f"DROPOUT: {params['dropout']}, EMBED_DIM: {params['embed_dim']}, "
        f"DNN_DIMS: {params['dnn_dims']}, PATIENCE: {patience}, EPOCHS: {epochs}"
    )

    return val_mae, test_mae, config, stdout, stderr


def objective(trial: optuna.Trial) -> float:
    """Optuna objective minimizing validation MAE using TPE sampler.

    Customizable via environment variables:
    - Ranges: OPT_LR_MIN/OPT_LR_MAX, OPT_DROPOUT_MIN/OPT_DROPOUT_MAX, OPT_WD_MIN/OPT_WD_MAX
    - Choices: OPT_EMBED_CHOICES (comma list), OPT_DNN_CHOICES (semicolon-separated comma lists)
    - Freeze: FIX_LR, FIX_DROPOUT, FIX_EMBED_DIM, FIX_WEIGHT_DECAY, FIX_DNN_DIMS
    - Budget: OPTUNA_EPOCHS/OPTUNA_PATIENCE or TUNE_EPOCHS/TUNE_PATIENCE
    - CSV logging: OPTUNA_CSV (filename in notebooks dir)
    """
    # Ranges and choices
    lr_min = float(os.getenv("OPT_LR_MIN", "1e-4"))
    lr_max = float(os.getenv("OPT_LR_MAX", "5e-3"))
    drop_min = float(os.getenv("OPT_DROPOUT_MIN", "0.15"))
    drop_max = float(os.getenv("OPT_DROPOUT_MAX", "0.5"))
    wd_min = float(os.getenv("OPT_WD_MIN", "1e-6"))
    wd_max = float(os.getenv("OPT_WD_MAX", "5e-5"))
    embed_choices = _parse_list(os.getenv("OPT_EMBED_CHOICES", "8,12,16"), int)
    dnn_choices = _parse_dnn_choices(
        os.getenv(
            "OPT_DNN_CHOICES",
            "256,128,64;384,192,96;512,256,128;256,256,128",
        )
    )

    # Freeze overrides
    fix_lr = os.getenv("FIX_LR")
    fix_dropout = os.getenv("FIX_DROPOUT")
    fix_embed = os.getenv("FIX_EMBED_DIM")
    fix_wd = os.getenv("FIX_WEIGHT_DECAY")
    fix_dnn = os.getenv("FIX_DNN_DIMS")

    # Suggest or use fixed values
    lr = float(fix_lr) if fix_lr is not None else trial.suggest_float("lr", lr_min, lr_max, log=True)
    dropout = float(fix_dropout) if fix_dropout is not None else trial.suggest_float("dropout", drop_min, drop_max)
    embed_dim = int(fix_embed) if fix_embed is not None else trial.suggest_categorical("embed_dim", embed_choices)
    weight_decay = float(fix_wd) if fix_wd is not None else trial.suggest_float("weight_decay", wd_min, wd_max, log=True)
    dnn_choice = fix_dnn if fix_dnn is not None else trial.suggest_categorical("dnn_dims", dnn_choices)

    # Training budget controls
    epochs = int(os.getenv("OPTUNA_EPOCHS", os.getenv("TUNE_EPOCHS", "40")))
    patience = int(os.getenv("OPTUNA_PATIENCE", os.getenv("TUNE_PATIENCE", "12")))

    params = {
        "lr": lr,
        "dropout": dropout,
        "embed_dim": embed_dim,
        "weight_decay": weight_decay,
        "dnn_dims": dnn_choice,
    }

    try:
        val_mae, test_mae, config, stdout, stderr = run_once(params, epochs=epochs, patience=patience)
    except Exception as e:
        # Penalize failures with a large value
        trial.set_user_attr("error", str(e))
        return float("inf")

    # Attach useful info to the trial for later inspection
    trial.set_user_attr("test_mae", test_mae)
    trial.set_user_attr("config", config)

    # Persist raw logs for troubleshooting
    notebooks_dir = os.path.dirname(__file__)
    try:
        with open(os.path.join(notebooks_dir, f"optuna_trial_{trial.number}.out.txt"), "w", encoding="utf-8") as f:
            f.write(stdout)
        with open(os.path.join(notebooks_dir, f"optuna_trial_{trial.number}.err.txt"), "w", encoding="utf-8") as f:
            f.write(stderr)
    except Exception:
        pass

    # Per-trial CSV logging
    csv_name = os.getenv("OPTUNA_CSV", "optuna_results.csv")
    csv_path = os.path.join(notebooks_dir, csv_name)
    first_write = not os.path.exists(csv_path)
    try:
        with open(csv_path, "a", encoding="utf-8") as f:
            if first_write:
                f.write("trial,val_mae,test_mae,lr,dropout,embed_dim,dnn_dims,weight_decay,epochs,patience\n")
            f.write(
                f"{trial.number},{val_mae:.6f},{test_mae:.6f},{lr},{dropout},{embed_dim},"\
                f"{dnn_choice},{weight_decay},{epochs},{patience}\n"
            )
    except Exception:
        # Non-fatal if logging fails; continue optimization
        pass

    return val_mae


def main():
    # Sampler with optional seed
    seed = int(os.getenv("OPT_SEED", "42"))
    sampler = optuna.samplers.TPESampler(seed=seed)

    # Optional pruner
    pruner_name = os.getenv("OPT_PRUNER", "median").lower()
    startup_trials = int(os.getenv("OPT_PRUNER_STARTUP_TRIALS", "3"))
    pruner = None
    if pruner_name == "median":
        pruner = optuna.pruners.MedianPruner(n_startup_trials=startup_trials)
    elif pruner_name == "halving":
        pruner = optuna.pruners.SuccessiveHalvingPruner()
    elif pruner_name == "none":
        pruner = None

    study_name = os.getenv("OPTUNA_STUDY", "deepfm_tpe_study")
    storage_url = os.getenv("OPT_STORAGE")  # e.g., sqlite:///optuna.db

    if storage_url:
        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
            study_name=study_name,
            storage=storage_url,
            load_if_exists=True,
        )
    else:
        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
            study_name=study_name,
        )

    n_trials = int(os.getenv("OPTUNA_TRIALS", "10"))
    print(f"Starting Optuna TPE optimization with {n_trials} trials (seed={seed}, pruner={pruner_name})...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    best_params = best.params
    best_val = best.value
    best_test = best.user_attrs.get("test_mae", None)
    best_config = best.user_attrs.get("config", json.dumps(best_params))

    print("\n=== Optuna TPE Optimization Results ===")
    print(f"Best validation MAE: {best_val:.4f}")
    if best_test is not None and best_test != float("inf"):
        print(f"Corresponding test MAE: {best_test:.4f}")
    print("Best params:")
    for k, v in best_params.items():
        print(f"- {k}: {v}")
    print("Config string:")
    print(best_config)

    # Persist best to JSON
    notebooks_dir = os.path.dirname(__file__)
    best_json = os.path.join(notebooks_dir, "optuna_best.json")
    try:
        with open(best_json, "w", encoding="utf-8") as f:
            json.dump({
                "best_val_mae": best_val,
                "best_test_mae": best_test,
                "best_params": best_params,
                "config": best_config,
            }, f, indent=2)
        print(f"Saved best trial to {best_json}")
    except Exception:
        pass

    # Repro command for convenience
    repro = (
        f"$env:LR={best_params['lr']}; $env:DROPOUT={best_params['dropout']}; "
        f"$env:EMBED_DIM={best_params['embed_dim']}; $env:DNN_DIMS='{best_params['dnn_dims']}'; "
        f"$env:WEIGHT_DECAY={best_params['weight_decay']}; python train_deepfm.py"
    )
    print("\nPowerShell repro command:")
    print(repro)


if __name__ == "__main__":
    main()
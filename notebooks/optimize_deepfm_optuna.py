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

    # Newly exposed knobs
    if "dnn_use_bn" in params:
        env["DNN_USE_BN"] = "1" if params["dnn_use_bn"] else "0"
    if "l2_reg_linear" in params:
        env["L2_REG_LINEAR"] = str(params["l2_reg_linear"])
    if "l2_reg_embedding" in params:
        env["L2_REG_EMBEDDING"] = str(params["l2_reg_embedding"])
    if "l2_reg_dnn" in params:
        env["L2_REG_DNN"] = str(params["l2_reg_dnn"])
    if "batch_size" in params:
        env["BATCH_SIZE"] = str(params["batch_size"])
    if "use_lr_scheduler" in params:
        env["USE_LR_SCHEDULER"] = "1" if params["use_lr_scheduler"] else "0"
    if "lr_sched_factor" in params:
        env["LR_SCHED_FACTOR"] = str(params["lr_sched_factor"])
    if "lr_sched_patience" in params:
        env["LR_SCHED_PATIENCE"] = str(params["lr_sched_patience"])
    if "lr_sched_min_lr" in params:
        env["LR_SCHED_MIN_LR"] = str(params["lr_sched_min_lr"])

    # Forward optional LOG_DENSE if set in current environment
    if os.getenv("LOG_DENSE"):
        env["LOG_DENSE"] = os.getenv("LOG_DENSE")

    # Execute training script
    timeout_sec = int(os.getenv("OPT_TRAIN_TIMEOUT_SEC", "0") or "0")
    try:
        proc = subprocess.run(
            [sys.executable, "train_deepfm.py"],
            cwd=os.path.dirname(__file__),
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_sec if timeout_sec > 0 else None,
        )
    except subprocess.TimeoutExpired as te:
        stdout = te.stdout or ""
        stderr = te.stderr or ""
        config = (
            f"TIMEOUT after {timeout_sec}s. Params: LR={params['lr']}, WD={params['weight_decay']}, "
            f"DROP={params['dropout']}, EMBED={params['embed_dim']}, DNN={params['dnn_dims']}"
        )
        return float("inf"), float("inf"), config, stdout, stderr

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""

    combined = stdout + "\n" + stderr
    val_mae, test_mae = parse_mae(combined)

    # Create a compact config string for logging/inspection
    config = (
        f"LR: {params['lr']}, WD: {params['weight_decay']}, "
        f"DROPOUT: {params['dropout']}, EMBED_DIM: {params['embed_dim']}, "
        f"DNN_DIMS: {params['dnn_dims']}, PATIENCE: {patience}, EPOCHS: {epochs}, "
        f"BN: {int(params.get('dnn_use_bn', 0))}, BATCH: {params.get('batch_size', 'n/a')}, "
        f"L2_lin: {params.get('l2_reg_linear', 0)}, L2_emb: {params.get('l2_reg_embedding', 0)}, L2_dnn: {params.get('l2_reg_dnn', 0)}, "
        f"Sched: {int(params.get('use_lr_scheduler', 0))}, factor: {params.get('lr_sched_factor', 'n/a')}, "
        f"sched_patience: {params.get('lr_sched_patience', 'n/a')}, min_lr: {params.get('lr_sched_min_lr', 'n/a')}"
    )

    return val_mae, test_mae, config, stdout, stderr


def objective(trial: optuna.Trial) -> float:
    """Optuna objective minimizing validation MAE using TPE sampler.

    Customizable via environment variables:
    - Ranges: OPT_LR_MIN/OPT_LR_MAX, OPT_DROPOUT_MIN/OPT_DROPOUT_MAX, OPT_WD_MIN/OPT_WD_MAX
    - Choices: OPT_EMBED_CHOICES (comma list), OPT_DNN_CHOICES (semicolon-separated comma lists)
      New: OPT_BN_CHOICES (comma list of 0/1), OPT_L2_CHOICES (comma list of values),
           OPT_BATCH_CHOICES (comma list), OPT_USE_SCHED_CHOICES (comma list of 0/1),
           OPT_SCHED_FACTOR_MIN/MAX (floats), OPT_SCHED_PATIENCE_MIN/MAX (ints), OPT_MINLR_CHOICES (comma list)
    - Freeze: FIX_LR, FIX_DROPOUT, FIX_EMBED_DIM, FIX_WEIGHT_DECAY, FIX_DNN_DIMS
      New: FIX_DNN_USE_BN, FIX_L2_REG_LINEAR, FIX_L2_REG_EMBEDDING, FIX_L2_REG_DNN, FIX_BATCH_SIZE,
           FIX_USE_LR_SCHEDULER, FIX_LR_SCHED_FACTOR, FIX_LR_SCHED_PATIENCE, FIX_LR_SCHED_MIN_LR
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

    # Newly tunable knobs
    bn_choices = _parse_list(os.getenv("OPT_BN_CHOICES", "0,1"), int)
    l2_choices = [float(x) for x in os.getenv("OPT_L2_CHOICES", "0,1e-7,1e-6,1e-5,1e-4").split(",") if x.strip()]
    batch_choices = _parse_list(os.getenv("OPT_BATCH_CHOICES", "256,512,1024"), int)
    use_sched_choices = _parse_list(os.getenv("OPT_USE_SCHED_CHOICES", "1"), int)
    sched_factor_min = float(os.getenv("OPT_SCHED_FACTOR_MIN", "0.3"))
    sched_factor_max = float(os.getenv("OPT_SCHED_FACTOR_MAX", "0.7"))
    sched_pat_min = int(os.getenv("OPT_SCHED_PATIENCE_MIN", "3"))
    sched_pat_max = int(os.getenv("OPT_SCHED_PATIENCE_MAX", "8"))
    min_lr_choices = [float(x) for x in os.getenv("OPT_MINLR_CHOICES", "1e-5,5e-6,1e-6").split(",") if x.strip()]

    # Freeze overrides
    fix_lr = os.getenv("FIX_LR")
    fix_dropout = os.getenv("FIX_DROPOUT")
    fix_embed = os.getenv("FIX_EMBED_DIM")
    fix_wd = os.getenv("FIX_WEIGHT_DECAY")
    fix_dnn = os.getenv("FIX_DNN_DIMS")
    fix_bn = os.getenv("FIX_DNN_USE_BN")
    fix_l2_lin = os.getenv("FIX_L2_REG_LINEAR")
    fix_l2_emb = os.getenv("FIX_L2_REG_EMBEDDING")
    fix_l2_dnn = os.getenv("FIX_L2_REG_DNN")
    fix_batch = os.getenv("FIX_BATCH_SIZE")
    fix_use_sched = os.getenv("FIX_USE_LR_SCHEDULER")
    fix_sched_factor = os.getenv("FIX_LR_SCHED_FACTOR")
    fix_sched_patience = os.getenv("FIX_LR_SCHED_PATIENCE")
    fix_sched_min_lr = os.getenv("FIX_LR_SCHED_MIN_LR")

    # Suggest or use fixed values
    lr = float(fix_lr) if fix_lr is not None else trial.suggest_float("lr", lr_min, lr_max, log=True)
    dropout = float(fix_dropout) if fix_dropout is not None else trial.suggest_float("dropout", drop_min, drop_max)
    embed_dim = int(fix_embed) if fix_embed is not None else trial.suggest_categorical("embed_dim", embed_choices)
    weight_decay = float(fix_wd) if fix_wd is not None else trial.suggest_float("weight_decay", wd_min, wd_max, log=True)
    dnn_choice = fix_dnn if fix_dnn is not None else trial.suggest_categorical("dnn_dims", dnn_choices)

    dnn_use_bn = int(fix_bn) if fix_bn is not None else trial.suggest_categorical("dnn_use_bn", bn_choices)
    l2_reg_linear = float(fix_l2_lin) if fix_l2_lin is not None else trial.suggest_categorical("l2_reg_linear", l2_choices)
    l2_reg_embedding = float(fix_l2_emb) if fix_l2_emb is not None else trial.suggest_categorical("l2_reg_embedding", l2_choices)
    l2_reg_dnn = float(fix_l2_dnn) if fix_l2_dnn is not None else trial.suggest_categorical("l2_reg_dnn", l2_choices)
    batch_size = int(fix_batch) if fix_batch is not None else trial.suggest_categorical("batch_size", batch_choices)
    use_lr_scheduler = int(fix_use_sched) if fix_use_sched is not None else trial.suggest_categorical("use_lr_scheduler", use_sched_choices)
    lr_sched_factor = float(fix_sched_factor) if fix_sched_factor is not None else trial.suggest_float("lr_sched_factor", sched_factor_min, sched_factor_max)
    lr_sched_patience = int(fix_sched_patience) if fix_sched_patience is not None else trial.suggest_int("lr_sched_patience", sched_pat_min, sched_pat_max)
    lr_sched_min_lr = float(fix_sched_min_lr) if fix_sched_min_lr is not None else trial.suggest_categorical("lr_sched_min_lr", min_lr_choices)

    # Training budget controls
    epochs = int(os.getenv("OPTUNA_EPOCHS", os.getenv("TUNE_EPOCHS", "40")))
    patience = int(os.getenv("OPTUNA_PATIENCE", os.getenv("TUNE_PATIENCE", "12")))

    params = {
        "lr": lr,
        "dropout": dropout,
        "embed_dim": embed_dim,
        "weight_decay": weight_decay,
        "dnn_dims": dnn_choice,
        "dnn_use_bn": bool(dnn_use_bn),
        "l2_reg_linear": l2_reg_linear,
        "l2_reg_embedding": l2_reg_embedding,
        "l2_reg_dnn": l2_reg_dnn,
        "batch_size": batch_size,
        "use_lr_scheduler": bool(use_lr_scheduler),
        "lr_sched_factor": lr_sched_factor,
        "lr_sched_patience": lr_sched_patience,
        "lr_sched_min_lr": lr_sched_min_lr,
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
                f.write("trial,val_mae,test_mae,lr,dropout,embed_dim,dnn_dims,weight_decay,epochs,patience,batch_size,dnn_use_bn,l2_reg_linear,l2_reg_embedding,l2_reg_dnn,use_lr_scheduler,lr_sched_factor,lr_sched_patience,lr_sched_min_lr\n")
            f.write(
                f"{trial.number},{val_mae:.6f},{test_mae:.6f},{lr},{dropout},{embed_dim},"\
                f"{dnn_choice},{weight_decay},{epochs},{patience},{batch_size},{int(dnn_use_bn)},{l2_reg_linear},{l2_reg_embedding},{l2_reg_dnn},{int(use_lr_scheduler)},{lr_sched_factor},{lr_sched_patience},{lr_sched_min_lr}\n"
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
    show_progress = os.getenv("OPT_SHOW_PROGRESS", "1") == "1"
    print(f"Starting Optuna TPE optimization with {n_trials} trials (seed={seed}, pruner={pruner_name})...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress)

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
        f"$env:DNN_USE_BN={(1 if best_params.get('dnn_use_bn', False) else 0)}; "
        f"$env:L2_REG_LINEAR={best_params.get('l2_reg_linear', 0)}; $env:L2_REG_EMBEDDING={best_params.get('l2_reg_embedding', 0)}; $env:L2_REG_DNN={best_params.get('l2_reg_dnn', 0)}; "
        f"$env:WEIGHT_DECAY={best_params['weight_decay']}; $env:BATCH_SIZE={best_params.get('batch_size', 512)}; "
        f"$env:USE_LR_SCHEDULER={(1 if best_params.get('use_lr_scheduler', False) else 0)}; $env:LR_SCHED_FACTOR={best_params.get('lr_sched_factor', 0.5)}; "
        f"$env:LR_SCHED_PATIENCE={best_params.get('lr_sched_patience', 4)}; $env:LR_SCHED_MIN_LR={best_params.get('lr_sched_min_lr', 1e-5)}; "
        f"python train_deepfm.py"
    )
    print("\nPowerShell repro command:")
    print(repro)
    # Explicitly flush and exit cleanly to avoid lingering progress bar/threads on some Windows shells
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    return


if __name__ == "__main__":
    main()
    # Force process termination in edge cases where non-daemon threads linger
    try:
        sys.exit(0)
    except SystemExit:
        pass
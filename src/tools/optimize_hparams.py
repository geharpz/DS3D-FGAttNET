"""
Hyperparameter optimization script for ViolenceDualStreamNet using Optuna.

This script explores various hyperparameter configurations by training and
evaluating the model with each one, logging the validation performance,
and exporting the best configuration to a YAML file. It supports dynamic
reporting based on the selected early stopping metric from the configuration.
Includes safe logging, traceability, and graceful interruption handling.
"""

import os
import json
import yaml
import sys
import signal
import time
from typing import Tuple, List, Dict, Any

import optuna
import torch
from torch.utils.data import DataLoader
from optuna.pruners import MedianPruner
from functools import partial

from model.cnn3d import ViolenceDualStreamNet
from datasets.rwf_npy_dataset import RWFNpyDataset
from datasets.tensor_transform import TensorVideoTransforms
from utils.config_loader import ConfigLoader
from training.trainer import Trainer
from utils.seed import set_seed
import matplotlib.pyplot as plt
from types import SimpleNamespace  # Asegúrate de tenerlo importado

CONFIG_PATH: str = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
CONFIG_BEST_PATH: str = os.path.join(os.path.dirname(__file__), "..", "config", "config_best.yaml")

torch.set_num_threads(4)

def str_presenter(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='"')

def namespace_to_dict(obj):
    if isinstance(obj, SimpleNamespace):
        fields = vars(obj)
        # Descomprime si el único campo es "state"
        if list(fields.keys()) == ["state"]:
            return namespace_to_dict(fields["state"])
        return {k: namespace_to_dict(v) for k, v in fields.items()}
    elif isinstance(obj, dict):
        return {k: namespace_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [namespace_to_dict(i) for i in obj]
    else:
        return obj

def validate_config(config: Any) -> None:
    required_keys = ["training", "model", "optuna"]
    for key in required_keys:
        if not hasattr(config, key):
            raise ValueError(f"Missing required key in config: {key}")

def load_config() -> object:
    """
    Load configuration from YAML, prioritizing best configuration if available.

    Returns
    -------
    object
        Parsed config object.
    """
    print(f"Loading configuration from: {CONFIG_PATH}")
    config = ConfigLoader(CONFIG_PATH).get()
    validate_config(config)
    return config


def collect_npy_paths(base_path: str) -> Tuple[List[str], List[int]]:
    """
    Recursively collect .npy file paths and their class labels.

    Parameters
    ----------
    base_path : str
        Root directory containing class subfolders (e.g., 'Fight', 'NonFight').

    Returns
    -------
    Tuple[List[str], List[int]]
        List of file paths and corresponding labels.
    """
    paths, labels = [], []
    for class_name, label in zip(["Fight", "NonFight"], [1, 0]):
        class_dir = os.path.join(base_path, class_name)
        if not os.path.exists(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.endswith(".npy"):
                paths.append(os.path.join(class_dir, fname))
                labels.append(label)
    return paths, labels



def objective(trial: optuna.trial.Trial, config: object) -> float:
    """
    Objective function for Optuna hyperparameter optimization.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Trial object used by Optuna to suggest hyperparameters and report progress.
    config : object
        Loaded configuration object.

    Returns
    -------
    float
        Value of the target metric (defined dynamically from config) for optimization.
    """
    start_time = time.time()
    try:
        space = config.optuna.search_space
        lr = trial.suggest_float("learning_rate", space.learning_rate.low, space.learning_rate.high, log=space.learning_rate.log)
        dropout = trial.suggest_float("dropout", space.dropout.low, space.dropout.high)
        batch_size = trial.suggest_categorical("batch_size", space.batch_size.choices)
        optimizer = trial.suggest_categorical("optimizer", space.optimizer.choices)
        classifier_hidden = trial.suggest_categorical("classifier_hidden", space.classifier_hidden.choices)
        lr_step_size = trial.suggest_int("lr_step_size", space.lr_step_size.low, space.lr_step_size.high)
        lr_gamma = trial.suggest_float("lr_gamma", space.lr_gamma.low, space.lr_gamma.high)

        config.training.learning_rate = lr
        config.training.batch_size = batch_size
        config.training.optimizer = optimizer
        config.training.lr_step_size = lr_step_size
        config.training.lr_gamma = lr_gamma
        config.model.dropout = dropout
        config.model.classifier_hidden = classifier_hidden

        device = torch.device(config.general.device if torch.cuda.is_available() else "cpu")

        train_paths, train_labels = collect_npy_paths(os.path.join(config.dataset.root, "train"))
        val_paths, val_labels = collect_npy_paths(os.path.join(config.dataset.root, "val"))

        if not train_paths or not val_paths:
            raise RuntimeError("Training or validation data not found.")

        train_dataset = RWFNpyDataset(train_paths, train_labels, transform=TensorVideoTransforms.from_config(config, mode="train"), normalize_global=config.dataset.normalize_global, expected_shape=(5, 32, 224, 224))
        val_dataset = RWFNpyDataset(val_paths, val_labels, transform=TensorVideoTransforms.from_config(config, mode="val"), normalize_global=config.dataset.normalize_global, expected_shape=(5, 32, 224, 224))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=config.dataset.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=config.dataset.num_workers)

        model = ViolenceDualStreamNet(
            num_classes=config.model.num_classes,
            dropout=dropout,
            classifier_hidden=classifier_hidden
        ).to(device)

        trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader, config=config, device=device, start_epoch=1, trial=trial, train_labels=train_labels)
        history = trainer.train()

        trial_dir = f"outputs/hparams/trial_{trial.number}"
        os.makedirs(trial_dir, exist_ok=True)

        with open(os.path.join(trial_dir, "params.json"), "w") as f:
            json.dump(trial.params, f, indent=4)
        with open(os.path.join(trial_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=4)
        with open(os.path.join(trial_dir, "config_used.yaml"), "w") as f:
            yaml.dump(namespace_to_dict(config), f, sort_keys=False, default_flow_style=False)
        with open(os.path.join(trial_dir, "raw_predictions.json"), "w") as f:
            json.dump({
                "y_true": trainer.last_y_true,
                "y_pred": trainer.last_y_pred,
                "y_prob": trainer.last_y_prob
            }, f, indent=4)

        plot_history(history, trial_dir)
        save_summary_metrics(trial_dir, trial.number, history, start_time)

        target_metric = getattr(config.training, "early_stopping_metric", "val_loss")
        metric_value = history.get(target_metric, [None])[-1] if history.get(target_metric) else None
        if metric_value is None:
            raise RuntimeError(f"Metric '{target_metric}' not found or empty in training history.")

        return metric_value

    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        raise optuna.TrialPruned()


def save_summary_metrics(trial_dir: str, trial_number: int, history: Dict[str, List[float]], start_time: float) -> None:
    """
    Save a summary of the last value of all monitored metrics from the training history
    along with start and end timestamps.

    Parameters
    ----------
    trial_dir : str
        Path to the directory where the summary file will be saved.
    trial_number : int
        Optuna trial number for identification.
    history : Dict[str, List[float]]
        Training history dictionary containing lists of metric values per epoch.
    start_time : float
        Timestamp of when the trial started.

    Returns
    -------
    None
    """
    end_time = time.time()
    summary = {"trial_number": trial_number, "start_time": start_time, "end_time": end_time, "duration_sec": end_time - start_time}
    for metric_name, values in history.items():
        summary[metric_name] = values[-1] if values else None
    with open(os.path.join(trial_dir, "summary_metrics.json"), "w") as f:
        json.dump(summary, f, indent=4)
    print(f"Saved summary metrics to {os.path.join(trial_dir, 'summary_metrics.json')}")


def export_best_config(best_params: Dict[str, Any], config: object) -> None:
    """
    Export the best hyperparameters from Optuna to a YAML configuration file.

    Parameters
    ----------
    best_params : Dict[str, Any]
        Best hyperparameter values suggested by Optuna.
    config : object
        Configuration object to update.

    Returns
    -------
    None
    """
    config.training.learning_rate = float(best_params["learning_rate"])
    config.training.batch_size = int(best_params["batch_size"])
    config.training.lr_step_size = int(best_params["lr_step_size"])
    config.training.lr_gamma = float(best_params["lr_gamma"])
    config.training.optimizer = best_params["optimizer"]
    config.model.dropout = float(best_params["dropout"])
    config.model.classifier_hidden = int(best_params["classifier_hidden"])

    with open(CONFIG_BEST_PATH, "w") as f:
        yaml.dump(namespace_to_dict(config), f, sort_keys=False, default_flow_style=False)

    print(f"\nBest config exported to: {CONFIG_BEST_PATH}")


def safe_exit(sig, frame) -> None:
    """
    Safely handle script interruption by the user.
    """
    print("\nInterruption detected. Exiting...")
    sys.exit(0)
    
    
def plot_history(history: Dict[str, List[float]], trial_dir: str) -> None:
    for key, values in history.items():
        if not isinstance(values, list):
            continue
        plt.figure()
        plt.plot(values, marker='o')
        plt.title(f"{key} over epochs")
        plt.xlabel("Epoch")
        plt.ylabel(key)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(trial_dir, f"{key}_curve.png"))
        plt.close()

def main() -> None:
    """
    Run Optuna hyperparameter search and export the best configuration.
    """
    print("🔍 Starting Optuna hyperparameter optimization...\n")
    config = load_config()
    set_seed(config.general.seed) 
    signal.signal(signal.SIGINT, safe_exit)
    print(f"Optimizing for: {config.training.early_stopping_metric}")
    yaml.add_representer(str, str_presenter)   
    pruner = MedianPruner(n_warmup_steps=5, interval_steps=1)
    study = optuna.create_study(
        direction="minimize",
        study_name="cnn3d_hparam_opt",
        sampler=optuna.samplers.TPESampler(seed=config.general.seed),
        pruner=pruner
    )

    try:
        config.output = config.output.optuna
        objective_with_config = partial(objective, config=config)
        study.optimize(objective_with_config, n_trials=config.optuna.n_trials, n_jobs=1, timeout=config.optuna.timeout)
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")

    try:
        best = study.best_trial
        print(f"\nBest metric ({best.value:.4f}):")
        for k, v in best.params.items():
            print(f"  {k}: {v}")

        os.makedirs(config.output.report_hparams, exist_ok=True)
        with open(config.output.report_best_hparams, "w") as f:
            json.dump(best.params, f, indent=4)

        export_best_config(best.params, config)
    except ValueError as e:
        print(f"No completed trials found: {e}")


if __name__ == "__main__":
    main()

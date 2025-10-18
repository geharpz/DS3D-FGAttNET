"""
Main training script for ViolenceDualStreamNet using structured training loop.

Loads configuration, initializes datasets, model, and trainer, and executes
the training process with Weights & Biases logging, checkpointing, confusion matrix,
ROC curve reporting, and optional export of slimmed/pruned model. Ensures reproducibility
by saving used configuration and recording training duration and hyperparameters
in final metrics.
"""

import os
import json
import signal
import sys
from datetime import datetime
from typing import List, Tuple, Dict, Optional

import torch
from torch.utils.data import DataLoader
import wandb

from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

from model.violence_dualstream_cbam_se_fgf import ViolenceDualStreamNet
from datasets.rwf_npy_dataset import RWFNpyDataset
from datasets.tensor_transform import TensorVideoTransforms
from utils.config_loader import ConfigLoader
from training.trainer import Trainer
from utils.seed import set_seed
from training.utils.model_utils import ModelUtils
from torchinfo import summary


BASE_DIR = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(BASE_DIR, "..", "config", "config.yaml")
CONFIG_BEST_PATH = os.path.join(BASE_DIR, "..", "config", "config_best_v2.yaml")


def safe_exit(sig, frame) -> None:
    """
    Handles safe interruption of the training process.

    Parameters
    ----------
    sig : int
        Signal number.
    frame : object
        Frame object (ignored).
    """
    print("\nInterruption detected. Safely stopping training...")
    try:
        wandb.finish()
    except Exception:
        pass
    sys.exit(0)


def collect_npy_paths(base_path: str) -> Tuple[List[str], List[int]]:
    """
    Collect .npy file paths and their labels from dataset directories.

    Parameters
    ----------
    base_path : str
        Path to dataset split directory.

    Returns
    -------
    Tuple[List[str], List[int]]
        File paths and integer labels.
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


def load_config() -> object:
    """
    Load configuration YAML, prioritizing best config if exists.

    Returns
    -------
    object
        Parsed config object.
    """
    path = CONFIG_BEST_PATH if os.path.exists(CONFIG_BEST_PATH) else CONFIG_PATH
    print(f"Loading configuration from: {path}")
    return ConfigLoader(path).get()


def init_wandb(config: object) -> None:
    """
    Initialize Weights & Biases session.

    Parameters
    ----------
    config : object
        Configuration object.
    """
    wandb.init(
        project=config.general.project_name,
        config=config,
        name=f"{config.general.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        reinit=True
    )


def safe_get_last(history: Dict[str, List[float]], key: str) -> Optional[float]:
    """
    Safely get last value from metric history.

    Parameters
    ----------
    history : dict
        History dictionary containing lists of metrics.
    key : str
        Metric name.

    Returns
    -------
    float or None
        Last value of the metric if exists, otherwise None.
    """
    return history[key][-1] if key in history and history[key] else None


def save_final_metrics(history: Dict[str, List[float]], config: object, output_path: str,
                       start_time: datetime, end_time: datetime) -> Dict[str, float]:
    """
    Save final metrics, timings, and hyperparameters used.

    Parameters
    ----------
    history : dict
        Training history.
    config : object
        Configuration object.
    output_path : str
        Path to save the metrics JSON.
    start_time : datetime
        Training start time.
    end_time : datetime
        Training end time.

    Returns
    -------
    dict
        Final metrics including timing and hyperparameters.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    final_metrics = {
        "final_train_loss": safe_get_last(history, "train_loss"),
        "final_val_loss": safe_get_last(history, "val_loss"),
        "final_val_accuracy": safe_get_last(history, "val_accuracy"),
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_minutes": (end_time - start_time).total_seconds() / 60.0,
        "learning_rate": config.training.learning_rate,
        "batch_size": config.training.batch_size,
        "optimizer": config.training.optimizer,
        "lr_step_size": config.training.lr_step_size,
        "lr_gamma": config.training.lr_gamma
    }

    for metric in getattr(config.training, "metrics_to_monitor", []):
        final_metrics[f"final_{metric}"] = safe_get_last(history, metric)

    with open(output_path, "w") as f:
        json.dump(final_metrics, f, indent=4)

    print(f"Final metrics saved to: {output_path}")
    return final_metrics


def plot_and_save_confusion_matrix(y_true, y_pred, save_path):
    """
    Plot and save confusion matrix.

    Parameters
    ----------
    y_true : list
        True labels.
    y_pred : list
        Predicted labels.
    save_path : str
        Path to save the confusion matrix PNG.
    """
    cm = confusion_matrix(y_true, y_pred)
    labels = ["NonViolence", "Violence"]
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Final Epoch)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved Confusion Matrix to {save_path}")


def plot_and_save_roc_curve(y_true, y_prob, save_path):
    """
    Plot and save ROC curve.

    Parameters
    ----------
    y_true : list
        True labels.
    y_prob : list
        Predicted probabilities.
    save_path : str
        Path to save the ROC curve PNG.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Final Epoch)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved ROC Curve to {save_path}")


def main() -> None:
    """
    Main entry point for the training pipeline.
    """
    try:
        torch.set_num_threads(4)
        config = load_config()
        config.output = config.output.train
        set_seed(config.general.seed)
        signal.signal(signal.SIGINT, safe_exit)

        device = torch.device(config.general.device if torch.cuda.is_available() else "cpu")

        if getattr(config.logging, "use_wandb", False):
            init_wandb(config)

        os.makedirs(config.output.metrics_dir, exist_ok=True)
        with open(os.path.join(config.output.metrics_dir, "config_used.yaml"), "w") as f:
            json.dump(config, f, default=lambda o: o.__dict__, indent=4)

        train_paths, train_labels = collect_npy_paths(os.path.join(config.dataset.root, "train"))
        val_paths, val_labels = collect_npy_paths(os.path.join(config.dataset.root, "val"))

        if not train_paths or not val_paths:
            raise RuntimeError("No training or validation data found.")
        
        normalize_global=getattr(config.dataset, "normalize_global", True)
        
        train_dataset = RWFNpyDataset(train_paths, train_labels, transform=TensorVideoTransforms.from_config(config, mode="train"), normalize_global=normalize_global, expected_shape=(5, 32, 224, 224))
        val_dataset = RWFNpyDataset(val_paths, val_labels, transform=TensorVideoTransforms.from_config(config, mode="val"), normalize_global=normalize_global, expected_shape=(5, 32, 224, 224))

        train_loader = DataLoader(train_dataset, batch_size=config.dataset.batch_size, shuffle=True, num_workers=config.dataset.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=config.dataset.batch_size, shuffle=False, num_workers=config.dataset.num_workers)

        model = ViolenceDualStreamNet(
            num_classes=config.model.num_classes,
            dropout=config.model.dropout,
            classifier_hidden=getattr(config.model, "classifier_hidden", 128)
        ).to(device)
        summary(model, input_size=(config.dataset.batch_size, 5, 32, 224, 224), device=device.type)
        trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader, config=config, device=device, start_epoch=1, train_labels=train_labels)

        start_time = datetime.now()
        result = trainer.train()
        history = result["history"]
        early_stopping = result["stopped_early"]
        end_time = datetime.now()
        
        if early_stopping:
            print("Training was stopped due to early stopping.")

        final_metrics = save_final_metrics(history, config, config.output.metrics_path, start_time, end_time)
        
        with open(os.path.join(config.output.metrics_dir, "history.json"), "w") as f:
           json.dump(history, f, indent=4)
        
        for metric, values in history.items():
            plt.figure()
            plt.plot(values, marker='o')
            plt.title(f'{metric} over epochs')
            plt.xlabel('Epoch')
            plt.ylabel(metric)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(config.output.metrics_dir, f"{metric}_curve.png"))
            plt.close()

        if getattr(config.output, "save_slimmed_model", False):
            ModelUtils.export_slimmed_model(model, config.output.slimmed_model_path)

        # Plot confusion matrix and ROC curve
        y_true, y_pred, y_prob = trainer.get_last_validation_results()
        with open(os.path.join(config.output.metrics_dir, "final_predictions.json"), "w") as f:
            json.dump({
                "y_true": y_true,
                "y_pred": y_pred,
                "y_prob": y_prob
            }, f, indent=4)
            
        plot_and_save_confusion_matrix(y_true, y_pred, os.path.join(config.output.metrics_dir, "confusion_matrix.png"))
        plot_and_save_roc_curve(y_true, y_prob, os.path.join(config.output.metrics_dir, "roc_curve.png"))

        if wandb.run:
            artifact = wandb.Artifact(name="config_used", type="config")
            artifact.add_file(os.path.join(config.output.metrics_dir, "config_used.yaml"))
            wandb.log_artifact(artifact)
            wandb.log(final_metrics)
            wandb.finish()

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        try:
            wandb.finish()
        except Exception:
            pass
        sys.exit(0)


if __name__ == "__main__":
    main()

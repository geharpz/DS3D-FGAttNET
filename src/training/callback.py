import os
import json
from typing import Optional, Dict
import torch
from torch.nn import Module
import wandb
import yaml 
from datetime import datetime



class EarlyStopping:
    """
    Flexible early stopping utility to terminate training when the target metric stops improving.

    Parameters
    ----------
    patience : int
        Number of epochs to wait without improvement before stopping.
    monitor : str
        Name of the metric to monitor (used for logging purposes only).
    mode : str
        One of {'min', 'max'}. Whether lower or higher values are better.
    """

    def __init__(self, patience: int = 5, monitor: str = "val_loss", mode: str = "min", delta: float = 0.0) -> None:
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.delta = delta
        self.best_score: Optional[float] = None
        self.counter = 0

        if self.mode not in ["min", "max"]:
            raise ValueError("EarlyStopping mode must be 'min' or 'max'.")

    def step(self, current_score: float) -> bool:
        """
        Updates the internal counter and checks whether to stop training.

        Parameters
        
        ----------
        current_score : float
            Current value of the monitored metric.

        Returns
        -------
        bool
            True if training should be stopped, False otherwise.
        """
        if self.best_score is None:
            self.best_score = current_score
            return False

        improved = (
            (self.mode == "min" and current_score < self.best_score - self.delta) or
            (self.mode == "max" and current_score > self.best_score + self.delta)
        )

        if improved:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            print(f"EarlyStopping counter {self.counter} of {self.patience} — no improvement in '{self.monitor}'")

            if self.counter >= self.patience:
                print(f"Early stopping triggered — '{self.monitor}' did not improve for {self.patience} epochs.")
                return True

        return False

class ModelCheckpoint:
    """
    Utility to save model checkpoints and log them with WandB.

    Attributes
    ----------
    best_model_path : str
        Path to save the best model file.
    save_all_epochs : bool
        Whether to save models at every epoch.
    all_models_dir : str
        Directory to save per-epoch models.
    """

    def __init__(
        self,
        best_model_path: str = "checkpoints/best_model.pth",
        save_all_epochs: bool = False,
        all_models_dir: str = "checkpoints/all_epochs"
    ) -> None:
        """
        Parameters
        ----------
        best_model_path : str, optional
            Path to save the best model (default="checkpoints/best_model.pth").
        save_all_epochs : bool, optional
            If True, saves the model at every epoch (default=False).
        all_models_dir : str, optional
            Directory for saving per-epoch models (default="checkpoints/all_epochs").
        """
        self.best_model_path = best_model_path
        self.save_all_epochs = save_all_epochs
        self.all_models_dir = all_models_dir

        os.makedirs(os.path.dirname(self.best_model_path), exist_ok=True)
        if self.save_all_epochs:
            os.makedirs(self.all_models_dir, exist_ok=True)

    def save_best(self, model: Module, metrics: Optional[Dict[str, float]] = None, config: Optional[object] = None) -> None:
        """
        Saves the best model and optional metrics.

        Parameters
        ----------
        model : torch.nn.Module
            The model to be saved.
        metrics : dict, optional
            Metrics to store with the model.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._safe_save(model.state_dict(), self.best_model_path)
        print(f"{timestamp}:Best model saved at: {self.best_model_path}")

        if metrics:
            self._save_metrics_json(self.best_model_path, metrics)

        self._log_wandb_artifact("best_model", self.best_model_path, metrics)
        
        if config:
            self._save_config_yaml(self.best_model_path, config)

    def save_epoch(self, model: Module, epoch: int, metrics: Optional[Dict[str, float]] = None) -> None:
        """
        Saves the model at a given epoch.

        Parameters
        ----------
        model : torch.nn.Module
            Model to be saved.
        epoch : int
            Epoch number.
        metrics : dict, optional
            Metrics for the epoch.
        """
        if not self.save_all_epochs:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"model_epoch_{epoch:03d}_{timestamp}.pth"
        path = os.path.join(self.all_models_dir, filename)

        self._safe_save(model.state_dict(), path)
        print(f"{timestamp}:Model for epoch {epoch} saved at: {path}")

        if metrics:
            self._save_metrics_json(path, metrics)

        self._log_wandb_artifact(f"model_epoch_{epoch:03d}", path, metrics)

    def _safe_save(self, state_dict: dict, path: str) -> None:
        try:
            torch.save(state_dict, path)
        except Exception as e:
            print(f"Failed to save model at {path}: {e}")

    def _save_metrics_json(self, base_path: str, metrics: Dict[str, float]) -> None:
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metrics_path = os.path.splitext(base_path)[0] + "_metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=4)
            print(f"{timestamp}:Metrics saved to: {metrics_path}")
        except Exception as e:
            print(f"Failed to save metrics: {e}")

    def _log_wandb_artifact(self, name: str, model_path: str, metrics: Optional[Dict[str, float]] = None) -> None:
        if wandb.run is None:
            return
        try:
            artifact = wandb.Artifact(name=name, type="model", metadata=metrics or {})
            artifact.add_file(model_path)
            if metrics:
                metrics_path = os.path.splitext(model_path)[0] + "_metrics.json"
                if os.path.exists(metrics_path):
                    artifact.add_file(metrics_path)
            wandb.log_artifact(artifact)
            print(f"Logged model '{name}' as WandB artifact.")
        except Exception as e:
            print(f"Failed to log WandB artifact '{name}': {e}")
            
    def _save_config_yaml(self, base_path: str, config_obj: object) -> None:
        """
        Save a YAML representation of the configuration used during training.

        Parameters
        ----------
        base_path : str
            Base path to derive the YAML output path.
        config_obj : object
            Configuration object (namespace or dict).
        """
        try:
            config_path = os.path.splitext(base_path)[0] + "_config.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config_obj, f, default_flow_style=False, allow_unicode=True)
            print(f"Saved config YAML to: {config_path}")
        except Exception as e:
            print(f"Failed to save config YAML: {e}")

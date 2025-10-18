import os
from typing import Optional, Dict, Any, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import optuna
import wandb

from training.callback import EarlyStopping, ModelCheckpoint
from training.metrics.metric_utils import MetricUtils
from training.pruning import Prunning
from training.utils.log_utils import LogUtils
from training.utils.optuna_utils import OptunaUtils
from reporting.confusion import Confusion
from reporting.classification import Classification
from reporting.roc import Roc
from reporting.save_predictions import RawPrediction
from utils.checkpoints import Checkpoint
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import time

class Trainer:
    """
    Trainer class for 3D CNN models including AMP, pruning, early stopping,
    Optuna integration, dynamic metric monitoring and detailed logging.

    Parameters
    ----------
    model : nn.Module
        Model to train.
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader
        Validation data loader.
    config : Any
        Configuration object parsed from YAML.
    device : torch.device
        Training device ('cuda' or 'cpu').
    start_epoch : int, optional
        Epoch to resume training from. Default is 1.
    trial : optuna.trial.Trial, optional
        Optuna trial for pruning. Default is None.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Any,
        device: torch.device,
        start_epoch: int = 1,
        trial: Optional[optuna.trial.Trial] = None,
        train_labels: Optional[List[int]] = None
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.start_epoch = start_epoch
        self.trial = trial

        if getattr(config.training, "freeze_backbone", False):
            print("Freezing backbone layers...")
            for name, param in model.named_parameters():
                if any(name.startswith(f"{layer}.") for layer in ["rgb_stream.0", "rgb_stream.1", "rgb_stream.2", "rgb_stream.3", "rgb_stream.4",
                                                                  "rgb_stream.5",
                                                                  "flow_stream.0", "flow_stream.1", "flow_stream.2", "flow_stream.3", "flow_stream.4",
                                                                  "flow_stream.5"]):
                    param.requires_grad = False
                    print(f"Layer frozen: {name}")
        self.initial_sparsity = getattr(config.model, "initial_pruning", 0.1)
        self.final_sparsity = getattr(config.model, "final_pruning", 0.5)
        self.sparsity_epochs = getattr(config.model, "pruning_epochs", 10)
        self.pruning_warmup = getattr(config.model, "pruning_warmup", 2)
        self.confusion = Confusion(self.config)
        self.classification = Classification(self.config)
        self.roc = Roc(self.config)
        self.raw_prediction = RawPrediction(self.config)
        self.prunning = Prunning(self.model, self.config, self.initial_sparsity,
                                 self.final_sparsity, self.sparsity_epochs, self.pruning_warmup)
        self.optuna_util = OptunaUtils(self.config, self.trial)
        smoothing = config.training.label_smoothing_value if config.training.use_label_smoothing else 0.0
        if getattr(config.training, "use_class_weight", False) and train_labels is not None:
            class_weights = compute_class_weight(
                class_weight="balanced",
                classes=np.array([0, 1]),
                y=np.array(train_labels)
            )
            assert len(
                class_weights) == 2, "Expected 2 class weights for binary classification."

            if getattr(config.training, "normalize_class_weights", False):
                class_weights = class_weights / class_weights.sum()
                print("Normalized class weights to sum to 1.")

            weight_tensor = torch.tensor(
                class_weights, dtype=torch.float32).to(device)
            self.criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing = smoothing)
            print(f"Using class weights: {class_weights}")
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing = smoothing)
            
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=config.training.learning_rate,
            weight_decay=getattr(config.training, "weight_decay", 0.0)
        )

        self.scheduler = None
        if getattr(config.training, "lr_scheduler", None) == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=config.training.optimizer_mode,
                factor=config.training.optimizer_factor,
                patience=config.training.optimizer_patience
            )

        self.scaler = GradScaler(device=device)
        self.metrics_to_monitor = getattr(config.training, "metrics_to_monitor", [
                                          "f1", "precision", "recall", "auc"])
        self.early_stopping_metric = getattr(
            config.training, "early_stopping_metric", "val_loss")
        self.early_stopping = EarlyStopping(
            patience = config.training.early_stopping_patience,
            monitor = self.early_stopping_metric,
            mode = config.training.early_stopping_mode,
            delta = config.training.early_stopping_delta
        )
        self.modelCheckpoint = ModelCheckpoint(
            best_model_path=config.output.checkpoints_path,
            save_all_epochs=True,
            all_models_dir=os.path.join(
                os.path.dirname(config.output.checkpoints_path), "all_epochs"
            )
        )
        self.checkpoint = Checkpoint(
            self.model, self.optimizer, self.scaler, self.scheduler, self.config, self.device, self.start_epoch, self.trial, self.early_stopping)
        self.checkpoint.load_checkpoint()
        
        self.start_epoch = self.checkpoint.get_start_epoch()
        
        if self.checkpoint and self.early_stopping.best_score is not None:
            self.best_metric_value = self.early_stopping.best_score
            print("self.best_metric_value restaurado como:", self.best_metric_value)
        else:
            self.best_metric_value = float("inf") if self.early_stopping_metric == "val_loss" else -float("inf")


        
        self.history = {metric: [] for metric in ["train_loss", "val_loss",
                                                  "val_accuracy", "epoch_duration_sec", "train_samples"] + self.metrics_to_monitor}

        self.last_y_true: List[int] = []
        self.last_y_pred: List[int] = []
        self.last_y_prob: List[float] = []
        
        #assert self.best_metric_value == self.early_stopping.best_score, \
        #   f"Inconsistent best score: {self.best_metric_value} != {self.early_stopping.best_score}"

    def train(self) -> Dict[str, List[Any]]:
        """
        Execute the complete training loop including validation, pruning, early stopping, and logging.

        Returns
        -------
        Dict[str, List[float]]
            History of training and validation metrics.
        """

        y_true, y_pred, y_prob = [], [], []
        stopped_early = False
        last_epoch = self.start_epoch
        
        for epoch in range(self.start_epoch, self.config.training.epochs):
            last_epoch = epoch 
            print(f"\nEpoch {epoch}/{self.config.training.epochs}")
            start_epoch_time = time.time()
            self.prunning .update_pruning(epoch)

            train_loss = self._train_one_epoch()
            val_loss, val_acc, y_true, y_pred, y_prob = self._validate_one_epoch()

            epoch_duration = time.time() - start_epoch_time
            try:
                # Robust metric computation with fallback on exception
                metrics = MetricUtils.compute(y_true, y_pred, y_prob)
            except Exception as e:
                print(f"Metric computation failed: {e}")
                metrics = {metric: 0.0 for metric in self.metrics_to_monitor}
                continue 

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_accuracy"].append(val_acc)
            self.history["epoch_duration_sec"].append(epoch_duration)
            self.history["train_samples"].append(
                len(self.train_loader.dataset))
            for metric in self.metrics_to_monitor:
                self.history[metric].append(metrics.get(metric, 0.0))

            log_data = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "epoch_duration_sec": epoch_duration,
                "train_samples": len(self.train_loader.dataset),
                **{metric: metrics.get(metric, 0.0) for metric in self.metrics_to_monitor}
            }

            if getattr(self.config.logging, "use_wandb", False) and wandb.run:
                wandb.log(log_data)

            LogUtils.save_metrics_json(
                self.config.output.metrics_dir, epoch, log_data)
            self.modelCheckpoint.save_epoch(self.model, epoch)

            current_metric = metrics.get(
                self.early_stopping_metric, val_loss if self.early_stopping_metric == "val_loss" else 0.0)
            
            if self.best_metric_value in [float("inf"), -float("inf")]:
                self.best_metric_value = current_metric
                print(f"Inicializado best_metric_value = {self.best_metric_value}")

            if (self.early_stopping_metric == "val_loss" and current_metric < self.best_metric_value) or \
                    (self.early_stopping_metric != "val_loss" and current_metric > self.best_metric_value):
                self.best_metric_value = current_metric
                self.modelCheckpoint.save_best(
                    self.model, metrics=log_data, config=self.config)

                
            if self.scheduler:
                prev_lr = self.optimizer.param_groups[0]["lr"]
                self.scheduler.step(current_metric)
                new_lr = self.optimizer.param_groups[0]["lr"]
                if new_lr != prev_lr:
                    print(f"Learning rate reduced: {prev_lr:.6f} ➜ {new_lr:.6f}")

            stopped_early = self.early_stopping.step(current_metric)
            self.checkpoint.save_last_checkpoint(epoch)
            
            if stopped_early:
                print("Early stopping triggered.")
                print("Training completed with early stopping at epoch", epoch)
                last_epoch = epoch
                break

            self.optuna_util.report_optuna(metrics, epoch)
            
        if y_true and y_pred:
            self.confusion.log_confusion_matrix(y_true, y_pred, last_epoch)
            self.classification.log_classification_report(y_true, y_pred)
            self.roc.log_roc_curve(y_true, y_prob)
            self.raw_prediction.save_raw_predictions(
                y_true, y_pred, y_prob, last_epoch)

        self.last_y_true = y_true
        self.last_y_pred = y_pred
        self.last_y_prob = y_prob

        return {
            "history": self.history,
            "stopped_early": stopped_early,
            "last_epoch": last_epoch,
            "best_metric_value": self.best_metric_value
        }

    def _train_one_epoch(self) -> float:
        """
        Train the model for one epoch.

        Returns
        -------
        float
            Average training loss.
        """
        self.model.train()
        total_loss = 0.0
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            if inputs.ndim != 5 or inputs.shape[1:] != (5, 32, 224, 224):
                raise ValueError(f"Expected input shape (5, 32, 224, 224), but got {inputs.shape}")
                
            self.optimizer.zero_grad()
            if self.config.training.use_amp:
                with autocast(device_type=self.device.type):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def _validate_one_epoch(self) -> Tuple[float, float, List[int], List[int], List[float]]:
        """
        Validate the model for one epoch.

        Returns
        -------
        Tuple[float, float, List[int], List[int], List[float]]
            Validation loss, accuracy, true labels, predicted labels, probabilities.
        """
        self.model.eval()
        val_loss, correct, total = 0.0, 0, 0
        y_true, y_pred, y_prob = [], [], []

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                if inputs.ndim != 5 or inputs.shape[1:] != (5, 32, 224, 224):
                    raise ValueError(f"Expected input shape (5, 32, 224, 224), but got {inputs.shape}")
                with autocast(device_type=self.device.type):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)[:, 1]
                preds = torch.argmax(outputs, dim=1)
                y_true.extend(labels.cpu().tolist())
                y_pred.extend(preds.cpu().tolist())
                y_prob.extend(probs.cpu().tolist())
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        self.last_y_true = y_true
        self.last_y_pred = y_pred
        self.last_y_prob = y_prob
        if not y_true:
            print("Warning: No validation data was collected.")
            return 1.0, 0.0, [0], [0], [0.0]

        return val_loss / len(self.val_loader), correct / total, y_true, y_pred, y_prob

    def get_last_validation_results(self) -> Tuple[List[int], List[int], List[float]]:
        """
        Retrieve the most recent validation predictions and labels.

        Returns
        -------
        Tuple[List[int], List[int], List[float]]
            - True labels from the last validation epoch.
            - Predicted labels.
            - Predicted probabilities for the positive class.
        """
        return (
            getattr(self, "last_y_true", []),
            getattr(self, "last_y_pred", []),
            getattr(self, "last_y_prob", []),
        )

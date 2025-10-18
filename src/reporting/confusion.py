import os
import json
from typing import Any, List
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import wandb


class Confusion:
    """
    Class for reporting confusion matrices of a 3D CNN model
    
    Parameters
    ----------
    config : Any
        Configuration object parsed from YAML.
    """

    def __init__(self, config: Any) -> None:
        self.config = config

    def log_confusion_matrix(self, y_true: List[int], y_pred: List[int], epoch: int) -> None:
        """
        Log and save confusion matrix.

        Parameters
        ----------
        y_true : List[int]
            True labels.
        y_pred : List[int]
            Predicted labels.
        epoch : int
            Current epoch.
        """
        if not y_true or not y_pred:
            print("⚠️ No data to log confusion matrix.")
            return

        cm = confusion_matrix(y_true, y_pred)
        labels = ["NonViolence", "Violence"]

        fig = plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", square=True,
                    xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix - Epoch {epoch}")

        # Log to wandb if enabled
        if getattr(self.config.logging, "use_wandb", False) and wandb.run:
            wandb.log({"confusion_matrix": wandb.Image(fig)})

        # Save to PNG if configured
        if hasattr(self.config.output, "report_confusion_matrix"):
            os.makedirs(os.path.dirname(
                self.config.output.report_confusion_matrix), exist_ok=True)
            fig.savefig(self.config.output.report_confusion_matrix)
            print(
                f"Saved confusion matrix to {self.config.output.report_confusion_matrix}")

        plt.close(fig)

        # Save to JSON if configured
        if getattr(self.config.output, "save_confusion_json", False):
            os.makedirs(self.config.output.confusion_dir, exist_ok=True)
            with open(f"{self.config.output.confusion_dir}/cm_epoch_{epoch}.json", "w") as f:
                json.dump(cm.tolist(), f, indent=4)

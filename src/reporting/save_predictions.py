import os
import json
from typing import Any, List

class RawPrediction:
    """
    Class for reporting RawPrediction of a 3D CNN model
    
    Parameters
    ----------
    config : Any
        Configuration object parsed from YAML.
    """
    
    def __init__(self, config: Any) -> None:
        self.config = config
        
    def save_raw_predictions(self, y_true: List[int], y_pred: List[int], y_prob: List[float], epoch: int) -> None:
        """
        Save raw predictions, probabilities, and true labels as a JSON file.

        This method saves the raw predictions, probabilities, and ground truth labels
        collected during validation to a JSON file, facilitating reproducibility and
        later auditing of the model outputs.

        Parameters
        ----------
        y_true : List[int]
            List of true labels.
        y_pred : List[int]
            List of predicted labels.
        y_prob : List[float]
            List of predicted probabilities for the positive class (violence).
        epoch : int
            Current training epoch, used to label the output file.

        Returns
        -------
        None
        """
        if hasattr(self.config.output, "metrics_dir"):
            os.makedirs(self.config.output.metrics_dir, exist_ok=True)
            with open(f"{self.config.output.metrics_dir}/raw_predictions_epoch_{epoch}.json", "w") as f:
                json.dump({
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "y_prob": y_prob
                }, f, indent=4)
            print(
                f"Saved raw predictions to {self.config.output.metrics_dir}/raw_predictions_epoch_{epoch}.json")
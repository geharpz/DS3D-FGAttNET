import os
import matplotlib.pyplot as plt
from typing import Any, List
from sklearn.metrics import roc_curve, auc

class Roc:
    """
    Class for log Roc of a 3D CNN model
    
    Parameters
    ----------
    config : Any
        Configuration object parsed from YAML.
    """
    def __init__(self, config: Any)->None:
        self.config = config
        
    def log_roc_curve(self, y_true: List[int], y_prob: List[float]) -> None:
        """
        Generate and save the ROC curve as an image file.

        The method computes the Receiver Operating Characteristic (ROC) curve and
        Area Under the Curve (AUC) using sklearn, then plots and saves it to the
        path specified in the configuration.

        Parameters
        ----------
        y_true : List[int]
            List of true labels.
        y_prob : List[float]
            List of predicted probabilities for the positive class (violence).

        Returns
        -------
        None
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")

        if hasattr(self.config.output, "report_roc_curve"):
            os.makedirs(os.path.dirname(
                self.config.output.report_roc_curve), exist_ok=True)
            plt.savefig(self.config.output.report_roc_curve)
            print(
                f"Saved ROC curve to {self.config.output.report_roc_curve}")
        plt.close()
        
    
        
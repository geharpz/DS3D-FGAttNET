
import os
import json
from typing import Any, List
from sklearn.metrics import classification_report

class Classification:
    """
    Class for classification reports of a 3D CNN model
    
    Parameters
    ----------
    config : Any
        Configuration object parsed from YAML.
    """
    
    def __init__(self, config: Any) -> None:
        self.config = config
        
    
    def log_classification_report(self, y_true: List[int], y_pred: List[int]) -> None:
        """
        Generate and save the classification report as a JSON file.

        The method computes the classification report using sklearn's `classification_report`
        with class names ["NonViolence", "Violence"] and saves the report as a JSON file
        to the path specified in the configuration.

        Parameters
        ----------
        y_true : List[int]
            List of true labels.
        y_pred : List[int]
            List of predicted labels.

        Returns
        -------
        None
        """
        report = classification_report(y_true, y_pred, target_names=[
                                       "NonViolence", "Violence"], output_dict=True)
        if hasattr(self.config.output, "report_classification_json"):
            os.makedirs(os.path.dirname(
                self.config.output.report_classification_json), exist_ok=True)
            with open(self.config.output.report_classification_json, "w") as f:
                json.dump(report, f, indent=4)
            print(
                f"Saved classification report to {self.config.output.report_classification_json}")
        
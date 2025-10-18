"""
Logging utilities for metrics and confusion matrices.

This module provides static methods to persist training logs such as metrics and confusion matrices in JSON format.
It ensures robustness by validating input data before saving.

Author
------
TFM VIU AI Thesis Assistant
"""

import os
import json
from typing import Dict, List
from datetime import datetime


class LogUtils:
    """
    Utility class for saving logs to disk in JSON format.

    Methods
    -------
    save_metrics_json(log_dir, epoch, metrics)
        Saves a dictionary of metrics into a JSON file per epoch.
    save_confusion_matrix_json(log_dir, epoch, matrix)
        Saves a confusion matrix into a JSON file per epoch.
    """

    @staticmethod
    def save_metrics_json(log_dir: str, epoch: int, metrics: Dict[str, float]) -> None:
        """
        Save a dictionary of metrics into a JSON file.

        Parameters
        ----------
        log_dir : str
            Directory where metrics will be saved.
        epoch : int
            Current epoch number.
        metrics : Dict[str, float]
            Dictionary containing metric names and their values.

        Raises
        ------
        ValueError
            If metrics is empty or not a dictionary.
        """
        if not isinstance(metrics, dict) or not metrics:
            raise ValueError("Metrics must be a non-empty dictionary.")

        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(log_dir, f"epoch_{epoch:03d}_{timestamp}.json")
        with open(filepath, "w") as f:
            json.dump(metrics, f, indent=4)

    @staticmethod
    def save_confusion_matrix_json(log_dir: str, epoch: int, matrix: List[List[int]]) -> None:
        """
        Save a confusion matrix as a JSON file.

        Parameters
        ----------
        log_dir : str
            Directory to save the matrix file.
        epoch : int
            Current epoch number.
        matrix : List[List[int]]
            2D confusion matrix as list of lists (raw values).

        Raises
        ------
        ValueError
            If matrix is empty or not a list of lists.
        """
        if not matrix or not all(isinstance(row, list) for row in matrix):
            raise ValueError("Matrix must be a non-empty list of lists.")

        os.makedirs(log_dir, exist_ok=True)
        filepath = os.path.join(log_dir, f"confusion_matrix_epoch_{epoch:03d}.json")
        with open(filepath, "w") as f:
            json.dump({"confusion_matrix": matrix}, f, indent=4)

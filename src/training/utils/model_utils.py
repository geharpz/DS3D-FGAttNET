"""
Model utility functions for model pruning removal and exporting.

This module provides utilities to remove pruning masks and save the model in a slimmed (pruning-free) format.
It is designed to work generically with PyTorch models and supports Conv3D layers as used in violence detection tasks.

Author
------
TFM VIU AI Thesis Assistant
"""

import os
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


class ModelUtils:
    """
    Utility class for handling model export operations like removing pruning hooks and saving slimmed models.
    """

    @staticmethod
    def export_slimmed_model(model: nn.Module, export_path: str) -> None:
        """
        Remove structured pruning reparameterizations from the model and export the slimmed version.

        This is essential to obtain a clean model for deployment without pruning masks that affect inference speed.

        Parameters
        ----------
        model : nn.Module
            The trained PyTorch model with potential pruning applied.
        export_path : str
            Path where the slimmed model's state_dict will be saved.

        Raises
        ------
        RuntimeError
            If the export path's directory does not exist or there is an issue saving the model.
        """
        if not os.path.exists(os.path.dirname(export_path)):
            os.makedirs(os.path.dirname(export_path), exist_ok=True)

        for module in model.modules():
            if isinstance(module, nn.Conv3d) and hasattr(module, "weight_mask"):
                try:
                    prune.remove(module, "weight")
                except Exception as e:
                    print(f"Failed to remove pruning from {module.__class__.__name__}: {e}")

        try:
            torch.save(model.state_dict(), export_path)
            print(f"Slimmed model successfully saved to: {export_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to save slimmed model to {export_path}: {e}")

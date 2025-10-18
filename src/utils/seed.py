"""
Random seed utility for reproducible experiments.

This module provides a function to set seeds across Python, NumPy, and PyTorch
to ensure deterministic behavior in deep learning experiments.

References
----------
- https://pytorch.org/docs/stable/notes/randomness.html
- https://numpy.org/doc/stable/reference/random/index.html
"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Sets random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Parameters
    ----------
    seed : int, optional
        The seed value to use for all random number generators. Default is 42.

    Notes
    -----
    This function sets:
    - Python built-in random module
    - NumPy random generator
    - PyTorch CPU and CUDA seeds
    - Disables cuDNN benchmarking and enables deterministic algorithms
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional, Callable
from .tensor_validator import TensorValidator 
from .tensor_transform import NormalizeVideo

class RWFNpyDataset(Dataset):
    """
    Dataset class for loading preprocessed RWF-2000 clips stored in `.npy` format.

    Each `.npy` file must contain a 4D array with shape `(D, H, W, 5)`, where the
    last dimension corresponds to 5 channels: 3 RGB channels and 2 optical flow
    channels. The data is transposed into PyTorch tensor format `(C=5, D, H, W)`.

    Optionally applies a global normalization scheme if no explicit normalization
    transform is provided.

    Parameters
    ----------
    npy_paths : list of str
        List of file paths pointing to `.npy` files containing video clips.
        Each file must have shape `(D, H, W, 5)`.
    labels : list of int
        List of labels aligned with `npy_paths`. Typically `0` or `1` for
        binary violence detection.
    transform : callable, optional
        A transform function or composition with signature
        `Callable[[torch.Tensor], torch.Tensor]`. If it contains a
        `NormalizeVideo` instance, the internal global normalization is skipped.
    normalize_global : bool, default=True
        If True, applies default normalization:
        
        * RGB channels scaled to `[-1, 1]`.
        * Optical flow channels clipped to `[-20, 20]` and scaled to `[-1, 1]`.
    expected_shape : tuple of int, optional
        Expected tensor shape `(C, D, H, W)` per sample.
        Default is `(5, 32, 224, 224)`. Raises error if mismatched.

    Raises
    ------
    FileNotFoundError
        If a `.npy` file in `npy_paths` does not exist.
    ValueError
        If the `.npy` array has unexpected shape or tensor validation fails.

    Examples
    --------
    >>> dataset = RWFNpyDataset(["clip1.npy", "clip2.npy"], [0, 1])
    >>> len(dataset)
    2
    >>> x, y = dataset[0]
    >>> x.shape
    torch.Size([5, 32, 224, 224])
    >>> y
    0
    """
    def __init__(
        self,
        npy_paths: List[str],
        labels: List[int],
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        normalize_global: bool = True,
        expected_shape: Optional[Tuple[int, int, int, int]] = (5, 32, 224, 224)
    ) -> None:
        self.npy_paths = npy_paths
        self.labels = labels
        self.transform = transform
        self.normalize_global = normalize_global
        self.expected_shape = expected_shape
        self.validator = TensorValidator()

    def __len__(self) -> int:
        return len(self.npy_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Load and return a single sample from the dataset.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        tuple
            A tuple `(tensor, label)` where:
            
            * `tensor` : torch.Tensor
                Video clip tensor of shape `(5, D, H, W)`.
            * `label` : int
                Label associated with the clip.
        """
        npy_path = self.npy_paths[idx]
        label = self.labels[idx]

        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"File not found: {npy_path}")

        data = np.load(npy_path)  # (D, H, W, 5)
        if data.ndim != 4 or data.shape[-1] != 5:
            raise ValueError(f"Expected shape (D, H, W, 5), got {data.shape}")

        tensor = torch.tensor(data.transpose(3, 0, 1, 2), dtype=torch.float32)  # (5, D, H, W)

        # Optional global normalization (if transform does not already normalize)
        if self.normalize_global and not self._has_normalization():
            rgb = (tensor[:3] / 255.0 - 0.5) / 0.5
            flow = torch.clamp(tensor[3:], -20.0, 20.0) / 20.0
            tensor = torch.cat([rgb, flow], dim=0)

        if self.transform:
            tensor = self.transform(tensor)

        if tensor.ndim == 5 and tensor.shape[0] > 5:
            raise ValueError(f"Got a batch (B, 5, D, H, W) inside __getitem__. This is unexpected.")
        if self.expected_shape and tensor.shape != self.expected_shape:
            raise ValueError(f"Shape mismatch: expected {self.expected_shape}, got {tensor.shape} (individual sample)")

        self.validator.validate(tensor, label)
        return tensor, label

    def _has_normalization(self) -> bool:
        """
        Check whether the provided transform already includes normalization.

        Returns
        -------
        bool
            True if the transform includes a `NormalizeVideo` instance, else False.
        """

        if not self.transform:
            return False
        return any(isinstance(t, NormalizeVideo) for t in getattr(self.transform, 'transforms', []))


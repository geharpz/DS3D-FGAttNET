import torch
from torch import Tensor
from typing import Tuple


class TensorValidator:
    """
    Validates tensors and labels for 3D CNN models using RGB + Optical Flow input.

    This class ensures:
    - Tensor shape is exactly as expected (C, D, H, W).
    - Tensor contains no NaN or Inf values.
    - Tensor has meaningful variation (i.e., not all values are constant).
    - Tensor is optionally within normalized range [-1, 1].
    - Label is an integer and within allowed classes.

    Parameters
    ----------
    expected_shape : Tuple[int, int, int, int], default=(5, 32, 224, 224)
        Expected shape of the input tensor (Channels, Depth, Height, Width).
    check_range : bool, default=True
        Whether to verify that values lie in normalized range [-1, 1].
    allowed_labels : Tuple[int, int], default=(0, 1)
        Permitted label classes (typically 0: NonViolent, 1: Violent).
    """

    def __init__(
        self,
        expected_shape: Tuple[int, int, int, int] = (5, 32, 224, 224),
        check_range: bool = True,
        allowed_labels: Tuple[int, int] = (0, 1)
    ) -> None:
        self.expected_shape = expected_shape
        self.check_range = check_range
        self.allowed_labels = allowed_labels

    def validate(self, tensor: Tensor, label: int) -> None:
        """
        Run all validation checks on a video tensor and its associated label.

        Parameters
        ----------
        tensor : Tensor
            Video tensor of shape (C, D, H, W), expected to contain 5 channels.
        label : int
            Associated class label for the video.

        Raises
        ------
        ValueError
            If the tensor fails any shape, range, or semantic checks.
        TypeError
            If the label is not an integer.
        """
        # Shape check
        if tensor.shape != self.expected_shape:
            raise ValueError(
                f"Shape mismatch: expected {self.expected_shape}, got {tensor.shape}"
            )

        # NaN and Inf check
        if torch.isnan(tensor).any():
            raise ValueError("Tensor contains NaN values.")
        if torch.isinf(tensor).any():
            raise ValueError("Tensor contains infinite values.")

        # Check range if normalized
        if self.check_range:
            min_val, max_val = tensor.min().item(), tensor.max().item()
            if min_val < -1.0 or max_val > 1.0:
                raise ValueError(
                    f"ensor values out of expected range [-1, 1]: min={min_val:.3f}, max={max_val:.3f}"
                )
                
        if torch.std(tensor) < 1e-5:
            raise ValueError("Tensor has near-zero variance; likely empty or uniform.")

        # Label type and value check
        if not isinstance(label, int):
            raise TypeError(f"Label must be of type int, got {type(label)}")
        if label not in self.allowed_labels:
            raise ValueError(
                f"Invalid label {label}. Allowed values are {self.allowed_labels}"
            )

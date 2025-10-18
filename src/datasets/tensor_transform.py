import torch
import torch.nn.functional as F
import random
from typing import Tuple, List


import torch
from typing import Optional, List


class NormalizeVideo:
    """
    Normalize RGB + Optical Flow video tensor using either:
    - Standard normalization to [-1, 1] for RGB and z-score for flow channels.
    - Statistical normalization using provided mean and std per channel.

    This transform expects a tensor of shape (C=5, D, H, W), where:
    - Channels 0:2 are RGB
    - Channels 3:4 are Optical Flow (X, Y)

    Parameters
    ----------
    mean : Optional[List[float]]
        Channel-wise mean values. Required if using statistical mode.
    std : Optional[List[float]]
        Channel-wise standard deviations. Required if using statistical mode.
    mode : str, default="standard"
        Type of normalization: "standard" or "statistical".

    Raises
    ------
    ValueError
        If statistical mode is selected but mean/std are missing or incorrect.
    """

    def __init__(
        self,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        mode: str = "standard"
    ) -> None:
        self.mean = mean
        self.std = std
        self.mode = mode

        if self.mode == "statistical":
            if self.mean is None or self.std is None:
                raise ValueError("Mean and std must be provided for statistical normalization.")
            if len(self.mean) != 5 or len(self.std) != 5:
                raise ValueError("Mean and std must be lists of 5 floats (RGB + FlowX + FlowY).")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply normalization to the video tensor.

        Parameters
        ----------
        x : torch.Tensor
            Video tensor of shape (5, D, H, W), values in range [0, 255].

        Returns
        -------
        torch.Tensor
            Normalized tensor.
        """
        if self.mode == "standard":
            # Normalize RGB to [-1, 1]
            x[:3] = (x[:3] / 255.0 - 0.5) / 0.5
            x[3:5] = torch.clamp(x[3:5], -20.0, 20.0) / 20.0

        elif self.mode == "statistical":
            # Normalize using provided mean and std per channel
            for c in range(5):
                x[c] = (x[c] / 255.0 - self.mean[c]) / self.std[c]

        else:
            raise ValueError(f"Unknown normalization mode: {self.mode}")

        return torch.clamp(x, -1.0, 1.0)
class MotionBasedCrop:
    """
    Crop video tensor around the region with highest motion intensity.

    Uses the magnitude of optical flow (channels 3 and 4) to locate the
    most active spatial region, and crops a fixed window around it.

    Parameters
    ----------
    crop_size : Tuple[int, int]
        Height and width of the crop window.
    resize_if_needed : bool, default=True
        Whether to resize the input spatially if it's smaller than the crop size.
    """

    def __init__(self, crop_size: Tuple[int, int], resize_if_needed: bool = True):
        self.th, self.tw = crop_size
        self.resize_if_needed = resize_if_needed

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape

        if self.resize_if_needed and (H < self.th or W < self.tw):
            x = F.interpolate(x, size=(self.th, self.tw), mode='bilinear', align_corners=False)
            return x

        flow = x[3:5]  # (2, D, H, W)
        mag = torch.norm(flow, dim=0).mean(dim=0)  # (H, W)

        cy, cx = torch.nonzero(mag == mag.max(), as_tuple=True)
        cy, cx = cy.item(), cx.item()

        i0 = max(0, min(cy - self.th // 2, H - self.th))
        j0 = max(0, min(cx - self.tw // 2, W - self.tw))

        return x[:, :, i0:i0 + self.th, j0:j0 + self.tw]
    
    
class ResizeVideo:
    """
    Resize entire video spatial dimensions.

    Parameters
    ----------
    size : Tuple[int, int]
        Desired output size (height, width).
    """

    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=self.size, mode='bilinear', align_corners=False)
    
class TemporalJitter:
    """
    Apply temporal crop or pad to a fixed length.

    Parameters
    ----------
    target_length : int
        Desired number of frames.
    """

    def __init__(self, target_length: int):
        self.target_length = target_length

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        _, D, _, _ = x.shape
        if D > self.target_length:
            start = random.randint(0, D - self.target_length)
            return x[:, start:start + self.target_length, :, :]
        elif D < self.target_length:
            pad = self.target_length - D
            return torch.cat([x] + [x[:, -1:, :, :]] * pad, dim=1)
        return x


class TemporalReverse:
    """
    Reverse video sequence with probability p.

    Parameters
    ----------
    p : float
        Probability of applying.
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.flip(dims=[1]) if random.random() < self.p else x


class RandomCrop:
    """
    Random spatial crop.

    Parameters
    ----------
    size : Tuple[int, int]
        Crop size (height, width).
    """

    def __init__(self, size: Tuple[int, int]):
        self.th, self.tw = size

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape
        if H < self.th or W < self.tw:
            x = F.interpolate(x, size=(self.th, self.tw), mode='bilinear', align_corners=False)
        i = random.randint(0, H - self.th)
        j = random.randint(0, W - self.tw)
        return x[:, :, i:i + self.th, j:j + self.tw]


class CenterCrop:
    """
    Center crop.

    Parameters
    ----------
    size : Tuple[int, int]
        Crop size (height, width).
    """

    def __init__(self, size: Tuple[int, int]):
        self.th, self.tw = size

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape
        if H < self.th or W < self.tw:
            x = F.interpolate(x, size=(self.th, self.tw), mode='bilinear', align_corners=False)
        i = (H - self.th) // 2
        j = (W - self.tw) // 2
        return x[:, :, i:i + self.th, j:j + self.tw]


class RandomHorizontalFlipTensor:
    """
    Horizontal flip with probability p.

    Parameters
    ----------
    p : float
        Probability of flipping.
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.flip(dims=[3]) if random.random() < self.p else x


class BrightnessJitter:
    """
    Brightness scaling for RGB channels.

    Parameters
    ----------
    p : float
        Probability of applying.
    min_factor : float
        Minimum brightness factor.
    max_factor : float
        Maximum brightness factor.
    """

    def __init__(self, p: float = 0.3, min_factor: float = 0.8, max_factor: float = 1.2):
        self.p = p
        self.min = min_factor
        self.max = max_factor

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            factor = random.uniform(self.min, self.max)
            x[:3] *= factor
        return x


class Cutout:
    """
    Apply cutout to simulate occlusion.

    Parameters
    ----------
    size : int
        Cutout square size.
    p : float
        Probability of applying.
    """

    def __init__(self, size: int = 20, p: float = 0.5):
        self.size = size
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            _, _, h, w = x.shape
            cx = random.randint(0, w - 1)
            cy = random.randint(0, h - 1)
            half = self.size // 2
            x[:, :, max(0, cy - half):cy + half, max(0, cx - half):cx + half] = 0
        return x


class FlowJitter:
    """
    Add noise to optical flow channels.

    Parameters
    ----------
    p : float
        Probability of applying.
    noise_std : float
        Noise standard deviation.
    """

    def __init__(self, p: float = 0.3, noise_std: float = 0.05):
        self.p = p
        self.noise_std = noise_std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            noise = torch.randn_like(x[3:5]) * self.noise_std
            x[3:5] += noise
        return x


class ComposeTensorTransforms:
    """
    Apply a sequence of video transforms.
    """

    def __init__(self, transforms: List):
        self.transforms = transforms

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            x = t(x)
        return x.contiguous()


class TensorVideoTransforms:
    """
    Create video transformation pipelines from config.

    Methods
    -------
    from_config(config: Dict, mode: str) -> ComposeTensorTransforms
        Create transformation pipeline.
    """

    @staticmethod
    def from_config(config: dict, mode: str = 'train') -> ComposeTensorTransforms:
        aug = config.augmentations
        transforms = []
        
        if hasattr(aug, "resize_video") and aug.resize_video.enabled:
            transforms.append(ResizeVideo(size=tuple(aug.resize_video.size)))


        if mode == 'train':
            if hasattr(aug, "motion_based_crop") and aug.motion_based_crop.enabled:
                transforms.append(MotionBasedCrop(crop_size=tuple(aug.motion_based_crop.crop_size)))

            if aug.temporal_jitter.enabled:
                transforms.append(TemporalJitter(aug.temporal_jitter.target_length))
            if aug.temporal_reverse.enabled:
                transforms.append(TemporalReverse(aug.temporal_reverse.p))
            if aug.random_crop.enabled:
                transforms.append(RandomCrop(tuple(aug.motion_based_crop.crop_size)))
            if aug.random_horizontal_flip.enabled:
                transforms.append(RandomHorizontalFlipTensor(aug.random_horizontal_flip.p))
            if aug.brightness_jitter.enabled:
                transforms.append(BrightnessJitter(aug.brightness_jitter.p, aug.brightness_jitter.min_factor, aug.brightness_jitter.max_factor))
            if aug.cutout.enabled:
                transforms.append(Cutout(aug.cutout.size, aug.cutout.p))
            if aug.flow_jitter.enabled:
                transforms.append(FlowJitter(aug.flow_jitter.p, aug.flow_jitter.noise_std))

        elif mode == 'val':
            if hasattr(aug, "resize_video") and aug.resize_video.enabled:
                transforms.append(ResizeVideo(size=tuple(aug.resize_video.size)))
            if hasattr(aug, "motion_based_crop") and aug.motion_based_crop.enabled:
                transforms.append(MotionBasedCrop(crop_size=tuple(aug.motion_based_crop.crop_size)))
            transforms.append(CenterCrop(tuple(aug.motion_based_crop.crop_size)))
            if aug.temporal_jitter.enabled:
                transforms.append(TemporalJitter(aug.temporal_jitter.target_length))

        if hasattr(aug, "normalize_video") and aug.normalize_video.enabled:
            mode = getattr(aug.normalize_video, "use_mode", "standard")
            
            if mode == "statistical":
                transforms.append(NormalizeVideo(
                    mean=aug.normalize_video.mean,
                    std=aug.normalize_video.std,
                    mode="statistical"
                ))
            elif mode == "standard":
                transforms.append(NormalizeVideo(mode="standard"))
            else:
                raise ValueError("Invalid normalize_video.use_mode. Choose 'standard' or 'statistical'.")
        return ComposeTensorTransforms(transforms)

    @staticmethod
    def get_basic_transform() -> ComposeTensorTransforms:
        return ComposeTensorTransforms([NormalizeVideo()])

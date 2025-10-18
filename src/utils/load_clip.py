import numpy as np
import torch
from typing import Tuple, List

def load_clip_from_npy(
    npy_path: str,
    start_frame: int = 0,
    clip_len: int = 32,
    mean_rgb: Tuple[float, float, float] = (0.43216, 0.394666, 0.37645),
    std_rgb: Tuple[float, float, float] = (0.22803, 0.22145, 0.216989),
    mean_flow: Tuple[float, float] = (0.5, 0.5),
    std_flow: Tuple[float, float] = (0.226, 0.226)
) -> Tuple[torch.Tensor, List[np.ndarray]]:
    """
    Load and preprocess a clip from .npy video data.

    Parameters
    ----------
    npy_path : str
        Path to .npy file (shape: [T, 224, 224, 5]).
    start_frame : int
        Starting index of the clip.
    clip_len : int
        Number of frames in the clip.
    mean_rgb : tuple
        Normalization mean for RGB channels.
    std_rgb : tuple
        Normalization std for RGB channels.
    mean_flow : tuple
        Normalization mean for Flow channels.
    std_flow : tuple
        Normalization std for Flow channels.

    Returns
    -------
    torch.Tensor
        Clip tensor ready for the model. Shape: (1, 5, 32, 224, 224)
    List[np.ndarray]
        List of 32 original RGB frames (uint8, shape (224,224,3)) for visualization.
    """
    data = np.load(npy_path)  # shape: (T, 224, 224, 5)
    assert data.shape[0] >= start_frame + clip_len, "Clip fuera de rango"

    clip = data[start_frame:start_frame + clip_len]  # (32, 224, 224, 5)

    rgb = clip[..., :3] / 255.0 
    flow = clip[..., 3:] / 255.0

    original_rgb_frames = [np.uint8(f * 255) for f in rgb]

    for i in range(3):
        rgb[..., i] = (rgb[..., i] - mean_rgb[i]) / std_rgb[i]
    for i in range(2):
        flow[..., i] = (flow[..., i] - mean_flow[i]) / std_flow[i]

    rgb = rgb.transpose(3, 0, 1, 2) 
    flow = flow.transpose(3, 0, 1, 2)
    clip_tensor = np.concatenate([rgb, flow], axis=0)

    return torch.tensor(clip_tensor, dtype=torch.float32).unsqueeze(0), original_rgb_frames

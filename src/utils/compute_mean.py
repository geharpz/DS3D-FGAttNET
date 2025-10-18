
import os
import numpy as np
from tqdm import tqdm

def compute_mean_std(npy_dir: str) -> tuple[list[float], list[float]]:
    """
    Compute mean and std across all video tensors in a dataset folder (RWF-2000 .npy format).
    
    Parameters
    ----------
    npy_dir : str
        Path to dataset directory (e.g., 'data/RWF-2000/train')
    
    Returns
    -------
    Tuple[List[float], List[float]]
        Channel-wise means and stds for C=5 channels (RGB + Flow)
    """
    sum_ = np.zeros(5)
    sum_sq = np.zeros(5)
    n_total = 0

    for class_name in ["Fight", "NonFight"]:
        class_dir = os.path.join(npy_dir, class_name)
        for fname in tqdm(os.listdir(class_dir), desc=f"Processing {class_name}"):
            if fname.endswith(".npy"):
                path = os.path.join(class_dir, fname)
                tensor = np.load(path)  # shape: (5, D, H, W)
                tensor = tensor.astype(np.float32) / 255.0

                # Flatten spatial and temporal dims: (5, D*H*W)
                flat = tensor.reshape(5, -1)
                sum_ += flat.mean(axis=1)
                sum_sq += (flat ** 2).mean(axis=1)
                n_total += 1

    mean = (sum_ / n_total).tolist()
    std = (np.sqrt(sum_sq / n_total - np.square(sum_ / n_total))).tolist()
    return mean, std


# Example usage:
train_dir = os.path.join("data", "npy", "train") 
mean, std = compute_mean_std(train_dir)
print("Mean:", mean)
print("Std :", std)

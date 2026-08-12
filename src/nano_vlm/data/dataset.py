import numpy as np
import torch

from torch.utils.data import Dataset


class SyntheticVLDataset(Dataset):
    """
    PyTorch Dataset for the synthetic NanoVLM data.
    """

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        # PIL image -> NumPy array
        image = np.array(sample["image"], dtype=np.float32)

        # [H, W, C] -> [C, H, W]
        image = torch.from_numpy(image).permute(2, 0, 1)

        # Normalize pixel values from [0, 255] -> [0, 1]
        image = image / 255.0

        return {
            "image": image,
            "caption": sample["caption"],
            "color": sample["color"],
            "shape": sample["shape"],
            "position": sample["position"],
        }
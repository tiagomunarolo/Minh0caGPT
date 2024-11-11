import numpy as np
import torch
from torch.utils.data import Dataset


class TorchDataSet(Dataset):
    """
    A PyTorch Dataset class for loading data.
    Args:
        x (np.ndarray): Input data array.
        y (np.ndarray): Target data array.
        device (str): Device for data loading.
    """

    def __init__(self,
                 x: np.ndarray,
                 y: np.ndarray,
                 device='cpu'):
        super().__init__()
        self.x = torch.tensor(x, dtype=torch.int32, device=device)
        self.y = torch.tensor(y, dtype=torch.int8, device=device)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

"""Module of wrapper datasets."""
from typing import Union, Tuple

from torch.utils.data import Dataset
import torch
import numpy as np


class Zip(Dataset):
    """Wrapper dataset behaves like `zip()`."""
    def __init__(self, *datas: Union[np.ndarray, torch.Tensor]):
        """init

        Args:
            *data (List of ArrayLike): A series of arraies.
        """

        assert len(datas) > 0
        self.datas = datas

    def __len__(self):
        return len(self.datas[0])

    def __getitem__(self, idx):
        return tuple(d[idx] for d in self.datas)


class Enumerate(Dataset):
    """Wrapper dataset behaves like `enumerate()`."""
    def __init__(self, data: Union[np.ndarray, torch.Tensor, Tuple[Union[np.ndarray, torch.Tensor]]]):
        """init

        Args:
            data (ArrayLike or Tuple): Input array or tuple of arraies.
        """
        assert len(data) > 0
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return idx, self.data[idx]

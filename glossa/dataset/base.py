from __future__ import annotations
from typing import List, Any

from glossa.preprocess.base import BasePreprocess
from glossa.encoder.base import BaseEncoder

from torch.utils.data import Dataset


class BaseDataset(Dataset):  # noqa
    """
    Base Dataset

    A PyTorch Dataset Base Class.
    """

    def __init__(self, preprocess: List[BasePreprocess | Any] = None, encoder: BaseEncoder | Any = None):
        self.preprocess_func = preprocess
        self.encoder_func = encoder

    def preprocess(self, x):
        for func in self.preprocess_func:
            x = func.preprocess(x)
        return x

    def encode(self, x, **kwargs):
        return self.encoder_func.encode(x, **kwargs)

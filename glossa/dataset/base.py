from __future__ import annotations
from typing import List, Any

from glossa.preprocess.base import BasePreprocess
from glossa.encoder.base import BaseEncoder

from torch.utils.data import Dataset, DataLoader


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


class PredictDataset(BaseDataset):  # noqa
    """
    Predict Dataset

    A PyTorch Dataset for Inference (without y or labels).
    """

    def __init__(self, x, preprocess: List[BasePreprocess | Any] = None, encoder: BaseEncoder | Any = None):
        super(PredictDataset, self).__init__(preprocess, encoder)

        if self.preprocess_func:
            x = [self.preprocess(data) for data in x]

        if self.encoder_func:
            x = [self.encoder_func.encode(data, suppress=True) for data in x]

        self.x = x

    def __getitem__(self, item):
        return self.x[item][0], self.x[item][1]

    def __len__(self):
        return len(self.x)


def create_predict_set(data: Any, encoder: BaseEncoder = None):
    """ Create a Predict Dataset

    :param data: Any form of data
    :param encoder: Any encoder (BaseEncoder)
    :return: PredictDataset
    """
    return PredictDataset(x=data, encoder=encoder)


def create_predict_loader(data: Any,
                          encoder: BaseEncoder = None,
                          batch_size: int = 1) -> DataLoader:
    """ Create a Predict DataLoader

    :param data: Any form of data
    :param encoder: Any encoder (BaseEncoder)
    :param batch_size: DataLoader Batch Size (default = 1)
    :return: DataLoader
    """
    pred_ds = create_predict_set(data=data, encoder=encoder)
    pred_dl = DataLoader(pred_ds, shuffle=False, batch_size=batch_size)
    return pred_dl

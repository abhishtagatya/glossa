from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from glossa.exception import GlossaValueError
from glossa.classification.language.method import lstm_clf


def train_lstm_clf(dataset: Dataset,
                   model: Any,
                   model_parameter: Dict = None,
                   epochs: int = 20,
                   batch_size: int = 1000,
                   lr: float = 0.001,
                   split_ratio: Tuple[int, int] = (0.8, 0.2),
                   save_as: str = None):
    train_set, val_set = torch.utils.data.random_split(dataset, split_ratio)
    train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_set, batch_size=batch_size)

    if model_parameter is None:
        raise GlossaValueError(code='MISSING_MODEL_PARAM')
    dec_model: nn.Module = model(**model_parameter)
    lstm_clf.train_model(
        model=dec_model,
        train_loader=train_dl,
        val_loader=val_dl,
        epochs=epochs,
        lr=lr
    )

    if save_as:
        torch.save(dec_model.state_dict(), save_as)

    return dec_model


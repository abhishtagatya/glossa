from __future__ import annotations

from typing import Tuple, Dict, Any, AnyStr

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

from glossa.exception import GlossaValueError
from glossa.encoder import BaseEncoder
from glossa.dataset import create_predict_loader
from glossa.classification.language.method import lstm_clf
from glossa.util.download import download_github_pretrained


def _lookup_pretrained_state_lstm_clf(key: AnyStr) -> Dict:
    """ Lookup Pretrained Models File and URL from Key.

    :param key: Key of pretrained model.
    :return: Dict (file, url) or None
    """
    sd_lookup = {
        'lstm16-3455-20': {
            'file': '/lstm_fm16-100-256-1000_20-3455_ord.pt',
            'url': 'https://github.com/abhishtagatya/glossa/raw/main/pretrained/lstm_clf/lstm_fm16-100-256-1000_20-3455_ord.pt',
            'param': {
                'vocab_size': 3455,
                'embedding_dim': 100,
                'hidden_dim': 256,
                'out_dim': 16,
                'drop': 0
            }
        },
        'lstm16-3455-30': {
            'file': '/lstm_fm16-100-256-1000_30-3455_ord.pt',
            'url': 'https://github.com/abhishtagatya/glossa/raw/main/pretrained/lstm_clf/lstm_fm16-100-256-1000_30-3455_ord.pt',
            'param': {
                'vocab_size': 3455,
                'embedding_dim': 100,
                'hidden_dim': 256,
                'out_dim': 16,
                'drop': 0
            }
        },
    }
    sd_lookup['default'] = sd_lookup['lstm16-3455-20']

    return sd_lookup.get(key, None)


def load_pretrained_lstm_clf(
        model: Any,
        pretrained: AnyStr = 'default',
        model_parameter: Dict = None,
        set_train: bool = False) -> nn.Module:
    """ Load Pretrained LSTM Classifier Model

    :param model: LSTM Model (glossa.classification.language.method.lstm_clf)
    :param pretrained: Pretrained model key
    :param model_parameter: Parameter of Model (Dict)
    :param set_train: Set as Train (bool)
    :return: nn.Module
    """
    sd_lookup = _lookup_pretrained_state_lstm_clf(pretrained)
    if sd_lookup is None:
        raise GlossaValueError('MISSING_OBJ_KEY', key=pretrained, dictobj=_lookup_pretrained_state_lstm_clf.__name__)

    sd_dl_path = download_github_pretrained(sd_lookup['url'], sd_lookup['file'])

    if model_parameter:
        dec_model: nn.Module = model(**model_parameter)
    else:
        dec_model: nn.Module = model(**sd_lookup['param'])
    dec_model.load_state_dict(torch.load(sd_dl_path))

    if set_train:
        dec_model.train()
    else:
        dec_model.eval()

    return dec_model


def train_lstm_clf(dataset: Dataset,
                   model: Any,
                   model_parameter: Dict = None,
                   epochs: int = 20,
                   batch_size: int = 1000,
                   lr: float = 0.001,
                   split_ratio: Tuple[int, int] = (0.8, 0.2),
                   save_as: str = None):
    """ Train an LSTM Classifier Model

    :param dataset: Dataset to train and validate (torch Dataset)
    :param model: Model to train
    :param model_parameter: Parameter for Model (Dict)
    :param epochs: Epochs
    :param batch_size: Batch Size
    :param lr: Learning Rate
    :param split_ratio: Train and Validation Split Ratio
    :param save_as: Model name to save
    :return:
    """
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


def use_lstm_clf(data: Any, encoder: BaseEncoder | Any, model: Any):
    """ Use an LSTM Classifier Model

    :param data: Any form of data
    :param encoder: Any encoder (BaseEncoder)
    :param model: Any torch model (nn.Module)
    :return:
    """
    pred_dl = create_predict_loader(data=data, encoder=encoder)
    outputs = []

    model.eval()
    for x, l in pred_dl:
        y_hat = model(x, l)
        pred = torch.max(y_hat, 1)[1]
        outputs.append(pred.item())

    return outputs

from __future__ import annotations

import os.path
from typing import List, Any

import numpy as np
import pandas as pd
import torch

from glossa.dataset import BaseDataset
from glossa.preprocess import BasePreprocess
from glossa.encoder import BaseEncoder, OrdEncoder
from glossa.exception import GlossaValueError
from glossa.util.download import download_kaggle_dataset


class LanguageDataset(BaseDataset):  # noqa
    """
    Language Dataset

    A PyTorch Dataset Class for Storing Text and their Language Category.\n
    `x = Text, y = Language.`
    """

    def __init__(self, x, y, preprocess: List[BasePreprocess | Any] = None, encoder: BaseEncoder | Any = None):
        super(LanguageDataset, self).__init__(preprocess, encoder)

        if self.preprocess_func:
            x = [self.preprocess(data) for data in x]

        if self.encoder_func:
            x = [self.encoder_func.encode(data, suppress=True) for data in x]

        self.x = x
        self.y = y
        self.label = {y: x for x, y in enumerate(np.unique(y))}

    def from_label(self, y):
        return list(self.label.keys())[list(self.label.values()).index(y)]

    def to_label(self, y):
        return self.label[y]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        return (
            self.x[item], self.y[item]
        )

    @classmethod
    def from_df(cls,
                df: pd.DataFrame,
                x_col: str = 'Text',
                y_col: str = 'Language',
                preprocess: List[BasePreprocess | Any] = None,
                encoder: BaseEncoder | Any = None):
        return cls(x=df[x_col].tolist(), y=df[y_col].tolist(), preprocess=preprocess, encoder=encoder)

    @classmethod
    def autoload_10k(cls,
                     preprocess: List[BasePreprocess | Any] = None,
                     encoder: BaseEncoder | Any = None):
        """
        Automatically downloads a selected Dataset from Kaggle for Language Detection.
        Loading the Dataset class with 10K and 16 classes.

        Class:
        `Arabic`, `Danish`, `Dutch`, `English`, `French`, `German`, `Greek`,
        `Italian`, `Kannada`, `Malayalam`, `Portuguese`, `Russian`, `Spanish`,
        `Swedish`, `Tamil`, `Turkish`

        :param preprocess: List of Preprocess Class or Functions (Subclass glossa.preprocess.BasePreprocess)
        :param encoder: An Encoder Class or Function (Subclass glossa.encoder.BaseEncoder)

        :return:
        """
        download_kaggle_dataset('basilb2s/language-detection')

        sub_folder = '/dataset'
        ds_path = os.getenv('GLOSSA_DATA', '.data') + sub_folder + '/Language Detection.csv'

        if not os.path.isfile(ds_path):
            raise GlossaValueError('MISSING_PATH', path=ds_path)

        x_col, y_col = 'Text', 'Language'
        support_lang = [
            'English', 'Malayalam', 'Tamil', 'Portugeese', 'French',
            'Dutch', 'Spanish', 'Greek', 'Russian', 'Danish', 'Italian',
            'Turkish', 'Sweedish', 'Arabic', 'German', 'Kannada'
        ]
        df = pd.read_csv(ds_path)
        df = df[df[y_col].isin(support_lang)]

        return cls.from_df(df=df, x_col='Text', y_col='Language', preprocess=preprocess, encoder=encoder)


class OrdLanguageDataset(LanguageDataset):
    """
    Language Dataset (with OrdEncoder Loaded)

    A PyTorch Dataset Class for Storing Text and their Language Category.\n
    `x = Text (Encoded), y = Language.`
    """

    def __init__(self, x, y, preprocess: List[BasePreprocess | Any] = None, encoder: BaseEncoder | Any = None):
        super(OrdLanguageDataset, self).__init__(x, y, preprocess, encoder or OrdEncoder())

    def __getitem__(self, item):
        return (
            self.x[item][0],
            self.label[self.y[item]],
            self.x[item][1]
        )

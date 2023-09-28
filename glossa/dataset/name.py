from __future__ import annotations

import os
import glob
from typing import List, Any

import numpy as np
import pandas as pd

from glossa.dataset import BaseDataset
from glossa.preprocess import BasePreprocess
from glossa.encoder import BaseEncoder
from glossa.util.download import download_torch_data


class ForeignNameDataset(BaseDataset): # noqa
    """
    Name Dataset

    A PyTorch Dataset Class for Storing Text that are foreign names.\n
    `x = Text, y = Origin.`
    """

    def __init__(self, x, y, preprocess: List[BasePreprocess | Any] = None, encoder: BaseEncoder | Any = None):
        super(ForeignNameDataset, self).__init__(preprocess, encoder)

        if self.preprocess_func:
            x = [self.preprocess(data) for data in x]

        if self.encoder_func:
            x = [self.encode(data, suppress=True) for data in x]

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
                x_col: str = 'Name',
                y_col: str = 'Origin',
                preprocess: List[BasePreprocess | Any] = None,
                encoder: BaseEncoder | Any = None):
        return cls(x=df[x_col].tolist(), y=df[y_col].tolist(), preprocess=preprocess, encoder=encoder)

    @classmethod
    def autoload_torch(cls,
                       preprocess: List[BasePreprocess | Any] = None,
                       encoder: BaseEncoder | Any = None):
        """
        Automatically downloads the dataset from the PyTorch Repository for Foreign Names.
        Loading the Dataset class with 20K and 18 classes.

        Class:
        `Arabic`, `Chinese`, `Czech`, `Dutch`, `English`, `French`, `German`, `Greek`,
        `Irish`, `Italian`, `Japanese`, `Korean`, `Polish`, `Portuguese`, `Russian`,
        `Scottish`, `Spanish`, `Vietnamese`

        :param preprocess: List of Preprocess Class or Functions (Subclass glossa.preprocess.BasePreprocess)
        :param encoder: An Encoder Class or Function (Subclass glossa.encoder.BaseEncoder)

        :return:
        """

        def find_files(fpath):
            return glob.glob(fpath)

        def read_lines(fname: str):
            with open(fname, encoding='utf-8') as nameFile:
                return nameFile.read().strip().split('\n')

        download_torch_data('https://download.pytorch.org/tutorial/data.zip')

        sub_folder = '/dataset/torch'
        ds_folder = '/data/names/*.txt'
        full_dir = os.getenv('GLOSSA_DATA', '.data') + sub_folder + ds_folder

        data = []

        for filename in sorted(find_files(full_dir)):
            category = os.path.splitext(os.path.basename(filename))[0]
            data.extend([(name, category) for name in read_lines(filename)])

        x_col, y_col = 'Name', 'Origin'
        df = pd.DataFrame(data=data, columns=[x_col, y_col])

        return cls.from_df(df=df, x_col=x_col, y_col=y_col, preprocess=preprocess, encoder=encoder)

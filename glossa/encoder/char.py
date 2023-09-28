import warnings
from typing import Any, AnyStr

import numpy as np
import torch

from glossa.common import *
from glossa.encoder.base import BaseEncoder
from glossa.exception import GlossaEncoderWarning


class OrdEncoder(BaseEncoder):
    """
    Ord Encoder

    Encodes a characters in a string object using ```ord()``` to get their unicode representation.
    """

    def __init__(self, unk_ord: int = 1, max_ord: int = 3455, max_length: int = 250):
        super(OrdEncoder, self).__init__()
        self.unk_ord = unk_ord
        self.max_ord = max_ord
        self.max_length = max_length

    def encode(self, x: AnyStr, suppress: bool = False):
        if any(ord(xi) > self.max_ord for xi in x) and not suppress:
            warnings.warn(
                f'{x} contains character above the `max_ord`. Characters will be casted to `unk_ord`.',
                GlossaEncoderWarning
            )

        encoded = np.zeros(self.max_length, dtype=int)
        encoder = np.array([ord(char) if ord(char) <= self.max_ord else self.unk_ord for char in list(x)])
        length = min(self.max_length, len(encoder))
        encoded[:length] = encoder[:length]

        return torch.from_numpy(encoded.astype(np.int32)), length

    @property
    def min(self):
        return 0

    @property
    def max(self):
        return self.max_ord


class CharOneHotEncoder(BaseEncoder):
    """
    Char One-Hot Encoder

    Encodes a character into a one-hot representation of a tensor.
    """

    def __init__(self, index_set: AnyStr = ASCII_CHARSET):
        super(CharOneHotEncoder, self).__init__()
        self.index_set = index_set

    def char_to_index(self, c):
        return self.index_set.find(c)
    
    def encode(self, x, **kwargs):
        x_tensor = torch.zeros(len(x), 1, len(self.index_set))
        for i, char in enumerate(x):
            x_tensor[i][0][self.char_to_index(char)] = 1
        return x_tensor

import warnings
from typing import Any, AnyStr

import numpy as np

from glossa.exception import GlossaEncoderWarning


class OrdEncoder:
    """
    Ord Encoder

    Encodes a characters in a string object using ```ord()``` to get their unicode representation.
    """

    def __init__(self, unk_ord: int = 1, max_ord: int = 3455, max_length: int = 250):
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

        return encoded, length

    @property
    def min(self):
        return 0

    @property
    def max(self):
        return self.max_ord

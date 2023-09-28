from typing import AnyStr

import unicodedata

from glossa.common import *
from glossa.preprocess.base import BasePreprocess


class NormalizeUnicodePreprocess(BasePreprocess):
    """
    Normalize Unicode Preprocess

    Preprocess unicode characters by normalizing its form.
    """

    def __init__(self,
                 target_set: AnyStr = ASCII_CHARSET,
                 unicode_form: AnyStr = 'NFD',
                 unicode_category: AnyStr = 'Mn'):
        super(NormalizeUnicodePreprocess, self).__init__()

        self.target_set = target_set
        self.unicode_form = unicode_form
        self.unicode_category = unicode_category

    def preprocess(self, x, **kwargs):
        return ''.join(
            c for c in unicodedata.normalize(self.unicode_form, x)
            if unicodedata.category(c) != self.unicode_category
            and c in self.target_set
        )

class BaseEncoder(object):

    def __init__(self):
        pass

    def encode(self, x, **kwargs):
        return x

    @property
    def min(self):
        return 0

    @property
    def max(self):
        return 0
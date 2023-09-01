import warnings


class GlossaEncoderWarning(UserWarning):
    pass


class GlossaPreprocessWarning(UserWarning):
    pass


class GlossaValueError(ValueError):

    msg_code = {
        'MISSING_PATH': 'Having trouble finding path `{path}`. Cannot proceed with missing value.',
        'MISSING_MODEL_PARAM': 'Model Parameter was not set. Receiving None as an invalid Model Parameter.'
    }

    def __init__(self, code=None, **kwargs):
        super().__init__(self.msg_code[code].format(**kwargs))
        self.code = code

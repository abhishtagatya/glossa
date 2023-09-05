import warnings


class GlossaEncoderWarning(UserWarning):
    pass


class GlossaPreprocessWarning(UserWarning):
    pass


class GlossaValueError(ValueError):

    msg_code = {
        'MISSING_PATH': 'Having trouble finding path `{path}`. Cannot proceed with missing value.',
        'MISSING_MODEL_PARAM': 'Model Parameter was not set. Receiving None as an invalid Model Parameter.',
        'MISSING_URL_PATH': 'Having trouble finding URL `{path}`. Cannot proceed with unreachable value.',
        'MISSING_OBJ_KEY': 'Failed to lookup key for `{key}` in `{dictobj}`. Cannot proceed with operation.',
        'MISSING_VALUE': 'Missing value for `{var}`. Object is empty, null, or unidentified.'
    }

    def __init__(self, code=None, **kwargs):
        super().__init__(self.msg_code[code].format(**kwargs))
        self.code = code

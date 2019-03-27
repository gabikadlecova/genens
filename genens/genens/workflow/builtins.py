# -*- coding: utf-8 -*-

import math

from sklearn.base import BaseEstimator, TransformerMixin


class WeightedPipeline(BaseEstimator):
    def __init__(self, pipe):
        self.pipe = pipe

    def fit(self, X, y, sample_weight=None):
            return self.pipe.fit(X, y, predictor__sample_weight=sample_weight)

    def __getattr__(self, item):
        try:
            super().__getattribute__(item)
        except AttributeError:
            val = getattr(self.pipe, item)
            setattr(self, item, val)
            return val


class RelativeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, transformer, feat_frac):
        self.transformer = transformer
        self.feat_frac = feat_frac

        if hasattr(self.transformer, 'k'):
            self.param_name = 'k'
        elif hasattr(self.transformer, 'n_components'):
            self.param_name = 'n_components'
        else:
            raise ValueError("Invalid feature transformer.")  # TODO specific

    def fit(self, X, y):
        param = int(math.ceil(self.feat_frac * X.shape[1]))

        if self.param_name == 'k':
            self.transformer.set_params(k=param)
        elif self.param_name == 'n_components':
            self.transformer.set_params(n_components=param)
        else:
            raise ValueError("Invalid feature transformer.")  # TODO specific

        self.transformer.fit(X, y)
        return self

    def __getattr__(self, item):
        try:
            super().__getattribute__(item)
        except AttributeError:
            val = getattr(self.transformer, item)
            setattr(self, item, val)
            return val

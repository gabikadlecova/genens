# -*- coding: utf-8 -*-
"""
This module contains builtin types which extend the standard scikit-learn models.
"""


import math

from sklearn.base import BaseEstimator, TransformerMixin


class WeightedPipeline(BaseEstimator):
    """
    Pipeline that supports weights and can be therefore used as a base estimator.
    """
    def __init__(self, pipe):
        self.pipe = pipe

    def __repr__(self):
        return "Weighted pipeline: " + self.pipe.__repr__()

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
    """
    Transformer wrapper that enables the n_components or k to be specified by a fraction of the dataset size.
    """
    def __init__(self, transformer, feat_frac):
        self.transformer = transformer
        self.feat_frac = feat_frac

        if hasattr(self.transformer, 'k'):
            self.param_name = 'k'
        elif hasattr(self.transformer, 'n_components'):
            self.param_name = 'n_components'
        else:
            raise ValueError("Invalid feature transformer.")  # TODO specific

    def __repr__(self):
        return "Transformer ({}): ".format(self.feat_frac) + self.transformer.__repr__()

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

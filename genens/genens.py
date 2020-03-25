# -*- coding: utf-8 -*-

"""
This module contains the Genens estimators.
"""

from .base import GenensBase
from .config import clf_config
from .config import regr_config

from sklearn.base import ClassifierMixin, RegressorMixin


class GenensClassifier(GenensBase, ClassifierMixin):
    def __init__(self, config=None, n_jobs=1,
                 scorer=None, pop_size=100, n_gen=15,
                 max_height=None, max_arity=None, timeout=None, evaluator=None,
                 max_evo_seconds=None, **kwargs):

        if config is None:
            config = clf_config()

        super().__init__(config,
                         n_jobs=n_jobs,
                         scorer=scorer,
                         pop_size=pop_size,
                         n_gen=n_gen,
                         max_height=max_height,
                         max_arity=max_arity,
                         timeout=timeout,
                         evaluator=evaluator,
                         max_evo_seconds=max_evo_seconds,
                         **kwargs)


class GenensRegressor(GenensBase, RegressorMixin):
    def __init__(self, config=None, n_jobs=1,
                 scorer=None, pop_size=100, n_gen=15,
                 max_height=None, max_arity=None, timeout=None, evaluator=None,
                 max_evo_seconds=None, **kwargs):
        if config is None:
            config = regr_config()

        super().__init__(config,
                         config,
                         n_jobs=n_jobs,
                         scorer=scorer,
                         pop_size=pop_size,
                         n_gen=n_gen,
                         max_height=max_height,
                         max_arity=max_arity,
                         timeout=timeout,
                         evaluator=evaluator,
                         max_evo_seconds=max_evo_seconds,
                         **kwargs)

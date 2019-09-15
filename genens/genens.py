# -*- coding: utf-8 -*-

"""
This module contains the Genens estimators.
"""

from .base import GenensBase
from .config import clf_config
from .config import regr_config

from sklearn.base import ClassifierMixin, RegressorMixin


class GenensClassifier(GenensBase, ClassifierMixin):
    def __init__(self, config=None, n_jobs=1, cx_pb=0.5, mut_pb=0.3, mut_args_pb=0.6, mut_node_pb=0.3,
                 scorer=None, pop_size=200, n_gen=15, hc_repeat=0, hc_keep_last=False,
                 weighted=True, use_groups=True,
                 max_height=None, max_arity=None, timeout=None, evaluator=None):
        if config is None:
            config = clf_config()

        super().__init__(config,
                         n_jobs=n_jobs,
                         cx_pb=cx_pb,
                         mut_pb=mut_pb,
                         mut_args_pb=mut_args_pb,
                         mut_node_pb=mut_node_pb,
                         scorer=scorer,
                         pop_size=pop_size,
                         n_gen=n_gen,
                         hc_repeat=hc_repeat,
                         hc_keep_last=hc_keep_last,
                         weighted=weighted,
                         use_groups=use_groups,
                         max_height=max_height,
                         max_arity=max_arity,
                         timeout=timeout,
                         evaluator=evaluator)


class GenensRegressor(GenensBase, RegressorMixin):
    def __init__(self, config=None, n_jobs=1, cx_pb=0.5, mut_pb=0.3, mut_args_pb=0.6, mut_node_pb=0.3,
                 scorer=None, pop_size=200, n_gen=15, hc_repeat=0, hc_keep_last=False,
                 weighted=True, use_groups=True,
                 max_height=None, max_arity=None, timeout=None, evaluator=None):
        if config is None:
            config = regr_config()

        super().__init__(config,
                         n_jobs=n_jobs,
                         cx_pb=cx_pb,
                         mut_pb=mut_pb,
                         mut_args_pb=mut_args_pb,
                         mut_node_pb=mut_node_pb,
                         scorer=scorer,
                         pop_size=pop_size,
                         n_gen=n_gen,
                         hc_repeat=hc_repeat,
                         hc_keep_last=hc_keep_last,
                         weighted=weighted,
                         use_groups=use_groups,
                         max_height=max_height,
                         max_arity=max_arity,
                         timeout=timeout,
                         evaluator=evaluator)
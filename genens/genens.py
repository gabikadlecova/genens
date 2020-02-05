# -*- coding: utf-8 -*-

"""
This module contains the Genens estimators.
"""

from .base import GenensBase
from .config import clf_config
from .config import regr_config

from sklearn.base import ClassifierMixin, RegressorMixin


class GenensClassifier(GenensBase, ClassifierMixin):
    def __init__(self, config=None, n_jobs=1, cx_pb=0.5, mut_pb=0.3, mut_args_pb=0.9, mut_node_pb=0.9,
                 scorer=None, pop_size=100, n_gen=15, hc_repeat=0, hc_keep_last=False,
                 mut_multiple_args=False, mut_multiple_nodes=False,
                 weighted=True, use_groups=True,
                 max_height=None, max_arity=None, timeout=None, evaluator=None, hc_n_nodes=3, hc_mut_pb=0.2,
                 max_evo_seconds=None, log_path=None,
                 **kwargs):
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
                         hc_n_nodes=hc_n_nodes,
                         hc_mut_pb=hc_mut_pb,
                         mut_multiple_args=mut_multiple_args,
                         mut_multiple_nodes=mut_multiple_nodes,
                         weighted=weighted,
                         use_groups=use_groups,
                         max_height=max_height,
                         max_arity=max_arity,
                         timeout=timeout,
                         evaluator=evaluator,
                         max_evo_seconds=max_evo_seconds,
                         log_path=log_path,
                         **kwargs)


class GenensRegressor(GenensBase, RegressorMixin):
    def __init__(self, config=None, n_jobs=1, cx_pb=0.5, mut_pb=0.3, mut_args_pb=0.9, mut_node_pb=0.9,
                 scorer=None, pop_size=100, n_gen=15, hc_repeat=0, hc_keep_last=False,
                 mut_multiple_args=False, mut_multiple_nodes=False,
                 weighted=True, use_groups=True,
                 max_height=None, max_arity=None, timeout=None, evaluator=None,
                 hc_n_nodes=3, hc_mut_pb=0.1, max_evo_seconds=None, **kwargs):
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
                         hc_n_nodes=hc_n_nodes,
                         hc_mut_pb=hc_mut_pb,
                         mut_multiple_args=mut_multiple_args,
                         mut_multiple_nodes=mut_multiple_nodes,
                         weighted=weighted,
                         use_groups=use_groups,
                         max_height=max_height,
                         max_arity=max_arity,
                         timeout=timeout,
                         evaluator=evaluator,
                         max_evo_seconds=max_evo_seconds,
                         **kwargs)

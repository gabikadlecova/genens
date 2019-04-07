# -*- coding: utf-8 -*-

from genens.base import GenensBase, CrossvalEvaluator
from genens.config.clf_default import create_clf_config
from genens.config.regr_default import create_regr_config

from sklearn.base import ClassifierMixin, RegressorMixin


class GenensClassifier(GenensBase, ClassifierMixin):
    def __init__(self, config=None, n_jobs=1, cx_pb=0.5, mut_pb=0.1, mut_args_pb=0.3, scorer=None,
                 pop_size=100, n_gen=10, hc_repeat=0, hc_keep_last=False, max_height=None,
                 max_arity=None, evaluator=CrossvalEvaluator()):
        if config is None:
            config = create_clf_config()

        super().__init__(config, n_jobs, cx_pb, mut_pb, mut_args_pb, scorer, pop_size,
                         n_gen, hc_repeat, hc_keep_last, max_height, max_arity, evaluator)


class GenensRegressor(GenensBase, RegressorMixin):
    def __init__(self, config=None, n_jobs=1, cx_pb=0.5, mut_pb=0.1, mut_args_pb=0.3, scorer=None,
                 pop_size=100, n_gen=10, hc_repeat=0, hc_keep_last=False, max_height=None,
                 max_arity=None, evaluator=CrossvalEvaluator()):
        if config is None:
            config = create_regr_config()

        super().__init__(config, n_jobs, cx_pb, mut_pb, mut_args_pb, scorer, pop_size,
                         n_gen, hc_repeat, hc_keep_last, max_height, max_arity, evaluator)
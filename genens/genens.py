# -*- coding: utf-8 -*-

"""
This module contains the Genens estimators.
"""
import os
from typing import Union

from genens.base import GenensBase
from genens.config.config_templates import clf_default_config, default_raw_config
from genens.config.genens_config import GenensConfig, parse_config
from genens.workflow.evaluate import EvaluatorBase

from sklearn.base import ClassifierMixin, RegressorMixin


class GenensClassifier(GenensBase, ClassifierMixin):
    def __init__(self, config: Union[GenensConfig, str] = None, use_base_config: bool = True, evo_kwargs: dict = None,
                 n_jobs: int = 1, scorer=None, pop_size: int = 100, n_gen: int = 15,
                 timeout: int = None, evaluator: EvaluatorBase = None,
                 max_evo_seconds: int = None, **kwargs):

        if config is None:
            config = clf_default_config()

        if isinstance(config, str):
            base_config = default_raw_config() if use_base_config else None
            config = parse_config(config, base_config=base_config, evo_kwargs=evo_kwargs)

        super().__init__(config,
                         n_jobs=n_jobs,
                         scorer=scorer,
                         pop_size=pop_size,
                         n_gen=n_gen,
                         timeout=timeout,
                         evaluator=evaluator,
                         max_evo_seconds=max_evo_seconds,
                         **kwargs)


class GenensRegressor(GenensBase, RegressorMixin):
    def __init__(self, config: Union[GenensConfig, str] = None, use_base_config: bool = True, evo_kwargs: dict = None,
                 n_jobs: int = 1, scorer=None, pop_size: int = 100, n_gen: int = 15,
                 timeout: int = None, evaluator: EvaluatorBase = None,
                 max_evo_seconds: int = None, **kwargs):
        if config is None:
            raise ValueError("Regressor default run_config is not available yet.")

        if isinstance(config, str):
            base_config = default_raw_config() if use_base_config else None
            config = parse_config(config, base_config=base_config, evo_kwargs=evo_kwargs)

        super().__init__(config,
                         n_jobs=n_jobs,
                         scorer=scorer,
                         pop_size=pop_size,
                         n_gen=n_gen,
                         timeout=timeout,
                         evaluator=evaluator,
                         max_evo_seconds=max_evo_seconds,
                         **kwargs)

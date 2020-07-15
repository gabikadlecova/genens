# -*- coding: utf-8 -*-

"""
This module contains helper functions for the creation of GP primitives and
construction functions for most common methods used in scikit-learn.

The functions create node templates and wrappers of ensembles, simple predictors and transformers.
Node templates are used in the GP evolution process, wrapper functions are used to convert nodes
to machine learning workflow methods.

Node templates are specified by input types, arities (one arity/arity range per type) and output types.

Wrapper functions have the signature ``func(child_list, kwarg_dict)``, where ``child_list`` are results
of other wrapper functions applied on child nodes and ``kwarg_dict`` is a dictionary of evolved keyword
arguments.
"""
import importlib
from functools import partial
from typing import Union, Callable

from genens.gp.types import GpFunctionTemplate, GpTerminalTemplate, TypeArity
from genens.workflow.model_creation import create_stacking
from genens.workflow.model_creation import create_estimator
from genens.workflow.model_creation import create_ensemble


def import_custom_func(func_path: str):
    func_path = func_path.split('.')
    func_name = func_path.pop()
    module_path = '.'.join(func_path)

    return getattr(importlib.import_module(module_path), func_name)


def _get_estimator_func(est_cls: Union[str, Callable]):
    if isinstance(est_cls, str):
        return import_custom_func(est_cls)

    return est_cls


def _call_func_with_cls(func, cls, cls_kwargs=None, *args, **kwargs):
    cls_kwargs = cls_kwargs if cls_kwargs is not None else {}
    return func(_get_estimator_func(cls), cls_kwargs, *args, **kwargs)


def estimator_func(cls, cls_kwargs=None, *args, **kwargs):
    """
    Creates a wrapper function which returns an instance of the argument estimator class.

    The function signature is ``func(child_list, kwarg_dict)``,
    keyword arguments in ``kwargs`` and ``kwarg_dict`` must be distinct.

    The ``child_list`` argument of the resulting function must be empty,
    as simple estimators cannot have sub-estimators.

    :param cls: Estimator class.
    :param kwargs: Keyword arguments of the estimator.
    :return: Function which constructs a new instance of the estimator.
    """
    return _call_func_with_cls(create_estimator, cls, cls_kwargs=cls_kwargs, *args, **kwargs)


def ensemble_func(cls, cls_kwargs=None, *args, **kwargs):
    """
    Creates a wrapper function which returns an instance of the argument ensemble class.

    The function signature is ``func(child_list, kwarg_dict)``,
    keyword arguments in ``kwargs`` and ``kwarg_dict`` must be distinct.

    The ``child_list`` argument contains estimators (or pipelines) which will be
    set as the ``base_estimator`` or ``estimators`` of the ensemble.

    :param cls: Ensemble class.
    :param kwargs: Keyword arguments of the ensemble.
    :return: Function which constructs a new instance of the ensemble.
    """
    return _call_func_with_cls(create_ensemble, cls, cls_kwargs=cls_kwargs, *args, **kwargs)


def stacking_func(cls, cls_kwargs=None, *args, **kwargs):
    return _call_func_with_cls(create_stacking, cls, cls_kwargs=cls_kwargs, *args, **kwargs)
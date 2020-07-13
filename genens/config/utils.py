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

#from genens.config.genens_config import GenensConfig

from ..gp.types import GpFunctionTemplate, GpTerminalTemplate, TypeArity
from ..workflow.model_creation import create_pipeline, create_stacking
from ..workflow.model_creation import create_data_union
from ..workflow.model_creation import create_transform_list
from ..workflow.model_creation import create_estimator
from ..workflow.model_creation import create_ensemble


# TODO this from json/yaml
def get_default_config(group_weights=None):
    """
    Creates the default config with pipeline construction nodes and functions.
    
    :param group_weights: Group weights of the config.
    :return: The default configuration.
    """
    
    func_config = {
        'cPred': create_pipeline,
        'cPipe': create_pipeline,
        # 'dUnion': create_data_union,
        'cData': create_transform_list,
        'cFeatSelect': create_transform_list,
        'cScale': create_transform_list
    }

    kwargs_config = {
        'cPred': {},
        'cPipe': {},
        'cData': {},
        'cFeatSelect': {},
        'cScale': {},
        # 'dUnion': {}
    }

    full_config = {
        'out': [
            GpFunctionTemplate('cPred', [TypeArity('ens', 1)], 'out',
                               group='pipeline'),
            GpFunctionTemplate('cPipe', [TypeArity('ens', 1), TypeArity('data', 1)], 'out',
                               group='pipeline'),
        ],
        'data': [
            # GpFunctionTemplate('dUnion', [TypeArity('data', (2,3))], 'data', group='union'),
            GpFunctionTemplate('cData', [TypeArity('featsel', 1), TypeArity('scale', 1)], 'data',
                               group='prepro'),
            GpFunctionTemplate('cFeatSelect', [TypeArity('featsel', 1)], 'data',
                               group='prepro'),
            GpFunctionTemplate('cScale', [TypeArity('scale', 1)], 'data',
                               group='prepro')
        ],
        'ens': []
    }

    term_config = {
        'out': [],
        'data': [],
        'ens': []
    }

    return None
    #return GenensConfig(func_config, full_config, term_config, kwargs_config,
    #                    group_weights=group_weights)


def import_custom_func(func_path: str):
    func_path = func_path.split('.')
    func_name = func_path.pop()
    module_path = '.'.join(func_path)

    return getattr(importlib.import_module(module_path), func_name)


def _get_estimator_func(est_cls: Union[str, Callable]):
    if isinstance(est_cls, str):
        return import_custom_func(est_cls)

    return est_cls


def estimator_func(cls, **kwargs):
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
    return partial(create_estimator, _get_estimator_func(cls), kwargs)


def ensemble_func(cls, **kwargs):
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
    return partial(create_ensemble, _get_estimator_func(cls), kwargs)


def stacking_func(cls, **kwargs):
    return partial(create_stacking, _get_estimator_func(cls), kwargs)


def ensemble_primitive(ens_name, in_arity, in_type='out', out_type='ens', group='ensemble'):
    """
    Creates a function template which represents an ensemble. The template can be used
    to create GP primitives with variable arity and different keyword argument dictionaries.

    The node must have child nodes of a type ``in_type``, their count is specified
    by ``in_arity``. The ``out_type`` is 'ens' by default, which is the output type
    of predictors.

    :param group:
    :param str ens_name: Name of the ensemble.
    :param int, (int, int), (int, 'n') in_arity:
        Arity of input nodes, either a constant, or a range (inclusive), or a range
        without a specified upper bound (in the case of (int, 'n')).

    :param str in_type: Type of all child nodes.
    :param str out_type: Node output type.
    :return: Function template which can be used to create an ensemble primitive.
    """
    return GpFunctionTemplate(ens_name, [TypeArity(in_type, in_arity)], out_type,
                              group=group)


def predictor_primitive(p_name, group='predictor'):
    """
    Creates a terminal template which represents a simple predictor. The template
    can be used to create GP primitives with different keyword argument dictionaries.

    The node has the output type 'ens', which is the output type of predictors.

    :param group:
    :param str p_name: Name of the predictor.
    :return: Terminal template which can be used to create a predictor primitive.
    """
    return GpTerminalTemplate(p_name, 'ens', group=group)


def predictor_terminal(p_name, group='predictor'):
    """
    Creates a terminal template which represents a simple predictor. The template
    can be used to create GP primitives with different keyword argument dictionaries.

    The node has the output type 'out', which is the output type of pipelines.
    This node should be used when maximum tree height would be exceeded; in other
    cases, nodes returned by ``predictor_primitive`` should be used.

    :param group:
    :param str p_name: Name of the predictor.
    :return: Terminal template which can be used to create a predictor primitive.
    """
    return GpTerminalTemplate(p_name, 'out', group=group)


def transformer_primitive(t_name, out_type, group='transform'):
    """
    Creates a terminal template which represents a simple transformer. The template
    can be used to create GP primitives with different keyword argument dictionaries.

    The node has the output type 'data', which is the output type of transformer nodes.

    :param group:
    :param str t_name: Name of the transformer.
    :param str out_type:
        Name of the transformer output type (most common are 'featsel' for feature
        selectors and 'scale' for scaling transformers.
    :return: Terminal template which can be used to create a transformer primitive.
    """
    return GpTerminalTemplate(t_name, out_type, group=group)

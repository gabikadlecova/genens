# -*- coding: utf-8 -*-

"""
This module contains helper functions for the creation of GP primitives and
construction functions of most common methods used in scikit-learn.

The functions create node templates and wrappers of ensembles, simple predictors and transformers.
Node templates are used in the GP evolution process, wrapper functions are used to convert nodes
to machine learning workflow methods.

Node templates are specified by input types, arities (one arity/arity range per type) and output types.

Wrapper functions have the signature ``func(child_list, kwarg_dict)``, where ``child_list`` are results
of other wrapper functions applied on child nodes and ``kwarg_dict`` is a dictionary of evolved keyword
arguments.
"""

import genens.workflow.model_creation as mc
from genens.gp.types import GpFunctionTemplate, GpTerminalTemplate, TypeArity

import math
from functools import partial


class GenensConfig:
    def __init__(self, func, full, term, kwargs_config, max_height, max_arity=10):
        self.func_config = func

        self.full_config = full
        self.term_config = term

        self.kwargs_config = kwargs_config

        self.max_height = max_height
        self.max_arity = max_arity

    def add_primitive(self, prim, term_only=False):
        # TODO warn if both not

        if not term_only:
            out_list = self.full_config.setdefault(prim.out_type, [])
            out_list.append(prim)

        if isinstance(prim, GpTerminalTemplate):
            out_list = self.term_config.setdefault(prim.out_type, [])
            out_list.append(prim)

    def add_functions_args(self, func_dict, kwarg_dict):
        for key, val in func_dict.items():
            if key in self.func_config.keys():
                raise ValueError("Cannot insert to func - duplicate value.")  # TODO specific

            kwarg_val = kwarg_dict.get(key, None)
            if kwarg_val is None:
                raise ValueError("Must provide keyword arguments for all names.")  # TODO specific

            self.func_config[key] = val
            self.kwargs_config[key] = kwarg_val


def get_default_config():
    func_config = {
        'cPipe': mc.create_pipeline,
        # 'dUnion': mc.create_data_union, #  todo handle data terminal
        'cData': mc.create_transform_list,
        'dTerm': mc.create_empty_data
    }

    kwargs_config = {
        'cPipe': {},
        'cData': {},
        'dTerm': {}
    }

    full_config = {
        'out': [GpFunctionTemplate('cPipe', [TypeArity('ens', 1), TypeArity('data', (0,1))], 'out')],
        'data': [
            # GpFunctionTemplate('dUnion', [TypeArity('data', (2,'n'))], 'data'),  # todo handle dTerm
            GpFunctionTemplate('cData', [TypeArity('featsel', 1), TypeArity('scale', 1)], 'data')
        ],
        'ens': []
    }

    term_config = {
        'out': [],
        'data': [
            GpTerminalTemplate('dTerm', 'data')
        ],
        'ens': []
    }

    # TODO BIG TODO height param!!!!

    return GenensConfig(func_config, full_config, term_config, kwargs_config, 7)


def get_n_components(feat_size, feat_fractions=None):
    """
    Returns list of feature counts which are fractions of the total count.

    :param int feat_size: Total feature count.
    :param list[float] feat_fractions: Fractions in the interval (0.0, 1.0].
    :return list[int]: Feature counts.
    """
    if feat_fractions is None:
        feat_fractions = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1]

    return [int(math.ceil(feat_size * fraction)) for fraction in feat_fractions]


def estimator_func(est_cls, **kwargs):
    """
    Creates a wrapper function which returns an instance of the argument estimator class.

    The function signature is ``func(child_list, kwarg_dict)``,
    keyword arguments in ``kwargs`` and ``kwarg_dict`` must be distinct.

    The ``child_list`` argument of the resulting function must be empty,
    as simple estimators cannot have sub-estimators.

    :param est_cls: Estimator class.
    :param kwargs: Keyword arguments of the estimator.
    :return: Function which constructs a new instance of the estimator.
    """
    return partial(mc.create_estimator, est_cls, kwargs)


def ensemble_func(ens_cls, **kwargs):
    """
    Creates a wrapper function which returns an instance of the argument ensemble class.

    The function signature is ``func(child_list, kwarg_dict)``,
    keyword arguments in ``kwargs`` and ``kwarg_dict`` must be distinct.

    The ``child_list`` argument contains estimators (or pipelines) which will be
    set as the ``base_estimator`` or ``estimators`` of the ensemble.

    :param ens_cls: Ensemble class.
    :param kwargs: Keyword arguments of the ensemble.
    :return: Function which constructs a new instance of the ensemble.
    """
    return partial(mc.create_ensemble, ens_cls, kwargs)


def ensemble_primitive(ens_name, in_arity, in_type='out', out_type='ens'):
    """
    Creates a function template which represents an ensemble. The template can be used
    to create GP primitives with variable arity and different keyword argument dictionaries.

    The node must have child nodes of a type ``in_type``, their count is specified
    by ``in_arity``. The ``out_type`` is 'ens' by default, which is the output type
    of predictors.

    :param str ens_name: Name of the ensemble.
    :param int, (int, int), (int, 'n') in_arity:
        Arity of input nodes, either a constant, or a range (inclusive), or a range
        without a specified upper bound (in the case of (int, 'n')).

    :param str in_type: Type of all child nodes.
    :param str out_type: Node output type.
    :return: Function template which can be used to create an ensemble primitive.
    """
    return GpFunctionTemplate(ens_name, [TypeArity(in_type, in_arity)], out_type)


def predictor_primitive(p_name):
    """
    Creates a terminal template which represents a simple predictor. The template
    can be used to create GP primitives with different keyword argument dictionaries.

    The node has the output type 'ens', which is the output type of predictors.

    :param str p_name: Name of the predictor.
    :return: Terminal template which can be used to create a predictor primitive.
    """
    return GpTerminalTemplate(p_name, 'ens')


def predictor_terminal(p_name):
    """
    Creates a terminal template which represents a simple predictor. The template
    can be used to create GP primitives with different keyword argument dictionaries.

    The node has the output type 'out', which is the output type of pipelines.
    This node should be used when maximum tree height would be exceeded; in other
    cases, nodes returned by ``predictor_primitive`` should be used.

    :param str p_name: Name of the predictor.
    :return: Terminal template which can be used to create a predictor primitive.
    """
    return GpTerminalTemplate(p_name, 'out')


def transformer_primitive(t_name, out_type):
    """
    Creates a terminal template which represents a simple transformer. The template
    can be used to create GP primitives with different keyword argument dictionaries.

    The node has the output type 'data', which is the output type of transformer nodes.

    :param str t_name: Name of the transformer.
    :param str out_type:
        Name of the transformer output type (most common are 'featsel' for feature
        selectors and 'scale' for scaling transformers.
    :return: Terminal template which can be used to create a transformer primitive.
    """
    return GpTerminalTemplate(t_name, out_type)

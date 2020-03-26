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

from functools import partial

from ..gp.types import GpFunctionTemplate, GpTerminalTemplate, TypeArity
from ..workflow.model_creation import create_pipeline, create_stacking
from ..workflow.model_creation import create_data_union
from ..workflow.model_creation import create_transform_list
from ..workflow.model_creation import create_estimator
from ..workflow.model_creation import create_ensemble

from warnings import warn


class GenensConfig:
    """
    Configuration of Genens estimators. Contains settings of GP and configuration of
    methods which decode nodes into scikit-learn methods.
    """
    def __init__(self, function_config=None, full_config=None, term_config=None, kwargs_config=None, min_height=1,
                 max_height=4, min_arity=2, max_arity=3, group_weights=None):
        """
        Creates a new instance of a Genens configuration. If ``group_weights`` are specified,
        validity check is performed.

        :param dict function_config: Function configuration dictionary, keys correspond do a particular primitive name.
        :param dict full_config: Node configuration dictionary, nodes are grouped by output types,
                          contains node templates used during the grow phase.

        :param dict term_config: Node configuration dictionary, nodes are grouped by output types,
                          contains node templates used to terminate a tree (in the last height level).

        :param kwargs_config: Keyword arguments for nodes, which are passed to functions during the decoding.
                              For every argument, there is a list of possibles values to choose from during the
                              evolution.

        :param min_height: Minimum height of trees.
        :param max_height: Maximum height of trees.
        :param min_arity: Minimum arity of a function node.
        :param max_arity: Maximum arity of a function node.
        :param group_weights: Group weight configuration.
        """
        self.func_config = function_config if function_config is not None else {}
        self.full_config = full_config if full_config is not None else {}
        self.term_config = term_config if term_config is not None else {}
        self.kwargs_config = kwargs_config if kwargs_config is not None else {}

        self.min_height = min_height
        self.max_height = max_height
        self.min_arity = min_arity
        self.max_arity = max_arity

        self.group_weights = group_weights if group_weights is not None else {}

    def __repr__(self):
        res = "max_height: {}, max_arity: {}".format(self.max_height, self.max_arity)
        res += ", group_weights: {}".format(str(self.group_weights))
        return res

    def add_terminal(self, prim, leaf_only=False):
        """
        Adds a new primitive to the configuration. If ``term_only`` is True,
        it is added only to the terminal set. If the primitive is a GpFunctionTemplate,
        it is added only to the grow set.

        :param GpPrimitive prim: Primitive to be added.
        :param bool leaf_only: Specifies whether the primitive should be added only to the terminal set.
        """

        if not leaf_only:
            out_list = self.full_config.setdefault(prim.out_type, [])
            out_list.append(prim)

        out_list = self.term_config.setdefault(prim.out_type, [])
        out_list.append(prim)

    def add_function(self, prim):
        out_list = self.full_config.setdefault(prim.out_type, [])
        out_list.append(prim)


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

    return GenensConfig(func_config, full_config, term_config, kwargs_config,
                        group_weights=group_weights)


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
    return partial(create_estimator, est_cls, kwargs)


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
    return partial(create_ensemble, ens_cls, kwargs)


def stacking_func(ens_cls, **kwargs):
    return partial(create_stacking, ens_cls, kwargs)


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

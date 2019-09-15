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
from ..workflow.model_creation import create_pipeline
from ..workflow.model_creation import create_data_union
from ..workflow.model_creation import create_transform_list
from ..workflow.model_creation import create_empty_data
from ..workflow.model_creation import create_estimator
from ..workflow.model_creation import create_ensemble

from warnings import warn


def default_group_weights(*args, groups=None):
    """
    Verifies whether all GpPrimitive instances contained in ``args`` (dictionaries of lists) belong to
    a particular group in ``groups``. If a GpPrimitive does not have a group specified, creates a new
    group for it with default weight.

    :param args: Dictionaries of lists of GpPrimitive (grouped by output types as in GenensConfig)
    :param groups: Dictionary of groups with weights
    :return: Adjusted groups
    """
    checked_groups = groups if groups is not None else {}
    for prim_list in args:
        # primitives are grouped by output nodes
        for out_prim_list in prim_list.values():
            for prim in out_prim_list:

                if prim.group is None:
                    # warn only if user provided config with missing values
                    if groups is not None:
                        warn("Primitive {} has no group, using its name as the group.".format(prim.name))

                    prim.group = prim.name

                if prim.group not in checked_groups.keys():
                    # warn only if user provided config with missing values
                    if groups is not None:
                        warn("Group {} is not present in the group list, adding with default"
                             "weight (1.0).".format(prim.group))

                    checked_groups[prim.group] = 1.0

    return checked_groups


class GenensConfig:
    """
    Configuration of Genens estimators. Contains settings of GP and configuration of
    methods which decode nodes into scikit-learn methods.
    """
    def __init__(self, func, full, term, kwargs_config, max_height=4, max_arity=3,
                 group_weights=None):
        """
        Creates a new instance of a Genens configuration. If ``group_weights`` are specified,
        validity check is performed.

        :param dict func: Function configuration dictionary, keys correspond do a particular primitive name.
        :param dict full: Node configuration dictionary, nodes are grouped by output types,
                          contains node templates used during the grow phase.

        :param dict term: Node configuration dictionary, nodes are grouped by output types,
                          contains node templates used to terminate a tree (in the last height level).

        :param kwargs_config: Keyword arguments for nodes, which are passed to functions during the decoding.
                              For every argument, there is a list of possibles values to choose from during the
                              evolution.

        :param max_height: Maximum height of trees.
        :param max_arity: Maximum arity of a function node.
        :param group_weights: Group weight configuration.
        """
        self.func_config = func

        self.full_config = full
        self.term_config = term

        self.kwargs_config = kwargs_config

        self.max_height = max_height
        self.max_arity = max_arity

        # perform check if specified
        if group_weights is None:
            self._group_weights = None
        else:
            self.group_weights = group_weights

    def __repr__(self):
        res = "max_height: {}, max_arity: {}".format(self.max_height, self.max_arity)
        res += ", group_weights: {}".format(str(self._group_weights))
        return res

    @property
    def group_weights(self):
        return self._group_weights

    @group_weights.setter
    def group_weights(self, value):
        """
        Sets the group weight and performs validity check, possibly adds a new group for primitives
        without a group specified.

        :param dict value: Dictionary of weights, keys are group names.
        """
        self._group_weights = default_group_weights(self.full_config, self.term_config,
                                                    groups=value)

    @group_weights.deleter
    def group_weights(self):
        del self._group_weights

    def add_primitive(self, prim, term_only=False):
        """
        Adds a new primitive to the configuration. If ``term_only`` is True,
        it is added only to the terminal set. If the primitive is a GpFunctionTemplate,
        it is added only to the grow set.

        :param GpPrimitive prim: Primitive to be added.
        :param bool term_only: Specifies whether the primitive should be added only to the terminal set.
        """
        if term_only and not isinstance(prim, GpTerminalTemplate):
            warn("Primitive is added neither to function nor terminal set.")

        if not term_only:
            out_list = self.full_config.setdefault(prim.out_type, [])
            out_list.append(prim)

        if isinstance(prim, GpTerminalTemplate):
            out_list = self.term_config.setdefault(prim.out_type, [])
            out_list.append(prim)

    def add_functions_args(self, func_dict, kwarg_dict):
        """
        Adds functions that decode a node along with keyword arguments for the node.
        The function has the signature ``func(child_list, kwargs)``, where child_list contains
        decoded values of child nodes, and kwargs are
        specific keyword arguments set during the evolution.

        :param func_dict: Dictionary of functions, the keys are names which should correspond to a node.
                          Every function must have a keyword argument dictionary specified.

        :param kwarg_dict: Dictionary of keyword argument dictionaries, the keys (names) corresponds to nodes
                           from grow/terminals sets and also to methods in the ``func_dict``.
        """
        for key, val in func_dict.items():
            if key in self.func_config.keys():
                raise ValueError("Cannot insert to func - duplicate value.")  # TODO specific

            kwarg_val = kwarg_dict.get(key, None)
            if kwarg_val is None:
                raise ValueError("Must provide keyword arguments for all names.")  # TODO specific

            self.func_config[key] = val
            self.kwargs_config[key] = kwarg_val


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
        'cScale': create_transform_list,
        'dTerm': create_empty_data
    }

    kwargs_config = {
        'cPred': {},
        'cPipe': {},
        'cData': {},
        'cFeatSelect': {},
        'cScale': {},
        # 'dUnion': {},
        'dTerm': {}
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

# -*- coding: utf-8 -*-

"""
This module provides methods which convert the tree representation to pipelines.
Every method operates on specific nodes along with its (already converted) child nodes.
"""


import inspect
from functools import reduce

from sklearn.pipeline import Pipeline, make_union

from .builtins import WeightedPipeline, RelativeTransformer


def create_workflow(gp_tree, config_dict):
    """
    Creates a pipeline (workflow) from the tree.

    :param gp_tree: Tree to be converted.
    :param config_dict: Configuration to be used for conversion.
    :return: A scikit-learn pipeline.
    """
    def wf_step_from_node(node, child_list):
        return config_dict[node.name](child_list, node.obj_kwargs)

    return gp_tree.run_tree(wf_step_from_node)


def create_stacking(ens_cls, const_kwargs, child_list, evolved_kwargs):
    if not len(child_list) or len(child_list) < 2:
        raise ValueError("Not enough estimators provided to the ensemble.")

    estimator_list = child_list[:-1]
    estimator_list = [(f'clf{i}', clf) for i, clf in enumerate(estimator_list)]
    final_estimator = child_list[-1]

    return ens_cls(estimators=estimator_list,
                   final_estimator=final_estimator,
                   **const_kwargs, **evolved_kwargs)


def create_ensemble(ens_cls, const_kwargs, child_list, evolved_kwargs):
    """
    Creates an ensemble with its children set as base-learners.

    :param ens_cls: Function that creates the ensemble
    :param const_kwargs: Keyword arguments which do not change during the evolution.
    :param child_list: List of converted child nodes.
    :param evolved_kwargs: Keyword arguments which are set during the evolution process.
    :return: A new ensemble.
    """
    if not len(child_list):
        raise ValueError("No base estimator provided to the ensemble.")  # TODO specific

    if 'base_estimator' in inspect.signature(ens_cls).parameters:
        if len(child_list) != 1:
            raise ValueError("Incorrect number of base estimators.")  # TODO specific

        ens = ens_cls(**const_kwargs, **evolved_kwargs, base_estimator=child_list[0])
    elif 'estimators' in inspect.signature(ens_cls).parameters:

        est_names = ['clf' + str(i) for i in range(0, len(child_list))]
        est_list = list(zip(est_names, child_list))

        ens = ens_cls(**const_kwargs, **evolved_kwargs, estimators=est_list)
    else:
        raise ValueError("Invalid ensemble - missing constructor parameters.")  # TODO specific

    return ens


def create_estimator(est_cls, const_kwargs, child_list, evolved_kwargs):
    """
    Creates an estimator.

    :param est_cls: Function that creates the estimator.
    :param const_kwargs: Keyword arguments which do not change during the evolution.
    :param child_list: List of converted child nodes - should me empty.
    :param evolved_kwargs: Keyword arguments which are set during the evolution process.
    :return: A new estimator.
    """

    if len(child_list) > 0:
        raise ValueError("Estimator cannot have sub-estimators.")  # TODO specific

    if 'feat_frac' in evolved_kwargs.keys():
        feat_frac = evolved_kwargs['feat_frac']
        evolved_kwargs = {key: val for key, val in evolved_kwargs.items()
                          if key != 'feat_frac'}

        est = est_cls(**const_kwargs, **evolved_kwargs)
        return RelativeTransformer(est, feat_frac)

    return est_cls(**const_kwargs, **evolved_kwargs)


def create_transform_list(child_list, evolved_kwargs):
    """
    Creates a transformer chain.

    :param child_list: List of transformers.
    :param evolved_kwargs: Keyword arguments, not used.
    :return:
    """
    return child_list


def create_empty_data(child_list, evolved_kwargs):
    """
    Creates an empty transformer list.

    :param child_list: Child list, must be empty.
    :param evolved_kwargs: Keyword arguments, not used.
    :return:
    """
    if len(child_list) > 0:
        raise ValueError("This can be assigned only to terminals.")  # TODO specific

    return []


def create_pipeline(child_list, evolved_kwargs):
    """
    Creates a pipeline from the child list.

    :param child_list: Should contain either an estimator or an estimator along with a transformer chain.
    :param evolved_kwargs: Keyword arguments, not used.
    :return: A scikit-learn pipeline.
    """
    if len(child_list) > 2 or not len(child_list):
        raise ValueError("Invalid child list for pipeline.")  # TODO specific

    predictor = child_list[0]

    if len(child_list) > 1:
        if isinstance(child_list[1], list):
            step_names = ['step' + str(i) for i in range(0, len(child_list[1]))]
            steps = list(zip(step_names, child_list[1]))
        else:
            steps = [('step', child_list[1])]
        steps.append(('predictor', predictor))

        pipe = Pipeline(steps=steps, **evolved_kwargs)
    else:
        pipe = Pipeline(steps=[('predictor', predictor)], **evolved_kwargs)

    if 'sample_weight' in inspect.signature(predictor.fit).parameters:
        wp = WeightedPipeline(pipe)
        return wp

    return pipe


def create_data_union(child_list, evolved_kwargs):
    """
    Creates a data union from the child list.
    :param child_list: List of transformers.
    :param evolved_kwargs: Keyword arguments of the union.
    :return: A scikit-learn FeatureUnion
    """
    if not len(child_list):
        raise ValueError("No base estimator provided to the feature union.")  # TODO specific

    def add_or_concat(res, child):
        if isinstance(child, list):
            return res + child
        res.append(child)
        return res

    reduced = reduce(add_or_concat, child_list, [])
    if not len(reduced):
        return []

    return [make_union(*reduced, **evolved_kwargs)]

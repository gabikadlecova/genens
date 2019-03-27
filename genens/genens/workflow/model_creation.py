# -*- coding: utf-8 -*-

import inspect
import operator
from functools import reduce

from sklearn.pipeline import Pipeline, make_union

from genens.workflow.builtins import WeightedPipeline, RelativeTransformer


def create_workflow(gp_tree, config_dict):
    def wf_step_from_node(node, child_list):
        return config_dict[node.name](child_list, node.obj_kwargs)

    return gp_tree.run_tree(wf_step_from_node)


def create_ensemble(ens_cls, const_kwargs, child_list, evolved_kwargs):
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
    return child_list


def create_empty_data(child_list, evolved_kwargs):
    if len(child_list) > 0:
        raise ValueError("This can be assigned only to terminals.")  # TODO specific

    return []


def create_pipeline(child_list, evolved_kwargs):
    if len(child_list) > 2 or not len(child_list):
        raise ValueError("Invalid child list for pipeline.")  # TODO specific

    predictor = child_list[0]

    if len(child_list) > 1 and len(child_list[1]) > 0:
        step_names = ['step' + str(i) for i in range(0, len(child_list[1]))]
        steps = list(zip(step_names, child_list[1]))
        steps.append(('predictor', predictor))

        pipe = Pipeline(steps=steps, **evolved_kwargs)
    else:
        pipe = Pipeline(steps=[('predictor', predictor)], **evolved_kwargs)

    if 'sample_weight' in inspect.signature(predictor.fit).parameters:
        wp = WeightedPipeline(pipe)
        return wp

    return pipe


def create_data_union(child_list, evolved_kwargs):
    if not len(child_list):
        raise ValueError("No base estimator provided to the feature union.")  # TODO specific

    reduced = reduce(operator.concat, child_list, [])
    if not len(reduced):
        return []

    return [make_union(*reduced, **evolved_kwargs)]

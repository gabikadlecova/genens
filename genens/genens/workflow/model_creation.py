# -*- coding: utf-8 -*-

from sklearn.pipeline import make_pipeline, make_union


def create_workflow(gp_tree, config_dict):
    def wf_step_from_node(node, child_list):
        return config_dict[node.name](child_list, node.obj_kwargs)

    gp_tree.run_tree(wf_step_from_node)


def create_ensemble(ens_cls, const_kwargs, child_list, evolved_kwargs):
    ens = ens_cls(**const_kwargs, **evolved_kwargs)

    if not len(child_list):
        raise ValueError("No base estimator provided to the ensemble.")  # TODO specific

    # TODO check for absence of parameters (suitable exception)

    if len(child_list) == 1:
        ens.base_estimator = child_list[0]
    else:
        ens.estimators = child_list

    return ens


def create_estimator(est_cls, const_kwargs, child_list, evolved_kwargs):
    if len(child_list) > 1:
        raise ValueError("Estimator cannot have sub-estimators.")  # TODO specific

    # create pipeline list
    estimator = est_cls(**const_kwargs, **evolved_kwargs)
    if not len(child_list):
        return [estimator]

    # append pipeline step
    child_list.append(estimator)
    return child_list


def create_pipeline(child_list, evolved_kwargs):
    if len(child_list) > 2 or not len(child_list):
        raise ValueError("Invalid child list for pipeline.")

    predictor = child_list[0]
    if len(child_list) > 1:
        return make_pipeline(*child_list[1], predictor, **evolved_kwargs)

    return make_pipeline(predictor, **evolved_kwargs)


def create_data_union(child_list, evolved_kwargs):
    if not len(child_list):
        raise ValueError("No base estimator provided to the feature union.")  # TODO specific

    return make_union(*child_list, **evolved_kwargs)

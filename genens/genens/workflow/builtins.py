# -*- coding: utf-8 -*-

import genens.workflow.eval as wfe
from sklearn.base import BaseEstimator


class ModelConfig:
    def __init__(self, node_dict=None, model_dict=None):
        self.node_dict = node_dict if node_dict is not None else {}
        self.model_dict = model_dict if model_dict is not None else {}

    def create_wf_node(self, name, model_kwargs, child_list):
        node_cls = self.node_dict[name]

        model_data = self.model_dict.get(name, None)
        if model_data is None:
            return node_cls(child_list, None)

        model_cls, const_kwargs = model_data

        model_kwargs = model_kwargs if model_kwargs is not None else {}
        const_kwargs = const_kwargs if const_kwargs is not None else {}
        model = model_cls(**const_kwargs, **model_kwargs)

        return node_cls(child_list, model)

    def add_sklearn_ensemble(self, ens_name, ens_cls, **kwargs):
        # TODO check if cls is in kwargs

        self.node_dict[ens_name] = wfe.EnsembleNode
        ens_cls = wrap_sklearn_ensemble(ens_cls)

        self.model_dict[ens_name] = (ens_cls, kwargs)

    def add_model(self, model_name, model_cls, **kwargs):
        self.node_dict[model_name] = wfe.ModelNode
        self.model_dict[model_name] = (model_cls, kwargs)

    def add_data_processor(self, prep_name, prep_cls, **kwargs):
        self.node_dict[prep_name] = wfe.DataProcessNode
        self.model_dict[prep_name] = (prep_cls, kwargs)


def wrap_sklearn_ensemble(ens):
    def accept_list(self, child_list):
        if not len(child_list):
            raise ValueError("No base estimator provided to the ensemble.")  # TODO specific

        # TODO check for absence of parameters (suitable exception)

        if len(child_list) == 1:
            self.base_estimator = child_list[0]
            return

        self.estimators = child_list

    ens.accept_list = accept_list
    return ens


def default_config():
    node_dict = {}

    node_dict['_cEns'] = wfe.EnsembleExecNode
    return ModelConfig(node_dict)
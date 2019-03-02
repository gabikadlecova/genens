# -*- coding: utf-8 -*-

import genens.workflow.eval as wfe


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

    def add_sklearn_ensemble(self, ens_name, ens_cls):
        self.node_dict[ens_name] = wfe.EnsembleNode
        self.model_dict[ens_name] = (SklearnEnsembleWrapper, {'cls' : ens_cls})

    def add_model(self, model_name, model_cls):
        self.node_dict[model_name] = wfe.ModelNode
        self.model_dict[model_name] = (model_cls, None)

    def add_data_processor(self, prep_name, prep_cls):
        self.node_dict[prep_name] = wfe.DataProcessNode
        self.model_dict[prep_name] = (prep_cls, None)


class SklearnEnsembleWrapper:
    def __init__(self, cls, **arg_dict):
        self.cls = cls
        self.arg_dict = arg_dict
        self.ens = None

    def accept_list(self, child_list):
        if not len(child_list):
            raise ValueError("No base estimator provided to the ensemble.")  # TODO specific

        # TODO check for absence of parameters (suitable exception)

        if len(child_list) == 1:
            self.ens = self.cls(**self.arg_dict, base_estimator=child_list[0])
            return

        self.ens = self.cls(**self.arg_dict, estimators=child_list)

    def fit(self, X, y, sample_weight=None):
        if self.ens is None:
            raise ValueError("No base estimators provided")  # TODO specific

        return self.ens.fit(X, y, sample_weight)

    def predict(self, X):
        if self.ens is None:
            raise ValueError("No base estimators provided")  # TODO specific

        return self.ens.predict(X)


def default_config():
    node_dict = {}

    node_dict['_cEns'] = wfe.EnsembleExecNode
    return ModelConfig(node_dict)
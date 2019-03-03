# -*- coding: utf-8 -*-

import inspect
from sklearn.base import BaseEstimator


class Workflow:
    def fit(self, X, y, sample_weight=None):
        # TODO args
        pass

    def predict(self, X):
        # TODO args
        pass


class WorkflowNodeBase(BaseEstimator):
    def __init__(self, out_type, child_list, obj=None):
        self.out_type = out_type
        self.child_list = child_list
        self.obj = obj

    def fit(self, X, y, sample_weight=None):
        pass

    def predict(self, X):
        pass


# TODO check for obj None everywhere


class EnsembleExecNode(WorkflowNodeBase):
    def __init__(self, child_list, obj=None):
        super().__init__('out', child_list, obj)

        self.ens = None
        self.data_p = None

        for child in child_list:
            if not isinstance(child, WorkflowNodeBase):
                raise ValueError("Child is not a workflow node.")

            if self.ens is None and child.out_type == 'ens':
                self.ens = child
                continue

            if self.data_p is None and child.out_type == 'data':
                self.data_p = child
                continue

            raise ValueError("Invalid child nodes.")  # TODO specific exception

    def fit(self, X, y, sample_weight=None):
        data = X
        if self.data_p is not None:
            data = self.data_p.fit(X, y, sample_weight)
        return self.ens.fit(data, y, sample_weight)

    def predict(self, X):
        data = X
        if self.data_p is not None:
            data = self.data_p.predict(X)
        return self.ens.predict(data)

    def __getattribute__(self, item):
        try:
            return super().__getattribute__(item)
        except AttributeError:
            ens_attr = getattr(self.ens, item)
            setattr(self, item, ens_attr)
            return ens_attr


class UnionExecNode(WorkflowNodeBase):
    def __init__(self, child_list, obj=None):
        super().__init__('data', child_list, obj)

        self.union = None
        self.data_p = None

        for child in child_list:
            if not isinstance(child, WorkflowNodeBase):
                raise ValueError("Child is not a workflow node.")

            if self.union is None and child.out_type == 'union':
                self.union = child
                continue

            if self.data_p is None and child.out_type == 'data':
                self.data_p = child
                continue

            raise ValueError("Invalid child node.")  # TODO specific exception

    def fit(self, X, y, sample_weight=None):
        processed = self.data_p.fit(X, y)
        return self.union.fit(processed, y)

    def predict(self, X):
        processed = self.data_p.predict(X)
        return self.union.predict(processed)


class EnsembleNode(WorkflowNodeBase):
    def __init__(self, child_list, obj=None):
        super().__init__('ens', child_list, obj)

        if not all(isinstance(child, WorkflowNodeBase) and child.out_type == 'out' for child in child_list):
            raise ValueError("Invalid child node.")  # TODO specific exception

        obj.accept_list(child_list)

    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            if 'sample_weight' not in inspect.signature(self.obj.fit).parameters:
                raise ValueError("Sample weight not supported.")
            return self.obj.fit(X, y, sample_weight=sample_weight)

        return self.obj.fit(X, y)

    def predict(self, X):
        return self.obj.predict(X)

    def __getattribute__(self, item):
        try:
            return super().__getattribute__(item)
        except AttributeError:
            obj_attr = getattr(self.obj, item)
            setattr(self, item, obj_attr)
            return obj_attr


class UnionNode(WorkflowNodeBase):
    def __init__(self, child_list, obj=None):
        super().__init__('union', child_list, obj)

        if not all(isinstance(child, WorkflowNodeBase) and
                   (child.out_type == 'data' or  child.out_type == 'out') for child in child_list):
            raise ValueError("Invalid child node.")  # TODO specific exception

        obj.accept_list(child_list)

    def fit(self, X, y, sample_weight=None):
        return self.obj.fit(X, y, sample_weight)

    def predict(self, X):
        return self.obj.predict(X)


class ModelNode(WorkflowNodeBase):
    def __init__(self, child_list, obj=None):
        super().__init__('ens', child_list, obj)

        # TODO list
        self.model = obj

    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            if 'sample_weight' not in inspect.signature(self.model.fit).parameters:
                raise ValueError("Sample weight not supported.")
            return self.model.fit(X, y, sample_weight=sample_weight)

        return self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def __getattribute__(self, item):
        try:
            return super().__getattribute__(item)
        except AttributeError:
            model_attr = getattr(self.model, item)
            setattr(self, item, model_attr)
            return model_attr


class DataProcessNode(WorkflowNodeBase):
    def __init__(self, child_list, obj=None):
        super().__init__('data', child_list, obj)

        self.sub_data_p = None

        for prep in child_list:
            if self.sub_data_p is not None:
                raise ValueError("Invalid child node.")  # TODO specific exception

            if not isinstance(prep, WorkflowNodeBase):
                raise ValueError("Invalid child node.")  # TODO specific exception

            self.sub_data_p = prep

    def fit(self, X, y, sample_weight=None):
        data = X
        if self.sub_data_p is not None:
            data = self.sub_data_p.fit(X, y, sample_weight)

        return self.obj.fit_transform(data, y)  # TODO does it copy?

    def predict(self, X):
        return self.obj.transform(X, copy=True)  # TODO copy or not?


def make_workflow(gp_tree, model_config):
    def get_workflow_node(node, child_list):
        return model_config.create_wf_node(node.name, node.obj_kwargs, child_list)

    return gp_tree.run_tree(get_workflow_node)

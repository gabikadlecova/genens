# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod


class Workflow:
    def fit(self, X, y, sample_weight=None):
        # TODO args
        pass

    def predict(self, X):
        # TODO args
        pass


class WorkflowNodeBase(ABC):
    def __init__(self, out_type):
        self.out_type = out_type

    @abstractmethod
    def fit(self, X, y, sample_weight=None):
        pass

    @abstractmethod
    def predict(self, X):
        pass


class EnsembleExecNode(WorkflowNodeBase):
    def __init__(self, child_list):
        super().__init__('out')

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

            raise ValueError("Invalid child nodes.") # TODO specific exception

    def fit(self, X, y, sample_weight=None):
        processed = self.data_p.fit(X, y, sample_weight)
        return self.ens.fit(processed, y, sample_weight)

    def predict(self, X):
        processed = self.data_p.predict(X)
        return self.ens.predict(processed)


class UnionExecNode(WorkflowNodeBase):
    def __init__(self, child_list):
        super().__init__('data')

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
    def __init__(self, obj, child_list):
        super().__init__('ens')

        if not all(isinstance(child, WorkflowNodeBase) and child.out_type == 'out' for child in child_list):
            raise ValueError("Invalid child node.")  # TODO specific exception

        obj.accept_list(child_list)
        self.obj = obj

    def fit(self, X, y, sample_weight=None):
        return self.obj.fit(X, y, sample_weight)

    def predict(self, X):
        return self.obj.predict(X)


class UnionNode(WorkflowNodeBase):
    def __init__(self, obj, child_list):
        super().__init__('union')

        if not all(isinstance(child, WorkflowNodeBase) and child.out_type == 'data' for child in child_list):
            raise ValueError("Invalid child node.")  # TODO specific exception

        obj.accept_list(child_list)
        self.obj = obj

    def fit(self, X, y, sample_weight=None):
        return self.obj.fit(X, y, sample_weight)

    def predict(self, X):
        return self.obj.predict(X)


class DataProcessNode(WorkflowNodeBase):
    def __init__(self, obj, child_list):
        super().__init__('data')

        self.sub_data_p = None
        for prep in child_list:
            if self.sub_data_p is not None:
                raise ValueError("Invalid child node.")  # TODO specific exception

            if not isinstance(prep, WorkflowNodeBase):
                raise ValueError("Invalid child node.")  # TODO specific exception

            self.sub_data_p = prep

        self.obj = obj

    def fit(self, X, y, sample_weight=None):
        data = X
        if self.sub_data_p is not None:
            data = self.sub_data_p.fit(X, y, sample_weight)

        return self.obj.fit_transform(data, y)

    def predict(self, X):
        return self.obj.transform(X, copy=True)  # TODO copy or not?


def make_workflow(gp_tree, model_dict):
    pass

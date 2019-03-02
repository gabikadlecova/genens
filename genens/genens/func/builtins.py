# -*- coding: utf-8 -*-


import genens.func.eval


class SklearnEnsembleWrapper:
    def __init__(self, cls, arg_dict):
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

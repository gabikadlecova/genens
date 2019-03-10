# -*- coding: utf-8 -*-

from sklearn.base import BaseEstimator

from deap.base import Toolbox


class GenensBase(BaseEstimator):
    def __init__(self):
        # accept config/load default config
        self._toolbox = Toolbox()

    def _setup_toolbox(self):
        pass
# -*- coding: utf-8 -*-

from genens.gp import types
from sklearn.base import BaseEstimator
from deap import base, tools, creator

import genens.gp.operators as ops


class GenensBase(BaseEstimator):
    def __init__(self, config):
        # accept config/load default config
        self.config = config
        self._setup_toolbox()

    def _setup_toolbox(self):
        self._toolbox = base.Toolbox()

        creator.create("GenensFitness", base.Fitness, weights=(1.0, -1.0))
        creator.create("TreeIndividual", types.GpTreeIndividual, fitness=creator.GenensFitness)

        self._toolbox.register("individual", ops.gen_tree, self._toolbox, self.config)
        self._toolbox.register("population", tools.initIterate, list,
                               ops.gen_population, self._toolbox)
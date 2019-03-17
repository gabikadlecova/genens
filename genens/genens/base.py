# -*- coding: utf-8 -*-

from genens.gp import types
from genens.workflow.model_creation import create_workflow

from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score

from deap import base, tools, creator
from functools import partial, wraps

import time
import numpy as np
import warnings
import genens.gp.operators as ops


class GenensBase(BaseEstimator):
    def __init__(self, config, cx_pb=0.5, mut_pb=0.1, scorer=None, hof_size=5):
        """

        :param config:
        :param cx_pb:
        :param mut_pb:
        :param scorer:
        :param default_score:
        """
        # accept config/load default config
        self.config = config

        self.cx_pb = cx_pb
        self.mut_pb = mut_pb

        self.scorer = scorer

        self._fitness_eval = FitnessEvaluator()
        self._hof = tools.HallOfFame(hof_size)  # TODO similar
        self._setup_toolbox()

    def _setup_toolbox(self):
        self._toolbox = base.Toolbox()

        creator.create("GenensFitness", base.Fitness, weights=(1.0, -1.0))
        creator.create("TreeIndividual", types.GpTreeIndividual, fitness=creator.GenensFitness)

        self._toolbox.register("individual", ops.gen_tree, self.config)

        pop_func = partial(ops.gen_population, self._toolbox, 10)  # TODO size!!!!
        self._toolbox.register("population", tools.initIterate, list, pop_func)

        self._toolbox.register("select", tools.selNSGA2)
        self._toolbox.register("mutate_subtree", ops.mutate_subtree, self._toolbox)
        self._toolbox.register("cx_one_point", ops.crossover_one_point)

        self._toolbox.register("compile", create_workflow, config_dict=self.config.func_config)
        self._toolbox.register("evaluate", self._eval_tree_individual)  # TODO maybe more params

    def _eval_tree_individual(self, gp_tree):

        wf = self._toolbox.compile(gp_tree)
        return self._fitness_eval.score(wf, self.scorer)

    def fit(self, train_X, train_Y):
        self._fitness_eval.fit(train_X, train_Y)

        pop = self._toolbox.population()
        ops.ea_run(pop, self._toolbox, 10, 10, self.cx_pb, self.mut_pb)  # TODO sizes!


def eval_time(fn):
    @wraps(fn)
    def with_time(*args, **kwargs):
        start_time = time.time()

        res = fn(*args, **kwargs)
        if res is None:
            return None

        elapsed_time = time.time() - start_time
        return res, elapsed_time

    return with_time


class FitnessEvaluator:
    def __init__(self):
        self.train_X = None
        self.train_Y = None

    def fit(self, train_X, train_Y):
        self.train_X = train_X
        self.train_Y = train_Y

    @eval_time
    def score(self, workflow, scorer=None):
        # TODO not fitted

        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')

                scores = cross_val_score(workflow, self.train_X, self.train_Y,
                                         cv=7, scoring=scorer)

                return np.mean(scores)
        # TODO think of a better exception handling
        except Exception as e:
            # TODO log exception
            return None

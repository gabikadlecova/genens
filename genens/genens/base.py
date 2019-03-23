# -*- coding: utf-8 -*-

from genens.gp import types
from genens.workflow.model_creation import create_workflow

from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score

from deap import base, tools, creator
from functools import partial, wraps

import math
import time
import numpy as np
import warnings
import genens.gp.operators as ops


class GenensBase(BaseEstimator):
    def __init__(self, config, cx_pb=0.5, mut_pb=0.1, scorer=None, hof_size=5, pop_size=100,
                 n_gen=10):
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
        self.pop_size = pop_size
        self.n_gen = n_gen

        self.scorer = scorer

        self._fitness_eval = FitnessEvaluator()
        self.pareto = tools.ParetoFront()

        self._setup_log()
        self._setup_toolbox()

    def _setup_toolbox(self):
        self._toolbox = base.Toolbox()

        creator.create("GenensFitness", base.Fitness, weights=(1.0, -1.0))
        creator.create("TreeIndividual", types.GpTreeIndividual, fitness=creator.GenensFitness)

        self._toolbox.register("individual", ops.gen_tree, self.config)

        pop_func = partial(ops.gen_population, self._toolbox)
        self._toolbox.register("population", tools.initRepeat, list, pop_func)

        self._toolbox.register("select", tools.selNSGA2)
        self._toolbox.register("mutate_subtree", ops.mutate_subtree, self._toolbox)
        self._toolbox.register("cx_one_point", ops.crossover_one_point)

        self._toolbox.register("compile", create_workflow, config_dict=self.config.func_config)
        self._toolbox.register("evaluate", self._eval_tree_individual)
        self._toolbox.register("log", self._log_pop_stats)

    def _setup_log(self):
        score_stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        self._mstats = tools.MultiStatistics(score=score_stats)

        self._mstats.register("avg", np.mean)
        self._mstats.register("std", np.std)
        self._mstats.register("min", np.min)
        self._mstats.register("max", np.max)

        self.logbook = tools.Logbook()

        self.logbook.header = "gen", "score"
        self.logbook.chapters["score"].header = "min", "avg", "max"

    def _eval_tree_individual(self, gp_tree):

        wf = self._toolbox.compile(gp_tree)
        return self._fitness_eval.score(wf, self.scorer)

    def _log_pop_stats(self, population, gen_id):
        self.pareto.update(population)

        record = self._mstats.compile(population)
        self.logbook.record(gen=gen_id, **record)

        # TODO if test_X and test_Y provided, compute these as well

    def fit(self, train_X, train_Y):
        self._fitness_eval.fit(train_X, train_Y)
        self.pareto.clear()

        pop = self._toolbox.population(n=self.pop_size)
        ops.ea_run(pop, self._toolbox, self.n_gen, self.pop_size, self.cx_pb, self.mut_pb)


def eval_time(fn):
    @wraps(fn)
    def with_time(*args, **kwargs):
        start_time = time.time()

        res = fn(*args, **kwargs)
        if res is None:
            return None

        # TODO modify time computation
        elapsed_time = np.log(time.time() - start_time + np.finfo(float).eps)
        return res, elapsed_time

    return with_time


class FitnessEvaluator:
    def __init__(self):
        self.train_X = None
        self.train_y = None

    def fit(self, train_X, train_y):
        self.train_X = train_X
        self.train_y = train_y

    @eval_time
    def score(self, workflow, scorer=None):
        if self.train_X is None or self.train_y is None:
            raise ValueError("Evaluator is not fitted with training data.")  # TODO specific

        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')

                scores = cross_val_score(workflow, self.train_X, self.train_y,
                                         cv=7, scoring=scorer)

                return np.mean(scores)
        # TODO think of a better exception handling
        except Exception as e:
            # TODO log exception
            return None

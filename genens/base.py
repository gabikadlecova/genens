# -*- coding: utf-8 -*-

"""
This module contains the base estimator of Genens.
"""

from .gp.operators import gen_tree
from .gp.operators import gen_individual
from .gp.operators import mutate_subtree
from .gp.operators import mutate_node_args
from .gp.operators import mutate_node_swap
from .gp.operators import crossover_one_point
from .gp.operators import ea_run

from .workflow.evaluate import CrossValEvaluator, TrainTestEvaluator, default_score
from .workflow.model_creation import create_workflow

from sklearn.base import BaseEstimator, is_classifier

from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

from deap import base, tools
from functools import partial
from joblib import delayed

import logging
import numpy as np


class GenensBase(BaseEstimator):
    def __init__(self, config, n_jobs=1, cx_pb=0.5, mut_pb=0.3, mut_args_pb=0.6,
                 mut_node_pb=0.3, scorer=None, pop_size=200,
                 n_gen=15, hc_repeat=0, hc_keep_last=False, weighted=True, use_groups=True, max_height=None,
                 max_arity=None, timeout=None, evaluator=None):
        """
        Creates a new Genens estimator.

        :param GenensConfig config: Configuration of Genens.
        :param n_jobs: The n_jobs parameter for the process of evolution.
        :param cx_pb: Crossover probability.
        :param mut_pb: Subtree mutation probability.
        :param mut_args_pb: Argument mutation probability.
        :param mut_node_pb: Point (node) mutation probability.
        :param scorer: Scorer to be used during all evaluation (must comply to scikit-learn scorer API).
        :param pop_size: Population size.
        :param n_gen: Number of generations.
        :param hc_repeat: Number of hill-climbing iteration. If set to 0, hill-climbing is not performed.
        :param hc_keep_last:
            Whether the last individual should be mutated if the hill-climbing did not find a better individual.

        :param weighted: Determines whether the selection of nodes is weighted (according to groups).

        :param bool use_groups: If false, primitive groups are ignored.
        :param max_height: Maximum height of tree individuals.
        :param max_arity: Maximum arity of all nodes.
        :param timeout: Timeout for a single method evaluation.
        :param evaluator: Evaluator to be used (see genens.worflow.evaluate)
        """

        # accept config/load default config
        self.config = config
        if max_height is not None:
            config.max_height = max_height

        if max_arity is not None:
            config.max_arity = max_arity

        self.n_jobs = n_jobs

        self.cx_pb = cx_pb
        self.mut_pb = mut_pb
        self.mut_args_pb = mut_args_pb
        self.mut_node_pb = mut_node_pb

        self.pop_size = pop_size
        self.n_gen = n_gen

        self.hc_repeat = hc_repeat
        self.hc_keep_last = hc_keep_last
        self.weighted = weighted
        self.use_groups = use_groups

        self.scorer = scorer

        self._timeout = timeout
        self._fitness_evaluator = evaluator if evaluator is not None \
            else CrossValEvaluator(timeout_s=timeout)
        self._fitness_evaluator.timeout = timeout

        self.test_evaluator = None

        self.pareto = tools.ParetoFront()
        self.fitted_wf = None
        self._population = None

        self._setup_log()
        self._setup_toolbox()

    def _setup_toolbox(self):
        self._toolbox = base.Toolbox()

        self._toolbox.register("individual", gen_tree, self.config,
                               weighted=self.weighted, use_groups=self.use_groups)

        pop_func = partial(gen_individual, self._toolbox, self.config)
        self._toolbox.register("population", tools.initRepeat, list, pop_func)

        self._toolbox.register("map", _map_parallel)

        self._toolbox.register("select", tools.selNSGA2)
        self._toolbox.register("mutate_subtree", mutate_subtree, self._toolbox)
        self._toolbox.register("mutate_node_args", mutate_node_args, self._toolbox, self.config,
                               hc_repeat=self.hc_repeat, keep_last=self.hc_keep_last)
        self._toolbox.register("mutate_node_swap", mutate_node_swap, self.config)
        self._toolbox.register("cx_one_point", crossover_one_point)

        self._toolbox.register("next_gen", self._prepare_next_gen)
        self._toolbox.register("compile", self._compile_pipe)
        self._toolbox.register("evaluate", self._eval_tree_individual)
        self._toolbox.register("log", self._log_pop_stats)

    def _setup_log(self):
        score_stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        test_stats = tools.Statistics(lambda ind: self._compute_test(ind))

        self._mstats = tools.MultiStatistics(score=score_stats, test_score=test_stats)

        self._mstats.register("avg", np.mean)
        self._mstats.register("std", np.std)
        self._mstats.register("min", np.min)
        self._mstats.register("max", np.max)

        self.logbook = tools.Logbook()

        self.logbook.header = "gen", "score", "test_score"
        self.logbook.chapters["score"].header = "min", "avg", "max", "std"
        self.logbook.chapters["test_score"].header = "min", "avg", "max", "std"

    def _compile_pipe(self, ind):
        if ind.compiled_pipe is not None:
            return ind.compiled_pipe

        return create_workflow(ind, self.config.func_config)

    def _eval_tree_individual(self, gp_tree):
        wf = self._toolbox.compile(gp_tree)
        return self._fitness_evaluator.score(wf, scorer=self.scorer)

    def _prepare_next_gen(self):
        self._fitness_evaluator.reset()

    @property
    def can_log_score(self):
        return self.test_evaluator is not None

    def setup_test_stats(self, train_X, train_y, test_X, test_y):
        self.test_evaluator = TrainTestEvaluator(test_X, test_y, timeout_s=self._timeout)
        self.test_evaluator.fit(train_X, train_y)

    def _compute_test(self, ind):
        if not self.can_log_score:
            return 0.0  # TODO, warn

        if ind.test_stats is not None:
            return ind.test_stats

        wf = self._toolbox.compile(ind)
        res = self.test_evaluator.score(wf, scorer=self.scorer)

        ind.test_stats = res
        return res if res is not None else 0.0  # TODO

    def _log_pop_stats(self, population, gen_id):
        self.pareto.update(population)

        record = self._mstats.compile(population)
        self.logbook.record(gen=gen_id, **record)

    def get_best_pipelines(self, as_individuals=False):
        check_is_fitted(self, 'is_fitted_')

        if as_individuals:
            [self._compute_test(ind) for ind in self.pareto]
            return self.pareto
        else:
            return list(map(self._toolbox.compile, self.pareto))

    def fit(self, train_X, train_y):
        train_X, train_y = check_X_y(train_X, train_y, accept_sparse=True)

        if is_classifier(self):
            self.classes_ = unique_labels(train_y)

        self._fitness_evaluator.fit(train_X, train_y)
        self.pareto.clear()

        self._population = self._toolbox.population(n=self.pop_size)
        ea_run(self._population, self._toolbox, n_gen=self.n_gen, pop_size=self.pop_size, cx_pb=self.cx_pb,
               mut_pb=self.mut_pb,
               mut_args_pb=self.mut_args_pb, mut_node_pb=self.mut_node_pb, n_jobs=self.n_jobs)

        # TODO change later
        tree = self.pareto[0]
        self.fitted_wf = self._toolbox.compile(tree)
        self.fitted_wf.fit(train_X, train_y)

        self.is_fitted_ = True
        return self

    def predict(self, test_X):
        test_X = check_array(test_X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')

        # TODO clf/regr specific

        res = self.fitted_wf.predict(test_X)
        return res

    def score(self, test_X, test_y):
        test_X, test_y = check_X_y(test_X, test_y, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')

        if self.scorer is not None:
            s = self.scorer(self.fitted_wf, test_X, test_y)
        else:
            s = default_score(self.fitted_wf, test_X, test_y)

        return s


def _map_parallel(func, population, parallel=None):
    if parallel is None:
        return list(map(func, population))

    return parallel(delayed(func)(ind) for ind in population)

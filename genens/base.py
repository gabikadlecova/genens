# -*- coding: utf-8 -*-

"""
This module contains the base estimator of Genens.
"""


import numpy as np
import os
import warnings

from genens.gp.tree import gen_tree
from genens.gp.evolution import gen_individual
from genens.gp.mutation import mutate_subtree, mutate_gradual_hillclimbing
from genens.gp.mutation import mutate_args
from genens.gp.mutation import mutate_node_swap
from genens.gp.crossover import crossover_one_point
from genens.gp.evolution import ea_run

from genens.log_utils import set_log_handler, GenensLogger

from genens.workflow.evaluate import CrossValEvaluator, TrainTestEvaluator, default_score
from genens.workflow.model_creation import create_workflow

from deap import base, tools
from functools import partial
from joblib import delayed, Parallel

from sklearn.base import BaseEstimator, is_classifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


file_dir = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOGGING_CONFIG = file_dir + '/.logging_config.json'


class GenensBase(BaseEstimator):
    def __init__(self, config, n_jobs=1, scorer=None, pop_size=100,
                 n_gen=15, weighted=True, use_groups=True, timeout=None, evaluator=None,
                 logging_config=None, log_path=None, disable_logging=True, max_evo_seconds=None):

        """
        Creates a new Genens estimator.

        :param GenensConfig config: Configuration of Genens.
        :param n_jobs: The n_jobs parameter for the process of evolution.
        :param scorer: Scorer to be used during all evaluation (must comply to scikit-learn scorer API).
        :param pop_size: Population size.
        :param n_gen: Number of generations.

        :param weighted: Determines whether the selection of nodes is weighted (according to groups).

        :param bool use_groups: If false, primitive groups are ignored.
        :param timeout: Timeout for a single method evaluation.
        :param evaluator: Evaluator to be used (see genens.workflow.evaluate)

        """

        # accept config/load default config
        self.config = config
        self.n_jobs = n_jobs

        self.pop_size = pop_size
        self.n_gen = n_gen

        self.weighted = weighted
        self.use_groups = use_groups

        self.scorer = scorer

        self._fitness_evaluator = evaluator if evaluator is not None else CrossValEvaluator(timeout_s=timeout)
        self._fitness_evaluator.timeout = timeout
        if timeout is not None and evaluator is not None:
            warnings.warn("Setting timeout on a different evaluator than the default. If there was a timeout set"
                          " before, it has been discarded. Set timeout=None when instantiating this class to avoid"
                          " this behavior.")

        self.max_evo_seconds = max_evo_seconds

        self.pareto = tools.ParetoFront()
        self.fitted_workflow = None
        self._population = None

        logging_config = logging_config if logging_config is not None else DEFAULT_LOGGING_CONFIG
        self.logger = GenensLogger(logging_config, log_file_name=log_path, n_jobs=self.n_jobs,
                                   disable_logging=disable_logging)

        self._setup_stats_logging()
        self._setup_toolbox()

    def _setup_arg_mut(self):
        mut_func = partial(mutate_args, self.config,
                           multiple_nodes=self.config.mut_multiple_nodes,
                           multiple_args=self.config.mut_multiple_args)

        self._toolbox.register("gradual_hillclimbing", mutate_gradual_hillclimbing,
                               self._toolbox, self.config,
                               hc_repeat=self.config.hc_repeat, keep_last=self.config.hc_keep_last)
        self._toolbox.register("mutate_args", mut_func)

    def _setup_toolbox(self):
        self._toolbox = base.Toolbox()

        self._toolbox.register("individual", gen_tree, self.config,
                               weighted=self.weighted, use_groups=self.use_groups)

        pop_func = partial(gen_individual, self._toolbox, self.config)
        self._toolbox.register("population", tools.initRepeat, list, pop_func)

        self._toolbox.register("map", _map_parallel)

        self._toolbox.register("select", tools.selNSGA2)
        self._toolbox.register("mutate_subtree", mutate_subtree, self._toolbox)
        self._setup_arg_mut()
        self._toolbox.register("mutate_node_swap", mutate_node_swap, self.config)
        self._toolbox.register("cx_one_point", crossover_one_point)

        self._toolbox.register("compile", self._compile_pipe)
        self._toolbox.register("evaluate", self._eval_tree_individual)

        self._toolbox.register("update", self._update_population)
        self._toolbox.register("log_setup", self.logger.setup_child_logging)

    def _setup_stats_logging(self):
        score_stats = tools.Statistics(lambda ind: ind.fitness.values[0])

        self._mstats = tools.MultiStatistics(score=score_stats)

        self._mstats.register("avg", np.mean)
        self._mstats.register("std", np.std)
        self._mstats.register("min", np.min)
        self._mstats.register("max", np.max)

        self.logbook = tools.Logbook()

        self.logbook.header = "gen", "score"
        self.logbook.chapters["score"].header = "min", "avg", "max", "std"

    def _update_population(self, population, gen_i):
        self._fitness_evaluator.reset()
        self._log_pop_stats(population, gen_i)

    def _compile_pipe(self, ind):
        if ind.compiled_pipe is not None:
            return ind.compiled_pipe

        return create_workflow(ind, self.config.func_config)

    def _eval_tree_individual(self, gp_tree):
        wf = self._toolbox.compile(gp_tree)
        return self._fitness_evaluator.score(wf, scorer=self.scorer)

    def _log_pop_stats(self, population, gen_id):
        self.pareto.update(population)

        record = self._mstats.compile(population)
        self.logbook.record(gen=gen_id, **record)

    def get_best_pipelines(self, as_individuals=False):
        check_is_fitted(self, 'is_fitted_')

        if as_individuals:
            return self.pareto

        return list(map(self._toolbox.compile, self.pareto))

    def fit(self, train_X, train_y, verbose=1):
        train_X, train_y = check_X_y(train_X, train_y, accept_sparse=True)

        if is_classifier(self):
            self.classes_ = unique_labels(train_y)

        self._fitness_evaluator.fit(train_X, train_y)
        self.pareto.clear()

        self._population = self._toolbox.population(n=self.pop_size)

        log_context = None
        try:
            log_context = self.logger.listen()

            ea_run(self._population, self._toolbox, self.n_gen, self.pop_size, self.config,
                   n_jobs=self.n_jobs, timeout=self.max_evo_seconds, verbose=verbose)
        finally:
            self.logger.close(log_context)

        if not len(self.pareto):
            # try to get individuals that were evaluated before time limit end
            evaluated_inds = [ind for ind in self._population if ind.fitness.valid]

            if not len(evaluated_inds):
                warnings.warn("The algorithm did not have enough time to evaluate first generation and was not fitted.")
                return self

            self.pareto.update(evaluated_inds)

        tree = self.pareto[0]
        self.fitted_workflow = self._toolbox.compile(tree)
        self.fitted_workflow.fit(train_X, train_y)

        self.is_fitted_ = True
        return self

    def predict(self, test_X):
        test_X = check_array(test_X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')

        return self.fitted_workflow.predict(test_X)

    def score(self, test_X, test_y):
        test_X, test_y = check_X_y(test_X, test_y, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')

        if self.scorer is not None:
            s = self.scorer(self.fitted_workflow, test_X, test_y)
        else:
            s = default_score(self.fitted_workflow, test_X, test_y)  # TODO remove this one

        return s


def _map_parallel(func, population, n_jobs=1, **kwargs):
    return Parallel(n_jobs=n_jobs, **kwargs)(delayed(func)(ind) for ind in population)

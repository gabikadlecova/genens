# -*- coding: utf-8 -*-

from genens.gp import types
from genens.workflow.model_creation import create_workflow

from sklearn.base import BaseEstimator, is_classifier

from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from deap import base, tools, creator
from functools import partial, wraps
from joblib import delayed

import time
import numpy as np
import warnings
import genens.gp.operators as ops


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
    def __init__(self, cv_k=7):
        self.train_X = None
        self.train_y = None

        if cv_k < 0:
            raise AttributeError("Cross validation k must be greater than 0.")

        if cv_k == 0:
            self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(self.train_X, self.train_y,
                                                                                    test_size=0.33)
        
        self.cv_k = cv_k

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

                if self.cv_k > 0:
                    scores = cross_val_score(workflow, self.train_X, self.train_y,
                                             cv=self.cv_k, scoring=scorer)
                else:
                    workflow.fit(self.train_X, self.train_y)
                    scores = scorer(workflow, self.test_X, self.test_y)

                return np.mean(scores)
        # TODO think of a better exception handling
        except Exception as e:
            # TODO log exception
            return None


class GenensBase(BaseEstimator):
    def __init__(self, config, n_jobs=1, cx_pb=0.5, mut_pb=0.1, mut_args_pb=0.3, scorer=None,
                 pop_size=100, n_gen=10, hc_repeat=0, hc_keep_last=False, max_height=None,
                 max_arity=None, evaluator=FitnessEvaluator()):
        """
        TODO all parameters

        :param config:
        :param cx_pb:
        :param mut_pb:
        :param scorer:
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

        self.pop_size = pop_size
        self.n_gen = n_gen

        self.hc_repeat = hc_repeat
        self.hc_keep_last = hc_keep_last

        self.scorer = scorer

        self._fitness_eval = evaluator
        self.pareto = tools.ParetoFront()

        self.fitted_wf = None

        self.train_X = None
        self.train_Y = None
        self.test_X = None
        self.test_Y = None
        self.can_log_score = False

        self._setup_log()
        self._setup_toolbox()

    def _setup_toolbox(self):
        self._toolbox = base.Toolbox()

        self._toolbox.register("individual", ops.gen_tree, self.config)

        pop_func = partial(ops.gen_population, self._toolbox, self.config)
        self._toolbox.register("population", tools.initRepeat, list, pop_func)

        self._toolbox.register("map", _map_parallel)

        self._toolbox.register("select", tools.selNSGA2)
        self._toolbox.register("mutate_subtree", ops.mutate_subtree, self._toolbox)
        self._toolbox.register("mutate_node_args", ops.mutate_node_args, self._toolbox, self.config,
                               hc_repeat=self.hc_repeat, keep_last=self.hc_keep_last)
        self._toolbox.register("cx_one_point", ops.crossover_one_point)

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
        return self._fitness_eval.score(wf, self.scorer)

    def set_test_stats(self, train_X, train_Y, test_X, test_Y):
        self.train_X = train_X
        self.train_Y = train_Y
        self.test_X = test_X
        self.test_Y = test_Y

        self.can_log_score = train_X is not None and train_Y is not None \
                             and test_X is not None and test_Y is not None

    def _compute_test(self, ind):
        if not self.can_log_score:
            return 0.0  # TODO

        if ind.test_stats is not None:
            return ind.test_stats

        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')

                wf = self._toolbox.compile(ind)
                wf.fit(self.train_X, self.train_Y)

                if self.scorer is not None:
                    res = self.scorer(wf, self.test_X, self.test_Y)
                else:
                    res = wf.score(self.test_X, self.test_Y)
        # TODO
        except Exception as e:
            return 0.0  # TODO

        ind.test_stats = res
        return res

    def _log_pop_stats(self, population, gen_id):
        self.pareto.update(population)

        record = self._mstats.compile(population)
        self.logbook.record(gen=gen_id, **record)

    def get_best_pipelines(self):
        check_is_fitted(self, 'is_fitted_')

        return list(map(self._toolbox.compile, self.pareto))

    def fit(self, train_X, train_Y):
        train_X, train_Y = check_X_y(train_X, train_Y, accept_sparse=True)

        if is_classifier(self):
            self.classes_ = unique_labels(train_Y)

        self._fitness_eval.fit(train_X, train_Y)
        self.pareto.clear()

        pop = self._toolbox.population(n=self.pop_size)
        ops.ea_run(pop, self._toolbox, self.n_gen, self.pop_size, self.cx_pb, self.mut_pb,
                   self.mut_args_pb, n_jobs=self.n_jobs)

        # TODO change later
        tree = self.pareto[0]
        self.fitted_wf = self._toolbox.compile(tree)
        self.fitted_wf.fit(train_X, train_Y)

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
            s = self.scorer(self.fitted_wf, self.test_X, self.test_Y)
        else:
            s = self.fitted_wf.score(self.test_X, self.test_Y)

        return s


def _map_parallel(func, population, parallel=None):
    if parallel is None:
        return list(map(func, population))

    return parallel(delayed(func)(ind) for ind in population)

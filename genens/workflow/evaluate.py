# -*- coding: utf-8 -*-

"""
This module provides evaluators that can be used as fitness evaluators in Genens.
"""

from abc import ABC, abstractmethod
from functools import wraps

import logging
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

from stopit import ThreadingTimeout as Timeout, TimeoutException

import time
import warnings


def timeout(fn):
    @wraps(fn)
    def with_timeout(self, workflow, *args, **kwargs):
        if not hasattr(self, 'timeout') or self.timeout is None:
            return fn(self, workflow, *args, **kwargs)

        try:
            with Timeout(self.timeout, swallow_exc=False):
                res = fn(self, workflow, *args, **kwargs)
        except TimeoutException:
            logger = logging.getLogger("genens")
            logger.debug(f"Timeouted:\n {workflow}")
            print(f"Timeouted:\n {workflow}")

            res = None

        return res
    return with_timeout


def eval_time(fn):
    @wraps(fn)
    def with_time(self, workflow, *args, **kwargs):
        start_time = time.process_time_ns()

        res = fn(self, workflow, *args, **kwargs)
        if res is None or np.isnan(res):
            return None

        elapsed_time = time.process_time_ns() - start_time

        logger = logging.getLogger("genens")
        logger.debug(f"Score: {res}, evaluation time - {elapsed_time / 10e9}s,\n{workflow}")

        return res, np.log(elapsed_time + np.finfo(float).eps)

    return with_time


def default_score(workflow, test_X, test_y):
    """
    Default scoring method for evaluators.

    :param workflow: Workflow to be evaluated.
    :param test_X: Features.
    :param test_y: Target vector.
    :return: Accuracy score of ``workflow`` on the data.
    """
    res_y = workflow.predict(test_X)
    return accuracy_score(test_y, res_y)


def _simple_eval(workflow, train_X, train_y, test_X, test_y, scorer=None):
    workflow.fit(train_X, train_y)
    if scorer is not None:
        return scorer(workflow, test_X, test_y)

    return default_score(workflow, test_X, test_y)


class EvaluatorBase(ABC):
    """
    Evaluator interface which is used in Genens during individual evaluation.
    """
    def __init__(self, timeout_s=None):
        self.train_X = None
        self.train_y = None

        self.timeout = timeout_s

    def __repr__(self):
        res = "{}".format(self.__class__.__name__)
        return res

    def fit(self, train_X, train_y):
        """
        Sets the training data for evaluation.

        :param train_X: Features.
        :param train_y: Target vector.
        """
        self.train_X = train_X
        self.train_y = train_y

    @abstractmethod
    def evaluate(self, workflow, scorer=None):
        """
        Evaluates a pipeline (workflow) learned on the training data from this evaluator. Evaluation method
        differs between different evaluators.
        :param workflow: Workflow to be evaluated.
        :param scorer: Scorer to be used, default is the accuracy scorer.
        :return: Score of the workflow.
        """
        pass

    @abstractmethod
    def reset(self):
        pass

    def check_is_fitted(self):
        if self.train_X is None or self.train_y is None:
            raise ValueError("Evaluator is not fitted with training data.")  # TODO specific

    @timeout
    @eval_time
    def score(self, workflow, scorer=None):
        """
        Computes the score of a pipeline (workflow).
        :param workflow: Workflow to be evaluated.
        :param scorer: Scorer to be used, default is the accuracy scorer.
        :return: Score of the workflow.
        """
        self.check_is_fitted()

        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')

                return self.evaluate(workflow, scorer=scorer)
        except Exception as e:
            logger = logging.getLogger("genens")
            logger.debug(f"Workflow failed:\n {e}\nWorkflow: {workflow}.")
            return None


class CrossValEvaluator(EvaluatorBase):
    """
    Evaluator which performs a k-fold cross-validation.
    """
    def __init__(self, cv_k=7, timeout_s=None):
        super().__init__(timeout_s)

        if cv_k < 0:
            raise AttributeError("Cross validation k must be greater than 0.")

        self.cv_k = cv_k

    def __repr__(self):
        res = super().__repr__()
        res += ", cv_k: {}".format(self.cv_k)
        return res

    def evaluate(self, workflow, scorer=None):
        scores = cross_val_score(workflow, self.train_X, self.train_y,
                                 cv=self.cv_k, scoring=scorer, error_score='raise')
        return np.mean(scores)

    def reset(self):
        pass


class FixedTrainTestEvaluator(EvaluatorBase):
    """
    Evaluator that uses fixed train and test samples.
    """
    def __init__(self, test_size=None, random_state=None, timeout_s=None):
        super().__init__(timeout_s)

        self.test_X = None
        self.test_y = None

        self.test_size = test_size
        self.random_state = random_state

    def __repr__(self):
        res = super().__repr__()
        res += ", test_size: {}".format(self.test_size)
        return res

    def fit(self, train_X, train_y):
        self.train_X, self.test_X, self.train_y, self.test_y = \
            train_test_split(train_X, train_y, test_size=self.test_size,
                             random_state=self.random_state)

    def check_is_fitted(self):
        super().check_is_fitted()
        if self.test_X is None or self.test_y is None:
            raise ValueError("Test data missing in evaluator.")

    def evaluate(self, workflow, scorer=None):
        return _simple_eval(workflow, self.train_X, self.train_y, self.test_X, self.test_y,
                            scorer=scorer)

    def reset(self):
        pass


class RandomTrainTestEvaluator(EvaluatorBase):
    """
    Evaluator which performs the train-test split for every evaluation.
    """
    def __init__(self, test_size=None, random_state=None, timeout_s=None):
        super().__init__(timeout_s)

        self.test_size = test_size

        self.rng = None
        if random_state is not None:
            if isinstance(random_state, np.random.RandomState):
                self.rng = random_state

            self.rng = np.random.RandomState(random_state)

    def __repr__(self):
        res = super().__repr__()
        res += ", test_size: {}".format(self.test_size)
        return res

    def evaluate(self, workflow, scorer=None):
        # random state is set only in the constructor
        train_X, test_X, train_y, test_y = train_test_split(self.train_X, self.train_y,
                                                            test_size=self.test_size,
                                                            random_state=self.rng)
        return _simple_eval(workflow, train_X, train_y, test_X, test_y,
                            scorer=scorer)

    def reset(self):
        pass


class TrainTestEvaluator(EvaluatorBase):
    """
    Evaluator with a specified validation set.
    """
    def __init__(self, test_X, test_y, timeout_s=None):
        super().__init__(timeout_s)

        # training set is provided in fit, validation set is set on initialization
        self.test_X = test_X
        self.test_y = test_y

    def evaluate(self, workflow, scorer=None):
        return _simple_eval(workflow, self.train_X, self.train_y, self.test_X, self.test_y,
                            scorer=scorer)

    def reset(self):
        pass


class DataSampler:
    """
    Provides random samples of a given dataset.
    """
    def __init__(self, sample_size=0.20, random_state=None, stratified=True):
        self.full_X = None
        self.full_y = None

        self.sample_size = sample_size
        self.stratified = stratified

        self.rng = None
        if random_state is not None:
            if isinstance(random_state, np.random.RandomState):
                self.rng = random_state

            self.rng = np.random.RandomState(random_state)

    def fit(self, full_X, full_y):
        self.full_X = full_X
        self.full_y = full_y

    def generate_sample(self):
        """
        Generates a random sample from full data.
        :return: A random sample.
        """
        stratify = self.full_y if self.stratified else None

        _, sample_X, _, sample_y = train_test_split(self.full_X, self.full_y,
                                                    test_size=self.sample_size,
                                                    random_state=self.rng,
                                                    stratify=stratify)
        return sample_X, sample_y


class SampleCrossValEvaluator(CrossValEvaluator):
    """
    Evaluator which uses a k-fold cross-validation based on sampling.
    """
    def __init__(self, cv_k=7, timeout_s=None, sample_size=0.20, per_gen=False, random_state=None):
        super().__init__(cv_k=cv_k, timeout_s=timeout_s)

        self.per_gen = per_gen
        self.sampler = DataSampler(sample_size=sample_size, random_state=random_state)

    def __repr__(self):
        res = super().__repr__()
        res += ", sample_size: {}".format(self.sampler.sample_size)
        res += ", per_gen: {}".format(self.per_gen)
        return res

    def _reset_data(self):
        self.train_X, self.train_y = self.sampler.generate_sample()

    def fit(self, train_X, train_y):
        self.sampler.fit(train_X, train_y)
        self._reset_data()

    def evaluate(self, workflow, scorer=None):
        # generate sample for every evaluation
        if not self.per_gen:
            self._reset_data()

        return super().evaluate(workflow, scorer=scorer)

    def reset(self):
        # generate sample once per reset
        if self.per_gen:
            self._reset_data()


class SampleTrainTestEvaluator(FixedTrainTestEvaluator):
    """
    Evaluator which uses a train-test evaluation on samples.
    """
    def __init__(self, test_size=None, timeout_s=None, sample_size=0.20,
                 per_gen=False, random_state=None):

        self.per_gen = per_gen
        self.sampler = DataSampler(sample_size=sample_size, random_state=random_state)

        # rng should be set in DataSampler ctor
        super().__init__(test_size=test_size, random_state=self.sampler.rng, timeout_s=timeout_s)

    def __repr__(self):
        res = super().__repr__()
        res += ", sample_size: {}".format(self.sampler.sample_size)
        res += ", per_gen: {}".format(self.per_gen)
        return res

    def _fit_sample(self):
        train_X, train_y = self.sampler.generate_sample()

        super().fit(train_X, train_y)

    def fit(self, train_X, train_y):
        self.sampler.fit(train_X, train_y)

        self._fit_sample()

    def evaluate(self, workflow, scorer=None):
        # one sample per evaluation
        if not self.per_gen:
            self._fit_sample()

        return super().evaluate(workflow, scorer=scorer)

    def reset(self):
        # generate new fixed sample
        if self.per_gen:
            self._fit_sample()


_eval_names = {
    'crossval': CrossValEvaluator,
    'fixed': FixedTrainTestEvaluator,
    'per_ind': RandomTrainTestEvaluator,
    'train_test': TrainTestEvaluator,
    'sample_crossval': SampleCrossValEvaluator,
    'sample_train_test': SampleTrainTestEvaluator
}


def get_evaluator_cls(name):
    return _eval_names[name]

# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

from stopit import ThreadingTimeout as Timeout, TimeoutException
from functools import wraps

import warnings
import time


def timeout(fn):
    @wraps(fn)
    def with_timeout(self, *args, **kwargs):
        if not hasattr(self, 'timeout') or self.timeout is None:
            return fn(self, *args, **kwargs)

        try:
            with Timeout(self.timeout, swallow_exc=False):
                res = fn(self, *args, **kwargs)
        except TimeoutException:
            # TODO log cause
            res = None

        return res
    return with_timeout


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


def _simple_eval(workflow, train_X, train_y, test_X, test_y, scorer=None):
    workflow.fit(train_X, train_y)
    if scorer is not None:
        return scorer(workflow, test_X, test_y)

    res_y = workflow.predict(test_X)
    return accuracy_score(test_y, res_y)


class EvaluatorBase(ABC):
    def __init__(self, timeout_s=None):
        self.train_X = None
        self.train_y = None

        self.timeout = timeout_s

    def fit(self, train_X, train_y):
        self.train_X = train_X
        self.train_y = train_y

    @abstractmethod
    def evaluate(self, workflow, scorer=None):
        pass

    def check_is_fitted(self):
        if self.train_X is None or self.train_y is None:
            raise ValueError("Evaluator is not fitted with training data.")  # TODO specific

    @timeout
    @eval_time
    def score(self, workflow, scorer=None):
        self.check_is_fitted()

        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')

                return self.evaluate(workflow, scorer)
        # TODO think of a better exception handling
        except Exception as e:
            # TODO log exception
            return None


class CrossvalEvaluator(EvaluatorBase):
    def __init__(self, cv_k=7, timeout_s=None):
        super().__init__(timeout_s)

        if cv_k < 0:
            raise AttributeError("Cross validation k must be greater than 0.")

        self.cv_k = cv_k

    def evaluate(self, workflow, scorer=None):
        scores = cross_val_score(workflow, self.train_X, self.train_y,
                                 cv=self.cv_k, scoring=scorer)
        return np.mean(scores)


class FixedSampleEvaluator(EvaluatorBase):
    def __init__(self, test_size=None, random_state=None, timeout_s=None):
        super().__init__(timeout_s)

        self.test_X = None
        self.test_y = None

        self.test_size = test_size
        self.random_state=random_state

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


class RandomSampleEvaluator(EvaluatorBase):
    def __init__(self, test_size=None, random_state=None, timeout_s=None):
        super().__init__(timeout_s)

        self.test_size = test_size

        if random_state is not None:
            np.random.seed(random_state)

    def evaluate(self, workflow, scorer=None):
        # random state is set only in the constructor
        train_X, test_X, train_y, test_y = train_test_split(self.train_X, self.train_y,
                                                            test_size=self.test_size)
        return _simple_eval(workflow, train_X, train_y, test_X, test_y,
                            scorer=scorer)


class TrainTestEvaluator(EvaluatorBase):
    def __init__(self, test_X, test_y, timeout_s=None):
        super().__init__(timeout_s)

        # training set is provided in fit, validation set is set on initialization
        self.test_X = test_X
        self.test_y = test_y

    def evaluate(self, workflow, scorer=None):
        return _simple_eval(workflow, self.train_X, self.train_y, self.test_X, self.test_y,
                            scorer=scorer)


_eval_names = {
    'crossval': CrossvalEvaluator,
    'fixed': FixedSampleEvaluator,
    'samplePerInd': RandomSampleEvaluator
}


def get_evaluator_cls(name):
    return _eval_names[name]
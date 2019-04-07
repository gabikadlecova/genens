# -*- coding: utf-8 -*-

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


from functools import wraps
import threading
import warnings
import time


def timeout(fn):
    @wraps(fn)
    def with_timeout(*args, **kwargs):
        # todo if self.timeout absent, warning and skip

        res = []

        def save_res():
            r = fn(*args, **kwargs)
            res.append(r)

        thread = threading.Thread(target=save_res)
        thread.start()

        thread.join(args[0].timeout)  # must be self

        if thread.is_alive():
            return None

        return res[0]

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


class CrossvalEvaluator:
    def __init__(self, cv_k=7, timeout_s=None):
        self.train_X = None
        self.train_y = None

        self.timeout = 5 * 60 if timeout_s is None else timeout_s

        if cv_k < 0:
            raise AttributeError("Cross validation k must be greater than 0.")

        self.cv_k = cv_k

    def fit(self, train_X, train_y):
        self.train_X = train_X
        self.train_y = train_y

    @timeout
    @eval_time
    def score(self, workflow, scorer=None):
        if self.train_X is None or self.train_y is None:
            raise ValueError("Evaluator is not fitted with training data.")  # TODO specific

        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')

                scores = cross_val_score(workflow, self.train_X, self.train_y,
                                             cv=self.cv_k, scoring=scorer)
                return np.mean(scores)
        # TODO think of a better exception handling
        except Exception as e:
            # TODO log exception
            return None


class FixedSampleEvaluator:
    def __init__(self, test_size=0.25, random_state=42, timeout_s=None):
        self.train_X = None
        self.train_y = None

        self.timeout = 5 * 60 if timeout_s is None else timeout_s

        self.test_size = test_size
        self.random_state=random_state

    def fit(self, train_X, train_y):
        self.train_X, self.test_X, self.train_y, self.test_y = \
            train_test_split(train_X, train_y, test_size=self.test_size, random_state=self.random_state)

    @timeout
    @eval_time
    def score(self, workflow, scorer=None):
        if self.train_X is None or self.train_y is None:
            raise ValueError("Evaluator is not fitted with training data.")  # TODO specific

        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')

                workflow.fit(self.train_X, self.train_y)
                scores = scorer(workflow, self.test_X, self.test_y)

                return np.mean(scores)
        except Exception as e:
            # TODO log exception
            return None


class RandomSampleEvaluator:
    def __init__(self, test_size=0.25, random_state=42, timeout_s=None):
        self.train_X = None
        self.train_y = None

        self.timeout = 8 * 60 if timeout_s is None else timeout_s

        self.test_size = test_size

        np.random.seed(random_state)

    def fit(self, train_X, train_y):
        self.train_X = train_X
        self.train_y = train_y

    @timeout
    @eval_time
    def score(self, workflow, scorer=None):
        if self.train_X is None or self.train_y is None:
            raise ValueError("Evaluator is not fitted with training data.")  # TODO specific

        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')

                train_X, test_X, train_y, test_y = \
                    train_test_split(self.train_X, self.train_y, test_size=self.test_size)

                workflow.fit(train_X, train_y)
                scores = scorer(workflow, test_X, test_y)

                return np.mean(scores)
        except Exception as e:
            # TODO log exception
            return None


_eval_names = {
    'crossval': CrossvalEvaluator,
    'fixed': FixedSampleEvaluator,
    'perInd': RandomSampleEvaluator
}


def get_evaluator_cls(name):
    return _eval_names[name]
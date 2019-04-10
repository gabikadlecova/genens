# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils import resample

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

    def __repr__(self):
        res = "{}".format(__class__.__name__)
        return res

    def fit(self, train_X, train_y):
        self.train_X = train_X
        self.train_y = train_y

    @abstractmethod
    def evaluate(self, workflow, scorer=None):
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
        self.check_is_fitted()

        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')

                return self.evaluate(workflow, scorer)
        # TODO think of a better exception handling
        except Exception as e:
            # TODO log exception
            return None


class CrossValEvaluator(EvaluatorBase):
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
                                 cv=self.cv_k, scoring=scorer)
        return np.mean(scores)

    def reset(self):
        pass


class FixedTrainTestEvaluator(EvaluatorBase):
    def __init__(self, test_size=None, random_state=None, timeout_s=None):
        super().__init__(timeout_s)

        self.test_X = None
        self.test_y = None

        self.test_size = test_size
        self.random_state = random_state

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
    def __init__(self, test_size=None, random_state=None, timeout_s=None):
        super().__init__(timeout_s)

        self.test_size = test_size

        self.rng = None
        if random_state is not None:
            self.rng = np.random.RandomState(random_state)

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
    def __init__(self, sample_size=0.20, random_state=None, replace=False):
        self.full_X = None
        self.full_y = None

        self.sample_size = sample_size

        self.replace = replace

        self.rng = None
        if random_state is not None:
            self.rng = np.random.RandomState(random_state)

    def fit(self, full_X, full_y):
        self.full_X = full_X
        self.full_y = full_y

    def generate_sample(self):
        return resample(self.full_X, self.full_y, replace=self.replace,
                        n_samples=self.sample_size * self.full_X.shape[0],
                        random_state=self.rng)


class SampleCrossValEvaluator(CrossValEvaluator):
    def __init__(self, cv_k=7, timeout_s=None, sample_size=0.20, per_gen=True, random_state=None,
                 replace=False):
        super().__init__(cv_k=cv_k, timeout_s=timeout_s)

        self.per_gen = per_gen
        self.sampler = DataSampler(sample_size=sample_size, random_state=random_state,
                                   replace=replace)

    def __repr__(self):
        res = super().__repr__()
        res += ", sample_size: {}".format(self.sampler.sample_size)
        res += ", per_gen: {}".format(self.per_gen)
        res += ", replace: {}".format(self.sampler.replace)
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


_eval_names = {
    'crossval': CrossValEvaluator,
    'fixed': FixedTrainTestEvaluator,
    'per_ind': RandomTrainTestEvaluator,
    'train_test': TrainTestEvaluator,
    'sample_crossval': SampleCrossValEvaluator
}


def get_evaluator_cls(name):
    return _eval_names[name]
import openml

import numpy as np
import pickle
import time

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import os

from genens import GenensClassifier
from genens.workflow.evaluate import SampleCrossValEvaluator
from genens.config.clf_default import create_clf_config
from genens.render.plot import export_plot
from genens.render.graph import create_graph


def _heuristic_sample_size(n_rows, n_cols):
    size = n_rows * n_cols

    # 'small' datasets
    if size < 10000:
        return 1.0

    if n_rows < 1000:
        return 0.5

    if n_rows < 5000:
        return 0.25

    if n_rows < 10000:
        return 0.1

    if n_rows < 20000:
        return 0.05

    if n_cols < 10:
        return 0.02

    return 0.01


def _conditional_imput(X, categorical):
    categorical_id = [i for i, val in enumerate(categorical) if val]
    numerical_id = [i for i, val in enumerate(categorical) if not val]

    imputer = ColumnTransformer(
        transformers=[
            ('mean_imputer', SimpleImputer(strategy='mean'), numerical_id),
            ('median_imputer', SimpleImputer(strategy='median'), categorical_id),
        ]
    )

    return imputer


def run_task(task, n_jobs=1, timeout=None, task_timeout=None, random_state=42):
    dataset = task.get_dataset()

    X, y, categorical = dataset.get_data(
        dataset_format='array',
        target=dataset.default_target_attribute,
        return_categorical_indicator=True,
    )

    imputer = _conditional_imput(X, categorical)
    X = imputer.fit_transform(X)

    sample_size = _heuristic_sample_size(X.shape[0], X.shape[1])
    evaluator = SampleCrossValEvaluator(cv_k=5, timeout_s=timeout, per_gen=False,
                                        sample_size=sample_size)

    # TODO bad practice warning - hardcoded height
    clf = GenensClassifier(
        n_jobs=n_jobs,
        timeout=timeout,
        evaluator=evaluator,
        max_height=8
    )

    clf.fit(X, y)

    # run on top 5 pipes

    return clf


def run_tests(test_ids, n_jobs=1, timeout=None, task_timeout=None):
    benchmark_suite = openml.study.get_study('OpenML-CC18', 'tasks')

    for task_id in benchmark_suite.tasks:
        task = openml.tasks.get_task(task_id)

        fitted_clf = run_task(task, n_jobs=n_jobs, timeout=timeout, task_timeout=task_timeout)



if __name__ == '__main__':
    pass

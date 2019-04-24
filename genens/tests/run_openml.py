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


def evaluate_pipeline(pipe, task, out_dir):
    pass


def _create_dir_check(out_dir):
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        # skip test
        print("Test directory {} already exists.".format(out_dir))
        return False
    except OSError as e:
        print("\nCannot create test directory.")
        raise e

    return True


def _log_evolution(fitted_clf, out_dir):
    # write logbook string representation to output dir
    with open(out_dir + '/logbook.txt', 'w+') as log_file:
        log_file.write(fitted_clf.logbook.__str__() + '\n')

    # evolution plot
    export_plot(fitted_clf, out_dir + '/result.png')

    # top 5 individual fitness values
    with open(out_dir + '/ind-fitness.txt', 'w+') as out_file:
        for i, ind in enumerate(fitted_clf.get_best_pipelines(as_individuals=True)[:5]):
            out_file.write('Individual {}: Score {}, Test {}\n'.format(i, ind.fitness.values,
                                                                       ind.test_stats))
            create_graph(ind, out_dir + '/graph{}.png'.format(i))

    # pickle top 5 best pipelines
    for i, pipe in enumerate(fitted_clf.get_best_pipelines()[:5]):
        with open(out_dir + '/pipeline{}.pickle'.format(i), 'wb') as pickle_file:
            pickle.dump(pipe, pickle_file, pickle.HIGHEST_PROTOCOL)


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


def run_task(task, out_dir, n_jobs=1, timeout=None, task_timeout=None):
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

    _log_evolution(clf, out_dir)

    # run top 5 pipelines
    for i, pipe in enumerate(clf.get_best_pipelines()[:5]):
        pipe_dir = out_dir + '/run{}'.format(i)
        _create_dir_check(pipe_dir)

        evaluate_pipeline(pipe, task, pipe_dir)


def run_tests(test_ids, out_dir='.', n_jobs=1, timeout=None, task_timeout=None):
    benchmark_suite = openml.study.get_study('OpenML-CC18', 'tasks')

    for task_id in benchmark_suite.tasks:
        task = openml.tasks.get_task(task_id)
        dataset = task.get_dataset()

        # create output directory
        dataset_dir = '/{}'.format(dataset.name)

        # skip existing directories
        if not _create_dir_check(dataset_dir):
            continue

        run_task(task,
                 out_dir=out_dir + dataset_dir,
                 n_jobs=n_jobs,
                 timeout=timeout,
                 task_timeout=task_timeout)


if __name__ == '__main__':
    pass

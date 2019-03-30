# -*- coding: utf-8 -*-
import argparse

import inspect
import json

import itertools

import os

import matplotlib.pyplot as plt

from genens import GenensClassifier, GenensRegressor
from genens.render.plot import export_plot
from genens.render.graph import create_graph
from tests.datasets.load_datasets import load_dataset


def run_tests(estimators, train_X, train_y, test_X, test_y, out_dir):
    for i, (est, kwarg_dict) in enumerate(estimators):
        try:
            test_dir = out_dir + '/{}'.format(i)
            os.mkdir(test_dir)
        except FileExistsError:
            # skip finished tests
            continue
        except OSError as e:
            print("Cannot create directory for test {}".format(i))
            raise e
        else:
            print("Running test {}".format(i))

        run_once(est, train_X, train_y, test_X, test_y, kwarg_dict, test_dir)


def run_once(estimator, train_X, train_y, test_X, test_y, kwarg_dict, out_dir):
    estimator.set_test_stats(train_X, train_y, test_X, test_y)

    estimator.fit(train_X, train_y)
    test_score = estimator.score(train_X, train_y)
    print('Test score: {}'.format(test_score))

    with open(out_dir + '/settings.txt', 'w+') as out_file:
        out_file.write(str(kwarg_dict))

    with open(out_dir + '/logbook.txt', 'w+') as log_file:
        log_file.write(estimator.logbook.__str__())

    with open(out_dir + '/pipelines.txt', 'w+') as out_file:
        for pipe in estimator.get_best_pipelines():
            out_file.write(str(pipe))
            out_file.write('\n\n')

    export_plot(estimator, out_dir + '/result.png')

    for i, ind in enumerate(estimator.pareto):
        create_graph(ind, out_dir + 'graph{}.png'.format(i))


def load_config(cmd_args):
    # read config
    with open(args.file) as f:
        config = json.load(f)

    dataset = config['dataset']
    train_X, train_Y, test_X, test_Y = load_dataset(dataset)

    params = config['parameters']

    def product_dict(**kwargs):
        keys = kwargs.keys()
        for values in itertools.product(*kwargs.values()):
            yield dict(zip(keys, values))

    param_product = product_dict(**params)

    def clf_iterate(prod):
        for kwargs in prod:
            if cmd_args.regression:
                yield GenensRegressor(**kwargs), kwargs
            else:
                yield GenensClassifier(**kwargs), kwargs

    run_tests(clf_iterate(param_product), train_X, train_Y, test_X, test_Y, cmd_args.out)


def load_from_args(cmd_args):
    arg_dict = vars(cmd_args)

    if cmd_args.regression:
        cls = GenensRegressor
    else:
        cls = GenensClassifier

    cls_params = inspect.signature(cls.__init__).parameters
    kwargs = dict((key, val) for key, val in arg_dict.items() if key in cls_params)

    dataset = cmd_args.dataset
    train_X, train_Y, test_X, test_Y = load_dataset(dataset)

    run_once(cls(**kwargs), train_X, train_Y, test_X, test_Y, kwargs, cmd_args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run Genens on datasets.")
    parser.add_argument('--out', required=True)
    parser.add_argument('--regression', action='store_true')

    subparsers = parser.add_subparsers()

    parser_config = subparsers.add_parser('config')
    parser_config.add_argument('file', type=str, help='Configuration file location.')

    parser_config.set_defaults(func=load_config)

    parser_run = subparsers.add_parser('run')

    parser_run.add_argument('--dataset', type=str, help='Dataset name.', required=True)
    parser_run.add_argument('--n_jobs', type=int, default=1, help='Sets n_jobs of Genens.')
    parser_run.add_argument('--cx_pb', type=float, default=0.5, help='Sets crossover probability.')
    parser_run.add_argument('--mut_pb', type=float, default=0.1, help='Sets mutation probability.')

    parser_run.add_argument('--mut_args_pb', type=float, default=0.3,
                            help='Sets argument mutation probability.')

    parser_run.add_argument('--scorer', type=str, help='Sets a custom scorer.')
    parser_run.add_argument('--pop_size', type=int, default=100, help='Sets population size.')

    parser_run.add_argument('--n_gen', type=int, default=10,
                            help='Sets the number of generations.')

    parser_run.add_argument('--hc_repeat', type=int, default=0,
                            help='Sets the number of hill-climbing repetitions.')

    parser_run.add_argument('--hc_keep_last', action='store_true',
                            help="If true, mutates the individual if hill-climbing hasn't been successful.")

    parser_run.set_defaults(func=load_from_args)

    args = parser.parse_args()
    args.func(args)

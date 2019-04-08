# -*- coding: utf-8 -*-
import argparse

import inspect
import json

import itertools

import os

from sklearn.metrics import make_scorer

import time

from genens import GenensClassifier, GenensRegressor
from genens.workflow.evaluate import get_evaluator_cls
from genens.config.clf_default import create_clf_config
from genens.render.plot import export_plot
from genens.render.graph import create_graph
from tests.datasets.load_datasets import load_dataset


def run_tests(estimators, train_X, train_y, test_X, test_y, out_dir):
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        print("Test directory already exists.")
    except OSError as e:
        print("\nCannot create test directory.")
        raise e

    for i, (est, kwarg_dict) in enumerate(estimators):
        try:
            test_dir = out_dir + '/{}'.format(i)
            os.mkdir(test_dir)
        except FileExistsError:
            # skip finished tests
            print("\nTest {} skipped.".format(i))
            continue
        except OSError as e:
            print("\nCannot create directory for test {}".format(i))
            raise e
        else:
            print("\nRunning test {}".format(i))

        run_once(est, train_X, train_y, test_X, test_y, kwarg_dict, test_dir)


def run_once(estimator, train_X, train_y, test_X, test_y, kwarg_dict, out_dir):
    # start test time measurement
    start_time = time.time()

    estimator.setup_test_stats(train_X, train_y, test_X, test_y)

    estimator.fit(train_X, train_y)
    test_score = estimator.score(test_X, test_y)

    # end test time measurement
    elapsed_time = time.time() - start_time

    print('Test score: {}'.format(test_score))
    print('Test time: {}'.format(elapsed_time))

    with open(out_dir + '/test-stats.txt', 'w+') as out_file:
        out_file.write('Test score: {}\n'.format(test_score))
        out_file.write('Test time: {}\n'.format(elapsed_time))

    with open(out_dir + '/settings.txt', 'w+') as out_file:
        out_file.write(str(kwarg_dict) + '\n')

    with open(out_dir + '/logbook.txt', 'w+') as log_file:
        log_file.write(estimator.logbook.__str__() + '\n')

    with open(out_dir + '/pipelines.txt', 'w+') as out_file:
        for pipe in estimator.get_best_pipelines():
            out_file.write(str(pipe))
            out_file.write('\n\n')

    export_plot(estimator, out_dir + '/result.png')

    with open(out_dir + '/ind-score.txt', 'w+') as out_file:
        for i, ind in enumerate(estimator.get_best_pipelines(as_individuals=True)):
            out_file.write('Individual {}: Score {}, Test {}\n'.format(i, ind.fitness.values,
                                                                       ind.test_stats))
            create_graph(ind, out_dir + '/graph{}.png'.format(i))


def create_scorer(scorer_path, scorer_kwargs):
    tmp_path = scorer_path.split('.')
    op_str = tmp_path.pop()
    import_str = '.'.join(tmp_path)

    try:
        exec('from {} import {}'.format(import_str, op_str))
    except ImportError as e:
        print("Could not import scoring function.")
        raise e

    cls = eval(op_str)

    return make_scorer(cls, **scorer_kwargs)


def load_config(cmd_args):
    # read config
    with open(cmd_args.file) as f:
        config = json.load(f)

    params = config['parameters']

    # set up scorer
    if 'scorer' in config.keys():
        func = config['scorer']['func']
        scorer_args = config['scorer']['kwargs']

        scorer = create_scorer(func, scorer_args)
        params['scorer'] = scorer

    def product_dict(**kwargs):
        """
        Makes a dictionary product; computes the product of all lists in the
        arguments, for every element of the product returns a dictionary of these
        arguments. If an argument isn't a list, it is added to every element
        of the product.
        :param kwargs: Keyword argument dictionary.
        :return: Keyword argument product.
        """
        only_lists = {key: val for key, val in kwargs.items() if isinstance(val, list)}
        other_args = {key: val for key, val in kwargs.items() if key not in only_lists.keys()}

        keys = only_lists.keys()
        for values in itertools.product(*only_lists.values()):
            yield dict(zip(keys, values), **other_args)

    def obj_kwargs_product(cls, kwargs_dict):
        for kwargs in product_dict(**kwargs_dict):
            yield cls(**kwargs)

    if 'evaluator' in config.keys():
        eval_cls = get_evaluator_cls(config['evaluator']['func'])
        params['evaluator'] = list(obj_kwargs_product(eval_cls, config['evaluator']['kwargs']))

    if 'group_weights' in config.keys():
        params['config'] = [create_clf_config(gweight) for gweight in
                            product_dict(**config['group_weights'])]

    param_product = product_dict(**params)

    def clf_iterate(param_prod):
        for kwargs in param_prod:
            if cmd_args.regression:
                yield GenensRegressor(**kwargs), {**kwargs}
            else:
                yield GenensClassifier(**kwargs), {**kwargs}

    datasets = config['datasets']

    for dataset in datasets:
        if 'split_validation' in dataset and dataset['split_validation']:
            train_X, train_Y, test_X, test_Y = load_dataset(**dataset)
            run_tests(clf_iterate(param_product), train_X, train_Y, test_X, test_Y,
                      cmd_args.out + '/' + dataset['dataset_name'])
        else:
            raise ValueError("TODO TODO TODO")


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

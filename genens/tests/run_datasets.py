# -*- coding: utf-8 -*-

import argparse

import inspect
import json

import itertools

import os

from sklearn.metrics import make_scorer

import pickle
import time

from genens import GenensClassifier, GenensRegressor
from genens.workflow.evaluate import get_evaluator_cls
from genens.config import clf_config
from genens.render.plot import export_plot
from genens.render.graph import create_graph
from .datasets.load_datasets import load_dataset


def run_tests(estimators, train_X, train_y, out_dir, test_X=None, test_y=None):
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

        run_once(est, train_X, train_y, kwarg_dict, test_dir, test_X=test_X, test_y=test_y)


def evaluate_workflows(clf, features, target, scorer=None, method='crossval', **kwargs):
    wfs = clf.get_best_pipelines()[:5]

    cls = get_evaluator_cls(method)

    eval = cls(**kwargs)
    eval.fit(features, target)

    s = clf.scorer if scorer is None else scorer
    return [eval.score(wf, s) for wf in wfs]


def run_once(estimator, train_X, train_y, kwarg_dict, out_dir, test_X=None, test_y=None):
    # start test time measurement
    start_time = time.time()

    if test_X is not None and test_y is not None:
        estimator.setup_test_stats(train_X, train_y, test_X, test_y)

    estimator.fit(train_X, train_y)

    # end test time measurement
    elapsed_time = time.time() - start_time

    print('Evolution time: {}'.format(elapsed_time))

    # test the pipelines
    with open(out_dir + '/pipe-eval.txt', 'w+') as out_file:
        # default evaluation methods
        if test_X is not None and test_y is not None:
            res = evaluate_workflows(estimator, train_X, train_y, method='train_test',
                                     test_X=test_X, test_y=test_y)
        else:
            res = evaluate_workflows(estimator, train_X, train_y, method='crossval', cv_k=10)

        for i, val in enumerate(res):
            out_file.write("Pipeline {} score: {}\n".format(i, val))

    # write config to output dir
    with open(out_dir + '/config.txt', 'w+') as out_file:
        out_file.write(str(kwarg_dict) + '\n')

    # write logbook string representation to output dir
    with open(out_dir + '/logbook.txt', 'w+') as log_file:
        log_file.write('Test time: {}\n\n'.format(elapsed_time))
        log_file.write(estimator.logbook.__str__() + '\n')

    # pickle best pipelines and print to file
    with open(out_dir + '/pipelines.txt', 'w+') as out_file:
        for i, pipe in enumerate(estimator.get_best_pipelines()[:5]):
            with open(out_dir + '/pipeline{}.pickle'.format(i), 'wb') as pickle_file:
                pickle.dump(pipe, pickle_file, pickle.HIGHEST_PROTOCOL)

            # todo check with wilt
            # pprint.pprint(pipe, out_file)
            out_file.write(repr(pipe))
            out_file.write('\n\n')

    # evolution plot
    export_plot(estimator, out_dir + '/result.png')

    # individual fitness values
    with open(out_dir + '/ind-fitness.txt', 'w+') as out_file:
        for i, ind in enumerate(estimator.get_best_pipelines(as_individuals=True)[:5]):
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
        params['config'] = [clf_config(gweight) for gweight in
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

            run_tests(clf_iterate(param_product), train_X, train_Y,
                      cmd_args.out + '/' + dataset['dataset_name'],
                      test_X=test_X, test_y=test_Y)
        else:
            features, target = load_dataset(**dataset)
            run_tests(clf_iterate(param_product), features, target,
                      cmd_args.out + '/' + dataset['dataset_name'])


def load_from_args(cmd_args):
    # TODO rewrite if needed (only for 'run' case)

    arg_dict = vars(cmd_args)

    if cmd_args.regression:
        cls = GenensRegressor
    else:
        cls = GenensClassifier

    cls_params = inspect.signature(cls.__init__).parameters
    kwargs = dict((key, val) for key, val in arg_dict.items() if key in cls_params)

    dataset = cmd_args.dataset
    train_X, train_Y, test_X, test_Y = load_dataset(dataset, split_validation=True)

    run_once(cls(**kwargs), train_X, train_Y, kwargs, cmd_args.out,
             test_X=test_X, test_y=test_Y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run Genens on datasets.")
    parser.add_argument('--out', required=True)
    parser.add_argument('--regression', action='store_true')

    subparsers = parser.add_subparsers()

    # pass arguments in a config file
    parser_config = subparsers.add_parser('config')
    parser_config.add_argument('file', type=str, help='Configuration file location.')

    parser_config.set_defaults(func=load_config)

    parser_run = subparsers.add_parser('run')

    # pass arguments on command line
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

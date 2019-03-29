# -*- coding: utf-8 -*-
import argparse
import configparser
import itertools

from genens.genens import GenensClassifier


def load_config(args):
    # subparsers func
    pass


def set_params(args):
    # subparsers func
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run Genens on datasets.")

    subparsers = parser.add_subparsers()

    parser_config = subparsers.add_parser('config')
    parser_config.add_argument('file', type=str, help='Configuration file location.')

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

    args = parser.parse_args()
    arg_dict = vars(args)

    if args.config is not None:
        # read config
        config = configparser.ConfigParser()

        config.read(args.config)

        params = config['Parameters']

        def product_dict(**kwargs):
            keys = kwargs.keys()
            for values in itertools.product(*kwargs.values()):
                yield dict(zip(keys, values))

        param_product = product_dict(params)

    # init via vars, setdiff, prefer config and warn
    # check if dir exists, skip; in shell "rerun"



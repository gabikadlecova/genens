# -*- coding: utf-8 -*-
import argparse
import configparser
import sys

from genens.genens import GenensClassifier

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run Genens on datasets.")

    parser.add_argument('--config', '-c', type=str, help='Configuration file location.')
    parser.add_argument('--dataset', type=str, help='Dataset name.', required=True)
    parser.add_argument('--n_jobs', type=int, default=1, help='Sets n_jobs of Genens.')
    parser.add_argument('--cx_pb', type=float, default=0.5, help='Sets crossover probability.')
    parser.add_argument('--mut_pb', type=float, default=0.1, help='Sets mutation probability.')

    parser.add_argument('--mut_args_pb', type=float, default=0.3,
                        help='Sets argument mutation probability.')

    parser.add_argument('--scorer', type=str, help='Sets a custom scorer.')
    parser.add_argument('--pop_size', type=int, default=100, help='Sets population size.')

    parser.add_argument('--n_gen', type=int, default=10,
                        help='Sets the number of generations.')

    parser.add_argument('--hc_repeat', type=int, default=0,
                        help='Sets the number of hill-climbing repetitions.')

    parser.add_argument('--hc_keep_last', action='store_true',
                        help="If true, mutates the individual if hill-climbing hasn't been successful.")

    args = parser.parse_args()
    arg_dict = vars(args)

    if args.config is not None:
        # read config
        pass

    # init via vars, override file config and warn
    # check if dir exists, skip; in shell "rerun"



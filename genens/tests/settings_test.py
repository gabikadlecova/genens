import argparse
import numpy as np
import random

from genens import GenensClassifier
from genens.workflow.evaluate import SampleCrossValEvaluator
from .datasets.load_datasets import load_dataset


def run_test(arg_dict):
    X, y = load_dataset(arg_dict['dataset'], random_state=arg_dict['random_state'])
    evaluator = SampleCrossValEvaluator(cv_k=arg_dict['cv_k'], sample_size=arg_dict['size'],
                                        per_gen=True, random_state=arg_dict['random_state'])

    np.random.seed(arg_dict['random_state'])
    random.seed(arg_dict['random_state'])

    clf = GenensClassifier(
        n_jobs=arg_dict['n_jobs'],
        cx_pb=arg_dict['cx_pb'],
        mut_pb=arg_dict['mut_pb'],
        mut_args_pb=arg_dict['mut_args_pb'],
        mut_node_pb=arg_dict['mut_node_pb'],
        pop_size=arg_dict['pop_size'],
        n_gen=arg_dict['n_gen'],
        weighted=arg_dict['weighted'],
        use_groups=arg_dict['use_groups'],
        max_arity=arg_dict['max_arity'],
        max_height=arg_dict['max_height'],
        timeout=arg_dict['timeout'],
        log_path=arg_dict['log_path']
    )

    clf.fit(X, y, verbose=2)
    return clf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--random_state', default=42, type=int)

    parser.add_argument('--n_jobs', default=-1, type=int)
    parser.add_argument('--pop_size', default=200, type=int)
    parser.add_argument('--n_gen', default=15, type=int)

    parser.add_argument('--max_height', default=4, type=int)
    parser.add_argument('--max_arity', default=3, type=int)

    parser.add_argument('--weighted', action='store_true')
    parser.add_argument('--use_groups', action='store_true')

    parser.add_argument('--cx_pb', default=0.5, type=float)
    parser.add_argument('--mut_pb', default=0.3, type=float)
    parser.add_argument('--mut_args_pb', default=0.6, type=float)
    parser.add_argument('--mut_node_pb', default=0.3, type=float)

    parser.add_argument('--cv_k', default=3, type=int)
    parser.add_argument('--size', default=0.25, type=float)
    parser.add_argument('--timeout', default=240, type=int)
    parser.add_argument('--log_path', default=None)

    args = parser.parse_args()

    run_test(vars(args))
